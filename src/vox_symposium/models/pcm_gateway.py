from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from websockets.asyncio.client import ClientConnection, connect

from vox_symposium.audio import PcmAudio, normalize_audio
from vox_symposium.models.base import QueueBackedRealtimeAudioModel


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PcmGatewayProtocol:
    session_event_type: str = "session.init"
    input_event_type: str = "input_audio_buffer.append"
    commit_event_type: str = "input_audio_buffer.commit"
    response_event_type: str = "response.create"
    close_event_type: str = "session.close"


class PcmGatewayRealtimeModel(QueueBackedRealtimeAudioModel):
    """OpenAI-like PCM16 JSON websocket adapter for self-hosted speech gateways."""

    def __init__(
        self,
        *,
        provider_name: str,
        url: str,
        instructions: str,
        model: str,
        api_key: str | None = None,
        input_sample_rate: int = 24_000,
        output_sample_rate: int = 24_000,
        manual_activity: bool = False,
        protocol: PcmGatewayProtocol | None = None,
    ) -> None:
        super().__init__()
        _validate_websocket_url(url, provider_name=provider_name)
        if input_sample_rate <= 0 or output_sample_rate <= 0:
            raise ValueError(f"{provider_name} sample rates must be positive")

        self.provider_name = provider_name
        self.url = url
        self.instructions = instructions
        self.model = model
        self.api_key = api_key
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.manual_activity = manual_activity
        self.protocol = protocol or PcmGatewayProtocol()
        self._ws: ClientConnection | None = None
        self._reader_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        self._ws = await connect(
            self.url,
            additional_headers=headers,
            max_size=None,
        )
        await self._send(self._session_event())
        self._reader_task = asyncio.create_task(
            self._read_loop(),
            name=f"{self.provider_name}-{self.model}-reader",
        )

    async def send_audio(self, audio: PcmAudio) -> None:
        pcm = normalize_audio(
            audio.data,
            from_rate=audio.sample_rate,
            to_rate=self.input_sample_rate,
            channels=audio.channels,
        )
        if not pcm:
            return
        await self._send(
            {
                "type": self.protocol.input_event_type,
                "audio": base64.b64encode(pcm).decode("ascii"),
                "format": "pcm16",
                "sample_rate": self.input_sample_rate,
                "channels": 1,
            }
        )

    async def end_audio_turn(self) -> None:
        if not self.manual_activity:
            return
        await self._send({"type": self.protocol.commit_event_type})
        await self._send({"type": self.protocol.response_event_type})

    async def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self._ws:
            try:
                await self._send({"type": self.protocol.close_event_type})
            except Exception:
                logger.debug("Failed to send %s close event", self.provider_name, exc_info=True)
            await self._ws.close()
        self.close_output_streams()

    def _session_event(self) -> dict[str, Any]:
        return {
            "type": self.protocol.session_event_type,
            "session": {
                "model": self.model,
                "instructions": self.instructions,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": "pcm16",
                        "sample_rate": self.input_sample_rate,
                        "channels": 1,
                    },
                    "output": {
                        "format": "pcm16",
                        "sample_rate": self.output_sample_rate,
                        "channels": 1,
                    },
                },
                "turn_detection": None if self.manual_activity else {"type": "server_vad"},
            },
        }

    async def _send(self, event: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError(f"{self.provider_name} websocket is not connected")
        await self._ws.send(json.dumps(event))

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    if _looks_like_json(raw):
                        event = json.loads(raw.decode("utf-8"))
                    else:
                        await self._audio_out.put(
                            PcmAudio(data=raw, sample_rate=self.output_sample_rate, channels=1)
                        )
                        continue
                else:
                    event = json.loads(raw)

                event_type = str(event.get("type") or "")
                if event_type == "error":
                    raise RuntimeError(f"{self.provider_name} realtime error: {event}")

                audio = _audio_from_event(event)
                if audio:
                    await self._audio_out.put(
                        PcmAudio(
                            data=audio,
                            sample_rate=_event_sample_rate(event, self.output_sample_rate),
                            channels=int(event.get("channels") or 1),
                        )
                    )

                text = _text_from_event(event)
                if text:
                    await self._text_out.put(text)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("%s realtime reader stopped with an error", self.provider_name)
        finally:
            self.close_output_streams()


def _validate_websocket_url(url: str, *, provider_name: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        raise ValueError(f"{provider_name} realtime URL must be a ws:// or wss:// URL")
    if parsed.hostname in {"0.0.0.0", "::"}:
        raise ValueError(
            f"{provider_name} realtime URL cannot use a wildcard address; "
            "use 127.0.0.1, a host name, or the gateway IP"
        )


def _audio_from_event(event: dict[str, Any]) -> bytes | None:
    is_audio_event = _is_audio_event(event)
    candidates = [
        event.get("audio"),
        event.get("delta"),
        _nested(event, "payload", "audio"),
        _nested(event, "payload", "delta"),
        _nested(event, "response", "audio"),
        _nested(event, "output", "audio"),
    ]
    if is_audio_event:
        candidates.append(event.get("data"))
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, dict):
            value = value.get("audio") or value.get("data") or value.get("delta")
        if isinstance(value, str):
            try:
                return base64.b64decode(value, validate=True)
            except Exception:
                logger.debug("Ignoring non-base64 audio payload from event: %s", event.get("type"))
                return None
        if isinstance(value, bytes):
            return value
    return None


def _text_from_event(event: dict[str, Any]) -> str | None:
    candidates = [
        event.get("text"),
        event.get("transcript"),
        _nested(event, "payload", "text"),
        _nested(event, "response", "text"),
        _nested(event, "output", "text"),
    ]
    if _is_text_event(event):
        candidates.append(event.get("data"))
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    return None


def _event_sample_rate(event: dict[str, Any], default: int) -> int:
    for key in ("sample_rate", "rate", "output_sample_rate"):
        value = event.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return default


def _nested(event: dict[str, Any], outer: str, inner: str) -> Any:
    value = event.get(outer)
    if isinstance(value, dict):
        return value.get(inner)
    return None


def _is_audio_event(event: dict[str, Any]) -> bool:
    event_type = str(event.get("type") or "").lower()
    kind = str(event.get("kind") or "").lower()
    return "audio" in event_type or kind == "audio"


def _is_text_event(event: dict[str, Any]) -> bool:
    event_type = str(event.get("type") or "").lower()
    kind = str(event.get("kind") or "").lower()
    return "text" in event_type or "transcript" in event_type or kind == "text"


def _looks_like_json(data: bytes) -> bool:
    stripped = data.lstrip()
    return stripped.startswith((b"{", b"["))
