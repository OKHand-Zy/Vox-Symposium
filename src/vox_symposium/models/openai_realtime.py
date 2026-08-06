from __future__ import annotations

import asyncio
import base64
import json
from urllib.parse import urlencode, urlsplit, urlunsplit

from websockets.asyncio.client import ClientConnection, connect

from vox_symposium.audio import PcmAudio, normalize_audio
from vox_symposium.models.base import (
    QueueBackedRealtimeAudioModel,
    RealtimeInterruption,
    cancel_task,
)

_AUDIO_TRANSCRIPT_DELTA_EVENTS = {
    "response.audio_transcript.delta",
    "response.output_audio_transcript.delta",
}
_AUDIO_TRANSCRIPT_DONE_EVENTS = {
    "response.audio_transcript.done",
    "response.output_audio_transcript.done",
}


class OpenAIRealtimeModel(QueueBackedRealtimeAudioModel):
    input_sample_rate = 24_000
    output_sample_rate = 24_000

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        voice: str,
        instructions: str,
        backend: str = "openai",
        endpoint: str | None = None,
        api_version: str | None = None,
        reasoning_effort: str | None = None,
        ping_interval: float = 20.0,
        ping_timeout: float = 20.0,
        manual_activity: bool = False,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.backend = backend
        self.endpoint = endpoint
        self.api_version = api_version
        self.reasoning_effort = reasoning_effort
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.manual_activity = manual_activity
        self._ws: ClientConnection | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._last_audio_item_id: str | None = None

    async def connect(self) -> None:
        url, headers = self._connection_config()
        self._ws = await connect(
            url,
            additional_headers=headers,
            max_size=None,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
        )
        session = {
            "type": "realtime",
            "instructions": self.instructions,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.input_sample_rate,
                    },
                    "turn_detection": (
                        None
                        if self.manual_activity
                        else {
                            "type": "semantic_vad",
                            "interrupt_response": True,
                        }
                    ),
                },
                "output": {
                    "voice": self.voice,
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.output_sample_rate,
                    },
                },
            },
        }
        if self.reasoning_effort:
            session["reasoning"] = {
                "effort": self.reasoning_effort,
            }
        # Azure selects the deployment in the URL and doesn't accept a deployment
        # alias as session.model. The official endpoint keeps the existing behavior.
        if self.backend == "openai":
            session["model"] = self.model
        await self._send(
            {
                "type": "session.update",
                "session": session,
            }
        )
        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"openai-{self.model}-reader"
        )

    async def end_audio_turn(self) -> None:
        if not self.manual_activity:
            return
        await self._send({"type": "input_audio_buffer.commit"})
        await self._send({"type": "response.create"})

    def _connection_config(self) -> tuple[str, dict[str, str]]:
        if self.backend == "openai":
            query = urlencode({"model": self.model})
            return (
                f"wss://api.openai.com/v1/realtime?{query}",
                {"Authorization": f"Bearer {self.api_key}"},
            )
        if self.backend != "azure":
            raise RuntimeError(f"Unsupported OpenAI backend: {self.backend}")
        if not self.endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is required for the Azure OpenAI backend")

        parsed = urlsplit(self.endpoint.rstrip("/"))
        if parsed.scheme not in {"https", "wss"} or not parsed.netloc:
            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT must be an HTTPS URL, for example "
                "https://your-resource.openai.azure.com"
            )
        if parsed.query or parsed.fragment:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT must not contain a query string or fragment")
        scheme = "wss"
        base_path = parsed.path.rstrip("/")
        if self.api_version:
            path = f"{base_path}/openai/realtime"
            query = urlencode({"api-version": self.api_version, "deployment": self.model})
        else:
            path = f"{base_path}/openai/v1/realtime"
            query = urlencode({"model": self.model})
        url = urlunsplit((scheme, parsed.netloc, path, query, ""))
        return url, {"api-key": self.api_key}

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
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )

    async def truncate_response(
        self,
        event: RealtimeInterruption,
        *,
        audio_end_ms: int,
    ) -> None:
        if not event.item_id:
            return
        await self._send(
            {
                "type": "conversation.item.truncate",
                "item_id": event.item_id,
                "content_index": 0,
                "audio_end_ms": max(0, audio_end_ms),
            }
        )

    async def close(self) -> None:
        reader_task = self._reader_task
        self._reader_task = None
        await cancel_task(reader_task)
        ws = self._ws
        self._ws = None
        if ws:
            await ws.close()
        self.close_output_streams()

    async def _send(self, event: dict) -> None:
        if self._ws is None:
            raise RuntimeError("OpenAI realtime websocket is not connected")
        await self._ws.send(json.dumps(event))

    async def _read_loop(self) -> None:
        assert self._ws is not None
        audio_transcript_parts: dict[str, list[str]] = {}
        try:
            async for raw in self._ws:
                event = json.loads(raw)
                event_type = event.get("type")
                if event_type in {"response.output_audio.delta", "response.audio.delta"}:
                    delta = event.get("delta")
                    if delta:
                        item_id = event.get("item_id")
                        if item_id:
                            self._last_audio_item_id = str(item_id)
                        await self._audio_out.put(
                            PcmAudio(
                                data=base64.b64decode(delta),
                                sample_rate=self.output_sample_rate,
                                channels=1,
                            )
                        )
                elif event_type == "input_audio_buffer.speech_started":
                    await self._emit_event(
                        RealtimeInterruption(
                            item_id=self._last_audio_item_id,
                            response_id=_optional_event_id(event, "response_id"),
                        )
                    )
                    self._last_audio_item_id = None
                elif event_type in _AUDIO_TRANSCRIPT_DELTA_EVENTS:
                    delta = event.get("delta")
                    if delta:
                        audio_transcript_parts.setdefault(_event_text_key(event), []).append(delta)
                elif event_type in _AUDIO_TRANSCRIPT_DONE_EVENTS:
                    key = _event_text_key(event)
                    parts = audio_transcript_parts.pop(key, [])
                    text = event.get("transcript") or "".join(parts)
                    if text:
                        await self._text_out.put(text)
                elif event_type == "error":
                    raise RuntimeError(f"OpenAI realtime error: {event}")
        finally:
            self.close_output_streams()


def _event_text_key(event: dict) -> str:
    return "|".join(
        str(event.get(key, ""))
        for key in ("response_id", "item_id", "output_index", "content_index")
    )


def _optional_event_id(event: dict, key: str) -> str | None:
    value = event.get(key)
    return str(value) if value else None
