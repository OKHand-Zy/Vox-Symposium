from __future__ import annotations

import asyncio
import base64
import json
import logging
from urllib.parse import urlsplit

from websockets.asyncio.client import ClientConnection, connect

from vox_symposium.audio import (
    PcmAudio,
    float32_to_pcm16,
    normalize_audio,
    pcm16_to_float32,
)
from vox_symposium.models.base import QueueBackedRealtimeAudioModel

logger = logging.getLogger(__name__)

EVALUATION_TURN_TAKING_POLICY = """Full-duplex speaking policy:
- The incoming audio is your conversation partner speaking directly to you.
- After the partner finishes a statement or question, respond immediately in spoken audio.
- Do not remain in listen mode after the partner has finished speaking."""


class MiniCPMRealtimeModel(QueueBackedRealtimeAudioModel):
    """MiniCPM-o 4.5 Audio Full-Duplex Gateway adapter."""

    input_sample_rate = 16_000
    output_sample_rate = 24_000

    def __init__(
        self,
        *,
        url: str,
        instructions: str,
        api_key: str | None = None,
        length_penalty: float = 1.1,
        input_chunk_ms: int = 1_000,
        queue_timeout: float = 300.0,
        ping_interval: float | None = 30.0,
        ping_timeout: float | None = 120.0,
        evaluation_turn_taking: bool = False,
    ) -> None:
        super().__init__()
        if not url.startswith(("ws://", "wss://")):
            raise ValueError("MiniCPM realtime URL must use ws:// or wss://")
        if urlsplit(url).hostname in {"0.0.0.0", "::"}:
            raise ValueError(
                "MiniCPM realtime URL cannot use a wildcard address; "
                "use 127.0.0.1, a host name, or the Gateway IP"
            )
        if input_chunk_ms <= 0:
            raise ValueError("MiniCPM input chunk duration must be positive")
        if queue_timeout <= 0:
            raise ValueError("MiniCPM queue timeout must be positive")
        if ping_interval is not None and ping_interval <= 0:
            raise ValueError("MiniCPM ping interval must be positive or None")
        if ping_timeout is not None and ping_timeout <= 0:
            raise ValueError("MiniCPM ping timeout must be positive or None")

        self.url = url
        self.instructions = instructions
        self.api_key = api_key
        self.length_penalty = length_penalty
        self.input_chunk_ms = input_chunk_ms
        self.queue_timeout = queue_timeout
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.evaluation_turn_taking = evaluation_turn_taking
        self.system_prompt = instructions.rstrip()
        if evaluation_turn_taking:
            self.system_prompt = f"{self.system_prompt}\n\n{EVALUATION_TURN_TAKING_POLICY}"
        self._input_chunk_bytes = int(self.input_sample_rate * input_chunk_ms / 1_000) * 4
        self._input_buffer = bytearray()
        self._ws: ClientConnection | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._silence_task: asyncio.Task[None] | None = None
        self._session_created = False
        self._response_active = False

    async def connect(self) -> None:
        if self._ws is not None:
            raise RuntimeError("MiniCPM realtime websocket is already connected")

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        self._ws = await connect(
            self.url,
            additional_headers=headers,
            max_size=128 * 1024 * 1024,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
        )
        try:
            await asyncio.wait_for(
                self._wait_for_event("session.queue_done"),
                timeout=self.queue_timeout,
            )
            await self._send(
                {
                    "type": "session.init",
                    "payload": {
                        "system_prompt": self.system_prompt,
                        "config": {"length_penalty": self.length_penalty},
                    },
                }
            )
            await asyncio.wait_for(
                self._wait_for_event("session.created"),
                timeout=self.queue_timeout,
            )
        except Exception:
            await self._ws.close()
            self._ws = None
            raise

        self._session_created = True
        self._reader_task = asyncio.create_task(
            self._read_loop(),
            name="minicpm-o-4_5-reader",
        )

    async def send_audio(self, audio: PcmAudio) -> None:
        if not self._session_created:
            raise RuntimeError("MiniCPM realtime session is not connected")
        pcm16 = normalize_audio(
            audio.data,
            from_rate=audio.sample_rate,
            to_rate=self.input_sample_rate,
            channels=audio.channels,
        )
        if not pcm16:
            return
        self._input_buffer.extend(pcm16_to_float32(pcm16))
        while len(self._input_buffer) >= self._input_chunk_bytes:
            chunk = bytes(self._input_buffer[: self._input_chunk_bytes])
            del self._input_buffer[: self._input_chunk_bytes]
            await self._send_audio_chunk(chunk)

    async def start_audio_turn(self) -> None:
        await self._stop_evaluation_silence()
        self._response_active = False

    async def end_audio_turn(self) -> None:
        # Full-duplex mode has no explicit commit event. Evaluation playback does,
        # however, need to flush a sub-second tail so short prompts are delivered.
        # It also needs continued silence input so an autonomous listen decision
        # gets another chance to become speak after the prerecorded turn ends.
        if self.evaluation_turn_taking:
            await self._flush_padded_input_buffer()
            self._silence_task = asyncio.create_task(
                self._pump_evaluation_silence(),
                name="minicpm-o-4_5-evaluation-silence",
            )
            return
        await self._flush_input_buffer()

    async def close(self) -> None:
        await self._stop_evaluation_silence()
        ws = self._ws
        if ws is None:
            self.close_output_streams()
            return

        if self._session_created:
            try:
                await self._flush_input_buffer()
                await self._send({"type": "session.close", "reason": "client_shutdown"})
            except Exception:
                logger.debug("Failed to close MiniCPM session cleanly", exc_info=True)

        if self._reader_task:
            try:
                await asyncio.wait_for(asyncio.shield(self._reader_task), timeout=2.0)
            except TimeoutError:
                self._reader_task.cancel()
                await asyncio.gather(self._reader_task, return_exceptions=True)
            except Exception:
                logger.debug(
                    "MiniCPM reader did not close cleanly",
                    exc_info=True,
                )
        await ws.close()
        self._ws = None
        self._session_created = False
        self.close_output_streams()

    async def _wait_for_event(self, expected_type: str) -> dict:
        if self._ws is None:
            raise RuntimeError("MiniCPM realtime websocket is not connected")
        while True:
            raw = await self._ws.recv()
            event = json.loads(raw)
            event_type = event.get("type")
            if event_type == "error":
                raise RuntimeError(f"MiniCPM realtime error: {event}")
            if event_type == "session.closed":
                reason = event.get("reason") or "unknown"
                raise RuntimeError(
                    f"MiniCPM session closed before {expected_type}: reason={reason}; event={event}"
                )
            if event_type == expected_type:
                return event

    async def _send_audio_chunk(self, chunk: bytes) -> None:
        await self._send(
            {
                "type": "input.append",
                "input": {
                    "audio": base64.b64encode(chunk).decode("ascii"),
                    "force_listen": False,
                },
            }
        )

    async def _flush_input_buffer(self) -> None:
        if not self._input_buffer or not self._session_created:
            return
        chunk = bytes(self._input_buffer)
        self._input_buffer.clear()
        await self._send_audio_chunk(chunk)

    async def _flush_padded_input_buffer(self) -> None:
        if not self._input_buffer or not self._session_created:
            return
        chunk = bytes(self._input_buffer)
        self._input_buffer.clear()
        if len(chunk) < self._input_chunk_bytes:
            chunk += b"\x00" * (self._input_chunk_bytes - len(chunk))
        await self._send_audio_chunk(chunk)

    async def _send(self, event: dict) -> None:
        if self._ws is None:
            raise RuntimeError("MiniCPM realtime websocket is not connected")
        await self._ws.send(json.dumps(event))

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                event = json.loads(raw)
                event_type = event.get("type")
                if event_type == "response.output.delta":
                    kind = event.get("kind")
                    if kind == "audio" and event.get("audio"):
                        self._response_active = True
                        float32_pcm = base64.b64decode(event["audio"])
                        await self._audio_out.put(
                            PcmAudio(
                                data=float32_to_pcm16(float32_pcm),
                                sample_rate=self.output_sample_rate,
                                channels=1,
                            )
                        )
                    elif kind == "text" and event.get("text"):
                        self._response_active = True
                        await self._text_out.put(str(event["text"]))
                    elif kind == "listen" and self._response_active:
                        # Full-duplex generation advances only while input keeps
                        # arriving. A listen after output marks the spoken turn end.
                        self._response_active = False
                        await self._stop_evaluation_silence()
                elif event_type == "session.closed":
                    return
                elif event_type == "error":
                    raise RuntimeError(f"MiniCPM realtime error: {event}")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("MiniCPM realtime reader stopped with an error")
        finally:
            self.close_output_streams()

    async def _pump_evaluation_silence(self) -> None:
        silence = b"\x00" * self._input_chunk_bytes
        interval = self.input_chunk_ms / 1_000
        try:
            while self._session_created:
                await self._send_audio_chunk(silence)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("MiniCPM evaluation silence pump stopped with an error")

    async def _stop_evaluation_silence(self) -> None:
        task = self._silence_task
        self._silence_task = None
        if task is None or task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
