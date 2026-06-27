from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import urlsplit

from vox_symposium.audio import PcmAudio, normalize_audio
from vox_symposium.models.base import QueueBackedRealtimeAudioModel


logger = logging.getLogger(__name__)


class FreezeOmniRealtimeModel(QueueBackedRealtimeAudioModel):
    """VITA-MLLM Freeze-Omni Flask-SocketIO demo server adapter."""

    input_sample_rate = 16_000
    output_sample_rate = 24_000

    def __init__(
        self,
        *,
        url: str,
        instructions: str,
        ssl_verify: bool = False,
        input_chunk_ms: int = 20,
        prompt_timeout: float = 30.0,
        post_turn_poll_seconds: float = 60.0,
        post_turn_idle_seconds: float = 3.0,
    ) -> None:
        super().__init__()
        _validate_socketio_url(url)
        if input_chunk_ms <= 0:
            raise ValueError("Freeze-Omni input chunk duration must be positive")
        if prompt_timeout <= 0:
            raise ValueError("Freeze-Omni prompt timeout must be positive")
        if post_turn_poll_seconds <= 0:
            raise ValueError("Freeze-Omni post-turn poll duration must be positive")
        if post_turn_idle_seconds <= 0:
            raise ValueError("Freeze-Omni post-turn idle timeout must be positive")

        self.url = _socketio_http_url(url)
        self.instructions = instructions
        self.ssl_verify = ssl_verify
        self.input_chunk_ms = input_chunk_ms
        self.prompt_timeout = prompt_timeout
        self.post_turn_poll_seconds = post_turn_poll_seconds
        self.post_turn_idle_seconds = post_turn_idle_seconds
        self._input_chunk_bytes = (
            int(self.input_sample_rate * input_chunk_ms / 1_000) * 2
        )
        self._input_buffer = bytearray()
        self._client = None
        self._connected = False
        self._prompt_ack: asyncio.Future[None] | None = None
        self._fatal_error: BaseException | None = None
        self._last_audio_at: float | None = None
        self._poll_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        if self._client is not None:
            raise RuntimeError("Freeze-Omni Socket.IO client is already connected")
        try:
            import socketio
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Freeze-Omni provider requires optional dependencies. "
                "Install them with `pip install 'vox-symposium[freeze-omni]'` "
                "or `pip install 'python-socketio[client]>=5.11,<6'`."
            ) from exc

        self._prompt_ack = asyncio.get_running_loop().create_future()
        client = socketio.AsyncClient(
            reconnection=False,
            logger=False,
            engineio_logger=False,
            ssl_verify=self.ssl_verify,
        )
        self._register_handlers(client)
        self._client = client

        try:
            await client.connect(self.url)
            self._connected = True
            await client.emit("prompt_text", self.instructions)
            await asyncio.wait_for(self._prompt_ack, timeout=self.prompt_timeout)
            await client.emit("recording-started")
        except Exception:
            await self.close()
            raise

    async def send_audio(self, audio: PcmAudio) -> None:
        self._raise_if_failed()
        if not self._connected:
            raise RuntimeError("Freeze-Omni Socket.IO session is not connected")
        pcm = normalize_audio(
            audio.data,
            from_rate=audio.sample_rate,
            to_rate=self.input_sample_rate,
            channels=audio.channels,
        )
        if not pcm:
            return
        self._input_buffer.extend(pcm)
        while len(self._input_buffer) >= self._input_chunk_bytes:
            chunk = bytes(self._input_buffer[: self._input_chunk_bytes])
            del self._input_buffer[: self._input_chunk_bytes]
            await self._send_audio_chunk(chunk)

    async def start_audio_turn(self) -> None:
        await self._stop_post_turn_polling()
        self._input_buffer.clear()
        self._last_audio_at = None
        if self._connected and self._client is not None:
            await self._client.emit("recording-started")

    async def end_audio_turn(self) -> None:
        if self._input_buffer:
            chunk = bytes(self._input_buffer)
            self._input_buffer.clear()
            await self._send_audio_chunk(chunk)
        await self._stop_post_turn_polling()
        self._poll_task = asyncio.create_task(
            self._poll_with_silence(),
            name="freeze-omni-post-turn-poll",
        )

    async def close(self) -> None:
        await self._stop_post_turn_polling()
        client = self._client
        self._client = None
        self._connected = False
        if client is not None:
            try:
                if client.connected:
                    await client.emit("recording-stopped")
                    await client.disconnect()
            except Exception:
                logger.debug("Failed to close Freeze-Omni session cleanly", exc_info=True)
        self.close_output_streams()

    def _register_handlers(self, client) -> None:
        @client.event
        async def disconnect() -> None:
            self._connected = False

        @client.on("prompt_success")
        async def on_prompt_success(_data=None) -> None:
            if self._prompt_ack is not None and not self._prompt_ack.done():
                self._prompt_ack.set_result(None)

        @client.on("too_many_users")
        async def on_too_many_users(_data=None) -> None:
            self._set_fatal_error(RuntimeError("Freeze-Omni server rejected the session: too many users"))

        @client.on("out_time")
        async def on_out_time(_data=None) -> None:
            self._set_fatal_error(RuntimeError("Freeze-Omni server disconnected the session after timeout"))

        @client.on("stop_tts")
        async def on_stop_tts(_data=None) -> None:
            self._last_audio_at = asyncio.get_running_loop().time()

        @client.on("audio")
        async def on_audio(data) -> None:
            if isinstance(data, str):
                payload = data.encode("latin1")
            else:
                payload = bytes(data)
            if not payload:
                return
            self._last_audio_at = asyncio.get_running_loop().time()
            await self._audio_out.put(
                PcmAudio(data=payload, sample_rate=self.output_sample_rate, channels=1)
            )

    async def _send_audio_chunk(self, chunk: bytes) -> None:
        if self._client is None or not self._connected:
            raise RuntimeError("Freeze-Omni Socket.IO session is not connected")
        payload = json.dumps(
            {
                "sample_rate": self.input_sample_rate,
                "audio": list(chunk),
            },
            separators=(",", ":"),
        )
        await self._client.emit("audio", payload)

    async def _poll_with_silence(self) -> None:
        silence = b"\x00" * self._input_chunk_bytes
        interval = self.input_chunk_ms / 1_000
        started_at = asyncio.get_running_loop().time()
        try:
            while self._connected:
                now = asyncio.get_running_loop().time()
                if now - started_at >= self.post_turn_poll_seconds:
                    return
                if (
                    self._last_audio_at is not None
                    and now - self._last_audio_at >= self.post_turn_idle_seconds
                ):
                    return
                await self._send_audio_chunk(silence)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Freeze-Omni post-turn silence polling stopped with an error")

    async def _stop_post_turn_polling(self) -> None:
        task = self._poll_task
        self._poll_task = None
        if task is None or task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _set_fatal_error(self, error: BaseException) -> None:
        self._fatal_error = error
        if self._prompt_ack is not None and not self._prompt_ack.done():
            self._prompt_ack.set_exception(error)

    def _raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError("Freeze-Omni Socket.IO session failed") from self._fatal_error


def _validate_socketio_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.netloc:
        raise ValueError("Freeze-Omni realtime URL must be an http(s) or ws(s) URL")
    if parsed.hostname in {"0.0.0.0", "::"}:
        raise ValueError(
            "Freeze-Omni realtime URL cannot use a wildcard address; "
            "use 127.0.0.1, a host name, or the server IP"
        )


def _socketio_http_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme == "ws":
        return parsed._replace(scheme="http").geturl()
    if parsed.scheme == "wss":
        return parsed._replace(scheme="https").geturl()
    return url
