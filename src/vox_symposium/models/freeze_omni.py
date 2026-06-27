from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import urlsplit

from vox_symposium.audio import PcmAudio, normalize_audio
from vox_symposium.models.base import QueueBackedRealtimeAudioModel


logger = logging.getLogger(__name__)


class FreezeOmniTooManyUsersError(RuntimeError):
    pass


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
        connect_timeout: float = 30.0,
        connect_retries: int = 5,
        connect_retry_delay: float = 5.0,
        prompt_timeout: float = 30.0,
        turn_start_delay: float = 1.0,
        turn_preroll_silence_ms: int = 800,
        post_turn_poll_seconds: float = 60.0,
        post_turn_idle_seconds: float = 3.0,
        post_turn_poll_chunk_ms: int = 160,
        stop_recording_after_turn: bool = True,
    ) -> None:
        super().__init__()
        _validate_socketio_url(url)
        if input_chunk_ms <= 0:
            raise ValueError("Freeze-Omni input chunk duration must be positive")
        if connect_timeout <= 0:
            raise ValueError("Freeze-Omni connect timeout must be positive")
        if connect_retries < 0:
            raise ValueError("Freeze-Omni connect retries cannot be negative")
        if connect_retry_delay <= 0:
            raise ValueError("Freeze-Omni connect retry delay must be positive")
        if prompt_timeout <= 0:
            raise ValueError("Freeze-Omni prompt timeout must be positive")
        if turn_start_delay < 0:
            raise ValueError("Freeze-Omni turn start delay cannot be negative")
        if turn_preroll_silence_ms < 0:
            raise ValueError("Freeze-Omni turn preroll silence cannot be negative")
        if post_turn_poll_seconds <= 0:
            raise ValueError("Freeze-Omni post-turn poll duration must be positive")
        if post_turn_idle_seconds <= 0:
            raise ValueError("Freeze-Omni post-turn idle timeout must be positive")
        if post_turn_poll_chunk_ms <= 0:
            raise ValueError("Freeze-Omni post-turn poll chunk duration must be positive")

        self.url = _socketio_http_url(url)
        self.instructions = instructions
        self.ssl_verify = ssl_verify
        self.input_chunk_ms = input_chunk_ms
        self.connect_timeout = connect_timeout
        self.connect_retries = connect_retries
        self.connect_retry_delay = connect_retry_delay
        self.prompt_timeout = prompt_timeout
        self.turn_start_delay = turn_start_delay
        self.turn_preroll_silence_ms = turn_preroll_silence_ms
        self.post_turn_poll_seconds = post_turn_poll_seconds
        self.post_turn_idle_seconds = post_turn_idle_seconds
        self.post_turn_poll_chunk_ms = post_turn_poll_chunk_ms
        self.stop_recording_after_turn = stop_recording_after_turn
        self._input_chunk_bytes = (
            int(self.input_sample_rate * input_chunk_ms / 1_000) * 2
        )
        self._poll_chunk_bytes = (
            int(self.input_sample_rate * post_turn_poll_chunk_ms / 1_000) * 2
        )
        self._input_buffer = bytearray()
        self._client = None
        self._connected = False
        self._prompt_ack: asyncio.Future[None] | None = None
        self._fatal_error: BaseException | None = None
        self._last_audio_at: float | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._saw_text_delta = False

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

        for attempt in range(self.connect_retries + 1):
            self._fatal_error = None
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
                await client.connect(self.url, wait_timeout=self.connect_timeout)
                self._connected = True
                self._raise_if_failed()
                await client.emit("prompt_text", self.instructions)
                await asyncio.wait_for(self._prompt_ack, timeout=self.prompt_timeout)
                await client.emit("recording-started")
                return
            except Exception as exc:
                error = self._fatal_error or exc
                await self._disconnect_client(client)
                if isinstance(error, FreezeOmniTooManyUsersError) and attempt < self.connect_retries:
                    logger.info(
                        "Freeze-Omni server is full; retrying connection in %.1fs (%s/%s)",
                        self.connect_retry_delay,
                        attempt + 1,
                        self.connect_retries,
                    )
                    await asyncio.sleep(self.connect_retry_delay)
                    continue
                if error is not exc:
                    raise error from exc
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
        self._saw_text_delta = False
        if self._connected and self._client is not None:
            await self._client.emit("recording-started")
            if self.turn_start_delay:
                await asyncio.sleep(self.turn_start_delay)
            if self.turn_preroll_silence_ms:
                await self._send_silence(self.turn_preroll_silence_ms)

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
            await self._disconnect_client(client, emit_recording_stopped=True)
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
            self._set_fatal_error(
                FreezeOmniTooManyUsersError("Freeze-Omni server rejected the session: too many users")
            )

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

        @client.on("text_delta")
        async def on_text_delta(data) -> None:
            text = _extract_text_payload(data)
            if text:
                self._saw_text_delta = True
                await self._text_out.put(text)

        @client.on("text")
        async def on_text(data) -> None:
            text = _extract_text_payload(data)
            if text:
                self._saw_text_delta = True
                await self._text_out.put(text)

        @client.on("text_done")
        async def on_text_done(data) -> None:
            if self._saw_text_delta:
                return
            text = _extract_text_payload(data)
            if text:
                await self._text_out.put(text)

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

    async def _disconnect_client(self, client, *, emit_recording_stopped: bool = False) -> None:
        if self._client is client:
            self._client = None
        self._connected = False
        self._consume_prompt_ack_exception()
        try:
            if client.connected:
                if emit_recording_stopped:
                    await client.emit("recording-stopped")
                await client.disconnect()
        except Exception:
            logger.debug("Failed to disconnect Freeze-Omni session cleanly", exc_info=True)

    def _consume_prompt_ack_exception(self) -> None:
        if self._prompt_ack is None or not self._prompt_ack.done():
            return
        try:
            self._prompt_ack.exception()
        except Exception:
            pass

    async def _poll_with_silence(self) -> None:
        silence = b"\x00" * self._poll_chunk_bytes
        interval = self.post_turn_poll_chunk_ms / 1_000
        started_at = asyncio.get_running_loop().time()
        stopped = False
        try:
            while self._connected:
                now = asyncio.get_running_loop().time()
                if now - started_at >= self.post_turn_poll_seconds:
                    stopped = True
                    return
                if (
                    self._last_audio_at is not None
                    and now - self._last_audio_at >= self.post_turn_idle_seconds
                ):
                    stopped = True
                    return
                await self._send_audio_chunk(silence)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Freeze-Omni post-turn silence polling stopped with an error")
        finally:
            if stopped and self.stop_recording_after_turn:
                await self._emit_recording_stopped()

    async def _send_silence(self, duration_ms: int) -> None:
        if duration_ms <= 0:
            return
        chunk_ms = self.post_turn_poll_chunk_ms
        silence = b"\x00" * self._poll_chunk_bytes
        remaining = duration_ms
        while remaining > 0:
            await self._send_audio_chunk(silence)
            await asyncio.sleep(chunk_ms / 1_000)
            remaining -= chunk_ms

    async def _emit_recording_stopped(self) -> None:
        client = self._client
        if client is None or not self._connected:
            return
        try:
            await client.emit("recording-stopped")
        except Exception:
            logger.debug("Failed to send Freeze-Omni recording-stopped", exc_info=True)

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


def _extract_text_payload(data) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        value = data.get("text")
        return value if isinstance(value, str) else ""
    return ""
