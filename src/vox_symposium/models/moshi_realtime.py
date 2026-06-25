from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlsplit

from websockets.asyncio.client import ClientConnection, connect

from vox_symposium.audio import PcmAudio, float32_to_pcm16, normalize_audio
from vox_symposium.models.base import QueueBackedRealtimeAudioModel


logger = logging.getLogger(__name__)


class MoshiRealtimeModel(QueueBackedRealtimeAudioModel):
    """Kyutai Moshi `/api/chat` websocket adapter using the official Opus protocol."""

    input_sample_rate = 24_000
    output_sample_rate = 24_000

    def __init__(
        self,
        *,
        url: str,
        api_key: str | None = None,
    ) -> None:
        super().__init__()
        _validate_moshi_url(url)
        self.url = url
        self.api_key = api_key
        self._ws: ClientConnection | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._np: Any = None
        self._opus_writer: Any = None
        self._opus_reader: Any = None

    async def connect(self) -> None:
        self._load_dependencies()
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        self._ws = await connect(
            self.url,
            additional_headers=headers,
            max_size=None,
        )
        self._reader_task = asyncio.create_task(self._read_loop(), name="moshi-reader")

    async def send_audio(self, audio: PcmAudio) -> None:
        if self._opus_writer is None:
            raise RuntimeError("Moshi realtime session is not connected")
        pcm = normalize_audio(
            audio.data,
            from_rate=audio.sample_rate,
            to_rate=self.input_sample_rate,
            channels=audio.channels,
        )
        if not pcm:
            return
        opus = self._opus_writer.append_pcm(self._pcm16_to_float32(pcm))
        if len(opus) > 0:
            await self._send_audio_page(opus)

    async def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self._ws:
            await self._ws.close()
        self.close_output_streams()

    def _load_dependencies(self) -> None:
        try:
            import numpy as np
            import sphn
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Moshi provider requires optional dependencies. "
                "Install them with `pip install 'vox-symposium[moshi]'` or "
                "`pip install 'numpy>=1.26,<2.3' 'sphn>=0.2.0,<0.3.0'`."
            ) from exc

        self._np = np
        self._opus_writer = sphn.OpusStreamWriter(self.input_sample_rate)
        self._opus_reader = sphn.OpusStreamReader(self.output_sample_rate)

    async def _send_audio_page(self, opus: bytes) -> None:
        if self._ws is None:
            raise RuntimeError("Moshi realtime websocket is not connected")
        await self._ws.send(b"\x01" + opus)

    async def _read_loop(self) -> None:
        assert self._ws is not None
        assert self._opus_reader is not None
        try:
            async for raw in self._ws:
                if not isinstance(raw, bytes) or not raw:
                    continue
                kind = raw[0]
                payload = raw[1:]
                if kind == 0:
                    continue
                if kind == 1:
                    pcm = self._opus_reader.append_bytes(payload)
                    if pcm is not None and len(pcm) > 0:
                        await self._audio_out.put(
                            PcmAudio(
                                data=self._float32_array_to_pcm16(pcm),
                                sample_rate=self.output_sample_rate,
                                channels=1,
                            )
                        )
                elif kind in {2, 7}:
                    text_payload = payload[1:] if kind == 7 and payload else payload
                    text = text_payload.decode("utf-8", errors="replace")
                    if text:
                        await self._text_out.put(text)
                elif kind == 5:
                    raise RuntimeError(
                        "Moshi realtime error: "
                        f"{payload.decode('utf-8', errors='replace')}"
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Moshi realtime reader stopped with an error")
        finally:
            self.close_output_streams()

    def _pcm16_to_float32(self, pcm: bytes):
        samples = self._np.frombuffer(pcm, dtype="<i2").astype(self._np.float32)
        return samples / 32768.0

    def _float32_array_to_pcm16(self, samples) -> bytes:
        float_samples = self._np.asarray(samples, dtype="<f4")
        return float32_to_pcm16(float_samples.tobytes())


def _validate_moshi_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        raise ValueError("Moshi realtime URL must be a ws:// or wss:// URL")
    if parsed.hostname in {"0.0.0.0", "::"}:
        raise ValueError(
            "Moshi realtime URL cannot use a wildcard address; "
            "use 127.0.0.1, a host name, or the server IP"
        )
