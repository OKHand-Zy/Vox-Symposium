from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from vox_symposium.audio import PcmAudio, normalize_audio
from vox_symposium.models.base import RealtimeAudioModel
from vox_symposium.models.local_backends import build_local_backend
from vox_symposium.models.local_backends.base import LocalVoiceBackend, LocalVoiceBackendConfig, LocalVoiceBackendError


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalVoiceConfig:
    provider: str
    model: str
    instructions: str
    device: str
    dtype: str
    attn_implementation: str
    ref_audio_path: str | None
    language: str
    turn_detection: bool
    silero_threshold: float
    turn_silence_ms: int
    min_turn_ms: int
    chunk_ms: int
    max_new_tokens: int
    qwen_speaker: str


class LocalVoiceModel(RealtimeAudioModel):
    input_sample_rate = 16_000
    output_sample_rate = 24_000

    def __init__(self, config: LocalVoiceConfig) -> None:
        self.config = config
        backend_config = LocalVoiceBackendConfig(
            model=config.model,
            instructions=config.instructions,
            device=config.device,
            dtype=config.dtype,
            attn_implementation=config.attn_implementation,
            ref_audio_path=config.ref_audio_path,
            language=config.language,
            chunk_ms=config.chunk_ms,
            max_new_tokens=config.max_new_tokens,
            qwen_speaker=config.qwen_speaker,
        )
        self.backend = build_local_backend(config.provider, backend_config)
        self.input_sample_rate = self.backend.input_sample_rate
        self.output_sample_rate = self.backend.output_sample_rate
        self._loop: asyncio.AbstractEventLoop | None = None
        self._audio_out: asyncio.Queue[PcmAudio | None] = asyncio.Queue(maxsize=100)
        self._turns: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=4)
        self._worker_task: asyncio.Task[None] | None = None
        self._model_lock = asyncio.Lock()
        self._vad_lock = asyncio.Lock()
        self._closed = False
        self._uses_model_turn_detection = False
        self._silero_model = None
        self._silero_remainder = b""
        self._turn_buffer = bytearray()
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._speech_started = False

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        await asyncio.to_thread(self.backend.load)
        if hasattr(self.backend, "set_session_id"):
            self.backend.set_session_id(f"vox-{uuid.uuid4().hex}")

        if self.config.turn_detection:
            try:
                await asyncio.to_thread(self.backend.prepare_model_turn_detection)
                self._uses_model_turn_detection = True
            except LocalVoiceBackendError as exc:
                logger.info(
                    "%s does not support model-side turn detection, falling back to Silero VAD: %s",
                    self.config.model,
                    exc,
                )

        if not self._uses_model_turn_detection:
            await asyncio.to_thread(self.backend.prepare_vad_turn_detection)
            await asyncio.to_thread(self._load_silero_vad)

        self._worker_task = asyncio.create_task(self._turn_worker(), name=f"local-{self.config.model}-worker")

    async def send_audio(self, audio: PcmAudio) -> None:
        pcm = normalize_audio(
            audio.data,
            from_rate=audio.sample_rate,
            to_rate=self.input_sample_rate,
            channels=audio.channels,
        )
        if not pcm or self._closed:
            return

        if self._uses_model_turn_detection:
            async with self._model_lock:
                try:
                    chunks = await asyncio.to_thread(
                        self._send_audio_chunk_with_model_turn_detection,
                        pcm,
                    )
                except Exception:
                    logger.exception("Local model-side turn detection failed")
                    return
            await self._publish_chunks(chunks)
            return

        await self._send_audio_with_silero_vad(pcm)

    async def receive_audio(self) -> AsyncIterator[PcmAudio]:
        while True:
            item = await self._audio_out.get()
            if item is None:
                return
            yield item

    async def close(self) -> None:
        self._closed = True
        if self._speech_started and self._turn_buffer:
            await self._enqueue_current_turn()
        await self._turns.put(None)
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=2)
            except TimeoutError:
                self._worker_task.cancel()
        await self._audio_out.put(None)

    async def _send_audio_with_silero_vad(self, pcm: bytes) -> None:
        async with self._vad_lock:
            decisions = await asyncio.to_thread(self._silero_decisions, pcm)
            for chunk, is_speech in decisions:
                audio_ms = len(chunk) / 2 / self.input_sample_rate * 1000
                if not self._speech_started:
                    if not is_speech:
                        continue
                    self._speech_started = True

                self._turn_buffer.extend(chunk)
                if is_speech:
                    self._speech_ms += audio_ms
                    self._silence_ms = 0.0
                else:
                    self._silence_ms += audio_ms

                has_min_speech = self._speech_ms >= self.config.min_turn_ms
                has_trailing_silence = self._silence_ms >= self.config.turn_silence_ms
                if has_min_speech and has_trailing_silence:
                    await self._enqueue_current_turn()

    async def _enqueue_current_turn(self) -> None:
        turn = bytes(self._turn_buffer)
        self._turn_buffer.clear()
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._speech_started = False
        if turn:
            await self._turns.put(turn)

    async def _turn_worker(self) -> None:
        while True:
            turn = await self._turns.get()
            if turn is None:
                return
            try:
                async with self._model_lock:
                    chunks = await asyncio.to_thread(self._generate_from_completed_turn, turn)
                await self._publish_chunks(chunks)
            except Exception:
                logger.exception("Local voice turn failed")

    def _send_audio_chunk_with_model_turn_detection(self, pcm: bytes) -> list[PcmAudio]:
        return list(self.backend.send_audio_chunk_with_model_turn_detection(pcm))

    def _generate_from_completed_turn(self, turn: bytes) -> list[PcmAudio]:
        return list(self.backend.generate_from_completed_turn(turn))

    async def _publish_chunks(self, chunks) -> None:
        for chunk in chunks:
            if self._closed:
                return
            await self._audio_out.put(chunk)

    def _load_silero_vad(self) -> None:
        try:
            from silero_vad import load_silero_vad
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Silero VAD fallback requires silero-vad. Install requirements.txt or install silero-vad directly."
            ) from exc

        self._silero_model = load_silero_vad()
        if hasattr(self._silero_model, "reset_states"):
            self._silero_model.reset_states()

    def _silero_decisions(self, pcm: bytes) -> list[tuple[bytes, bool]]:
        if self._silero_model is None:
            raise RuntimeError("Silero VAD is not initialized")

        frame_size = 512 * 2
        data = self._silero_remainder + pcm
        decisions: list[tuple[bytes, bool]] = []
        offset = 0
        while offset + frame_size <= len(data):
            chunk = data[offset : offset + frame_size]
            probability = _silero_speech_probability(self._silero_model, chunk, self.input_sample_rate)
            decisions.append((chunk, probability >= self.config.silero_threshold))
            offset += frame_size
        self._silero_remainder = data[offset:]
        return decisions


def _silero_speech_probability(model, data: bytes, sample_rate: int) -> float:
    import numpy as np
    import torch

    samples = np.frombuffer(data, dtype="<i2").astype("float32") / 32768.0
    with torch.no_grad():
        probability = model(torch.from_numpy(samples), sample_rate)
    return float(probability.item())
