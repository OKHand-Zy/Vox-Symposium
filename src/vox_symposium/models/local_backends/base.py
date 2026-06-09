from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from vox_symposium.audio import PcmAudio


class LocalVoiceBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalVoiceBackendConfig:
    model: str
    instructions: str
    device: str
    dtype: str
    attn_implementation: str
    ref_audio_path: str | None
    language: str
    chunk_ms: int
    max_new_tokens: int
    qwen_speaker: str


class LocalVoiceBackend:
    input_sample_rate = 16_000
    output_sample_rate = 24_000

    def __init__(self, config: LocalVoiceBackendConfig) -> None:
        self.config = config

    def load(self) -> None:
        raise NotImplementedError

    def prepare_model_turn_detection(self) -> None:
        raise LocalVoiceBackendError(
            f"{type(self).__name__} does not support model-side turn detection"
        )

    def prepare_vad_turn_detection(self) -> None:
        raise NotImplementedError

    def send_audio_chunk_with_model_turn_detection(self, pcm: bytes) -> Iterable[PcmAudio]:
        raise LocalVoiceBackendError(
            f"{type(self).__name__} does not support model-side turn detection"
        )

    def generate_from_completed_turn(self, pcm: bytes) -> Iterable[PcmAudio]:
        raise NotImplementedError
