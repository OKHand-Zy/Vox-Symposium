from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterable
from pathlib import Path

from vox_symposium.audio import PcmAudio
from vox_symposium.models.local_backends.base import (
    LocalVoiceBackend,
    LocalVoiceBackendConfig,
    LocalVoiceBackendError,
)


logger = logging.getLogger(__name__)


class Qwen3OmniBackend(LocalVoiceBackend):
    def __init__(self, config: LocalVoiceBackendConfig) -> None:
        super().__init__(config)
        self._model = None
        self._processor = None
        self._process_mm_info = None

    def load(self) -> None:
        if "thinking" in self.config.model.lower():
            raise LocalVoiceBackendError(
                "Qwen3-Omni Thinking checkpoints support audio input with text output only. "
                "Use Qwen3-Omni Instruct for audio output, or add a text-to-speech backend."
            )

        try:
            from qwen_omni_utils import process_mm_info
            from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Qwen3-Omni local backend requires optional dependencies. Install qwen-omni-utils "
                "and a Qwen3-Omni-compatible transformers build."
            ) from exc

        kwargs = {
            "dtype": self.config.dtype,
            "device_map": self.config.device,
        }
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation

        logger.info("Loading Qwen3-Omni local model %s", self.config.model)
        self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(self.config.model, **kwargs)
        self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.config.model)
        self._process_mm_info = process_mm_info

    def prepare_model_turn_detection(self) -> None:
        raise LocalVoiceBackendError(
            "Qwen3-Omni Transformers backend does not expose model-side streaming turn_detection=True. "
            "Silero VAD will be used instead."
        )

    def prepare_vad_turn_detection(self) -> None:
        self._require_model()
        self._require_processor()

    def generate_from_completed_turn(self, pcm: bytes) -> Iterable[PcmAudio]:
        import soundfile as sf

        model = self._require_model()
        processor = self._require_processor()
        process_mm_info = self._require_process_mm_info()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as audio_file:
            sf.write(audio_file.name, _pcm16_to_float32(pcm), self.input_sample_rate)
            conversation = self._conversation(Path(audio_file.name))
            text = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
            inputs = inputs.to(model.device).to(model.dtype)
            _text_ids, audio = model.generate(
                **inputs,
                speaker=self.config.qwen_speaker,
                thinker_return_dict_in_generate=True,
                use_audio_in_video=False,
                max_new_tokens=self.config.max_new_tokens,
            )

        if audio is None:
            return []
        pcm16 = _waveform_to_pcm16(audio)
        if not pcm16:
            return []
        return [PcmAudio(data=pcm16, sample_rate=self.output_sample_rate, channels=1)]

    def _conversation(self, audio_path: Path) -> list[dict]:
        conversation: list[dict] = []
        if self.config.instructions:
            conversation.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.config.instructions}],
                }
            )
        conversation.append(
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": "Reply conversationally using speech."},
                ],
            }
        )
        return conversation

    def _require_model(self):
        if self._model is None:
            raise RuntimeError("Qwen3-Omni local backend is not loaded")
        return self._model

    def _require_processor(self):
        if self._processor is None:
            raise RuntimeError("Qwen3-Omni processor is not loaded")
        return self._processor

    def _require_process_mm_info(self):
        if self._process_mm_info is None:
            raise RuntimeError("qwen_omni_utils.process_mm_info is not loaded")
        return self._process_mm_info


def _pcm16_to_float32(data: bytes):
    import numpy as np

    samples = np.frombuffer(data, dtype="<i2").astype("float32")
    return samples / 32768.0


def _waveform_to_pcm16(waveform) -> bytes:
    import numpy as np
    import torch

    if isinstance(waveform, torch.Tensor):
        array = waveform.detach().cpu().float().numpy()
    else:
        array = np.asarray(waveform, dtype=np.float32)
    array = np.asarray(array).reshape(-1)
    if array.size == 0:
        return b""
    array = np.clip(array, -1.0, 1.0)
    return (array * 32767.0).astype("<i2").tobytes()
