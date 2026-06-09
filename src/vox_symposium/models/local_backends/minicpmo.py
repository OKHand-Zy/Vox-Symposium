from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable

from vox_symposium.audio import PcmAudio
from vox_symposium.models.local_backends.base import (
    LocalVoiceBackend,
    LocalVoiceBackendConfig,
    LocalVoiceBackendError,
)


logger = logging.getLogger(__name__)


class MiniCpmOBackend(LocalVoiceBackend):
    def __init__(self, config: LocalVoiceBackendConfig) -> None:
        super().__init__(config)
        self._model = None
        self._ref_audio = None
        self._session_id: str | None = None

    def load(self) -> None:
        try:
            import librosa
            import torch
            from transformers import AutoModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MiniCPM-o local backend requires optional dependencies. Install with: "
                'pip install -e ".[local]"'
            ) from exc

        dtype = _torch_dtype(torch, self.config.dtype)
        device = _device(torch, self.config.device)
        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "init_vision": False,
            "init_audio": True,
            "init_tts": True,
        }
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation

        logger.info("Loading MiniCPM-o local model %s on %s", self.config.model, device)
        model = AutoModel.from_pretrained(self.config.model, **kwargs)
        model.eval()
        model.to(device)

        if self.config.ref_audio_path:
            self._ref_audio, _ = librosa.load(
                self.config.ref_audio_path,
                sr=self.input_sample_rate,
                mono=True,
            )
        self._model = model

    def set_session_id(self, session_id: str) -> None:
        self._session_id = session_id

    def prepare_model_turn_detection(self) -> None:
        model = self._require_model()
        if hasattr(model, "init_tts"):
            model.init_tts()
        if hasattr(model, "as_duplex"):
            model = model.as_duplex()
        if not hasattr(model, "prepare"):
            raise LocalVoiceBackendError(
                f"{self.config.model} does not expose a duplex prepare() API. "
                "Silero VAD will be used instead."
            )
        if not hasattr(model, "streaming_prefill") or not hasattr(model, "streaming_generate"):
            raise LocalVoiceBackendError(
                f"{self.config.model} does not expose streaming_prefill()/streaming_generate(). "
                "Silero VAD will be used instead."
            )
        if not _supports_kwarg(model.streaming_prefill, "turn_detection"):
            raise LocalVoiceBackendError(
                f"{self.config.model} streaming_prefill() does not support turn_detection=True. "
                "Silero VAD will be used instead."
            )
        if not _supports_kwarg(model.streaming_generate, "turn_detection"):
            raise LocalVoiceBackendError(
                f"{self.config.model} streaming_generate() does not support turn_detection=True. "
                "Silero VAD will be used instead."
            )

        _call_with_supported_kwargs(
            model.prepare,
            {
                "prefix_system_prompt": self.config.instructions,
                "ref_audio": self._ref_audio,
                "prompt_wav_path": self.config.ref_audio_path,
            },
        )
        self._model = model

    def prepare_vad_turn_detection(self) -> None:
        model = self._require_model()
        model.init_tts()
        model.reset_session(reset_token2wav_cache=True)
        if self._ref_audio is not None:
            model.init_token2wav_cache(prompt_speech_16k=self._ref_audio)

        model.streaming_prefill(
            session_id=self._require_session_id(),
            msgs=[self._build_system_msg()],
            omni_mode=False,
            is_last_chunk=True,
        )

    def send_audio_chunk_with_model_turn_detection(self, pcm: bytes) -> Iterable[PcmAudio]:
        model = self._require_model()
        waveform = _pcm16_to_float32(pcm)
        _call_with_supported_kwargs(
            model.streaming_prefill,
            {
                "audio_waveform": waveform,
                "frame_list": [],
                "max_slice_nums": 1,
                "batch_vision_feed": False,
                "turn_detection": True,
            },
        )

        result = _call_with_supported_kwargs(
            model.streaming_generate,
            {
                "prompt_wav_path": self.config.ref_audio_path,
                "max_new_speak_tokens_per_chunk": 20,
                "decode_mode": "sampling",
                "turn_detection": True,
            },
        )
        if not isinstance(result, dict):
            return []
        audio = result.get("audio_waveform")
        if audio is None:
            return []
        pcm16 = _waveform_to_pcm16(audio)
        if not pcm16:
            return []
        return [PcmAudio(data=pcm16, sample_rate=self.output_sample_rate, channels=1)]

    def generate_from_completed_turn(self, pcm: bytes) -> Iterable[PcmAudio]:
        model = self._require_model()
        chunks = _chunk_pcm16(pcm, self.input_sample_rate, self.config.chunk_ms)
        for idx, chunk in enumerate(chunks):
            model.streaming_prefill(
                session_id=self._require_session_id(),
                msgs=[{"role": "user", "content": [_pcm16_to_float32(chunk)]}],
                omni_mode=False,
                is_last_chunk=idx == len(chunks) - 1,
            )

        iter_gen = model.streaming_generate(
            session_id=self._require_session_id(),
            generate_audio=True,
            use_tts_template=True,
            enable_thinking=False,
            do_sample=True,
            max_new_tokens=self.config.max_new_tokens,
            length_penalty=1.1,
        )
        for wav_chunk, _text_chunk in iter_gen:
            audio = _waveform_to_pcm16(wav_chunk)
            if audio:
                yield PcmAudio(data=audio, sample_rate=self.output_sample_rate, channels=1)

    def _build_system_msg(self) -> dict:
        if self._ref_audio is None:
            return {"role": "system", "content": [self.config.instructions]}

        if self.config.language.lower().startswith("zh"):
            return {
                "role": "system",
                "content": ["模仿输入音频中的声音特征。", self._ref_audio, self.config.instructions],
            }

        return {
            "role": "system",
            "content": ["Clone the voice in the provided audio prompt.", self._ref_audio, self.config.instructions],
        }

    def _require_model(self):
        if self._model is None:
            raise RuntimeError("MiniCPM-o local backend is not loaded")
        return self._model

    def _require_session_id(self) -> str:
        if self._session_id is None:
            raise RuntimeError("MiniCPM-o local backend session id is not set")
        return self._session_id


def _torch_dtype(torch, value: str):
    normalized = value.lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise RuntimeError(f"Unsupported LOCAL_MODEL_DTYPE: {value!r}")


def _device(torch, value: str) -> str:
    normalized = value.lower()
    if normalized != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _call_with_supported_kwargs(callable_obj, kwargs: dict):
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return callable_obj(**kwargs)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return callable_obj(**supported)


def _supports_kwarg(callable_obj, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )


def _chunk_bytes(sample_rate: int, chunk_ms: int) -> int:
    return int(sample_rate * chunk_ms / 1000) * 2


def _chunk_pcm16(data: bytes, sample_rate: int, chunk_ms: int) -> list[bytes]:
    size = _chunk_bytes(sample_rate, chunk_ms)
    if size <= 0:
        raise RuntimeError("LOCAL_MODEL_CHUNK_MS must be positive")
    chunks = [data[offset : offset + size] for offset in range(0, len(data), size)]
    if chunks and len(chunks[-1]) < size:
        chunks[-1] = chunks[-1] + bytes(size - len(chunks[-1]))
    return chunks


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
