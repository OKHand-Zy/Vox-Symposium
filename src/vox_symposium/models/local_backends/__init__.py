from __future__ import annotations

from vox_symposium.models.local_backends.base import LocalVoiceBackend, LocalVoiceBackendConfig
from vox_symposium.models.local_backends.minicpmo import MiniCpmOBackend
from vox_symposium.models.local_backends.qwen3_omni import Qwen3OmniBackend


def build_local_backend(provider: str, config: LocalVoiceBackendConfig) -> LocalVoiceBackend:
    normalized = provider.lower()
    if normalized == "auto":
        model = config.model.lower()
        if "minicpm" in model:
            normalized = "minicpmo"
        elif "qwen3-omni" in model or "qwen3_omni" in model:
            normalized = "qwen3_omni"
        else:
            raise RuntimeError(
                "LOCAL_MODEL_PROVIDER=auto could not infer the local backend. "
                "Set LOCAL_MODEL_PROVIDER=minicpmo or LOCAL_MODEL_PROVIDER=qwen3_omni."
            )

    if normalized in {"minicpmo", "minicpm-o", "minicpm_o"}:
        return MiniCpmOBackend(config)
    if normalized in {"qwen3_omni", "qwen3-omni", "qwen"}:
        return Qwen3OmniBackend(config)
    raise RuntimeError(
        "LOCAL_MODEL_PROVIDER must be 'auto', 'minicpmo', or 'qwen3_omni', "
        f"got {provider!r}"
    )
