from __future__ import annotations

from typing import Any

from vox_symposium.config import (
    FreezeOmniSettings,
    GeminiSettings,
    MiniCPMSettings,
    MoshiSettings,
    OpenAISettings,
    gemini_uses_thinking_level,
    load_freeze_omni_settings,
    load_gemini_settings,
    load_minicpm_settings,
    load_moshi_settings,
    load_openai_settings,
)
from vox_symposium.models.base import RealtimeAudioModel
from vox_symposium.providers import (
    MOSHI_PROTOCOL_PROVIDERS,
    normalize_provider,
    provider_label,
)


def build_evaluation_model_from_env(
    provider: str,
    instructions: str,
    *,
    initial_history: tuple[dict[str, Any], ...] = (),
) -> RealtimeAudioModel:
    provider = normalize_provider(provider)
    if provider == "openai":
        return _build_openai_model(load_openai_settings(), instructions)
    if provider == "gemini":
        return _build_gemini_model(
            load_gemini_settings(), instructions, initial_history=initial_history
        )
    if provider == "minicpm":
        return _build_minicpm_model(load_minicpm_settings(), instructions)
    if provider == "freeze_omni":
        return _build_freeze_omni_model(load_freeze_omni_settings(), instructions)
    if provider in MOSHI_PROTOCOL_PROVIDERS:
        config = load_moshi_settings(provider, required=True)
        if config is None:
            raise RuntimeError(
                f"{provider_label(provider)} realtime URL is required when provider={provider}"
            )
        return _build_moshi_model(config, instructions)
    raise RuntimeError(f"Unsupported provider: {provider}")


def _build_openai_model(config: OpenAISettings, instructions: str) -> RealtimeAudioModel:
    from vox_symposium.models.openai_realtime import OpenAIRealtimeModel

    return OpenAIRealtimeModel(
        api_key=config.auth.api_key,
        backend=config.auth.backend,
        endpoint=config.auth.endpoint,
        api_version=config.auth.api_version,
        model=config.auth.model,
        voice=config.voice,
        reasoning_effort=config.reasoning_effort,
        ping_interval=config.ping_interval,
        ping_timeout=config.ping_timeout,
        instructions=instructions,
        manual_activity=True,
    )


def _build_gemini_model(
    config: GeminiSettings,
    instructions: str,
    *,
    initial_history: tuple[dict[str, Any], ...],
) -> RealtimeAudioModel:
    from vox_symposium.models.gemini_live import GeminiLiveModel

    history: tuple[dict[str, Any], ...] = ()
    if gemini_uses_thinking_level(config.model):
        history = initial_history or config.initial_history
    return GeminiLiveModel(
        api_key=config.auth.api_key,
        backend=config.auth.backend,
        vertex_project=config.auth.project,
        vertex_location=config.auth.location,
        credentials_file=config.auth.credentials_file,
        model=config.model,
        instructions=instructions,
        thinking_level=config.thinking_level,
        thinking_budget=config.thinking_budget,
        enable_affective_dialog=config.enable_affective_dialog,
        initial_history=history,
        manual_activity=True,
    )


def _build_minicpm_model(config: MiniCPMSettings, instructions: str) -> RealtimeAudioModel:
    from vox_symposium.models.minicpm_realtime import MiniCPMRealtimeModel

    return MiniCPMRealtimeModel(
        url=config.url,
        api_key=config.api_key,
        instructions=instructions,
        length_penalty=config.length_penalty,
        input_chunk_ms=config.input_chunk_ms,
        queue_timeout=config.queue_timeout,
        ping_interval=config.ping_interval,
        ping_timeout=config.ping_timeout,
        evaluation_turn_taking=True,
    )


def _build_freeze_omni_model(config: FreezeOmniSettings, instructions: str) -> RealtimeAudioModel:
    from vox_symposium.models.freeze_omni import FreezeOmniRealtimeModel

    return FreezeOmniRealtimeModel(
        url=config.url,
        instructions=instructions,
        ssl_verify=config.ssl_verify,
        input_chunk_ms=config.input_chunk_ms,
        connect_timeout=config.connect_timeout,
        connect_retries=config.connect_retries,
        connect_retry_delay=config.connect_retry_delay,
        prompt_timeout=config.prompt_timeout,
        turn_start_delay=config.turn_start_delay,
        turn_preroll_silence_ms=config.turn_preroll_silence_ms,
        max_input_silence_ms=config.max_input_silence_ms,
        input_silence_rms_threshold=config.input_silence_rms_threshold,
        post_turn_poll_seconds=config.post_turn_poll_seconds,
        post_turn_idle_seconds=config.post_turn_idle_seconds,
        post_turn_poll_chunk_ms=config.post_turn_poll_chunk_ms,
        stop_recording_after_turn=config.stop_recording_after_turn,
        evaluation_turn_taking=True,
    )


def _build_moshi_model(
    config: MoshiSettings,
    instructions: str,
) -> RealtimeAudioModel:
    from vox_symposium.models.moshi_realtime import MoshiRealtimeModel

    return MoshiRealtimeModel(
        url=config.url,
        api_key=config.api_key,
        text_prompt=instructions,
    )
