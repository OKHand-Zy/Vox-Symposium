from __future__ import annotations

import os
from typing import TYPE_CHECKING

from vox_symposium.config import (
    gemini_live_model,
    load_gemini_auth,
    load_moshi_settings,
    load_openai_auth,
)
from vox_symposium.env import bool_env, float_env, int_env, required_env
from vox_symposium.models.base import RealtimeAudioModel
from vox_symposium.providers import (
    MOSHI_PROTOCOL_PROVIDERS,
    normalize_provider,
    provider_label,
)

if TYPE_CHECKING:
    from vox_symposium.config import AgentConfig, Settings


def build_model_from_settings(
    settings: Settings,
    agent: AgentConfig,
    *,
    evaluation_mode: bool = False,
) -> RealtimeAudioModel:
    provider = normalize_provider(agent.provider)
    if provider == "openai":
        if settings.openai_api_key is None:
            raise RuntimeError(
                "OpenAI credentials are required when a participant uses provider=openai"
            )
        from vox_symposium.models.openai_realtime import OpenAIRealtimeModel

        return OpenAIRealtimeModel(
            api_key=settings.openai_api_key,
            backend=settings.openai_backend,
            endpoint=settings.openai_endpoint,
            api_version=settings.openai_api_version,
            model=settings.openai_model,
            voice=settings.openai_voice,
            instructions=agent.instructions,
            manual_activity=evaluation_mode,
        )

    if provider == "gemini":
        from vox_symposium.models.gemini_live import GeminiLiveModel

        return GeminiLiveModel(
            api_key=settings.gemini_api_key,
            backend=settings.gemini_backend,
            vertex_project=settings.gemini_vertex_project,
            vertex_location=settings.gemini_vertex_location,
            credentials_file=settings.gemini_credentials_file,
            model=settings.gemini_model,
            instructions=agent.instructions,
            manual_activity=evaluation_mode,
        )

    if provider == "minicpm":
        if settings.minicpm_realtime_url is None:
            raise RuntimeError(
                "MINICPM_REALTIME_URL is required when a participant uses provider=minicpm"
            )
        from vox_symposium.models.minicpm_realtime import MiniCPMRealtimeModel

        return MiniCPMRealtimeModel(
            url=settings.minicpm_realtime_url,
            api_key=settings.minicpm_api_key,
            instructions=agent.instructions,
            length_penalty=settings.minicpm_length_penalty,
            input_chunk_ms=settings.minicpm_input_chunk_ms,
            queue_timeout=settings.minicpm_queue_timeout,
            evaluation_turn_taking=evaluation_mode,
        )

    if provider == "freeze_omni":
        if settings.freeze_omni_realtime_url is None:
            raise RuntimeError(
                "FREEZE_OMNI_REALTIME_URL is required when a participant uses provider=freeze_omni"
            )
        from vox_symposium.models.freeze_omni import FreezeOmniRealtimeModel

        return FreezeOmniRealtimeModel(
            url=settings.freeze_omni_realtime_url,
            instructions=agent.instructions,
            ssl_verify=settings.freeze_omni_ssl_verify,
            input_chunk_ms=settings.freeze_omni_input_chunk_ms,
            connect_timeout=settings.freeze_omni_connect_timeout,
            connect_retries=settings.freeze_omni_connect_retries,
            connect_retry_delay=settings.freeze_omni_connect_retry_delay,
            prompt_timeout=settings.freeze_omni_prompt_timeout,
            turn_start_delay=settings.freeze_omni_turn_start_delay,
            turn_preroll_silence_ms=settings.freeze_omni_turn_preroll_silence_ms,
            max_input_silence_ms=settings.freeze_omni_max_input_silence_ms,
            input_silence_rms_threshold=settings.freeze_omni_input_silence_rms_threshold,
            post_turn_poll_seconds=settings.freeze_omni_post_turn_poll_seconds,
            post_turn_idle_seconds=settings.freeze_omni_post_turn_idle_seconds,
            post_turn_poll_chunk_ms=settings.freeze_omni_post_turn_poll_chunk_ms,
            stop_recording_after_turn=settings.freeze_omni_stop_recording_after_turn,
        )

    if provider in MOSHI_PROTOCOL_PROVIDERS:
        if provider not in settings.moshi_protocols:
            raise RuntimeError(
                f"{provider_label(provider)} realtime URL is required when a participant uses "
                f"provider={provider}"
            )
        from vox_symposium.models.moshi_realtime import MoshiRealtimeModel

        protocol_settings = settings.moshi_protocols[provider]
        return MoshiRealtimeModel(
            url=protocol_settings.url,
            api_key=protocol_settings.api_key,
            text_prompt=agent.instructions,
        )

    raise RuntimeError(f"Unsupported provider for {agent.identity}: {agent.provider}")


def build_model_from_env(
    provider: str,
    instructions: str,
    *,
    evaluation_mode: bool = False,
) -> RealtimeAudioModel:
    provider = normalize_provider(provider)
    if provider == "openai":
        from vox_symposium.models.openai_realtime import OpenAIRealtimeModel

        auth = load_openai_auth()
        return OpenAIRealtimeModel(
            api_key=auth.api_key,
            backend=auth.backend,
            endpoint=auth.endpoint,
            api_version=auth.api_version,
            model=auth.model,
            voice=os.getenv("OPENAI_REALTIME_VOICE", "marin"),
            instructions=instructions,
            manual_activity=evaluation_mode,
        )

    if provider == "gemini":
        from vox_symposium.models.gemini_live import GeminiLiveModel

        auth = load_gemini_auth()
        return GeminiLiveModel(
            api_key=auth.api_key,
            backend=auth.backend,
            vertex_project=auth.project,
            vertex_location=auth.location,
            credentials_file=auth.credentials_file,
            model=gemini_live_model(auth.backend),
            instructions=instructions,
            manual_activity=evaluation_mode,
        )

    if provider == "minicpm":
        from vox_symposium.models.minicpm_realtime import MiniCPMRealtimeModel

        return MiniCPMRealtimeModel(
            url=required_env("MINICPM_REALTIME_URL"),
            api_key=os.getenv("MINICPM_API_KEY") or None,
            instructions=instructions,
            length_penalty=float_env("MINICPM_LENGTH_PENALTY", 1.1),
            input_chunk_ms=int_env("MINICPM_INPUT_CHUNK_MS", 1_000),
            queue_timeout=float_env("MINICPM_QUEUE_TIMEOUT", 300.0),
            evaluation_turn_taking=evaluation_mode,
        )

    if provider == "freeze_omni":
        from vox_symposium.models.freeze_omni import FreezeOmniRealtimeModel

        return FreezeOmniRealtimeModel(
            url=required_env("FREEZE_OMNI_REALTIME_URL"),
            instructions=instructions,
            ssl_verify=bool_env("FREEZE_OMNI_SSL_VERIFY", False),
            input_chunk_ms=int_env("FREEZE_OMNI_INPUT_CHUNK_MS", 20),
            connect_timeout=float_env("FREEZE_OMNI_CONNECT_TIMEOUT", 30.0),
            connect_retries=int_env("FREEZE_OMNI_CONNECT_RETRIES", 5),
            connect_retry_delay=float_env("FREEZE_OMNI_CONNECT_RETRY_DELAY", 5.0),
            prompt_timeout=float_env("FREEZE_OMNI_PROMPT_TIMEOUT", 30.0),
            turn_start_delay=float_env("FREEZE_OMNI_TURN_START_DELAY", 1.0),
            turn_preroll_silence_ms=int_env("FREEZE_OMNI_TURN_PREROLL_SILENCE_MS", 800),
            max_input_silence_ms=int_env("FREEZE_OMNI_MAX_INPUT_SILENCE_MS", 40),
            input_silence_rms_threshold=float_env(
                "FREEZE_OMNI_INPUT_SILENCE_RMS_THRESHOLD",
                1800.0,
            ),
            post_turn_poll_seconds=float_env(
                "FREEZE_OMNI_POST_TURN_POLL_SECONDS",
                60.0,
            ),
            post_turn_idle_seconds=float_env(
                "FREEZE_OMNI_POST_TURN_IDLE_SECONDS",
                3.0,
            ),
            post_turn_poll_chunk_ms=int_env("FREEZE_OMNI_POST_TURN_POLL_CHUNK_MS", 160),
            stop_recording_after_turn=bool_env(
                "FREEZE_OMNI_STOP_RECORDING_AFTER_TURN",
                True,
            ),
        )

    if provider in MOSHI_PROTOCOL_PROVIDERS:
        from vox_symposium.models.moshi_realtime import MoshiRealtimeModel

        settings = load_moshi_settings(provider, required=True)
        if settings is None:
            raise RuntimeError(
                f"{provider_label(provider)} realtime URL is required when provider={provider}"
            )
        return MoshiRealtimeModel(
            url=settings.url,
            api_key=settings.api_key,
            text_prompt=instructions,
        )

    raise RuntimeError(f"Unsupported provider: {provider}")
