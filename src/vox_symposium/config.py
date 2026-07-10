from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False

from vox_symposium.env import (
    bool_env,
    env_with_legacy,
    first_env,
    float_env,
    int_env,
    normalized_env,
    optional_float_env,
    optional_int_env,
    required_env,
)
from vox_symposium.providers import (
    MOSHI_PROTOCOL_PROVIDERS,
    SUPPORTED_PROVIDERS,
    normalize_provider,
    provider_env_prefix,
)


@dataclass(frozen=True)
class AgentConfig:
    identity: str
    provider: str
    instructions: str


@dataclass(frozen=True)
class Settings:
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    livekit_room: str
    publish_sample_rate: int
    frame_ms: int
    agent_citizen: AgentConfig
    agent_scholar: AgentConfig
    openai_api_key: str | None
    openai_backend: str
    openai_endpoint: str | None
    openai_api_version: str | None
    openai_model: str
    openai_voice: str
    openai_reasoning_effort: str | None
    openai_ping_interval: float
    openai_ping_timeout: float
    gemini_api_key: str | None
    gemini_backend: str
    gemini_vertex_project: str | None
    gemini_vertex_location: str | None
    gemini_credentials_file: str | None
    gemini_model: str
    gemini_thinking_level: str
    gemini_thinking_budget: int | None
    gemini_enable_affective_dialog: bool
    gemini_initial_history: tuple[dict[str, Any], ...]
    minicpm_realtime_url: str | None
    minicpm_api_key: str | None
    minicpm_length_penalty: float
    minicpm_input_chunk_ms: int
    minicpm_queue_timeout: float
    minicpm_ping_interval: float | None
    minicpm_ping_timeout: float | None
    freeze_omni_realtime_url: str | None
    freeze_omni_ssl_verify: bool
    freeze_omni_input_chunk_ms: int
    freeze_omni_connect_timeout: float
    freeze_omni_connect_retries: int
    freeze_omni_connect_retry_delay: float
    freeze_omni_prompt_timeout: float
    freeze_omni_turn_start_delay: float
    freeze_omni_turn_preroll_silence_ms: int
    freeze_omni_max_input_silence_ms: int
    freeze_omni_input_silence_rms_threshold: float
    freeze_omni_post_turn_poll_seconds: float
    freeze_omni_post_turn_idle_seconds: float
    freeze_omni_post_turn_poll_chunk_ms: int
    freeze_omni_stop_recording_after_turn: bool
    moshi_protocols: dict[str, MoshiSettings]


@dataclass(frozen=True)
class GeminiAuthConfig:
    backend: str
    api_key: str | None = None
    project: str | None = None
    location: str | None = None
    credentials_file: str | None = None


@dataclass(frozen=True)
class OpenAIAuthConfig:
    backend: str
    api_key: str
    model: str
    endpoint: str | None = None
    api_version: str | None = None


@dataclass(frozen=True)
class MoshiSettings:
    url: str
    api_key: str | None


def load_settings() -> Settings:
    load_dotenv()
    agent_citizen = AgentConfig(
        identity=env_with_legacy("AGENT_CITIZEN_IDENTITY", "AGENT_A_IDENTITY", default="agent-citizen"),
        provider=_provider_env("AGENT_CITIZEN_PROVIDER", "AGENT_A_PROVIDER", default="openai"),
        instructions=env_with_legacy(
            "AGENT_CITIZEN_INSTRUCTIONS",
            "AGENT_A_INSTRUCTIONS",
            default="You are Agent-Citizen, representing a human user. Keep replies concise and conversational.",
        ),
    )
    agent_scholar = AgentConfig(
        identity=env_with_legacy("AGENT_SCHOLAR_IDENTITY", "AGENT_B_IDENTITY", default="agent-scholar"),
        provider=_provider_env("AGENT_SCHOLAR_PROVIDER", "AGENT_B_PROVIDER", default="gemini"),
        instructions=env_with_legacy(
            "AGENT_SCHOLAR_INSTRUCTIONS",
            "AGENT_B_INSTRUCTIONS",
            default="You are Agent-Scholar, the voice agent under test. Keep replies concise and conversational.",
        ),
    )
    agent_citizen, agent_scholar = _apply_scenario_instructions(agent_citizen, agent_scholar)
    _validate_provider(agent_citizen)
    _validate_provider(agent_scholar)
    openai_auth = (
        load_openai_auth()
        if _uses_provider("openai", agent_citizen, agent_scholar)
        else None
    )
    gemini_auth = load_gemini_auth() if _uses_provider("gemini", agent_citizen, agent_scholar) else None
    minicpm_url = (
        required_env("MINICPM_REALTIME_URL")
        if _uses_provider("minicpm", agent_citizen, agent_scholar)
        else os.getenv("MINICPM_REALTIME_URL")
    )
    freeze_omni_url = (
        required_env("FREEZE_OMNI_REALTIME_URL")
        if _uses_provider("freeze_omni", agent_citizen, agent_scholar)
        else os.getenv("FREEZE_OMNI_REALTIME_URL")
    )
    moshi_protocols = {
        provider: settings
        for provider in MOSHI_PROTOCOL_PROVIDERS
        if _uses_provider(provider, agent_citizen, agent_scholar)
        for settings in [load_moshi_settings(provider, required=True)]
        if settings is not None
    }

    gemini_backend = gemini_auth.backend if gemini_auth else "ai_studio"
    gemini_model = gemini_live_model(gemini_backend)
    uses_gemini_thinking_level = gemini_uses_thinking_level(gemini_model)

    return Settings(
        livekit_url=required_env("LIVEKIT_URL"),
        livekit_api_key=required_env("LIVEKIT_API_KEY"),
        livekit_api_secret=required_env("LIVEKIT_API_SECRET"),
        livekit_room=os.getenv("LIVEKIT_ROOM", "vox-symposium"),
        publish_sample_rate=int_env("LIVEKIT_PUBLISH_SAMPLE_RATE", 48_000),
        frame_ms=int_env("LIVEKIT_FRAME_MS", 20),
        agent_citizen=agent_citizen,
        agent_scholar=agent_scholar,
        openai_api_key=openai_auth.api_key if openai_auth else None,
        openai_backend=openai_auth.backend if openai_auth else "openai",
        openai_endpoint=openai_auth.endpoint if openai_auth else None,
        openai_api_version=openai_auth.api_version if openai_auth else None,
        openai_model=(
            openai_auth.model
            if openai_auth
            else os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2")
        ),
        openai_voice=os.getenv("OPENAI_REALTIME_VOICE", "marin"),
        openai_reasoning_effort=os.getenv("OPENAI_REALTIME_REASONING_EFFORT") or None,
        openai_ping_interval=float_env("OPENAI_REALTIME_PING_INTERVAL", 20.0),
        openai_ping_timeout=float_env("OPENAI_REALTIME_PING_TIMEOUT", 20.0),
        gemini_api_key=gemini_auth.api_key if gemini_auth else None,
        gemini_backend=gemini_backend,
        gemini_vertex_project=gemini_auth.project if gemini_auth else None,
        gemini_vertex_location=gemini_auth.location if gemini_auth else None,
        gemini_credentials_file=gemini_auth.credentials_file if gemini_auth else None,
        gemini_model=gemini_model,
        gemini_thinking_level=(gemini_thinking_level() if uses_gemini_thinking_level else "minimal"),
        gemini_thinking_budget=(None if uses_gemini_thinking_level else gemini_thinking_budget()),
        gemini_enable_affective_dialog=bool_env("GEMINI_LIVE_ENABLE_AFFECTIVE_DIALOG", False),
        gemini_initial_history=gemini_live_initial_history(),
        minicpm_realtime_url=minicpm_url,
        minicpm_api_key=os.getenv("MINICPM_API_KEY") or None,
        minicpm_length_penalty=float_env("MINICPM_LENGTH_PENALTY", 1.1),
        minicpm_input_chunk_ms=int_env("MINICPM_INPUT_CHUNK_MS", 1_000),
        minicpm_queue_timeout=float_env("MINICPM_QUEUE_TIMEOUT", 300.0),
        minicpm_ping_interval=optional_float_env("MINICPM_PING_INTERVAL", 30.0),
        minicpm_ping_timeout=optional_float_env("MINICPM_PING_TIMEOUT", 120.0),
        freeze_omni_realtime_url=(
            _socketio_url(freeze_omni_url) if freeze_omni_url else None
        ),
        freeze_omni_ssl_verify=bool_env("FREEZE_OMNI_SSL_VERIFY", False),
        freeze_omni_input_chunk_ms=int_env("FREEZE_OMNI_INPUT_CHUNK_MS", 20),
        freeze_omni_connect_timeout=float_env("FREEZE_OMNI_CONNECT_TIMEOUT", 60.0),
        freeze_omni_connect_retries=int_env("FREEZE_OMNI_CONNECT_RETRIES", 10),
        freeze_omni_connect_retry_delay=float_env("FREEZE_OMNI_CONNECT_RETRY_DELAY", 5.0),
        freeze_omni_prompt_timeout=float_env("FREEZE_OMNI_PROMPT_TIMEOUT", 60.0),
        freeze_omni_turn_start_delay=float_env("FREEZE_OMNI_TURN_START_DELAY", 3.0),
        freeze_omni_turn_preroll_silence_ms=int_env(
            "FREEZE_OMNI_TURN_PREROLL_SILENCE_MS",
            1200,
        ),
        freeze_omni_max_input_silence_ms=int_env(
            "FREEZE_OMNI_MAX_INPUT_SILENCE_MS",
            200,
        ),
        freeze_omni_input_silence_rms_threshold=float_env(
            "FREEZE_OMNI_INPUT_SILENCE_RMS_THRESHOLD",
            800.0,
        ),
        freeze_omni_post_turn_poll_seconds=float_env(
            "FREEZE_OMNI_POST_TURN_POLL_SECONDS",
            60.0,
        ),
        freeze_omni_post_turn_idle_seconds=float_env(
            "FREEZE_OMNI_POST_TURN_IDLE_SECONDS",
            3.0,
        ),
        freeze_omni_post_turn_poll_chunk_ms=int_env(
            "FREEZE_OMNI_POST_TURN_POLL_CHUNK_MS",
            160,
        ),
        freeze_omni_stop_recording_after_turn=bool_env(
            "FREEZE_OMNI_STOP_RECORDING_AFTER_TURN",
            True,
        ),
        moshi_protocols=moshi_protocols,
    )


def load_openai_auth() -> OpenAIAuthConfig:
    backend = normalized_env("OPENAI_BACKEND", "openai")
    if backend in {"openai", "official"}:
        return OpenAIAuthConfig(
            backend="openai",
            api_key=required_env("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2"),
        )
    if backend not in {"azure", "azure_openai"}:
        raise RuntimeError(
            f"OPENAI_BACKEND must be 'openai' or 'azure', got {backend!r}"
        )

    return OpenAIAuthConfig(
        backend="azure",
        api_key=required_env("AZURE_OPENAI_API_KEY"),
        endpoint=required_env("AZURE_OPENAI_ENDPOINT"),
        model=required_env("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION") or None,
    )


def load_gemini_auth() -> GeminiAuthConfig:
    backend = normalized_env("GEMINI_BACKEND", "ai_studio")
    if backend == "ai_studio":
        return GeminiAuthConfig(backend=backend, api_key=required_env("GEMINI_API_KEY"))
    if backend != "vertex":
        raise RuntimeError(
            f"GEMINI_BACKEND must be 'ai_studio' or 'vertex', got {backend!r}"
        )

    credentials_file = required_env("GOOGLE_APPLICATION_CREDENTIALS")
    credentials_path = Path(credentials_file).expanduser()
    if not credentials_path.is_file():
        raise RuntimeError(
            f"GOOGLE_APPLICATION_CREDENTIALS does not point to a file: {credentials_file}"
        )
    return GeminiAuthConfig(
        backend=backend,
        project=required_env("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        credentials_file=str(credentials_path),
    )


def gemini_live_model(backend: str) -> str:
    default = (
        "gemini-live-2.5-flash-native-audio"
        if backend == "vertex"
        else "gemini-3.1-flash-live-preview"
    )
    return os.getenv("GEMINI_LIVE_MODEL", default)


def gemini_uses_thinking_level(model: str) -> bool:
    return model.removeprefix("models/").startswith("gemini-3.")


def gemini_thinking_level() -> str:
    level = normalized_env("GEMINI_LIVE_THINKING_LEVEL", "minimal")
    allowed = {"minimal", "low", "medium", "high"}
    if level not in allowed:
        values = "', '".join(sorted(allowed))
        raise RuntimeError(
            "GEMINI_LIVE_THINKING_LEVEL must be one of "
            f"'{values}', got {level!r}"
        )
    return level


def gemini_thinking_budget() -> int | None:
    budget = optional_int_env("GEMINI_LIVE_THINKING_BUDGET")
    if budget is not None and budget != -1 and not 0 <= budget <= 24_576:
        raise RuntimeError(
            "GEMINI_LIVE_THINKING_BUDGET must be -1 or an integer from 0 to 24576"
        )
    return budget


def gemini_live_initial_history() -> tuple[dict[str, Any], ...]:
    """Load Live API initial history used with send_client_content.

    Gemini 3.1 accepts client content only while seeding a session's history;
    live audio and text must continue through send_realtime_input.
    """
    raw = os.getenv("GEMINI_LIVE_INITIAL_HISTORY_JSON")
    if raw is None or not raw.strip():
        return ()
    try:
        history = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GEMINI_LIVE_INITIAL_HISTORY_JSON must be valid JSON") from exc
    if not isinstance(history, list) or not history:
        raise RuntimeError("GEMINI_LIVE_INITIAL_HISTORY_JSON must be a non-empty JSON array")
    for turn in history:
        if not isinstance(turn, dict):
            raise RuntimeError("GEMINI_LIVE_INITIAL_HISTORY_JSON entries must be JSON objects")
        if turn.get("role") not in {"user", "model"}:
            raise RuntimeError(
                "GEMINI_LIVE_INITIAL_HISTORY_JSON entries must have role 'user' or 'model'"
            )
        if not isinstance(turn.get("parts"), list) or not turn["parts"]:
            raise RuntimeError(
                "GEMINI_LIVE_INITIAL_HISTORY_JSON entries must have a non-empty parts array"
            )
    return tuple(history)


def load_moshi_settings(provider: str = "moshi", *, required: bool) -> MoshiSettings | None:
    provider = normalize_provider(provider)
    prefix = provider_env_prefix(provider)
    url = first_env(_env_names(provider, "REALTIME_URL"))
    if required and not url:
        raise RuntimeError(f"Missing required environment variable: {prefix}_REALTIME_URL")
    if not url:
        return None
    return MoshiSettings(
        url=_websocket_url(url, default_path="/api/chat"),
        api_key=first_env(_env_names(provider, "API_KEY")) or None,
    )


def _provider_env(primary: str, legacy: str, *, default: str) -> str:
    return normalize_provider(env_with_legacy(primary, legacy, default=default))


def _apply_scenario_instructions(
    agent_citizen: AgentConfig,
    agent_scholar: AgentConfig,
) -> tuple[AgentConfig, AgentConfig]:
    scenario_file = os.getenv("SCENARIO_FILE")
    if not scenario_file:
        return agent_citizen, agent_scholar

    from vox_symposium.scenario import load_scenario

    scenario = load_scenario(
        scenario_file,
        scenario_id=os.getenv("SCENARIO_ID"),
        scenario_index=optional_int_env("SCENARIO_INDEX"),
        audio_dir=os.getenv("SCENARIO_AUDIO_DIR"),
        dialogue_turns=int_env("SCENARIO_DIALOGUE_TURNS", 5),
    )
    return (
        replace(
            agent_citizen,
            instructions=scenario.build_instructions(
                "citizen",
                dialogue_behavior_extra=_provider_dialogue_behavior_extra(agent_citizen.provider),
            ),
        ),
        replace(
            agent_scholar,
            instructions=scenario.build_instructions(
                "scholar",
                dialogue_behavior_extra=_provider_dialogue_behavior_extra(agent_scholar.provider),
            ),
        ),
    )


def _provider_dialogue_behavior_extra(provider: str) -> str | None:
    if normalize_provider(provider) != "minicpm":
        return None

    from vox_symposium.scenario import MINICPM_DIALOGUE_BEHAVIOR

    return MINICPM_DIALOGUE_BEHAVIOR


def _uses_provider(provider: str, *agents: AgentConfig) -> bool:
    return any(normalize_provider(agent.provider) == provider for agent in agents)


def _validate_provider(agent: AgentConfig) -> None:
    if normalize_provider(agent.provider) not in SUPPORTED_PROVIDERS:
        supported = "', '".join(sorted(SUPPORTED_PROVIDERS))
        raise RuntimeError(
            f"{agent.identity} provider must be one of '{supported}', "
            f"got {agent.provider!r}"
        )


def _env_names(provider: str, suffix: str) -> list[str]:
    prefix = provider_env_prefix(provider)
    return [f"{prefix}_{suffix}"]


def _websocket_url(url: str, *, default_path: str | None = None) -> str:
    parsed = urlsplit(url)
    scheme = parsed.scheme
    if scheme in {"http", "https"}:
        scheme = "wss" if scheme == "https" else "ws"
    if scheme not in {"ws", "wss"} or not parsed.netloc:
        raise RuntimeError(f"Realtime URL must be ws:// or wss://, got {url!r}")

    path = parsed.path
    if default_path and path in {"", "/"}:
        path = default_path
    return urlunsplit((scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def _socketio_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.netloc:
        raise RuntimeError(f"Socket.IO URL must be http(s):// or ws(s)://, got {url!r}")
    if parsed.hostname in {"0.0.0.0", "::"}:
        raise RuntimeError(
            "Socket.IO URL cannot use a wildcard address; "
            "use 127.0.0.1, a host name, or the server IP"
        )
    return url
