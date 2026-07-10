from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from vox_symposium.env import (
    bool_env,
    env_with_legacy,
    first_env,
    float_env,
    int_env,
    load_environment,
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
    initial_history: tuple[dict[str, Any], ...] = ()


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


@dataclass(frozen=True)
class OpenAISettings:
    auth: OpenAIAuthConfig
    voice: str
    reasoning_effort: str | None
    ping_interval: float
    ping_timeout: float


@dataclass(frozen=True)
class GeminiSettings:
    auth: GeminiAuthConfig
    model: str
    thinking_level: str
    thinking_budget: int | None
    enable_affective_dialog: bool
    initial_history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class MiniCPMSettings:
    url: str
    api_key: str | None
    length_penalty: float
    input_chunk_ms: int
    queue_timeout: float
    ping_interval: float | None
    ping_timeout: float | None


@dataclass(frozen=True)
class FreezeOmniSettings:
    url: str
    ssl_verify: bool
    input_chunk_ms: int
    connect_timeout: float
    connect_retries: int
    connect_retry_delay: float
    prompt_timeout: float
    turn_start_delay: float
    turn_preroll_silence_ms: int
    max_input_silence_ms: int
    input_silence_rms_threshold: float
    post_turn_poll_seconds: float
    post_turn_idle_seconds: float
    post_turn_poll_chunk_ms: int
    stop_recording_after_turn: bool


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
    openai: OpenAISettings | None
    gemini: GeminiSettings | None
    minicpm: MiniCPMSettings | None
    freeze_omni: FreezeOmniSettings | None
    moshi_protocols: dict[str, MoshiSettings]


def load_settings() -> Settings:
    load_environment()
    agent_citizen = _load_agent(
        role="citizen",
        legacy_role="A",
        default_provider="openai",
        default_instructions=(
            "You are Agent-Citizen, representing a human user. "
            "Keep replies concise and conversational."
        ),
    )
    agent_scholar = _load_agent(
        role="scholar",
        legacy_role="B",
        default_provider="gemini",
        default_instructions=(
            "You are Agent-Scholar, the voice agent under test. "
            "Keep replies concise and conversational."
        ),
    )
    agent_citizen, agent_scholar = _apply_scenario_instructions(agent_citizen, agent_scholar)
    _validate_provider(agent_citizen)
    _validate_provider(agent_scholar)
    agents = (agent_citizen, agent_scholar)
    moshi_protocols = {
        provider: settings
        for provider in MOSHI_PROTOCOL_PROVIDERS
        if _uses_provider(provider, *agents)
        for settings in [load_moshi_settings(provider, required=True)]
        if settings is not None
    }

    return Settings(
        livekit_url=required_env("LIVEKIT_URL"),
        livekit_api_key=required_env("LIVEKIT_API_KEY"),
        livekit_api_secret=required_env("LIVEKIT_API_SECRET"),
        livekit_room=os.getenv("LIVEKIT_ROOM", "vox-symposium"),
        publish_sample_rate=int_env("LIVEKIT_PUBLISH_SAMPLE_RATE", 48_000),
        frame_ms=int_env("LIVEKIT_FRAME_MS", 20),
        agent_citizen=agent_citizen,
        agent_scholar=agent_scholar,
        openai=(load_openai_settings() if _uses_provider("openai", *agents) else None),
        gemini=(load_gemini_settings() if _uses_provider("gemini", *agents) else None),
        minicpm=(load_minicpm_settings() if _uses_provider("minicpm", *agents) else None),
        freeze_omni=(
            load_freeze_omni_settings() if _uses_provider("freeze_omni", *agents) else None
        ),
        moshi_protocols=moshi_protocols,
    )


def _load_agent(
    *,
    role: str,
    legacy_role: str,
    default_provider: str,
    default_instructions: str,
) -> AgentConfig:
    prefix = f"AGENT_{role.upper()}"
    legacy_prefix = f"AGENT_{legacy_role}"
    return AgentConfig(
        identity=env_with_legacy(
            f"{prefix}_IDENTITY",
            f"{legacy_prefix}_IDENTITY",
            default=f"agent-{role}",
        ),
        provider=_provider_env(
            f"{prefix}_PROVIDER",
            f"{legacy_prefix}_PROVIDER",
            default=default_provider,
        ),
        instructions=env_with_legacy(
            f"{prefix}_INSTRUCTIONS",
            f"{legacy_prefix}_INSTRUCTIONS",
            default=default_instructions,
        ),
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
        raise RuntimeError(f"OPENAI_BACKEND must be 'openai' or 'azure', got {backend!r}")

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
        raise RuntimeError(f"GEMINI_BACKEND must be 'ai_studio' or 'vertex', got {backend!r}")

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


def load_openai_settings() -> OpenAISettings:
    return OpenAISettings(
        auth=load_openai_auth(),
        voice=os.getenv("OPENAI_REALTIME_VOICE", "marin"),
        reasoning_effort=os.getenv("OPENAI_REALTIME_REASONING_EFFORT") or None,
        ping_interval=float_env("OPENAI_REALTIME_PING_INTERVAL", 20.0),
        ping_timeout=float_env("OPENAI_REALTIME_PING_TIMEOUT", 20.0),
    )


def load_gemini_settings() -> GeminiSettings:
    auth = load_gemini_auth()
    model = gemini_live_model(auth.backend)
    uses_thinking_level = gemini_uses_thinking_level(model)
    return GeminiSettings(
        auth=auth,
        model=model,
        thinking_level=gemini_thinking_level() if uses_thinking_level else "minimal",
        thinking_budget=None if uses_thinking_level else gemini_thinking_budget(),
        enable_affective_dialog=bool_env("GEMINI_LIVE_ENABLE_AFFECTIVE_DIALOG", False),
        initial_history=(gemini_live_initial_history() if uses_thinking_level else ()),
    )


def load_minicpm_settings() -> MiniCPMSettings:
    return MiniCPMSettings(
        url=required_env("MINICPM_REALTIME_URL"),
        api_key=os.getenv("MINICPM_API_KEY") or None,
        length_penalty=float_env("MINICPM_LENGTH_PENALTY", 1.1),
        input_chunk_ms=int_env("MINICPM_INPUT_CHUNK_MS", 1_000),
        queue_timeout=float_env("MINICPM_QUEUE_TIMEOUT", 300.0),
        ping_interval=optional_float_env("MINICPM_PING_INTERVAL", 30.0),
        ping_timeout=optional_float_env("MINICPM_PING_TIMEOUT", 120.0),
    )


def load_freeze_omni_settings() -> FreezeOmniSettings:
    return FreezeOmniSettings(
        url=_socketio_url(required_env("FREEZE_OMNI_REALTIME_URL")),
        ssl_verify=bool_env("FREEZE_OMNI_SSL_VERIFY", False),
        input_chunk_ms=int_env("FREEZE_OMNI_INPUT_CHUNK_MS", 20),
        connect_timeout=float_env("FREEZE_OMNI_CONNECT_TIMEOUT", 60.0),
        connect_retries=int_env("FREEZE_OMNI_CONNECT_RETRIES", 10),
        connect_retry_delay=float_env("FREEZE_OMNI_CONNECT_RETRY_DELAY", 5.0),
        prompt_timeout=float_env("FREEZE_OMNI_PROMPT_TIMEOUT", 60.0),
        turn_start_delay=float_env("FREEZE_OMNI_TURN_START_DELAY", 3.0),
        turn_preroll_silence_ms=int_env("FREEZE_OMNI_TURN_PREROLL_SILENCE_MS", 1200),
        max_input_silence_ms=int_env("FREEZE_OMNI_MAX_INPUT_SILENCE_MS", 200),
        input_silence_rms_threshold=float_env("FREEZE_OMNI_INPUT_SILENCE_RMS_THRESHOLD", 800.0),
        post_turn_poll_seconds=float_env("FREEZE_OMNI_POST_TURN_POLL_SECONDS", 60.0),
        post_turn_idle_seconds=float_env("FREEZE_OMNI_POST_TURN_IDLE_SECONDS", 3.0),
        post_turn_poll_chunk_ms=int_env("FREEZE_OMNI_POST_TURN_POLL_CHUNK_MS", 160),
        stop_recording_after_turn=bool_env("FREEZE_OMNI_STOP_RECORDING_AFTER_TURN", True),
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
        raise RuntimeError(f"GEMINI_LIVE_THINKING_LEVEL must be one of '{values}', got {level!r}")
    return level


def gemini_thinking_budget() -> int | None:
    budget = optional_int_env("GEMINI_LIVE_THINKING_BUDGET")
    if budget is not None and budget != -1 and not 0 <= budget <= 24_576:
        raise RuntimeError("GEMINI_LIVE_THINKING_BUDGET must be -1 or an integer from 0 to 24576")
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
    citizen_uses_initial_history = provider_uses_structured_history(agent_citizen.provider)
    scholar_uses_initial_history = provider_uses_structured_history(agent_scholar.provider)
    citizen_prompt = scenario.build_prompt(
        "citizen",
        provider=agent_citizen.provider,
        use_structured_history=citizen_uses_initial_history,
    )
    scholar_prompt = scenario.build_prompt(
        "scholar",
        provider=agent_scholar.provider,
        use_structured_history=scholar_uses_initial_history,
    )
    return (
        replace(
            agent_citizen,
            instructions=citizen_prompt.instructions,
            initial_history=citizen_prompt.initial_history,
        ),
        replace(
            agent_scholar,
            instructions=scholar_prompt.instructions,
            initial_history=scholar_prompt.initial_history,
        ),
    )


def provider_uses_structured_history(provider: str) -> bool:
    """Return whether the selected provider/model accepts structured scenario history."""
    if normalize_provider(provider) != "gemini":
        return False
    backend = normalized_env("GEMINI_BACKEND", "ai_studio")
    return gemini_uses_thinking_level(gemini_live_model(backend))


def _uses_provider(provider: str, *agents: AgentConfig) -> bool:
    return any(normalize_provider(agent.provider) == provider for agent in agents)


def _validate_provider(agent: AgentConfig) -> None:
    if normalize_provider(agent.provider) not in SUPPORTED_PROVIDERS:
        supported = "', '".join(sorted(SUPPORTED_PROVIDERS))
        raise RuntimeError(
            f"{agent.identity} provider must be one of '{supported}', got {agent.provider!r}"
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
