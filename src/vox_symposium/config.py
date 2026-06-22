from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False


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
    gemini_api_key: str | None
    gemini_backend: str
    gemini_vertex_project: str | None
    gemini_vertex_location: str | None
    gemini_credentials_file: str | None
    gemini_model: str
    minicpm_realtime_url: str | None
    minicpm_api_key: str | None
    minicpm_length_penalty: float
    minicpm_input_chunk_ms: int
    minicpm_queue_timeout: float


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


def load_settings() -> Settings:
    load_dotenv()
    agent_citizen = AgentConfig(
        identity=_env("AGENT_CITIZEN_IDENTITY", "AGENT_A_IDENTITY", default="agent-citizen"),
        provider=_env("AGENT_CITIZEN_PROVIDER", "AGENT_A_PROVIDER", default="openai").lower(),
        instructions=_env(
            "AGENT_CITIZEN_INSTRUCTIONS",
            "AGENT_A_INSTRUCTIONS",
            default="You are Agent-Citizen, representing a human user. Keep replies concise and conversational.",
        ),
    )
    agent_scholar = AgentConfig(
        identity=_env("AGENT_SCHOLAR_IDENTITY", "AGENT_B_IDENTITY", default="agent-scholar"),
        provider=_env("AGENT_SCHOLAR_PROVIDER", "AGENT_B_PROVIDER", default="gemini").lower(),
        instructions=_env(
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
        _required("MINICPM_REALTIME_URL")
        if _uses_provider("minicpm", agent_citizen, agent_scholar)
        else os.getenv("MINICPM_REALTIME_URL")
    )

    return Settings(
        livekit_url=_required("LIVEKIT_URL"),
        livekit_api_key=_required("LIVEKIT_API_KEY"),
        livekit_api_secret=_required("LIVEKIT_API_SECRET"),
        livekit_room=os.getenv("LIVEKIT_ROOM", "vox-symposium"),
        publish_sample_rate=_int_env("LIVEKIT_PUBLISH_SAMPLE_RATE", 48_000),
        frame_ms=_int_env("LIVEKIT_FRAME_MS", 20),
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
        gemini_api_key=gemini_auth.api_key if gemini_auth else None,
        gemini_backend=gemini_auth.backend if gemini_auth else "ai_studio",
        gemini_vertex_project=gemini_auth.project if gemini_auth else None,
        gemini_vertex_location=gemini_auth.location if gemini_auth else None,
        gemini_credentials_file=gemini_auth.credentials_file if gemini_auth else None,
        gemini_model=gemini_live_model(gemini_auth.backend if gemini_auth else "ai_studio"),
        minicpm_realtime_url=minicpm_url,
        minicpm_api_key=os.getenv("MINICPM_API_KEY") or None,
        minicpm_length_penalty=_float_env("MINICPM_LENGTH_PENALTY", 1.1),
        minicpm_input_chunk_ms=_int_env("MINICPM_INPUT_CHUNK_MS", 1_000),
        minicpm_queue_timeout=_float_env("MINICPM_QUEUE_TIMEOUT", 300.0),
    )


def load_openai_auth() -> OpenAIAuthConfig:
    backend = os.getenv("OPENAI_BACKEND", "openai").strip().lower().replace("-", "_")
    if backend in {"openai", "official"}:
        return OpenAIAuthConfig(
            backend="openai",
            api_key=_required("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2"),
        )
    if backend not in {"azure", "azure_openai"}:
        raise RuntimeError(
            f"OPENAI_BACKEND must be 'openai' or 'azure', got {backend!r}"
        )

    return OpenAIAuthConfig(
        backend="azure",
        api_key=_required("AZURE_OPENAI_API_KEY"),
        endpoint=_required("AZURE_OPENAI_ENDPOINT"),
        model=_required("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION") or None,
    )


def load_gemini_auth() -> GeminiAuthConfig:
    backend = os.getenv("GEMINI_BACKEND", "ai_studio").strip().lower().replace("-", "_")
    if backend == "ai_studio":
        return GeminiAuthConfig(backend=backend, api_key=_required("GEMINI_API_KEY"))
    if backend != "vertex":
        raise RuntimeError(
            f"GEMINI_BACKEND must be 'ai_studio' or 'vertex', got {backend!r}"
        )

    credentials_file = _required("GOOGLE_APPLICATION_CREDENTIALS")
    credentials_path = Path(credentials_file).expanduser()
    if not credentials_path.is_file():
        raise RuntimeError(
            f"GOOGLE_APPLICATION_CREDENTIALS does not point to a file: {credentials_file}"
        )
    return GeminiAuthConfig(
        backend=backend,
        project=_required("GOOGLE_CLOUD_PROJECT"),
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


def _env(primary: str, legacy: str, *, default: str) -> str:
    return os.getenv(primary) or os.getenv(legacy) or default


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
        scenario_index=_optional_int_env("SCENARIO_INDEX"),
        audio_dir=os.getenv("SCENARIO_AUDIO_DIR"),
        dialogue_turns=_int_env("SCENARIO_DIALOGUE_TURNS", 5),
    )
    return (
        replace(agent_citizen, instructions=scenario.build_instructions("citizen")),
        replace(agent_scholar, instructions=scenario.build_instructions("scholar")),
    )


def _uses_provider(provider: str, *agents: AgentConfig) -> bool:
    return any(agent.provider == provider for agent in agents)


def _validate_provider(agent: AgentConfig) -> None:
    if agent.provider not in {"openai", "gemini", "minicpm"}:
        raise RuntimeError(
            f"{agent.identity} provider must be 'openai', 'gemini', or 'minicpm', "
            f"got {agent.provider!r}"
        )


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from exc


def _optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
