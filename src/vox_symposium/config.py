from __future__ import annotations

import os
from dataclasses import dataclass

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
    openai_model: str
    openai_voice: str
    gemini_api_key: str | None
    gemini_model: str
    local_model_provider: str
    local_model: str
    local_model_device: str
    local_model_dtype: str
    local_model_attn_implementation: str
    local_model_ref_audio: str | None
    local_model_language: str
    local_model_turn_detection: bool
    local_model_silero_threshold: float
    local_model_turn_silence_ms: int
    local_model_min_turn_ms: int
    local_model_chunk_ms: int
    local_model_max_new_tokens: int
    local_model_qwen_speaker: str


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
    _validate_provider(agent_citizen)
    _validate_provider(agent_scholar)

    return Settings(
        livekit_url=_required("LIVEKIT_URL"),
        livekit_api_key=_required("LIVEKIT_API_KEY"),
        livekit_api_secret=_required("LIVEKIT_API_SECRET"),
        livekit_room=os.getenv("LIVEKIT_ROOM", "vox-symposium"),
        publish_sample_rate=_int_env("LIVEKIT_PUBLISH_SAMPLE_RATE", 48_000),
        frame_ms=_int_env("LIVEKIT_FRAME_MS", 20),
        agent_citizen=agent_citizen,
        agent_scholar=agent_scholar,
        openai_api_key=_required("OPENAI_API_KEY") if _uses_provider("openai", agent_citizen, agent_scholar) else None,
        openai_model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2"),
        openai_voice=os.getenv("OPENAI_REALTIME_VOICE", "marin"),
        gemini_api_key=_required("GEMINI_API_KEY") if _uses_provider("gemini", agent_citizen, agent_scholar) else None,
        gemini_model=os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview"),
        local_model_provider=os.getenv("LOCAL_MODEL_PROVIDER", "auto"),
        local_model=os.getenv("LOCAL_MODEL") or os.getenv("LOCAL_MODEL_NAME", "openbmb/MiniCPM-o-4_5"),
        local_model_device=os.getenv("LOCAL_MODEL_DEVICE", "auto"),
        local_model_dtype=os.getenv("LOCAL_MODEL_DTYPE", "auto"),
        local_model_attn_implementation=os.getenv("LOCAL_MODEL_ATTN_IMPLEMENTATION", "sdpa"),
        local_model_ref_audio=os.getenv("LOCAL_MODEL_REF_AUDIO") or None,
        local_model_language=os.getenv("LOCAL_MODEL_LANGUAGE", "en"),
        local_model_turn_detection=_bool_env("LOCAL_MODEL_TURN_DETECTION", True),
        local_model_silero_threshold=_float_env("LOCAL_MODEL_SILERO_THRESHOLD", 0.5),
        local_model_turn_silence_ms=_int_env("LOCAL_MODEL_TURN_SILENCE_MS", 700),
        local_model_min_turn_ms=_int_env("LOCAL_MODEL_MIN_TURN_MS", 400),
        local_model_chunk_ms=_int_env("LOCAL_MODEL_CHUNK_MS", 1000),
        local_model_max_new_tokens=_int_env("LOCAL_MODEL_MAX_NEW_TOKENS", 512),
        local_model_qwen_speaker=os.getenv("LOCAL_MODEL_QWEN_SPEAKER", "Ethan"),
    )


def _env(primary: str, legacy: str, *, default: str) -> str:
    return os.getenv(primary) or os.getenv(legacy) or default


def _uses_provider(provider: str, *agents: AgentConfig) -> bool:
    return any(agent.provider == provider for agent in agents)


def _validate_provider(agent: AgentConfig) -> None:
    if agent.provider not in {"openai", "gemini", "local", "hf"}:
        raise RuntimeError(
            f"{agent.identity} provider must be 'openai', 'gemini', 'local', or 'hf', got {agent.provider!r}"
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


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean, got {raw!r}")
