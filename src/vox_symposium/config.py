from __future__ import annotations

import os
from dataclasses import dataclass, replace

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
    )


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
    if agent.provider not in {"openai", "gemini"}:
        raise RuntimeError(
            f"{agent.identity} provider must be 'openai' or 'gemini', got {agent.provider!r}"
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


def _optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
