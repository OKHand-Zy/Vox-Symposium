from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from vox_symposium.config import gemini_live_model
from vox_symposium.env import env_with_legacy, normalized_env
from vox_symposium.json_io import read_json, write_json
from vox_symposium.providers import normalize_provider
from vox_symposium.scenario import AGENT_KEYS, HUMAN_AGENT, AgentKey, validate_agent


@dataclass(frozen=True)
class EvaluationArtifacts:
    root: Path
    console_log: Path
    summary: Path
    env_snapshot: Path

    @classmethod
    def create(
        cls,
        result_path: Path,
        *,
        run_id: str,
        artifact_dir: str | None,
    ) -> EvaluationArtifacts:
        root = Path(artifact_dir) if artifact_dir else result_path.parent / f"{run_id}-artifacts"
        root.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            console_log=root / "console-log.txt",
            summary=root / "summary.json",
            env_snapshot=root / "run-env.txt",
        )


def scenario_artifact_dir(artifact_root: Path, scenario: dict[str, Any]) -> Path:
    return artifact_root / _safe_path_segment(_scenario_row_id(scenario))


def scenario_run_id(
    base_run_id: str,
    scenario: dict[str, Any],
    *,
    index: int,
    total: int,
) -> str:
    if total == 1:
        return base_run_id
    row_id = _safe_path_segment(_scenario_row_id(scenario))
    return f"{base_run_id}-{index + 1:04d}-{row_id}"


def remove_failed_case_artifacts(artifact_root: Path, scenario: dict[str, Any]) -> None:
    artifact_dir = scenario_artifact_dir(artifact_root, scenario)
    if not artifact_dir.exists():
        return

    root = artifact_root.resolve()
    target = artifact_dir.resolve()
    if target == root or not target.is_relative_to(root):
        raise RuntimeError(
            f"Refusing to delete case artifacts outside artifact root: {artifact_dir}"
        )

    shutil.rmtree(artifact_dir)
    print(f"Deleted failed case artifact directory: {artifact_dir}")


def result_payload(
    results: list[dict[str, Any]], *, total: int
) -> dict[str, Any] | list[dict[str, Any]]:
    if total == 1:
        return results[0]
    return results


def load_resume_results(path: Path, *, start_index: int) -> list[dict[str, Any]]:
    if start_index <= 0 or not path.is_file():
        return []

    payload = read_json(path)
    existing = payload if isinstance(payload, list) else [payload]
    results: list[dict[str, Any]] = []
    for fallback_index, result in enumerate(existing):
        if not isinstance(result, dict):
            continue
        case_index = _result_case_index(result, fallback_index)
        if case_index >= start_index:
            continue
        resumed = dict(result)
        resumed["case_index"] = case_index
        resumed["case_number"] = case_index + 1
        results.append(resumed)
    return results


def build_summary_report(
    results: list[dict[str, Any]],
    *,
    run_id: str,
    scenario_path: str,
    result_path: Path,
    artifacts: EvaluationArtifacts,
    total_cases: int | None = None,
) -> dict[str, Any]:
    cases = [_summary_case(index, result) for index, result in enumerate(results)]
    passed = sum(1 for case in cases if case["status"] == "passed")
    failed = sum(1 for case in cases if case["status"] == "failed")
    unknown = sum(1 for case in cases if case["status"] == "unknown")
    completed = len(cases)
    total = total_cases if total_cases is not None else completed
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scenario": scenario_path,
        "result": str(result_path),
        "artifact_root": str(artifacts.root),
        "env_snapshot": str(artifacts.env_snapshot),
        "console_log": str(artifacts.console_log),
        "total_cases": total,
        "completed_cases": completed,
        "passed_cases": passed,
        "failed_cases": failed,
        "unknown_cases": unknown,
        "pass_rate": passed / total if total else None,
        "completed_pass_rate": passed / completed if completed else None,
        "cases": cases,
    }


def write_summary_report(
    results: list[dict[str, Any]],
    *,
    run_id: str,
    scenario_path: str,
    result_path: Path,
    artifacts: EvaluationArtifacts,
    total_cases: int | None = None,
) -> None:
    report = build_summary_report(
        results,
        run_id=run_id,
        scenario_path=scenario_path,
        result_path=result_path,
        artifacts=artifacts,
        total_cases=total_cases,
    )
    write_json(artifacts.summary, report)


def _summary_case(index: int, result: dict[str, Any]) -> dict[str, Any]:
    response = result.get("response") or {}
    evaluation = result.get("evaluation") or {}
    artifacts = result.get("artifacts") or {}
    is_correct = response.get("is_correct")
    status = result.get("status") or (
        "passed" if is_correct is True else "failed" if is_correct is False else "unknown"
    )
    case_index = _result_case_index(result, index)
    return {
        "index": case_index,
        "case_number": result.get("case_number", case_index + 1),
        "scenario_id": result.get("scenario_id"),
        "run_id": result.get("run_id"),
        "status": status,
        "is_correct": is_correct,
        "choice": response.get("choice"),
        "correct_answer": evaluation.get("correct_answer"),
        "response_text": response.get("text"),
        "response_audio": response.get("audio"),
        "dialogue_log": artifacts.get("dialogue_log"),
        "error": result.get("error"),
    }


class _TeeStream:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(stream.isatty() for stream in self._streams)

    @property
    def encoding(self) -> str:
        return getattr(self._streams[0], "encoding", None) or "utf-8"


@contextmanager
def tee_console(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    should_separate = path.is_file() and path.stat().st_size > 0
    separator_prefix = ""
    if should_separate:
        with path.open("rb") as existing:
            existing.seek(-1, os.SEEK_END)
            if existing.read(1) != b"\n":
                separator_prefix = "\n"
    with path.open("a", encoding="utf-8") as file:
        if should_separate:
            file.write(f"{separator_prefix}###################################\n")
        stdout = _TeeStream(sys.stdout, file)
        stderr = _TeeStream(sys.stderr, file)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield


def format_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def format_error(exc: Exception) -> str:
    if isinstance(exc, RuntimeError):
        return f"Error: {exc}"
    return f"Error: {type(exc).__name__}: {exc}"


def write_run_env_snapshot(
    path: Path,
    *,
    run_id: str,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Vox Symposium evaluation environment snapshot.",
        "# Secrets such as API keys and API secrets are intentionally omitted.",
        "# Values reflect the process environment after .env loading and runner defaults.",
        "",
        f"RUN_ID={_env_value(run_id)}",
        f"SCENARIO={_env_value(args.scenario)}",
        f"RESULT={_env_value(args.result)}",
        f"DIALOGUE_TURNS={args.dialogue_turns}",
        f"START_INDEX={args.start_index}",
        f"LIMIT={_env_value(args.limit or '')}",
        f"CASE_DELAY={args.case_delay}",
        f"CASE_RETRIES={args.case_retries}",
        f"CASE_RETRY_DELAY={args.case_retry_delay}",
        f"OVERNIGHT={args.overnight}",
        f"AUDIO_SPEED={args.audio_speed}",
        f"FRAME_MS={args.frame_ms}",
        "",
    ]

    for agent in AGENT_KEYS:
        lines.extend(_agent_env_snapshot(agent))
        lines.append("")

    lines.extend(_provider_env_snapshot())
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Saved environment snapshot: {path}")


def _agent_env_snapshot(agent: AgentKey) -> list[str]:
    agent = validate_agent(agent)
    legacy_agent = "A" if agent == HUMAN_AGENT else "B"
    prefix = f"AGENT_{agent.upper()}"
    provider = normalize_provider(
        env_with_legacy(
            f"{prefix}_PROVIDER",
            f"AGENT_{legacy_agent}_PROVIDER",
            default=_default_provider(agent),
        )
    )
    identity = env_with_legacy(
        f"{prefix}_IDENTITY",
        f"AGENT_{legacy_agent}_IDENTITY",
        default=f"agent-{agent}",
    )
    model, backend, extra = _effective_model_snapshot(provider)
    lines = [
        f"{prefix}_IDENTITY={_env_value(identity)}",
        f"{prefix}_PROVIDER={_env_value(provider)}",
        f"{prefix}_MODEL={_env_value(model)}",
    ]
    if backend:
        lines.append(f"{prefix}_BACKEND={_env_value(backend)}")
    lines.extend(f"{prefix}_{key}={_env_value(value)}" for key, value in extra.items())
    return lines


def _effective_model_snapshot(provider: str) -> tuple[str, str | None, dict[str, str]]:
    provider = normalize_provider(provider)
    if provider == "openai":
        backend = normalized_env("OPENAI_BACKEND", "openai")
        if backend in {"azure", "azure_openai"}:
            return os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""), "azure", {}
        return (
            os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2"),
            "openai",
            {
                "VOICE": os.getenv("OPENAI_REALTIME_VOICE", "marin"),
                "REASONING_EFFORT": os.getenv("OPENAI_REALTIME_REASONING_EFFORT", ""),
                "PING_INTERVAL": os.getenv("OPENAI_REALTIME_PING_INTERVAL", "20.0"),
                "PING_TIMEOUT": os.getenv("OPENAI_REALTIME_PING_TIMEOUT", "20.0"),
            },
        )
    if provider == "gemini":
        backend = normalized_env("GEMINI_BACKEND", "ai_studio")
        return gemini_live_model(backend), backend, {}
    if provider == "minicpm":
        return (
            os.getenv("MINICPM_REALTIME_MODEL", "minicpm-realtime-gateway"),
            None,
            {
                "PING_INTERVAL": os.getenv("MINICPM_PING_INTERVAL", "30.0"),
                "PING_TIMEOUT": os.getenv("MINICPM_PING_TIMEOUT", "120.0"),
            },
        )
    if provider == "freeze_omni":
        return (
            os.getenv("FREEZE_OMNI_MODEL", "freeze-omni"),
            "socketio",
            {
                "REALTIME_URL": _redacted_url(os.getenv("FREEZE_OMNI_REALTIME_URL", "")),
            },
        )
    if provider == "moshi":
        return (
            os.getenv("MOSHI_MODEL", "moshi"),
            "moshi",
            {
                "REALTIME_URL": _redacted_url(os.getenv("MOSHI_REALTIME_URL", "")),
            },
        )
    if provider == "personaplex":
        return (
            os.getenv("PERSONAPLEX_MODEL", "personaplex"),
            "moshi",
            {
                "REALTIME_URL": _redacted_url(os.getenv("PERSONAPLEX_REALTIME_URL", "")),
            },
        )
    return "", None, {}


def _provider_env_snapshot() -> list[str]:
    openai_backend = normalized_env("OPENAI_BACKEND", "openai")
    gemini_backend = normalized_env("GEMINI_BACKEND", "ai_studio")
    groups = [
        [
            ("OPENAI_BACKEND", openai_backend),
            ("OPENAI_REALTIME_MODEL", os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2")),
            ("OPENAI_REALTIME_VOICE", os.getenv("OPENAI_REALTIME_VOICE", "marin")),
            ("OPENAI_REALTIME_REASONING_EFFORT", os.getenv("OPENAI_REALTIME_REASONING_EFFORT", "")),
            ("OPENAI_REALTIME_PING_INTERVAL", os.getenv("OPENAI_REALTIME_PING_INTERVAL", "20.0")),
            ("OPENAI_REALTIME_PING_TIMEOUT", os.getenv("OPENAI_REALTIME_PING_TIMEOUT", "20.0")),
            ("AZURE_OPENAI_DEPLOYMENT_NAME", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")),
            ("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT", "")),
            ("AZURE_OPENAI_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "")),
        ],
        [
            ("GEMINI_BACKEND", gemini_backend),
            ("GEMINI_LIVE_MODEL", gemini_live_model(gemini_backend)),
            ("GOOGLE_CLOUD_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", "")),
            ("GOOGLE_CLOUD_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")),
        ],
        [
            (
                "MINICPM_REALTIME_MODEL",
                os.getenv("MINICPM_REALTIME_MODEL", "minicpm-realtime-gateway"),
            ),
            ("MINICPM_REALTIME_URL", _redacted_url(os.getenv("MINICPM_REALTIME_URL", ""))),
            ("MINICPM_LENGTH_PENALTY", os.getenv("MINICPM_LENGTH_PENALTY", "1.1")),
            ("MINICPM_INPUT_CHUNK_MS", os.getenv("MINICPM_INPUT_CHUNK_MS", "1000")),
            ("MINICPM_QUEUE_TIMEOUT", os.getenv("MINICPM_QUEUE_TIMEOUT", "300.0")),
            ("MINICPM_PING_INTERVAL", os.getenv("MINICPM_PING_INTERVAL", "30.0")),
            ("MINICPM_PING_TIMEOUT", os.getenv("MINICPM_PING_TIMEOUT", "120.0")),
        ],
        [
            ("FREEZE_OMNI_MODEL", os.getenv("FREEZE_OMNI_MODEL", "freeze-omni")),
            ("FREEZE_OMNI_REALTIME_URL", _redacted_url(os.getenv("FREEZE_OMNI_REALTIME_URL", ""))),
            ("FREEZE_OMNI_SSL_VERIFY", os.getenv("FREEZE_OMNI_SSL_VERIFY", "false")),
            ("FREEZE_OMNI_INPUT_CHUNK_MS", os.getenv("FREEZE_OMNI_INPUT_CHUNK_MS", "20")),
            ("FREEZE_OMNI_CONNECT_TIMEOUT", os.getenv("FREEZE_OMNI_CONNECT_TIMEOUT", "60.0")),
            ("FREEZE_OMNI_CONNECT_RETRIES", os.getenv("FREEZE_OMNI_CONNECT_RETRIES", "10")),
            (
                "FREEZE_OMNI_CONNECT_RETRY_DELAY",
                os.getenv("FREEZE_OMNI_CONNECT_RETRY_DELAY", "5.0"),
            ),
            ("FREEZE_OMNI_PROMPT_TIMEOUT", os.getenv("FREEZE_OMNI_PROMPT_TIMEOUT", "60.0")),
            ("FREEZE_OMNI_TURN_START_DELAY", os.getenv("FREEZE_OMNI_TURN_START_DELAY", "3.0")),
            (
                "FREEZE_OMNI_TURN_PREROLL_SILENCE_MS",
                os.getenv("FREEZE_OMNI_TURN_PREROLL_SILENCE_MS", "1200"),
            ),
            (
                "FREEZE_OMNI_MAX_INPUT_SILENCE_MS",
                os.getenv("FREEZE_OMNI_MAX_INPUT_SILENCE_MS", "200"),
            ),
            (
                "FREEZE_OMNI_INPUT_SILENCE_RMS_THRESHOLD",
                os.getenv("FREEZE_OMNI_INPUT_SILENCE_RMS_THRESHOLD", "800.0"),
            ),
            (
                "FREEZE_OMNI_POST_TURN_POLL_SECONDS",
                os.getenv("FREEZE_OMNI_POST_TURN_POLL_SECONDS", "60.0"),
            ),
            (
                "FREEZE_OMNI_POST_TURN_IDLE_SECONDS",
                os.getenv("FREEZE_OMNI_POST_TURN_IDLE_SECONDS", "3.0"),
            ),
            (
                "FREEZE_OMNI_POST_TURN_POLL_CHUNK_MS",
                os.getenv("FREEZE_OMNI_POST_TURN_POLL_CHUNK_MS", "160"),
            ),
            (
                "FREEZE_OMNI_STOP_RECORDING_AFTER_TURN",
                os.getenv("FREEZE_OMNI_STOP_RECORDING_AFTER_TURN", "true"),
            ),
        ],
        [
            ("MOSHI_REALTIME_URL", _redacted_url(os.getenv("MOSHI_REALTIME_URL", ""))),
            ("MOSHI_MODEL", os.getenv("MOSHI_MODEL", "moshi")),
        ],
        [
            ("PERSONAPLEX_REALTIME_URL", _redacted_url(os.getenv("PERSONAPLEX_REALTIME_URL", ""))),
            ("PERSONAPLEX_MODEL", os.getenv("PERSONAPLEX_MODEL", "personaplex")),
        ],
    ]
    lines: list[str] = []
    for index, group in enumerate(groups):
        if index:
            lines.append("")
        lines.extend(f"{name}={_env_value(value)}" for name, value in group)
    return lines


def _result_case_index(result: dict[str, Any], fallback_index: int) -> int:
    try:
        return int(result.get("case_index", fallback_index))
    except (TypeError, ValueError):
        return fallback_index


def _scenario_row_id(scenario: dict[str, Any]) -> str:
    source = scenario.get("source") or {}
    value = (
        scenario.get("row_id") or source.get("row_id") or source.get("raw_id") or scenario.get("id")
    )
    return str(value or "row")


def _safe_path_segment(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "-"
        for character in value
    )
    return cleaned.strip(".-") or "row"


def _env_value(value: object) -> str:
    text = str(value)
    if text == "" or any(
        character.isspace() or character in {'"', "'", "#", "="} for character in text
    ):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def _redacted_url(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value

    netloc = parsed.netloc
    if "@" in netloc:
        netloc = f"***@{netloc.rsplit('@', 1)[1]}"

    query = urlencode(
        [
            (
                key,
                "***"
                if any(secret in key.lower() for secret in ("key", "token", "secret", "password"))
                else val,
            )
            for key, val in parse_qsl(parsed.query, keep_blank_values=True)
        ],
        safe="*",
    )
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))


def _default_provider(agent: AgentKey) -> str:
    agent = validate_agent(agent)
    return "openai" if agent == HUMAN_AGENT else "gemini"
