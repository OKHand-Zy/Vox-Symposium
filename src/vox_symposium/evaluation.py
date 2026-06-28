from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
import wave
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from vox_symposium.audio import PcmAudio, rechunk_pcm16
from vox_symposium.config import gemini_live_model
from vox_symposium.env import env_with_legacy, normalized_env
from vox_symposium.models.base import RealtimeAudioModel
from vox_symposium.models.factory import build_model_from_env
from vox_symposium.recording import RecordedAudio, audio_event_fields, write_wav
from vox_symposium.scenario import (
    LoadedScenario,
    build_evaluation_result,
    load_scenarios,
    write_evaluation_result,
)
from vox_symposium.providers import normalize_provider

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False


@dataclass(frozen=True)
class AudioUtterance:
    agent: str
    audio: PcmAudio


async def run() -> None:
    load_dotenv()
    args = _parse_args()
    result_path = Path(args.result)
    base_run_id = args.run_id or result_path.stem
    artifact_root = Path(args.artifact_dir) if args.artifact_dir else result_path.parent / f"{base_run_id}-artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    console_log_path = artifact_root / "console-log.txt"
    summary_path = artifact_root / "summary.json"

    with _tee_console(console_log_path):
        print(f"Saving console log: {console_log_path}")
        print(f"Command: {_format_command(sys.argv)}")
        try:
            await _run_evaluations(
                args,
                result_path=result_path,
                base_run_id=base_run_id,
                artifact_root=artifact_root,
                console_log_path=console_log_path,
                summary_path=summary_path,
            )
        except Exception as exc:
            print(_format_error(exc), file=sys.stderr)
            setattr(exc, "_vox_console_logged", True)
            raise


async def _run_evaluations(
    args: argparse.Namespace,
    *,
    result_path: Path,
    base_run_id: str,
    artifact_root: Path,
    console_log_path: Path,
    summary_path: Path,
) -> None:
    scenarios = [
        LoadedScenario(data)
        for data in load_scenarios(
            args.scenario,
            scenario_id=args.scenario_id,
            scenario_index=args.scenario_index,
            audio_dir=args.audio_dir,
            dialogue_turns=args.dialogue_turns,
        )
    ]
    scenarios = _limit_scenarios(
        scenarios,
        limit=args.limit,
        scenario_id=args.scenario_id,
        scenario_index=args.scenario_index,
    )
    if not scenarios:
        raise RuntimeError("No scenarios found")
    if (args.scenario_id is not None or args.scenario_index is not None) and len(scenarios) != 1:
        raise RuntimeError(f"Expected exactly one scenario, got {len(scenarios)}")

    env_snapshot_path = _write_run_env_snapshot(artifact_root, run_id=base_run_id, args=args)

    results: list[dict[str, Any]] = []
    total = len(scenarios)
    if total > 1 and args.answer_audio:
        raise RuntimeError("--answer-audio can only be used when one scenario is selected with --id or --index")

    _write_summary_report(
        summary_path,
        results,
        run_id=base_run_id,
        scenario_path=args.scenario,
        result_path=result_path,
        artifact_root=artifact_root,
        env_snapshot_path=env_snapshot_path,
        console_log_path=console_log_path,
        total_cases=total,
    )
    print("==========")
    for index, scenario in enumerate(scenarios):
        run_id = _scenario_run_id(base_run_id, scenario.data, index=index, total=total)
        if index > 0:
            print("----------")
        if total > 1:
            print(f"Running scenario {index + 1}/{total}: {scenario.id} (run_id={run_id})")

        result = await _run_scenario_evaluation(
            args,
            scenario=scenario,
            run_id=run_id,
            artifact_root=artifact_root,
            env_snapshot_path=env_snapshot_path,
        )
        result["artifacts"]["console_log"] = str(console_log_path)
        result["artifacts"]["summary"] = str(summary_path)
        results.append(result)
        write_evaluation_result(result_path, _result_payload(results, total=total))
        _write_summary_report(
            summary_path,
            results,
            run_id=base_run_id,
            scenario_path=args.scenario,
            result_path=result_path,
            artifact_root=artifact_root,
            env_snapshot_path=env_snapshot_path,
            console_log_path=console_log_path,
            total_cases=total,
        )
        response = result["response"]
        if total == 1:
            print(
                "Saved evaluation result: "
                f"{result_path} "
                f"(choice={response['choice'] or 'unknown'}, "
                f"is_correct={response['is_correct']})"
            )
        else:
            print(
                "Saved evaluation result batch: "
                f"{result_path} ({index + 1}/{total}, "
                f"choice={response['choice'] or 'unknown'}, "
                f"is_correct={response['is_correct']})"
            )


async def _run_scenario_evaluation(
    args: argparse.Namespace,
    *,
    scenario: LoadedScenario,
    run_id: str,
    artifact_root: Path,
    env_snapshot_path: Path,
) -> dict[str, Any]:
    artifact_dir = _scenario_artifact_dir(artifact_root, scenario.data)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    citizen = _build_model("citizen", scenario.build_instructions("citizen"))
    scholar = _build_model("scholar", scenario.build_instructions("scholar"))
    models = {"citizen": citizen, "scholar": scholar}

    audio_queues: dict[str, asyncio.Queue[PcmAudio | None]] = {
        "citizen": asyncio.Queue(),
        "scholar": asyncio.Queue(),
    }
    text_queues: dict[str, asyncio.Queue[str | None]] = {
        "citizen": asyncio.Queue(),
        "scholar": asyncio.Queue(),
    }
    reader_tasks: list[asyncio.Task[None]] = []
    dialogue_log: dict[str, Any] = {
        "scenario_id": scenario.id,
        "run_id": run_id,
        "events": [],
    }

    try:
        await asyncio.gather(citizen.connect(), scholar.connect())
        for agent, model in models.items():
            reader_tasks.append(asyncio.create_task(_read_audio(model, audio_queues[agent]), name=f"{agent}-audio"))
            reader_tasks.append(asyncio.create_task(_read_text(model, text_queues[agent]), name=f"{agent}-text"))

        next_agent = await _play_opening(
            scenario.data,
            models,
            artifact_dir=artifact_dir,
            frame_ms=args.frame_ms,
            audio_speed=args.audio_speed,
            dialogue_log=dialogue_log,
        )
        scholar_turns = await _run_dialogue_turns(
            models,
            audio_queues,
            text_queues,
            dialogue_log=dialogue_log,
            artifact_dir=artifact_dir,
            start_agent=next_agent,
            target_turns=args.dialogue_turns,
            idle_timeout=args.idle_timeout,
            max_utterance_seconds=args.max_utterance_seconds,
            frame_ms=args.frame_ms,
            audio_speed=args.audio_speed,
        )

        _drain_queue(text_queues["scholar"])
        _drain_queue(audio_queues["scholar"])
        question_audio = _question_audio(
            scenario.data,
            question_audio=args.question_audio,
            artifact_dir=artifact_dir,
            scenario_dir=Path(args.scenario).parent,
            disable_tts=args.no_tts,
        )
        dialogue_log["events"].append(
            {
                "type": "evaluation_question",
                "target_agent": "scholar",
                "audio": str(question_audio),
                "after_scholar_turns": scholar_turns,
            }
        )
        print(f"Playing evaluation question into scholar: {question_audio}")
        await _send_audio_file(question_audio, scholar, frame_ms=args.frame_ms, audio_speed=args.audio_speed)

        answer = await _collect_utterance(
            "scholar",
            audio_queues["scholar"],
            idle_timeout=args.idle_timeout,
            max_seconds=args.max_utterance_seconds,
        )
        answer_audio = Path(args.answer_audio) if args.answer_audio else artifact_dir / "scholar-answer.wav"
        answer_recording = _write_wav(answer_audio, answer.audio)
        answer_text = await _collect_text_after_audio(text_queues["scholar"])
        dialogue_log["events"].append(
            {
                "type": "evaluation_answer",
                "agent": "scholar",
                "text": answer_text,
                **audio_event_fields(answer_recording),
            }
        )
        print(f"Captured scholar answer evaluation question: {answer_audio}")

        dialogue_log_path = artifact_dir / "dialogue-log.json"
        _write_json(dialogue_log_path, dialogue_log)
        result = build_evaluation_result(
            scenario.data,
            response_text=answer_text,
            response_audio=str(answer_audio),
            dialogue_log=str(dialogue_log_path),
            run_id=run_id,
        )
        result["artifacts"]["env_snapshot"] = str(env_snapshot_path)
        response = result["response"]
        print(
            "Completed evaluation scenario: "
            f"{scenario.id} "
            f"(choice={response['choice'] or 'unknown'}, "
            f"is_correct={response['is_correct']}, "
            f"answer_audio={answer_audio})"
        )
        return result
    finally:
        for task in reader_tasks:
            task.cancel()
        await asyncio.gather(citizen.close(), scholar.close(), return_exceptions=True)


def main() -> None:
    try:
        asyncio.run(run())
    except RuntimeError as exc:
        if getattr(exc, "_vox_console_logged", False):
            raise SystemExit(1) from exc
        raise SystemExit(_format_error(exc)) from exc
    except Exception as exc:
        if getattr(exc, "_vox_console_logged", False):
            raise SystemExit(1) from exc
        raise SystemExit(_format_error(exc)) from exc


async def _play_opening(
    scenario: dict[str, Any],
    models: dict[str, RealtimeAudioModel],
    *,
    artifact_dir: Path,
    frame_ms: int,
    audio_speed: float,
    dialogue_log: dict[str, Any],
) -> str:
    opening = scenario.get("opening")
    if not opening:
        return "citizen"

    opening_agent = opening["agent"]
    receiver = _other_agent(opening_agent)
    opening_audio = _turn_audio(opening, artifact_dir=artifact_dir)
    dialogue_log["events"].append(
        {
            "type": "opening_playback",
            "speaker_agent": opening_agent,
            "receiver_agent": receiver,
            "audio": str(opening_audio),
            "text": opening.get("text"),
        }
    )
    print(f"Playing opening from {opening_agent} into {receiver}: {opening_audio}")
    await _send_audio_file(opening_audio, models[receiver], frame_ms=frame_ms, audio_speed=audio_speed)
    return receiver


async def _run_dialogue_turns(
    models: dict[str, RealtimeAudioModel],
    audio_queues: dict[str, asyncio.Queue[PcmAudio | None]],
    text_queues: dict[str, asyncio.Queue[str | None]],
    *,
    dialogue_log: dict[str, Any],
    artifact_dir: Path,
    start_agent: str,
    target_turns: int,
    idle_timeout: float,
    max_utterance_seconds: float,
    frame_ms: int,
    audio_speed: float,
) -> int:
    current_agent = start_agent
    scholar_turns = 0
    event_index = 0
    turn_counts = {"citizen": 0, "scholar": 0}

    while scholar_turns < target_turns:
        utterance = await _collect_utterance(
            current_agent,
            audio_queues[current_agent],
            idle_timeout=idle_timeout,
            max_seconds=max_utterance_seconds,
        )
        event_index += 1
        turn_counts[current_agent] += 1
        turn_index = turn_counts[current_agent]
        if current_agent == "scholar":
            scholar_turns += 1
        utterance_path = artifact_dir / f"dialogue-{event_index:02d}-{current_agent}.wav"
        recording = _write_wav(utterance_path, utterance.audio)
        text = await _collect_text_after_audio(text_queues[current_agent])
        dialogue_log["events"].append(
            {
                "type": "dialogue_turn",
                "agent": current_agent,
                "turn_index": turn_index,
                "scholar_turns": scholar_turns,
                "text": text,
                **audio_event_fields(recording),
            }
        )
        print(_format_dialogue_capture(current_agent, turn_index))

        if current_agent == "scholar" and scholar_turns >= target_turns:
            break

        receiver = _other_agent(current_agent)
        _drain_queue(text_queues[receiver])
        await _send_audio(utterance.audio, models[receiver], frame_ms=frame_ms, audio_speed=audio_speed)
        current_agent = receiver

    return scholar_turns


def _format_dialogue_capture(agent: str, turn_index: int) -> str:
    turn_word = "turns" if agent == "scholar" else "turn"
    return f"Captured {agent} {turn_word} {turn_index}"


async def _collect_utterance(
    agent: str,
    queue: asyncio.Queue[PcmAudio | None],
    *,
    idle_timeout: float,
    max_seconds: float,
) -> AudioUtterance:
    try:
        first = await asyncio.wait_for(queue.get(), timeout=max_seconds)
    except TimeoutError as exc:
        raise RuntimeError(f"Timed out waiting for {agent} audio after {max_seconds:.1f}s") from exc
    if first is None:
        raise RuntimeError(f"{agent} model audio stream closed before an utterance was captured")

    chunks = [first]
    sample_rate = first.sample_rate
    channels = first.channels
    started_at = asyncio.get_running_loop().time()

    while True:
        remaining = max_seconds - (asyncio.get_running_loop().time() - started_at)
        if remaining <= 0:
            break
        try:
            item = await asyncio.wait_for(queue.get(), timeout=min(idle_timeout, remaining))
        except TimeoutError:
            break
        if item is None:
            break
        chunks.append(item)

    data = b"".join(chunk.data for chunk in chunks)
    return AudioUtterance(agent=agent, audio=PcmAudio(data=data, sample_rate=sample_rate, channels=channels))


async def _send_audio_file(path: Path, model: RealtimeAudioModel, *, frame_ms: int, audio_speed: float) -> None:
    audio = _read_wav(path)
    await _send_audio(audio, model, frame_ms=frame_ms, audio_speed=audio_speed)


async def _send_audio(audio: PcmAudio, model: RealtimeAudioModel, *, frame_ms: int, audio_speed: float) -> None:
    frame_seconds = _frame_sleep_seconds(frame_ms, audio_speed)
    await model.start_audio_turn()
    try:
        for chunk in rechunk_pcm16(audio.data, audio.sample_rate, frame_ms):
            await model.send_audio(PcmAudio(data=chunk, sample_rate=audio.sample_rate, channels=audio.channels))
            await asyncio.sleep(frame_seconds)

        silence_bytes = int(audio.sample_rate * 0.2) * 2
        silence = b"\x00" * silence_bytes
        for chunk in rechunk_pcm16(silence, audio.sample_rate, frame_ms):
            await model.send_audio(PcmAudio(data=chunk, sample_rate=audio.sample_rate, channels=1))
            await asyncio.sleep(frame_seconds)
    finally:
        await model.end_audio_turn()


async def _read_audio(model: RealtimeAudioModel, queue: asyncio.Queue[PcmAudio | None]) -> None:
    try:
        async for audio in model.receive_audio():
            await queue.put(audio)
    finally:
        await queue.put(None)


async def _read_text(model: RealtimeAudioModel, queue: asyncio.Queue[str | None]) -> None:
    try:
        async for text in model.receive_text():
            await queue.put(text)
    finally:
        await queue.put(None)


async def _collect_text_after_audio(queue: asyncio.Queue[str | None]) -> str:
    await asyncio.sleep(0.7)
    parts: list[str] = []
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if item:
            parts.append(item)
    return "".join(parts).strip()


def _build_model(agent: str, instructions: str) -> RealtimeAudioModel:
    return build_model_from_env(_agent_provider(agent), instructions, evaluation_mode=True)


def _agent_provider(agent: str) -> str:
    return normalize_provider(
        env_with_legacy(
            f"AGENT_{agent.upper()}_PROVIDER",
            f"AGENT_{'A' if agent == 'citizen' else 'B'}_PROVIDER",
            default=_default_provider(agent),
        )
    )


def _question_audio(
    scenario: dict[str, Any],
    *,
    question_audio: str | None,
    artifact_dir: Path,
    scenario_dir: Path,
    disable_tts: bool,
) -> Path:
    explicit = question_audio or (scenario.get("evaluation") or {}).get("question_audio")
    if explicit:
        path = _resolve_question_audio_path(explicit, scenario_dir=scenario_dir)
        return _ensure_wav(path, artifact_dir / "question.wav")

    if disable_tts:
        raise RuntimeError("No evaluation.question_audio is set and --no-tts was used")

    path = artifact_dir / "question.wav"
    _synthesize_question_audio(_question_text(scenario), path)
    return path


def _resolve_question_audio_path(path_text: str, *, scenario_dir: Path) -> Path:
    path = Path(path_text)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                scenario_dir / path,
                scenario_dir.parent / "question" / path.name,
                Path("data/question") / path.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"Question audio does not exist. Checked: {checked}")


def _ensure_wav(path: Path, output: Path) -> Path:
    if path.suffix.lower() == ".wav":
        return path
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(f"Question audio is {path.suffix}, but ffmpeg is not available to convert it to wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "24000",
            "-sample_fmt",
            "s16",
            str(output),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output


def _question_text(scenario: dict[str, Any]) -> str:
    evaluation = scenario.get("evaluation") or {}
    choices = " ".join(str(choice) for choice in evaluation.get("choices") or [])
    return f"{evaluation.get('question', '')} {choices} Answer with A, B, C, or D."


def _synthesize_question_audio(text: str, output: Path) -> None:
    say = shutil.which("say")
    afconvert = shutil.which("afconvert")
    if not say or not afconvert:
        raise RuntimeError("No question_audio is set, and macOS say/afconvert are not available")

    output.parent.mkdir(parents=True, exist_ok=True)
    aiff = output.with_suffix(".aiff")
    subprocess.run([say, "-o", str(aiff), text], check=True)
    subprocess.run([afconvert, "-f", "WAVE", "-d", "LEI16@24000", str(aiff), str(output)], check=True)
    audio = _read_wav(output)
    duration = len(audio.data) / (audio.sample_rate * 2)
    if duration < 0.5:
        raise RuntimeError(
            "Generated question audio is too short. Set evaluation.question_audio or pass --question-audio."
        )


def _turn_audio(turn: dict[str, Any], *, artifact_dir: Path) -> Path:
    audio = turn.get("audio")
    if audio:
        path = Path(audio)
        if path.exists():
            return path
    raise RuntimeError(
        f"Opening turn has no usable audio. Expected an existing audio path, got {audio!r}."
    )


def _read_wav(path: Path) -> PcmAudio:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        if sample_width != 2:
            raise RuntimeError(f"{path} must be 16-bit PCM wav, got sample width {sample_width}")
        data = wav.readframes(wav.getnframes())
    return PcmAudio(data=data, sample_rate=sample_rate, channels=channels)


def _write_wav(path: Path, audio: PcmAudio) -> RecordedAudio:
    return write_wav(path, audio)


def _scenario_artifact_dir(artifact_root: Path, scenario: dict[str, Any]) -> Path:
    return artifact_root / _safe_path_segment(_scenario_row_id(scenario))


def _scenario_run_id(base_run_id: str, scenario: dict[str, Any], *, index: int, total: int) -> str:
    if total == 1:
        return base_run_id
    return f"{base_run_id}-{index + 1:04d}-{_safe_path_segment(_scenario_row_id(scenario))}"


def _limit_scenarios(
    scenarios: list[LoadedScenario],
    *,
    limit: int | None,
    scenario_id: str | None,
    scenario_index: int | None,
) -> list[LoadedScenario]:
    if limit is None:
        return scenarios
    if limit < 1:
        raise RuntimeError("--limit must be at least 1")
    if scenario_id is not None or scenario_index is not None:
        raise RuntimeError("--limit cannot be used with --id or --index")
    return scenarios[:limit]


def _result_payload(results: list[dict[str, Any]], *, total: int) -> dict[str, Any] | list[dict[str, Any]]:
    if total == 1:
        return results[0]
    return results


def _summary_report(
    results: list[dict[str, Any]],
    *,
    run_id: str,
    scenario_path: str,
    result_path: Path,
    artifact_root: Path,
    env_snapshot_path: Path,
    console_log_path: Path,
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
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scenario": scenario_path,
        "result": str(result_path),
        "artifact_root": str(artifact_root),
        "env_snapshot": str(env_snapshot_path),
        "console_log": str(console_log_path),
        "total_cases": total,
        "completed_cases": completed,
        "passed_cases": passed,
        "failed_cases": failed,
        "unknown_cases": unknown,
        "pass_rate": passed / total if total else None,
        "completed_pass_rate": passed / completed if completed else None,
        "cases": cases,
    }


def _summary_case(index: int, result: dict[str, Any]) -> dict[str, Any]:
    response = result.get("response") or {}
    evaluation = result.get("evaluation") or {}
    artifacts = result.get("artifacts") or {}
    is_correct = response.get("is_correct")
    status = "passed" if is_correct is True else "failed" if is_correct is False else "unknown"
    return {
        "index": index,
        "scenario_id": result.get("scenario_id"),
        "run_id": result.get("run_id"),
        "status": status,
        "is_correct": is_correct,
        "choice": response.get("choice"),
        "correct_answer": evaluation.get("correct_answer"),
        "response_text": response.get("text"),
        "response_audio": response.get("audio"),
        "dialogue_log": artifacts.get("dialogue_log"),
    }


def _write_summary_report(
    path: Path,
    results: list[dict[str, Any]],
    *,
    run_id: str,
    scenario_path: str,
    result_path: Path,
    artifact_root: Path,
    env_snapshot_path: Path,
    console_log_path: Path,
    total_cases: int | None = None,
) -> None:
    report = _summary_report(
        results,
        run_id=run_id,
        scenario_path=scenario_path,
        result_path=result_path,
        artifact_root=artifact_root,
        env_snapshot_path=env_snapshot_path,
        console_log_path=console_log_path,
        total_cases=total_cases,
    )
    _write_json(path, report)


def _scenario_row_id(scenario: dict[str, Any]) -> str:
    source = scenario.get("source") or {}
    value = scenario.get("row_id") or source.get("row_id") or source.get("raw_id") or scenario.get("id")
    return str(value or "row")


def _safe_path_segment(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"-", "_", "."} else "-" for character in value)
    return cleaned.strip(".-") or "row"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


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
def _tee_console(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        stdout = _TeeStream(sys.stdout, file)
        stderr = _TeeStream(sys.stderr, file)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield


def _format_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def _format_error(exc: Exception) -> str:
    if isinstance(exc, RuntimeError):
        return f"Error: {exc}"
    return f"Error: {type(exc).__name__}: {exc}"


def _write_run_env_snapshot(artifact_dir: Path, *, run_id: str, args: argparse.Namespace) -> Path:
    path = artifact_dir / "run-env.txt"
    lines = [
        "# Vox Symposium evaluation environment snapshot.",
        "# Secrets such as API keys and API secrets are intentionally omitted.",
        "# Values reflect the process environment after .env loading and runner defaults.",
        "",
        f"RUN_ID={_env_value(run_id)}",
        f"SCENARIO={_env_value(args.scenario)}",
        f"RESULT={_env_value(args.result)}",
        f"DIALOGUE_TURNS={args.dialogue_turns}",
        f"LIMIT={_env_value(args.limit or '')}",
        f"AUDIO_SPEED={args.audio_speed}",
        f"FRAME_MS={args.frame_ms}",
        "",
    ]

    for agent in ("citizen", "scholar"):
        lines.extend(_agent_env_snapshot(agent))
        lines.append("")

    lines.extend(_provider_env_snapshot())
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Saved environment snapshot: {path}")
    return path


def _agent_env_snapshot(agent: str) -> list[str]:
    legacy_agent = "A" if agent == "citizen" else "B"
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
        return os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2"), "openai", {
            "VOICE": os.getenv("OPENAI_REALTIME_VOICE", "marin"),
        }
    if provider == "gemini":
        backend = normalized_env("GEMINI_BACKEND", "ai_studio")
        return gemini_live_model(backend), backend, {}
    if provider == "minicpm":
        return os.getenv("MINICPM_REALTIME_MODEL", "minicpm-realtime-gateway"), None, {}
    if provider == "freeze_omni":
        return os.getenv("FREEZE_OMNI_MODEL", "freeze-omni"), "socketio", {
            "REALTIME_URL": _redacted_url(os.getenv("FREEZE_OMNI_REALTIME_URL", "")),
        }
    if provider == "moshi":
        return os.getenv("MOSHI_MODEL", "moshi"), "moshi", {
            "REALTIME_URL": _redacted_url(os.getenv("MOSHI_REALTIME_URL", "")),
        }
    if provider == "personaplex":
        return os.getenv("PERSONAPLEX_MODEL", "personaplex"), "moshi", {
            "REALTIME_URL": _redacted_url(os.getenv("PERSONAPLEX_REALTIME_URL", "")),
        }
    return "", None, {}


def _provider_env_snapshot() -> list[str]:
    lines = [
        "OPENAI_BACKEND=" + _env_value(normalized_env("OPENAI_BACKEND", "openai")),
        "OPENAI_REALTIME_MODEL=" + _env_value(os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2")),
        "OPENAI_REALTIME_VOICE=" + _env_value(os.getenv("OPENAI_REALTIME_VOICE", "marin")),
        "AZURE_OPENAI_DEPLOYMENT_NAME=" + _env_value(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")),
        "AZURE_OPENAI_ENDPOINT=" + _env_value(os.getenv("AZURE_OPENAI_ENDPOINT", "")),
        "AZURE_OPENAI_API_VERSION=" + _env_value(os.getenv("AZURE_OPENAI_API_VERSION", "")),
        "",
        "GEMINI_BACKEND=" + _env_value(normalized_env("GEMINI_BACKEND", "ai_studio")),
        "GEMINI_LIVE_MODEL=" + _env_value(gemini_live_model(normalized_env("GEMINI_BACKEND", "ai_studio"))),
        "GOOGLE_CLOUD_PROJECT=" + _env_value(os.getenv("GOOGLE_CLOUD_PROJECT", "")),
        "GOOGLE_CLOUD_LOCATION=" + _env_value(os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")),
        "",
        "MINICPM_REALTIME_MODEL=" + _env_value(os.getenv("MINICPM_REALTIME_MODEL", "minicpm-realtime-gateway")),
        "MINICPM_REALTIME_URL=" + _env_value(_redacted_url(os.getenv("MINICPM_REALTIME_URL", ""))),
        "MINICPM_LENGTH_PENALTY=" + _env_value(os.getenv("MINICPM_LENGTH_PENALTY", "1.1")),
        "MINICPM_INPUT_CHUNK_MS=" + _env_value(os.getenv("MINICPM_INPUT_CHUNK_MS", "1000")),
        "MINICPM_QUEUE_TIMEOUT=" + _env_value(os.getenv("MINICPM_QUEUE_TIMEOUT", "300.0")),
        "",
        "FREEZE_OMNI_MODEL=" + _env_value(os.getenv("FREEZE_OMNI_MODEL", "freeze-omni")),
        "FREEZE_OMNI_REALTIME_URL=" + _env_value(_redacted_url(os.getenv("FREEZE_OMNI_REALTIME_URL", ""))),
        "FREEZE_OMNI_SSL_VERIFY=" + _env_value(os.getenv("FREEZE_OMNI_SSL_VERIFY", "false")),
        "FREEZE_OMNI_INPUT_CHUNK_MS=" + _env_value(os.getenv("FREEZE_OMNI_INPUT_CHUNK_MS", "20")),
        "FREEZE_OMNI_CONNECT_TIMEOUT=" + _env_value(os.getenv("FREEZE_OMNI_CONNECT_TIMEOUT", "60.0")),
        "FREEZE_OMNI_CONNECT_RETRIES=" + _env_value(os.getenv("FREEZE_OMNI_CONNECT_RETRIES", "10")),
        "FREEZE_OMNI_CONNECT_RETRY_DELAY=" + _env_value(os.getenv("FREEZE_OMNI_CONNECT_RETRY_DELAY", "5.0")),
        "FREEZE_OMNI_PROMPT_TIMEOUT=" + _env_value(os.getenv("FREEZE_OMNI_PROMPT_TIMEOUT", "60.0")),
        "FREEZE_OMNI_TURN_START_DELAY=" + _env_value(os.getenv("FREEZE_OMNI_TURN_START_DELAY", "3.0")),
        "FREEZE_OMNI_TURN_PREROLL_SILENCE_MS=" + _env_value(os.getenv("FREEZE_OMNI_TURN_PREROLL_SILENCE_MS", "1200")),
        "FREEZE_OMNI_MAX_INPUT_SILENCE_MS=" + _env_value(os.getenv("FREEZE_OMNI_MAX_INPUT_SILENCE_MS", "200")),
        "FREEZE_OMNI_INPUT_SILENCE_RMS_THRESHOLD=" + _env_value(os.getenv("FREEZE_OMNI_INPUT_SILENCE_RMS_THRESHOLD", "800.0")),
        "FREEZE_OMNI_POST_TURN_POLL_SECONDS=" + _env_value(os.getenv("FREEZE_OMNI_POST_TURN_POLL_SECONDS", "60.0")),
        "FREEZE_OMNI_POST_TURN_IDLE_SECONDS=" + _env_value(os.getenv("FREEZE_OMNI_POST_TURN_IDLE_SECONDS", "3.0")),
        "FREEZE_OMNI_POST_TURN_POLL_CHUNK_MS=" + _env_value(os.getenv("FREEZE_OMNI_POST_TURN_POLL_CHUNK_MS", "160")),
        "FREEZE_OMNI_STOP_RECORDING_AFTER_TURN=" + _env_value(os.getenv("FREEZE_OMNI_STOP_RECORDING_AFTER_TURN", "true")),
        "",
        "MOSHI_REALTIME_URL=" + _env_value(_redacted_url(os.getenv("MOSHI_REALTIME_URL", ""))),
        "MOSHI_MODEL=" + _env_value(os.getenv("MOSHI_MODEL", "moshi")),
        "",
        "PERSONAPLEX_REALTIME_URL=" + _env_value(_redacted_url(os.getenv("PERSONAPLEX_REALTIME_URL", ""))),
        "PERSONAPLEX_MODEL=" + _env_value(os.getenv("PERSONAPLEX_MODEL", "personaplex")),
    ]
    return lines


def _env_value(value: object) -> str:
    text = str(value)
    if text == "" or any(character.isspace() or character in {'"', "'", "#", "="} for character in text):
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
            (key, "***" if any(secret in key.lower() for secret in ("key", "token", "secret", "password")) else val)
            for key, val in parse_qsl(parsed.query, keep_blank_values=True)
        ],
        safe="*",
    )
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))


def _drain_queue(queue: asyncio.Queue[Any]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return


def _other_agent(agent: str) -> str:
    return "scholar" if agent == "citizen" else "citizen"


def _default_provider(agent: str) -> str:
    return "openai" if agent == "citizen" else "gemini"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automated Vox Symposium scenario evaluations.")
    parser.add_argument("scenario", help="Normalized scenario JSON path, or source dataset.")
    parser.add_argument("result", help="Output evaluation result JSON path.")
    parser.add_argument("--id", dest="scenario_id", help="Select a scenario by id when scenario is a dataset.")
    parser.add_argument("--index", dest="scenario_index", type=int, help="Select a scenario by zero-based index.")
    parser.add_argument("--limit", type=int, help="Run only the first N scenarios when scenario is a dataset.")
    parser.add_argument("--audio-dir", help="Directory containing original speech wav files.")
    parser.add_argument("--dialogue-turns", type=int, default=5, help="Scholar turns before evaluation.")
    parser.add_argument(
        "--question-audio",
        help="Evaluation question wav file. Overrides scenario evaluation.question_audio.",
    )
    parser.add_argument("--answer-audio", help="Where to save the evaluated scholar answer wav.")
    parser.add_argument("--artifact-dir", help="Directory for dialogue logs and captured wav files.")
    parser.add_argument("--run-id", help="Stable run id for this evaluation.")
    parser.add_argument("--frame-ms", type=int, default=20, help="Audio frame size used to stream wav files.")
    parser.add_argument(
        "--audio-speed",
        type=float,
        default=float(os.getenv("EVALUATION_AUDIO_SPEED", "1.0")),
        help=(
            "Audio injection speed. 1.0 is realtime; higher values send audio faster and may "
            "affect streaming VAD/turn detection; 0 disables sleeps."
        ),
    )
    parser.add_argument("--idle-timeout", type=float, default=1.5, help="Silence timeout used to end an utterance.")
    parser.add_argument(
        "--max-utterance-seconds",
        type=float,
        default=30.0,
        help="Maximum seconds to wait for one utterance.",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Require question audio instead of generating it with macOS say.",
    )
    return parser.parse_args()


def _frame_sleep_seconds(frame_ms: int, audio_speed: float) -> float:
    if audio_speed <= 0:
        return 0
    return (frame_ms / 1000) / audio_speed


if __name__ == "__main__":
    main()
