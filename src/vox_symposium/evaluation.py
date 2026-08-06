from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path
from typing import Any, cast

from vox_symposium.audio import PcmAudio
from vox_symposium.config import provider_uses_structured_history
from vox_symposium.env import env_with_legacy, float_env, int_env, load_environment
from vox_symposium.evaluation_artifacts import (
    EvaluationArtifacts,
    write_run_env_snapshot,
    write_summary_report,
)
from vox_symposium.evaluation_artifacts import (
    format_command as _format_command,
)
from vox_symposium.evaluation_artifacts import (
    format_error as _format_error,
)
from vox_symposium.evaluation_artifacts import (
    load_resume_results as _load_resume_results,
)
from vox_symposium.evaluation_artifacts import (
    remove_failed_case_artifacts as _remove_failed_case_artifacts,
)
from vox_symposium.evaluation_artifacts import (
    result_payload as _result_payload,
)
from vox_symposium.evaluation_artifacts import (
    scenario_artifact_dir as _scenario_artifact_dir,
)
from vox_symposium.evaluation_artifacts import (
    scenario_run_id as _scenario_run_id,
)
from vox_symposium.evaluation_artifacts import (
    tee_console as _tee_console,
)
from vox_symposium.evaluation_audio import (
    collect_text_after_audio as _collect_text_after_audio,
)
from vox_symposium.evaluation_audio import (
    collect_utterance as _collect_utterance,
)
from vox_symposium.evaluation_audio import (
    question_audio as _question_audio,
)
from vox_symposium.evaluation_audio import (
    read_audio_stream as _read_audio,
)
from vox_symposium.evaluation_audio import (
    read_event_stream as _read_event,
)
from vox_symposium.evaluation_audio import (
    read_text_stream as _read_text,
)
from vox_symposium.evaluation_audio import (
    send_audio_file as _send_audio_file,
)
from vox_symposium.evaluation_audio import (
    turn_audio as _turn_audio,
)
from vox_symposium.json_io import write_json
from vox_symposium.models.base import RealtimeAudioModel, RealtimeInterruption
from vox_symposium.models.factory import build_evaluation_model_from_env
from vox_symposium.providers import normalize_provider
from vox_symposium.recording import audio_event_fields, write_wav
from vox_symposium.scenario import (
    AGENT_KEYS,
    HUMAN_AGENT,
    ROBOT_AGENT,
    AgentKey,
    LoadedScenario,
    build_evaluation_result,
    load_scenarios,
    other_agent,
    validate_agent,
    write_evaluation_result,
)
from vox_symposium.tick import TickAudioBuffer, TickResult, tick_result_fields


class _ConsoleLoggedError(Exception):
    """Signal that the original exception was already written to the run log."""


async def run() -> None:
    load_environment()
    args = _parse_args()
    _validate_args(args)
    result_path = Path(args.result)
    base_run_id = args.run_id or result_path.stem
    artifacts = EvaluationArtifacts.create(
        result_path,
        run_id=base_run_id,
        artifact_dir=args.artifact_dir,
    )

    with _tee_console(artifacts.console_log):
        print(f"Saving console log: {artifacts.console_log}")
        print(f"Command: {_format_command(sys.argv)}")
        try:
            await _run_evaluations(
                args,
                result_path=result_path,
                base_run_id=base_run_id,
                artifacts=artifacts,
            )
        except Exception as exc:
            print(_format_error(exc), file=sys.stderr)
            raise _ConsoleLoggedError from exc


async def _run_evaluations(
    args: argparse.Namespace,
    *,
    result_path: Path,
    base_run_id: str,
    artifacts: EvaluationArtifacts,
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
    loaded_total = len(scenarios)
    scenarios = _limit_scenarios(
        scenarios,
        limit=args.limit,
        scenario_id=args.scenario_id,
        scenario_index=args.scenario_index,
        start_index=args.start_index,
    )
    if not scenarios:
        raise RuntimeError("No scenarios found")
    if (args.scenario_id is not None or args.scenario_index is not None) and len(scenarios) != 1:
        raise RuntimeError(f"Expected exactly one scenario, got {len(scenarios)}")

    write_run_env_snapshot(artifacts.env_snapshot, run_id=base_run_id, args=args)

    start_index = args.start_index or 0
    total = loaded_total if start_index else len(scenarios)
    results = _load_resume_results(result_path, start_index=start_index)
    if total > 1 and args.answer_audio:
        raise RuntimeError(
            "--answer-audio can only be used when one scenario is selected with --id or --index"
        )
    if results:
        print(
            f"Loaded {len(results)} completed result(s) before --start-index {start_index}: "
            f"{result_path}"
        )

    write_summary_report(
        results,
        run_id=base_run_id,
        scenario_path=args.scenario,
        result_path=result_path,
        artifacts=artifacts,
        total_cases=total,
    )
    print("==========")
    for selected_index, scenario in enumerate(scenarios):
        index = start_index + selected_index
        run_id = _scenario_run_id(base_run_id, scenario.data, index=index, total=total)
        if selected_index > 0:
            print("----------")
        if total > 1:
            print(f"Running scenario {index + 1}/{total}: {scenario.id} (run_id={run_id})")

        result = await _run_scenario_with_retries(
            args,
            scenario=scenario,
            run_id=run_id,
            artifact_root=artifacts.root,
            env_snapshot_path=artifacts.env_snapshot,
        )
        result["artifacts"]["console_log"] = str(artifacts.console_log)
        result["artifacts"]["summary"] = str(artifacts.summary)
        result["case_index"] = index
        result["case_number"] = index + 1
        results.append(result)
        write_evaluation_result(result_path, _result_payload(results, total=total))
        write_summary_report(
            results,
            run_id=base_run_id,
            scenario_path=args.scenario,
            result_path=result_path,
            artifacts=artifacts,
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
        if args.case_delay > 0 and selected_index < len(scenarios) - 1:
            print(f"Waiting {args.case_delay:.1f}s before next scenario")
            await asyncio.sleep(args.case_delay)


async def _run_scenario_with_retries(
    args: argparse.Namespace,
    *,
    scenario: LoadedScenario,
    run_id: str,
    artifact_root: Path,
    env_snapshot_path: Path,
) -> dict[str, Any]:
    attempts = args.case_retries
    for attempt in range(1, attempts + 1):
        try:
            if attempt > 1:
                print(f"Retrying scenario {scenario.id}: attempt {attempt}/{attempts}")
            return await _run_scenario_evaluation(
                args,
                scenario=scenario,
                run_id=run_id,
                artifact_root=artifact_root,
                env_snapshot_path=env_snapshot_path,
            )
        except Exception as exc:
            _remove_failed_case_artifacts(artifact_root, scenario.data)
            if attempt >= attempts:
                print(f"Scenario {scenario.id} failed after {attempts} attempt(s)")
                if args.overnight:
                    print("Overnight mode is enabled; recording failed case and continuing")
                    return _build_failed_case_result(
                        scenario.data,
                        run_id=run_id,
                        exc=exc,
                        attempts=attempts,
                        env_snapshot_path=env_snapshot_path,
                    )
                raise
            print(
                f"Scenario {scenario.id} failed on attempt {attempt}/{attempts}: "
                f"{_format_error(exc)}"
            )
            print(f"Deleted failed case artifacts; retrying in {args.case_retry_delay:.1f}s")
            await asyncio.sleep(args.case_retry_delay)

    raise RuntimeError(f"Scenario {scenario.id} failed after {attempts} attempt(s)")


def _build_failed_case_result(
    scenario: dict[str, Any],
    *,
    run_id: str,
    exc: Exception,
    attempts: int,
    env_snapshot_path: Path,
) -> dict[str, Any]:
    result = build_evaluation_result(
        scenario,
        response_text="",
        response_audio=None,
        dialogue_log=None,
        run_id=run_id,
    )
    result["status"] = "failed"
    result["error"] = {
        "message": _format_error(exc),
        "attempts": attempts,
    }
    result["response"]["is_correct"] = False
    result["artifacts"]["env_snapshot"] = str(env_snapshot_path)
    return result


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

    providers: dict[AgentKey, str] = {agent: _agent_provider(agent) for agent in AGENT_KEYS}
    prompts = {
        agent: scenario.build_prompt(
            agent,
            provider=providers[agent],
            use_structured_history=provider_uses_structured_history(providers[agent]),
        )
        for agent in AGENT_KEYS
    }
    models: dict[AgentKey, RealtimeAudioModel] = {
        agent: _build_model(
            agent,
            prompts[agent].instructions,
            initial_history=prompts[agent].initial_history,
            continuous_audio=True,
        )
        for agent in AGENT_KEYS
    }

    audio_queues: dict[AgentKey, asyncio.Queue[PcmAudio | None]] = {
        agent: asyncio.Queue() for agent in AGENT_KEYS
    }
    text_queues: dict[AgentKey, asyncio.Queue[str | None]] = {
        agent: asyncio.Queue() for agent in AGENT_KEYS
    }
    event_queues: dict[AgentKey, asyncio.Queue[RealtimeInterruption | None]] = {
        agent: asyncio.Queue() for agent in AGENT_KEYS
    }
    reader_tasks: list[asyncio.Task[None]] = []
    dialogue_log: dict[str, Any] = {
        "scenario_id": scenario.id,
        "run_id": run_id,
        "events": [],
    }

    try:
        await asyncio.gather(*(model.connect() for model in models.values()))
        for agent, model in models.items():
            reader_tasks.append(
                asyncio.create_task(_read_audio(model, audio_queues[agent]), name=f"{agent}-audio")
            )
            reader_tasks.append(
                asyncio.create_task(_read_text(model, text_queues[agent]), name=f"{agent}-text")
            )
            reader_tasks.append(
                asyncio.create_task(_read_event(model, event_queues[agent]), name=f"{agent}-events")
            )

        await _play_opening(
            scenario.data,
            models,
            artifact_dir=artifact_dir,
            frame_ms=args.frame_ms,
            audio_speed=args.audio_speed,
            dialogue_log=dialogue_log,
        )
        robot_turns = await _run_dialogue_turns(
            models,
            audio_queues,
            text_queues,
            event_queues,
            dialogue_log=dialogue_log,
            artifact_dir=artifact_dir,
            target_turns=args.dialogue_turns,
            idle_timeout=args.idle_timeout,
            max_utterance_seconds=args.max_utterance_seconds,
            audio_speed=args.audio_speed,
            tick_duration_ms=args.tick_duration_ms,
        )

        _drain_queue(text_queues[ROBOT_AGENT])
        _drain_queue(audio_queues[ROBOT_AGENT])
        question_audio = _question_audio(
            scenario.data,
            question_audio_override=args.question_audio,
            artifact_dir=artifact_dir,
            scenario_dir=Path(args.scenario).parent,
            disable_tts=args.no_tts,
        )
        dialogue_log["events"].append(
            {
                "type": "evaluation_question",
                "target_agent": ROBOT_AGENT,
                "audio": str(question_audio),
                "after_robot_turns": robot_turns,
            }
        )
        print(f"Playing evaluation question into {ROBOT_AGENT}: {question_audio}")
        await _send_audio_file(
            question_audio,
            models[ROBOT_AGENT],
            frame_ms=args.frame_ms,
            audio_speed=args.audio_speed,
        )

        answer = await _collect_utterance(
            ROBOT_AGENT,
            audio_queues[ROBOT_AGENT],
            idle_timeout=args.idle_timeout,
            max_seconds=args.max_utterance_seconds,
        )
        answer_audio = (
            Path(args.answer_audio)
            if args.answer_audio
            else artifact_dir / f"{ROBOT_AGENT}-answer.wav"
        )
        answer_recording = write_wav(answer_audio, answer.audio)
        answer_text = await _collect_text_after_audio(
            text_queues[ROBOT_AGENT],
            idle_timeout=args.text_idle_timeout,
            max_wait=args.text_max_wait,
        )
        dialogue_log["events"].append(
            {
                "type": "evaluation_answer",
                "agent": ROBOT_AGENT,
                "text": answer_text,
                **audio_event_fields(answer_recording),
            }
        )
        print(f"Captured {ROBOT_AGENT} answer evaluation question: {answer_audio}")

        dialogue_log_path = artifact_dir / "dialogue-log.json"
        write_json(dialogue_log_path, dialogue_log)
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
        await asyncio.gather(*(model.close() for model in models.values()), return_exceptions=True)
        if reader_tasks:
            await asyncio.gather(*reader_tasks, return_exceptions=True)


def main() -> None:
    try:
        asyncio.run(run())
    except _ConsoleLoggedError as exc:
        raise SystemExit(1) from exc
    except RuntimeError as exc:
        raise SystemExit(_format_error(exc)) from exc
    except Exception as exc:
        raise SystemExit(_format_error(exc)) from exc


async def _play_opening(
    scenario: dict[str, Any],
    models: dict[AgentKey, RealtimeAudioModel],
    *,
    artifact_dir: Path,
    frame_ms: int,
    audio_speed: float,
    dialogue_log: dict[str, Any],
) -> None:
    opening = scenario.get("opening")
    if not opening:
        return

    opening_agent = validate_agent(str(opening["agent"]))
    receiver = other_agent(opening_agent)
    opening_audio = _turn_audio(opening)
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
    await _send_audio_file(
        opening_audio, models[receiver], frame_ms=frame_ms, audio_speed=audio_speed
    )


async def _run_dialogue_turns(
    models: dict[AgentKey, RealtimeAudioModel],
    audio_queues: dict[AgentKey, asyncio.Queue[PcmAudio | None]],
    text_queues: dict[AgentKey, asyncio.Queue[str | None]],
    event_queues: dict[AgentKey, asyncio.Queue[RealtimeInterruption | None]],
    *,
    dialogue_log: dict[str, Any],
    artifact_dir: Path,
    target_turns: int,
    idle_timeout: float,
    max_utterance_seconds: float,
    audio_speed: float,
    tick_duration_ms: int,
) -> int:
    if tick_duration_ms <= 0:
        raise RuntimeError("--tick-duration-ms must be greater than 0")

    tick_seconds = tick_duration_ms / 1000
    silence_ticks_to_end_turn = max(1, math.ceil(idle_timeout / tick_seconds))
    max_dialogue_seconds = max_utterance_seconds * max(1, target_turns * 2 + 1)
    started_at = asyncio.get_running_loop().time()

    buffers: dict[AgentKey, TickAudioBuffer] = {agent: TickAudioBuffer() for agent in AGENT_KEYS}
    active_audio: dict[AgentKey, bytearray] = {agent: bytearray() for agent in AGENT_KEYS}
    active_text: dict[AgentKey, list[str]] = {agent: [] for agent in AGENT_KEYS}
    turn_start_ticks: dict[AgentKey, int | None] = {agent: None for agent in AGENT_KEYS}
    silence_ticks: dict[AgentKey, int] = {agent: 0 for agent in AGENT_KEYS}
    interrupted_turn: dict[AgentKey, bool] = {agent: False for agent in AGENT_KEYS}
    forwarded_audio_ms: dict[AgentKey, float] = {agent: 0.0 for agent in AGENT_KEYS}
    robot_turns = 0
    event_index = 0
    tick_number = 0
    turn_counts: dict[AgentKey, int] = {agent: 0 for agent in AGENT_KEYS}
    tick_budget = tick_seconds / audio_speed if audio_speed > 0 else 0
    tick_wait = tick_budget if tick_budget > 0 else tick_seconds
    dialogue_log.setdefault("ticks", [])
    dialogue_log["tick_config"] = {
        "tick_duration_ms": tick_duration_ms,
        "silence_timeout_ms": idle_timeout * 1000,
        "audio_speed": audio_speed,
    }

    while robot_turns < target_turns:
        if asyncio.get_running_loop().time() - started_at >= max_dialogue_seconds:
            raise RuntimeError(
                f"Timed out waiting for dialogue turns after {max_dialogue_seconds:.1f}s"
            )

        tick_started_at = asyncio.get_running_loop().time()
        tick_number += 1
        interrupted_agents = await _consume_provider_interruptions(
            models=models,
            audio_queues=audio_queues,
            event_queues=event_queues,
            buffers=buffers,
            interrupted_turn=interrupted_turn,
            forwarded_audio_ms=forwarded_audio_ms,
            dialogue_log=dialogue_log,
            tick_number=tick_number,
        )
        await asyncio.gather(
            *(
                _fill_tick_buffer(
                    audio_queues[agent],
                    buffers[agent],
                    tick_duration_ms=tick_duration_ms,
                    max_wait=tick_wait,
                    agent=agent,
                )
                for agent in AGENT_KEYS
            )
        )

        interrupted_agents.update(
            await _consume_provider_interruptions(
                models=models,
                audio_queues=audio_queues,
                event_queues=event_queues,
                buffers=buffers,
                interrupted_turn=interrupted_turn,
                forwarded_audio_ms=forwarded_audio_ms,
                dialogue_log=dialogue_log,
                tick_number=tick_number,
            )
        )

        for agent in AGENT_KEYS:
            _drain_text_to_parts(text_queues[agent], active_text[agent])

        fixed_audio: dict[AgentKey, PcmAudio] = {}
        captured_audio: dict[AgentKey, PcmAudio] = {}
        truncated: dict[AgentKey, bool] = {}
        for agent in AGENT_KEYS:
            fixed, captured, was_truncated = buffers[agent].pop_tick(tick_duration_ms)
            fixed_audio[agent] = fixed
            captured_audio[agent] = captured
            truncated[agent] = was_truncated
            if captured.data:
                active_audio[agent].extend(captured.data)

            if captured.data:
                if turn_start_ticks[agent] is None:
                    turn_start_ticks[agent] = tick_number
                silence_ticks[agent] = 0
            elif turn_start_ticks[agent] is not None:
                silence_ticks[agent] += 1

        tick_result = TickResult(
            tick_number=tick_number,
            tick_duration_ms=tick_duration_ms,
            audio=cast(dict[str, PcmAudio], fixed_audio),
            captured_audio=cast(dict[str, PcmAudio], captured_audio),
            truncated=cast(dict[str, bool], truncated),
            interrupted_agents=tuple(sorted(interrupted_agents)),
        )
        dialogue_log.setdefault("ticks", []).append(
            {
                "type": "dialogue_tick",
                **tick_result_fields(tick_result),
            }
        )

        await asyncio.gather(
            models[HUMAN_AGENT].send_audio(fixed_audio[ROBOT_AGENT]),
            models[ROBOT_AGENT].send_audio(fixed_audio[HUMAN_AGENT]),
        )

        for agent in AGENT_KEYS:
            if turn_start_ticks[agent] is not None:
                forwarded_audio_ms[agent] += captured_audio[agent].duration_seconds * 1000

        for agent in AGENT_KEYS:
            _drain_text_to_parts(text_queues[agent], active_text[agent])
            if turn_start_ticks[agent] is None:
                continue
            if silence_ticks[agent] < silence_ticks_to_end_turn:
                continue

            event_index += 1
            turn_counts[agent] += 1
            turn_index = turn_counts[agent]
            if agent == ROBOT_AGENT:
                robot_turns += 1

            audio = PcmAudio(
                data=bytes(active_audio[agent]),
                sample_rate=fixed_audio[agent].sample_rate,
                channels=fixed_audio[agent].channels,
            )
            utterance_path = artifact_dir / f"dialogue-{event_index:02d}-{agent}.wav"
            recording = write_wav(utterance_path, audio)
            text = "".join(active_text[agent]).strip()
            dialogue_log["events"].append(
                {
                    "type": "dialogue_turn",
                    "agent": agent,
                    "turn_index": turn_index,
                    "robot_turns": robot_turns,
                    "text": text,
                    "start_tick": turn_start_ticks[agent],
                    "end_tick": tick_number,
                    "interrupted": interrupted_turn[agent],
                    **audio_event_fields(recording),
                }
            )
            print(_format_dialogue_capture(agent, turn_index, text))
            active_audio[agent].clear()
            active_text[agent].clear()
            turn_start_ticks[agent] = None
            silence_ticks[agent] = 0
            interrupted_turn[agent] = False
            forwarded_audio_ms[agent] = 0.0

        remaining = tick_budget - (asyncio.get_running_loop().time() - tick_started_at)
        if remaining > 0:
            await asyncio.sleep(remaining)

    return robot_turns


async def _consume_provider_interruptions(
    *,
    models: dict[AgentKey, RealtimeAudioModel],
    audio_queues: dict[AgentKey, asyncio.Queue[PcmAudio | None]],
    event_queues: dict[AgentKey, asyncio.Queue[RealtimeInterruption | None]],
    buffers: dict[AgentKey, TickAudioBuffer],
    interrupted_turn: dict[AgentKey, bool],
    forwarded_audio_ms: dict[AgentKey, float],
    dialogue_log: dict[str, Any],
    tick_number: int,
) -> set[AgentKey]:
    interrupted_agents: set[AgentKey] = set()
    for agent in AGENT_KEYS:
        while True:
            try:
                event = event_queues[agent].get_nowait()
            except asyncio.QueueEmpty:
                break
            if event is None:
                continue

            audio_end_ms = max(0, round(forwarded_audio_ms[agent]))
            buffers[agent].clear()
            _clear_pending_audio(audio_queues[agent])
            interrupted_turn[agent] = True
            forwarded_audio_ms[agent] = 0.0
            await models[agent].truncate_response(event, audio_end_ms=audio_end_ms)
            interrupted_agents.add(agent)
            dialogue_log["events"].append(
                {
                    "type": "provider_interruption",
                    "agent": agent,
                    "tick": tick_number,
                    "reason": event.reason,
                    "item_id": event.item_id,
                    "response_id": event.response_id,
                    "audio_end_ms": audio_end_ms,
                }
            )
    return interrupted_agents


async def _fill_tick_buffer(
    queue: asyncio.Queue[PcmAudio | None],
    buffer: TickAudioBuffer,
    *,
    tick_duration_ms: int,
    max_wait: float,
    agent: AgentKey,
) -> None:
    """Wait for enough provider output to fill a tick, then leave excess buffered."""
    deadline = asyncio.get_running_loop().time() + max_wait
    while buffer.pending_duration_ms < tick_duration_ms:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return
        try:
            item = await asyncio.wait_for(queue.get(), timeout=remaining)
        except TimeoutError:
            return
        if item is None:
            raise RuntimeError(f"{agent} model audio stream closed during tick evaluation")
        buffer.append(item)

        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                raise RuntimeError(f"{agent} model audio stream closed during tick evaluation")
            buffer.append(item)


def _drain_text_to_parts(queue: asyncio.Queue[str | None], parts: list[str]) -> None:
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        if item:
            parts.append(item)


def _format_dialogue_capture(agent: AgentKey, turn_index: int, text: str = "") -> str:
    turn_word = "turns" if agent == ROBOT_AGENT else "turn"
    message = f"Captured {agent} {turn_word} {turn_index}"
    if text:
        return f"{message}: {text}"
    return message


def _build_model(
    agent: AgentKey,
    instructions: str,
    *,
    initial_history: tuple[dict[str, Any], ...] = (),
    continuous_audio: bool = False,
) -> RealtimeAudioModel:
    return build_evaluation_model_from_env(
        _agent_provider(agent),
        instructions,
        initial_history=initial_history,
        continuous_audio=continuous_audio,
    )


def _agent_provider(agent: AgentKey) -> str:
    agent = validate_agent(agent)
    return normalize_provider(
        env_with_legacy(
            f"AGENT_{agent.upper()}_PROVIDER",
            f"AGENT_{'A' if agent == HUMAN_AGENT else 'B'}_PROVIDER",
            default=_default_provider(agent),
        )
    )


def _limit_scenarios(
    scenarios: list[LoadedScenario],
    *,
    limit: int | None,
    scenario_id: str | None,
    scenario_index: int | None,
    start_index: int = 0,
) -> list[LoadedScenario]:
    if start_index < 0:
        raise RuntimeError("--start-index must be at least 0")
    if limit is not None and limit < 1:
        raise RuntimeError("--limit must be at least 1")
    if scenario_id is not None or scenario_index is not None:
        if start_index:
            raise RuntimeError("--start-index cannot be used with --id or --index")
        if limit is not None:
            raise RuntimeError("--limit cannot be used with --id or --index")
        return scenarios
    if start_index >= len(scenarios) and scenarios:
        raise RuntimeError(f"--start-index out of range: {start_index}")

    selected = scenarios[start_index:]
    if limit is None:
        return selected
    return selected[:limit]


def _drain_queue(queue: asyncio.Queue[Any]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return


def _clear_pending_audio(queue: asyncio.Queue[PcmAudio | None]) -> None:
    """Drop stale audio while preserving a reader's closed-stream sentinel."""
    closed = False
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if item is None:
            closed = True
    if closed:
        queue.put_nowait(None)


def _default_provider(agent: AgentKey) -> str:
    agent = validate_agent(agent)
    return "openai" if agent == HUMAN_AGENT else "gemini"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run automated Vox Symposium scenario evaluations."
    )
    parser.add_argument("scenario", help="Normalized scenario JSON path, or source dataset.")
    parser.add_argument("result", help="Output evaluation result JSON path.")
    parser.add_argument(
        "--id", dest="scenario_id", help="Select a scenario by id when scenario is a dataset."
    )
    parser.add_argument(
        "--index", dest="scenario_index", type=int, help="Select a scenario by zero-based index."
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start a batch at this zero-based scenario index, e.g. 6 starts at case 7.",
    )
    parser.add_argument(
        "--limit", type=int, help="Run only the first N scenarios when scenario is a dataset."
    )
    parser.add_argument("--audio-dir", help="Directory containing original speech wav files.")
    parser.add_argument(
        "--dialogue-turns", type=int, default=5, help="Robot turns before evaluation."
    )
    parser.add_argument(
        "--question-audio",
        help="Evaluation question wav file. Overrides scenario evaluation.question_audio.",
    )
    parser.add_argument("--answer-audio", help="Where to save the evaluated robot answer wav.")
    parser.add_argument(
        "--artifact-dir", help="Directory for dialogue logs and captured wav files."
    )
    parser.add_argument("--run-id", help="Stable run id for this evaluation.")
    parser.add_argument(
        "--frame-ms", type=int, default=20, help="Audio frame size used to stream wav files."
    )
    parser.add_argument(
        "--tick-duration-ms",
        type=int,
        default=int_env("EVALUATION_TICK_DURATION_MS", 200),
        help=(
            "Full-duplex TickResult duration in milliseconds. "
            "Defaults to EVALUATION_TICK_DURATION_MS or 200."
        ),
    )
    parser.add_argument(
        "--audio-speed",
        type=float,
        default=float_env("EVALUATION_AUDIO_SPEED", 1.0),
        help=(
            "Audio injection speed. 1.0 is realtime; higher values send audio faster and may "
            "affect streaming VAD/turn detection; 0 disables sleeps."
        ),
    )
    parser.add_argument(
        "--idle-timeout", type=float, default=1.5, help="Silence timeout used to end an utterance."
    )
    parser.add_argument(
        "--max-utterance-seconds",
        type=float,
        default=30.0,
        help="Maximum seconds to wait for one utterance.",
    )
    parser.add_argument(
        "--text-idle-timeout",
        type=float,
        default=0.7,
        help="Seconds without new text deltas before captured text is considered complete.",
    )
    parser.add_argument(
        "--text-max-wait",
        type=float,
        default=5.0,
        help="Maximum seconds to wait for delayed text deltas after an utterance ends.",
    )
    parser.add_argument(
        "--case-retries",
        type=int,
        default=3,
        help="Maximum attempts per scenario before failing the evaluation.",
    )
    parser.add_argument(
        "--case-delay",
        type=float,
        default=0.0,
        help="Seconds to wait after a successful scenario before starting the next one.",
    )
    parser.add_argument(
        "--case-retry-delay",
        type=float,
        default=30.0,
        help="Seconds to wait before retrying a failed scenario.",
    )
    parser.add_argument(
        "--overnight",
        action="store_true",
        help=(
            "Continue to the next scenario after retry attempts are exhausted, "
            "recording the case as failed."
        ),
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Require question audio instead of generating it with macOS say.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    minimums = {
        "--start-index": (args.start_index, 0, True),
        "--dialogue-turns": (args.dialogue_turns, 0, True),
        "--frame-ms": (args.frame_ms, 0, False),
        "--tick-duration-ms": (getattr(args, "tick_duration_ms", 200), 0, False),
        "--audio-speed": (args.audio_speed, 0, True),
        "--idle-timeout": (args.idle_timeout, 0, False),
        "--max-utterance-seconds": (args.max_utterance_seconds, 0, False),
        "--text-idle-timeout": (args.text_idle_timeout, 0, True),
        "--text-max-wait": (args.text_max_wait, 0, True),
        "--case-retries": (args.case_retries, 1, True),
        "--case-delay": (args.case_delay, 0, True),
        "--case-retry-delay": (args.case_retry_delay, 0, True),
    }
    for option, (value, minimum, inclusive) in minimums.items():
        is_valid = value >= minimum if inclusive else value > minimum
        if not is_valid:
            comparison = "at least" if inclusive else "greater than"
            raise RuntimeError(f"{option} must be {comparison} {minimum}")

    if args.limit is not None and args.limit < 1:
        raise RuntimeError("--limit must be at least 1")


if __name__ == "__main__":
    main()
