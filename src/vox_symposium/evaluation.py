from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vox_symposium.audio import PcmAudio, rechunk_pcm16
from vox_symposium.models.base import RealtimeAudioModel
from vox_symposium.scenario import (
    build_evaluation_result,
    load_scenario,
    write_evaluation_result,
)

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
    scenario = load_scenario(
        args.scenario,
        scenario_id=args.scenario_id,
        scenario_index=args.scenario_index,
        audio_dir=args.audio_dir,
        dialogue_turns=args.dialogue_turns,
    )

    result_path = Path(args.result)
    run_id = args.run_id or result_path.stem
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else result_path.parent / f"{run_id}-artifacts"
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
        await _send_audio_file(question_audio, scholar, frame_ms=args.frame_ms, audio_speed=args.audio_speed)

        answer = await _collect_utterance(
            "scholar",
            audio_queues["scholar"],
            idle_timeout=args.idle_timeout,
            max_seconds=args.max_utterance_seconds,
        )
        answer_audio = Path(args.answer_audio) if args.answer_audio else artifact_dir / "scholar-answer.wav"
        _write_wav(answer_audio, answer.audio)
        answer_text = await _collect_text_after_audio(text_queues["scholar"])
        dialogue_log["events"].append(
            {
                "type": "evaluation_answer",
                "agent": "scholar",
                "audio": str(answer_audio),
                "text": answer_text,
            }
        )

        dialogue_log_path = artifact_dir / "dialogue-log.json"
        _write_json(dialogue_log_path, dialogue_log)
        result = build_evaluation_result(
            scenario.data,
            response_text=answer_text,
            response_audio=str(answer_audio),
            dialogue_log=str(dialogue_log_path),
            run_id=run_id,
        )
        write_evaluation_result(result_path, result)
        response = result["response"]
        print(
            "Saved evaluation result: "
            f"{result_path} "
            f"(choice={response['choice'] or 'unknown'}, "
            f"is_correct={response['is_correct']}, "
            f"answer_audio={answer_audio})"
        )
    finally:
        for task in reader_tasks:
            task.cancel()
        await asyncio.gather(citizen.close(), scholar.close(), return_exceptions=True)


def main() -> None:
    try:
        asyncio.run(run())
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    except Exception as exc:
        raise SystemExit(f"Error: {type(exc).__name__}: {exc}") from exc


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

    while scholar_turns < target_turns:
        utterance = await _collect_utterance(
            current_agent,
            audio_queues[current_agent],
            idle_timeout=idle_timeout,
            max_seconds=max_utterance_seconds,
        )
        event_index += 1
        if current_agent == "scholar":
            scholar_turns += 1
        utterance_path = artifact_dir / f"dialogue-{event_index:02d}-{current_agent}.wav"
        _write_wav(utterance_path, utterance.audio)
        dialogue_log["events"].append(
            {
                "type": "dialogue_turn",
                "agent": current_agent,
                "scholar_turns": scholar_turns,
                "audio": str(utterance_path),
            }
        )
        print(f"Captured {current_agent} turn {event_index}; scholar_turns={scholar_turns}")

        if current_agent == "scholar" and scholar_turns >= target_turns:
            break

        receiver = _other_agent(current_agent)
        await _send_audio(utterance.audio, models[receiver], frame_ms=frame_ms, audio_speed=audio_speed)
        current_agent = receiver

    return scholar_turns


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
    provider = _env(f"AGENT_{agent.upper()}_PROVIDER", f"AGENT_{'A' if agent == 'citizen' else 'B'}_PROVIDER", default=_default_provider(agent)).lower()
    if provider == "openai":
        try:
            from vox_symposium.models.openai_realtime import OpenAIRealtimeModel
        except ModuleNotFoundError as exc:
            raise RuntimeError("OpenAI provider dependencies are missing. Run `pip install -r requirements.txt`.") from exc

        return OpenAIRealtimeModel(
            api_key=_required("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2"),
            voice=os.getenv("OPENAI_REALTIME_VOICE", "marin"),
            instructions=instructions,
        )
    if provider == "gemini":
        try:
            from vox_symposium.models.gemini_live import GeminiLiveModel
        except ModuleNotFoundError as exc:
            raise RuntimeError("Gemini provider dependencies are missing. Run `pip install -r requirements.txt`.") from exc

        return GeminiLiveModel(
            api_key=_required("GEMINI_API_KEY"),
            model=os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview"),
            instructions=instructions,
            manual_activity=True,
        )
    raise RuntimeError(f"Unsupported provider for {agent}: {provider}")


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


def _write_wav(path: Path, audio: PcmAudio) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(audio.channels)
        wav.setsampwidth(2)
        wav.setframerate(audio.sample_rate)
        wav.writeframes(audio.data)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _drain_queue(queue: asyncio.Queue[str | None]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return


def _other_agent(agent: str) -> str:
    return "scholar" if agent == "citizen" else "citizen"


def _env(primary: str, legacy: str, *, default: str) -> str:
    return os.getenv(primary) or os.getenv(legacy) or default


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _default_provider(agent: str) -> str:
    return "openai" if agent == "citizen" else "gemini"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one automated Vox Symposium scenario evaluation.")
    parser.add_argument("scenario", help="Normalized scenario JSON path, or source dataset with --id/--index.")
    parser.add_argument("result", help="Output evaluation result JSON path.")
    parser.add_argument("--id", dest="scenario_id", help="Select a scenario by id when scenario is a dataset.")
    parser.add_argument("--index", dest="scenario_index", type=int, help="Select a scenario by zero-based index.")
    parser.add_argument("--audio-dir", help="Directory containing original speech wav files.")
    parser.add_argument("--dialogue-turns", type=int, default=5, help="Scholar turns before evaluation.")
    parser.add_argument("--question-audio", help="Evaluation question wav file. Overrides scenario evaluation.question_audio.")
    parser.add_argument("--answer-audio", help="Where to save the evaluated scholar answer wav.")
    parser.add_argument("--artifact-dir", help="Directory for dialogue logs and captured wav files.")
    parser.add_argument("--run-id", help="Stable run id for this evaluation.")
    parser.add_argument("--frame-ms", type=int, default=20, help="Audio frame size used to stream wav files.")
    parser.add_argument("--audio-speed", type=float, default=float(os.getenv("EVALUATION_AUDIO_SPEED", "1.0")), help="Audio injection speed. 1.0 is realtime; higher values send audio faster and may affect streaming VAD/turn detection; 0 disables sleeps.")
    parser.add_argument("--idle-timeout", type=float, default=1.5, help="Silence timeout used to end an utterance.")
    parser.add_argument("--max-utterance-seconds", type=float, default=30.0, help="Maximum seconds to wait for one utterance.")
    parser.add_argument("--no-tts", action="store_true", help="Require question audio instead of generating it with macOS say.")
    return parser.parse_args()


def _frame_sleep_seconds(frame_ms: int, audio_speed: float) -> float:
    if audio_speed <= 0:
        return 0
    return (frame_ms / 1000) / audio_speed


if __name__ == "__main__":
    main()
