from __future__ import annotations

import asyncio
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vox_symposium.audio import (
    PCM_SAMPLE_WIDTH_BYTES,
    PcmAudio,
    concatenate_pcm_audio,
    rechunk_pcm16,
)
from vox_symposium.models.base import RealtimeAudioModel, RealtimeInterruption


@dataclass(frozen=True)
class AudioUtterance:
    agent: str
    audio: PcmAudio


async def collect_utterance(
    agent: str,
    queue: asyncio.Queue[PcmAudio | None],
    *,
    idle_timeout: float,
    max_seconds: float,
) -> AudioUtterance:
    if idle_timeout < 0:
        raise ValueError("idle_timeout must be at least 0")
    if max_seconds <= 0:
        raise ValueError("max_seconds must be greater than 0")

    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_seconds
    try:
        first = await asyncio.wait_for(queue.get(), timeout=max_seconds)
    except TimeoutError as exc:
        raise RuntimeError(f"Timed out waiting for {agent} audio after {max_seconds:.1f}s") from exc
    if first is None:
        raise RuntimeError(f"{agent} model audio stream closed before an utterance was captured")

    chunks = [first]
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            item = await asyncio.wait_for(queue.get(), timeout=min(idle_timeout, remaining))
        except TimeoutError:
            break
        if item is None:
            break
        chunks.append(item)

    try:
        audio = concatenate_pcm_audio(chunks)
    except ValueError as exc:
        raise RuntimeError(f"{agent} changed audio format during an utterance") from exc
    return AudioUtterance(agent=agent, audio=audio)


async def send_audio_file(
    path: Path,
    model: RealtimeAudioModel,
    *,
    frame_ms: int,
    audio_speed: float,
) -> None:
    await send_audio(
        read_wav(path),
        model,
        frame_ms=frame_ms,
        audio_speed=audio_speed,
    )


async def send_audio(
    audio: PcmAudio,
    model: RealtimeAudioModel,
    *,
    frame_ms: int,
    audio_speed: float,
) -> None:
    frame_seconds = frame_sleep_seconds(frame_ms, audio_speed)
    await model.start_audio_turn()
    try:
        for chunk in rechunk_pcm16(
            audio.data,
            audio.sample_rate,
            frame_ms,
            channels=audio.channels,
        ):
            await model.send_audio(
                PcmAudio(
                    data=chunk,
                    sample_rate=audio.sample_rate,
                    channels=audio.channels,
                )
            )
            await asyncio.sleep(frame_seconds)

        silence = b"\x00" * (int(audio.sample_rate * 0.2) * PCM_SAMPLE_WIDTH_BYTES * audio.channels)
        for chunk in rechunk_pcm16(
            silence,
            audio.sample_rate,
            frame_ms,
            channels=audio.channels,
        ):
            await model.send_audio(
                PcmAudio(data=chunk, sample_rate=audio.sample_rate, channels=audio.channels)
            )
            await asyncio.sleep(frame_seconds)
    finally:
        await model.flush_input_stream()


async def read_audio_stream(
    model: RealtimeAudioModel,
    queue: asyncio.Queue[PcmAudio | None],
) -> None:
    try:
        async for audio in model.receive_audio():
            await queue.put(audio)
    finally:
        await queue.put(None)


async def read_text_stream(
    model: RealtimeAudioModel,
    queue: asyncio.Queue[str | None],
) -> None:
    try:
        async for text in model.receive_text():
            await queue.put(text)
    finally:
        await queue.put(None)


async def read_event_stream(
    model: RealtimeAudioModel,
    queue: asyncio.Queue[RealtimeInterruption | None],
) -> None:
    try:
        async for event in model.receive_events():
            await queue.put(event)
    finally:
        await queue.put(None)


async def collect_text_after_audio(
    queue: asyncio.Queue[str | None],
    *,
    idle_timeout: float,
    max_wait: float,
) -> str:
    if idle_timeout < 0:
        raise RuntimeError("--text-idle-timeout must be at least 0")
    if max_wait < 0:
        raise RuntimeError("--text-max-wait must be at least 0")

    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_wait
    parts: list[str] = []
    received_text = False
    while True:
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                return "".join(parts).strip()
            if item:
                parts.append(item)
                received_text = True

        remaining = deadline - loop.time()
        if remaining <= 0:
            break

        try:
            # Text can start later than the audio stream. Before the first
            # chunk arrives, wait for the full grace period; once generation
            # has started, use the shorter idle timeout to detect completion.
            timeout = min(idle_timeout, remaining) if received_text else remaining
            item = await asyncio.wait_for(queue.get(), timeout=timeout)
        except TimeoutError:
            break
        if item is None:
            break
        if item:
            parts.append(item)
            received_text = True
    return "".join(parts).strip()


def question_audio(
    scenario: dict[str, Any],
    *,
    question_audio_override: str | None,
    artifact_dir: Path,
    scenario_dir: Path,
    disable_tts: bool,
) -> Path:
    configured_path = question_audio_override or (scenario.get("evaluation") or {}).get(
        "question_audio"
    )
    if configured_path:
        path = resolve_question_audio_path(
            configured_path,
            scenario_dir=scenario_dir,
        )
        return ensure_wav(path, artifact_dir / "question.wav")

    if disable_tts:
        raise RuntimeError("No evaluation.question_audio is set and --no-tts was used")

    path = artifact_dir / "question.wav"
    synthesize_question_audio(_question_text(scenario), path)
    return path


def resolve_question_audio_path(path_text: str, *, scenario_dir: Path) -> Path:
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
        if candidate.is_file():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"Question audio does not exist. Checked: {checked}")


def ensure_wav(path: Path, output: Path) -> Path:
    if path.suffix.lower() == ".wav":
        output.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() == output.resolve():
            return path
        shutil.copyfile(path, output)
        return output
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            f"Question audio is {path.suffix}, but ffmpeg is not available to convert it to wav"
        )
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


def synthesize_question_audio(text: str, output: Path) -> None:
    say = shutil.which("say")
    afconvert = shutil.which("afconvert")
    if not say or not afconvert:
        raise RuntimeError("No question_audio is set, and macOS say/afconvert are not available")

    output.parent.mkdir(parents=True, exist_ok=True)
    aiff = output.with_suffix(".aiff")
    try:
        subprocess.run([say, "-o", str(aiff), text], check=True)
        subprocess.run(
            [
                afconvert,
                "-f",
                "WAVE",
                "-d",
                "LEI16@24000",
                str(aiff),
                str(output),
            ],
            check=True,
        )
    finally:
        aiff.unlink(missing_ok=True)

    if read_wav(output).duration_seconds < 0.5:
        raise RuntimeError(
            "Generated question audio is too short. Set evaluation.question_audio "
            "or pass --question-audio."
        )


def turn_audio(turn: dict[str, Any]) -> Path:
    audio = turn.get("audio")
    if audio:
        path = Path(audio)
        if path.is_file():
            return path
    raise RuntimeError(
        f"Opening turn has no usable audio. Expected an existing audio path, got {audio!r}."
    )


def read_wav(path: Path) -> PcmAudio:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        if sample_width != 2:
            raise RuntimeError(f"{path} must be 16-bit PCM wav, got sample width {sample_width}")
        data = wav.readframes(wav.getnframes())
    return PcmAudio(data=data, sample_rate=sample_rate, channels=channels)


def frame_sleep_seconds(frame_ms: int, audio_speed: float) -> float:
    if frame_ms <= 0:
        raise ValueError("frame_ms must be greater than 0")
    if audio_speed < 0:
        raise ValueError("audio_speed must be at least 0")
    if audio_speed <= 0:
        return 0
    return (frame_ms / 1000) / audio_speed


def _question_text(scenario: dict[str, Any]) -> str:
    evaluation = scenario.get("evaluation") or {}
    choices = " ".join(str(choice) for choice in evaluation.get("choices") or [])
    return f"{evaluation.get('question', '')} {choices} Answer with A, B, C, or D."
