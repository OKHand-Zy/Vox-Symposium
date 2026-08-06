from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vox_symposium.audio import PcmAudio


@dataclass(frozen=True)
class RecordedAudio:
    path: Path
    sample_rate: int
    channels: int
    duration_seconds: float


def write_wav(path: str | Path, audio: PcmAudio) -> RecordedAudio:
    if audio.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if audio.channels < 1:
        raise ValueError("channels must be at least 1")
    frame_width = audio.channels * 2
    if len(audio.data) % frame_width:
        raise ValueError("PCM data length must be a whole number of audio frames")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(audio.channels)
        wav.setsampwidth(2)
        wav.setframerate(audio.sample_rate)
        wav.writeframes(audio.data)

    return RecordedAudio(
        path=output_path,
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        duration_seconds=audio.duration_seconds,
    )


def audio_event_fields(recording: RecordedAudio) -> dict[str, Any]:
    return {
        "audio": str(recording.path),
        "sample_rate": recording.sample_rate,
        "channels": recording.channels,
        "duration_seconds": round(recording.duration_seconds, 3),
    }
