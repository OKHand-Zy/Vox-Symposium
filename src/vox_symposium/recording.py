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
