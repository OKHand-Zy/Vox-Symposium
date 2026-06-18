from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vox_symposium.audio import PcmAudio


@dataclass(frozen=True)
class RecordedAudio:
    path: Path
    sample_rate: int
    channels: int
    duration_seconds: float


class ConversationRecorder:
    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": _utc_now(),
            "metadata": metadata or {},
            "events": [],
        }

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.payload["events"]

    def append(self, event: dict[str, Any]) -> None:
        self.events.append(
            {
                "index": len(self.events) + 1,
                "recorded_at": _utc_now(),
                **event,
            }
        )
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(self.payload, file, ensure_ascii=False, indent=2)
            file.write("\n")


def write_wav(path: str | Path, audio: PcmAudio) -> RecordedAudio:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(audio.channels)
        wav.setsampwidth(2)
        wav.setframerate(audio.sample_rate)
        wav.writeframes(audio.data)

    bytes_per_second = audio.sample_rate * audio.channels * 2
    duration = len(audio.data) / bytes_per_second if bytes_per_second else 0.0
    return RecordedAudio(
        path=output_path,
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        duration_seconds=duration,
    )


def audio_event_fields(recording: RecordedAudio) -> dict[str, Any]:
    return {
        "audio": str(recording.path),
        "sample_rate": recording.sample_rate,
        "channels": recording.channels,
        "duration_seconds": round(recording.duration_seconds, 3),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
