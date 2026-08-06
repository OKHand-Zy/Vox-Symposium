from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vox_symposium.audio import PCM_SAMPLE_WIDTH_BYTES, PcmAudio, normalize_audio

TICK_AUDIO_SAMPLE_RATE = 24_000
TICK_AUDIO_CHANNELS = 1


@dataclass(frozen=True)
class TickResult:
    """The audio exchanged during one discrete full-duplex simulation tick.

    ``audio`` is always exactly ``tick_duration_ms`` long so it can be sent to
    the other realtime model immediately. ``captured_audio`` contains only the
    bytes that were actually available from the model; padding is deliberately
    excluded from recordings and turn logs.
    """

    tick_number: int
    tick_duration_ms: int
    audio: Mapping[str, PcmAudio]
    captured_audio: Mapping[str, PcmAudio]
    truncated: Mapping[str, bool]
    interrupted_agents: tuple[str, ...] = ()

    @property
    def simulation_time_ms(self) -> int:
        return self.tick_number * self.tick_duration_ms


class TickAudioBuffer:
    """Buffer model output and expose fixed-size audio ticks.

    Realtime providers do not necessarily return audio chunks aligned to the
    evaluator's tick size. This buffer carries excess output to the next tick,
    pads short ticks with silence, and can discard pending output when a
    barge-in interrupts the current response.
    """

    def __init__(
        self,
        *,
        sample_rate: int = TICK_AUDIO_SAMPLE_RATE,
        channels: int = TICK_AUDIO_CHANNELS,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels != 1:
            raise ValueError("TickAudioBuffer currently supports mono audio only")
        self.sample_rate = sample_rate
        self.channels = channels
        self._data = bytearray()

    @property
    def pending_bytes(self) -> int:
        return len(self._data)

    @property
    def pending_duration_ms(self) -> float:
        bytes_per_second = self.sample_rate * self.channels * PCM_SAMPLE_WIDTH_BYTES
        return len(self._data) * 1000 / bytes_per_second

    def append(self, audio: PcmAudio) -> None:
        normalized = normalize_audio(
            audio.data,
            from_rate=audio.sample_rate,
            to_rate=self.sample_rate,
            channels=audio.channels,
        )
        if normalized:
            self._data.extend(normalized)

    def clear(self) -> None:
        self._data.clear()

    def pop_tick(self, tick_duration_ms: int) -> tuple[PcmAudio, PcmAudio, bool]:
        """Return ``(fixed_audio, captured_audio, truncated)`` for one tick."""
        if tick_duration_ms <= 0:
            raise ValueError("tick_duration_ms must be positive")

        tick_bytes = self._tick_bytes(tick_duration_ms)
        captured_bytes = bytes(self._data[:tick_bytes])
        truncated = len(self._data) > tick_bytes
        del self._data[:tick_bytes]

        fixed_bytes = captured_bytes
        if len(fixed_bytes) < tick_bytes:
            fixed_bytes += b"\x00" * (tick_bytes - len(fixed_bytes))

        return (
            PcmAudio(
                data=fixed_bytes,
                sample_rate=self.sample_rate,
                channels=self.channels,
            ),
            PcmAudio(
                data=captured_bytes,
                sample_rate=self.sample_rate,
                channels=self.channels,
            ),
            truncated,
        )

    def _tick_bytes(self, tick_duration_ms: int) -> int:
        samples = self.sample_rate * tick_duration_ms / 1000
        if not samples.is_integer():
            raise ValueError(
                "tick_duration_ms must produce a whole number of samples for the buffer format"
            )
        return int(samples) * self.channels * PCM_SAMPLE_WIDTH_BYTES


def tick_result_fields(result: TickResult) -> dict[str, Any]:
    """Return JSON-friendly metadata for a tick log entry."""
    return {
        "tick_number": result.tick_number,
        "simulation_time_ms": result.simulation_time_ms,
        "tick_duration_ms": result.tick_duration_ms,
        "truncated": dict(result.truncated),
        "interrupted_agents": list(result.interrupted_agents),
    }
