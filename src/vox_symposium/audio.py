from __future__ import annotations

import sys
from array import array
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

PCM_SAMPLE_WIDTH_BYTES = 2


@dataclass(frozen=True)
class PcmAudio:
    data: bytes
    sample_rate: int
    channels: int = 1

    @property
    def duration_seconds(self) -> float:
        bytes_per_second = self.sample_rate * self.channels * PCM_SAMPLE_WIDTH_BYTES
        return len(self.data) / bytes_per_second if bytes_per_second > 0 else 0.0


def concatenate_pcm_audio(chunks: Iterable[PcmAudio]) -> PcmAudio:
    """Join compatible PCM chunks and reject accidental format changes."""
    iterator = iter(chunks)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("at least one PCM audio chunk is required") from exc

    data = bytearray(first.data)
    for chunk in iterator:
        if (chunk.sample_rate, chunk.channels) != (first.sample_rate, first.channels):
            raise ValueError("PCM audio chunks must use the same sample rate and channel count")
        data.extend(chunk.data)
    return PcmAudio(
        data=bytes(data),
        sample_rate=first.sample_rate,
        channels=first.channels,
    )


def ensure_mono_pcm16(data: bytes, channels: int) -> bytes:
    if channels == 1:
        return data
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}")

    samples = _pcm16_array(data)
    mono = array("h")
    usable = len(samples) - (len(samples) % channels)
    for idx in range(0, usable, channels):
        mono.append(int(sum(samples[idx : idx + channels]) / channels))
    return _array_to_le_bytes(mono)


def resample_pcm16_mono(data: bytes, from_rate: int, to_rate: int) -> bytes:
    if from_rate <= 0 or to_rate <= 0:
        raise ValueError("sample rates must be positive")
    if from_rate == to_rate or not data:
        return data

    src = _pcm16_array(data)
    if len(src) <= 1:
        return data

    dst_len = max(1, int(round(len(src) * to_rate / from_rate)))
    ratio = from_rate / to_rate
    dst = array("h")

    for dst_idx in range(dst_len):
        src_pos = dst_idx * ratio
        left_idx = int(src_pos)
        right_idx = min(left_idx + 1, len(src) - 1)
        frac = src_pos - left_idx
        value = int(src[left_idx] * (1.0 - frac) + src[right_idx] * frac)
        dst.append(_clamp_pcm16(value))

    return _array_to_le_bytes(dst)


def rechunk_pcm16(
    data: bytes,
    sample_rate: int,
    frame_ms: int,
    *,
    channels: int = 1,
) -> list[bytes]:
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}")
    frame_bytes = int(sample_rate * frame_ms / 1000) * PCM_SAMPLE_WIDTH_BYTES * channels
    if frame_bytes <= 0:
        raise ValueError("frame size must be positive")

    chunks: list[bytes] = []
    for offset in range(0, len(data), frame_bytes):
        chunk = data[offset : offset + frame_bytes]
        if len(chunk) == frame_bytes:
            chunks.append(chunk)
    return chunks


def normalize_audio(data: bytes, *, from_rate: int, to_rate: int, channels: int) -> bytes:
    mono = ensure_mono_pcm16(data, channels)
    return resample_pcm16_mono(mono, from_rate, to_rate)


def pcm16_to_float32(data: bytes) -> bytes:
    """Convert little-endian signed PCM16 bytes to little-endian float32 PCM."""
    samples = _pcm16_array(data)
    floats = array("f", (sample / 32768.0 for sample in samples))
    if _is_big_endian():
        floats.byteswap()
    return floats.tobytes()


def float32_to_pcm16(data: bytes) -> bytes:
    """Convert little-endian float32 PCM bytes to clipped little-endian PCM16."""
    usable = len(data) - (len(data) % 4)
    samples = array("f")
    samples.frombytes(data[:usable])
    if _is_big_endian():
        samples.byteswap()

    pcm = array("h")
    for sample in samples:
        if not isfinite(sample):
            sample = 0.0
        value = int(round(sample * 32768.0))
        pcm.append(_clamp_pcm16(value))
    return _array_to_le_bytes(pcm)


def _pcm16_array(data: bytes) -> array:
    if len(data) % PCM_SAMPLE_WIDTH_BYTES:
        data = data[:-1]
    samples = array("h")
    samples.frombytes(data)
    if samples.itemsize != PCM_SAMPLE_WIDTH_BYTES:
        raise RuntimeError("platform does not expose 16-bit signed shorts")
    if _is_big_endian():
        samples.byteswap()
    return samples


def _array_to_le_bytes(samples: array) -> bytes:
    out = array("h", samples)
    if _is_big_endian():
        out.byteswap()
    return out.tobytes()


def _is_big_endian() -> bool:
    return sys.byteorder == "big"


def _clamp_pcm16(value: int) -> int:
    return max(-32768, min(32767, value))
