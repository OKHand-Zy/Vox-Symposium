from __future__ import annotations

from array import array
from dataclasses import dataclass


PCM_SAMPLE_WIDTH_BYTES = 2


@dataclass(frozen=True)
class PcmAudio:
    data: bytes
    sample_rate: int
    channels: int = 1


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
    if from_rate == to_rate or not data:
        return data
    if from_rate <= 0 or to_rate <= 0:
        raise ValueError("sample rates must be positive")

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


def rechunk_pcm16(data: bytes, sample_rate: int, frame_ms: int) -> list[bytes]:
    frame_bytes = int(sample_rate * frame_ms / 1000) * PCM_SAMPLE_WIDTH_BYTES
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


def _pcm16_array(data: bytes) -> array:
    if len(data) % PCM_SAMPLE_WIDTH_BYTES:
        data = data[:-1]
    samples = array("h")
    samples.frombytes(data)
    if samples.itemsize != PCM_SAMPLE_WIDTH_BYTES:
        raise RuntimeError("platform does not expose 16-bit signed shorts")
    if _is_big_endian(samples):
        samples.byteswap()
    return samples


def _array_to_le_bytes(samples: array) -> bytes:
    out = array("h", samples)
    if _is_big_endian(out):
        out.byteswap()
    return out.tobytes()


def _is_big_endian(samples: array) -> bool:
    probe = array(samples.typecode, [1])
    return probe.tobytes() == b"\x00\x01"


def _clamp_pcm16(value: int) -> int:
    return max(-32768, min(32767, value))
