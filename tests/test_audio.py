from __future__ import annotations

import struct
import unittest

from vox_symposium.audio import (
    PcmAudio,
    concatenate_pcm_audio,
    ensure_mono_pcm16,
    rechunk_pcm16,
)


class AudioTests(unittest.TestCase):
    def test_ensure_mono_pcm16_averages_channels(self) -> None:
        stereo = struct.pack("<hhhh", 1000, 3000, -1000, 1000)

        self.assertEqual(ensure_mono_pcm16(stereo, channels=2), struct.pack("<hh", 2000, 0))

    def test_rechunk_pcm16_drops_partial_frame(self) -> None:
        sample_rate = 1_000
        frame_ms = 20
        frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        data = b"\x01" * (frame_bytes * 2 + 1)

        self.assertEqual(rechunk_pcm16(data, sample_rate, frame_ms), [b"\x01" * frame_bytes] * 2)

    def test_rechunk_pcm16_preserves_stereo_frame_duration(self) -> None:
        sample_rate = 1_000
        frame_ms = 20
        stereo_frame_bytes = int(sample_rate * frame_ms / 1000) * 2 * 2
        data = b"\x01" * (stereo_frame_bytes * 2)

        self.assertEqual(
            rechunk_pcm16(data, sample_rate, frame_ms, channels=2),
            [b"\x01" * stereo_frame_bytes] * 2,
        )

    def test_concatenate_pcm_audio_rejects_format_changes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same sample rate"):
            concatenate_pcm_audio(
                [
                    PcmAudio(b"\x00\x00", sample_rate=16_000),
                    PcmAudio(b"\x00\x00", sample_rate=24_000),
                ]
            )


if __name__ == "__main__":
    unittest.main()
