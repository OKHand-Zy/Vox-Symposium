from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import cast

from vox_symposium.audio import PcmAudio
from vox_symposium.evaluation_audio import (
    collect_text_after_audio,
    collect_utterance,
    ensure_wav,
    send_audio,
)
from vox_symposium.models.base import RealtimeAudioModel


class EvaluationAudioTests(unittest.IsolatedAsyncioTestCase):
    async def test_ensure_wav_copies_wav_input_into_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            output = root / "artifacts" / "question.wav"
            source.write_bytes(b"wav data")

            self.assertEqual(ensure_wav(source, output), output)
            self.assertEqual(output.read_bytes(), b"wav data")

    async def test_collect_utterance_max_seconds_includes_initial_wait(self) -> None:
        queue: asyncio.Queue[PcmAudio | None] = asyncio.Queue()
        first = PcmAudio(b"\x01\x00", sample_rate=1_000)
        second = PcmAudio(b"\x02\x00", sample_rate=1_000)

        async def publish_audio() -> None:
            await asyncio.sleep(0.1)
            await queue.put(first)
            await asyncio.sleep(0.06)
            await queue.put(second)

        producer = asyncio.create_task(publish_audio())
        try:
            utterance = await collect_utterance(
                "robot",
                queue,
                idle_timeout=1.0,
                max_seconds=0.13,
            )
        finally:
            await producer

        self.assertEqual(utterance.audio.data, first.data)

    async def test_send_audio_preserves_input_channel_count_for_silence(self) -> None:
        class RecordingModel:
            def __init__(self) -> None:
                self.chunks: list[PcmAudio] = []
                self.flushed = False

            async def start_audio_turn(self) -> None:
                pass

            async def send_audio(self, audio: PcmAudio) -> None:
                self.chunks.append(audio)

            async def flush_input_stream(self) -> None:
                self.flushed = True

        model = RecordingModel()
        audio = PcmAudio(b"\x01\x00" * 200, sample_rate=1_000, channels=2)

        await send_audio(audio, cast(RealtimeAudioModel, model), frame_ms=100, audio_speed=0)

        self.assertTrue(model.flushed)
        self.assertEqual(len(model.chunks), 3)
        self.assertTrue(all(chunk.channels == 2 for chunk in model.chunks))
        self.assertTrue(all(len(chunk.data) == 400 for chunk in model.chunks))

    async def test_collect_text_waits_for_first_delayed_text_until_max_wait(self) -> None:
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def publish_late_text() -> None:
            await asyncio.sleep(0.05)
            await queue.put("late text")

        producer = asyncio.create_task(publish_late_text())
        try:
            text = await collect_text_after_audio(
                queue,
                idle_timeout=0.01,
                max_wait=0.2,
            )
        finally:
            await producer

        self.assertEqual(text, "late text")


if __name__ == "__main__":
    unittest.main()
