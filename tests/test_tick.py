from __future__ import annotations

import asyncio
import unittest

from vox_symposium.audio import PcmAudio
from vox_symposium.evaluation import _consume_provider_interruptions
from vox_symposium.models.base import RealtimeInterruption
from vox_symposium.scenario import AGENT_KEYS
from vox_symposium.tick import TickAudioBuffer, TickResult, tick_result_fields


class TickAudioBufferTests(unittest.TestCase):
    def test_pop_tick_carries_excess_audio_to_the_next_tick(self) -> None:
        buffer = TickAudioBuffer(sample_rate=1000)
        audio = PcmAudio(data=b"\x01\x00" * 300, sample_rate=1000)
        buffer.append(audio)

        first, captured, truncated = buffer.pop_tick(200)

        self.assertEqual(first.duration_seconds, 0.2)
        self.assertEqual(captured.data, b"\x01\x00" * 200)
        self.assertTrue(truncated)
        self.assertEqual(buffer.pending_duration_ms, 100)

        second, second_captured, second_truncated = buffer.pop_tick(200)

        self.assertEqual(second_captured.data, b"\x01\x00" * 100)
        self.assertEqual(len(second.data), 400)
        self.assertEqual(second.data[:200], b"\x01\x00" * 100)
        self.assertEqual(second.data[200:], b"\x00" * 200)
        self.assertFalse(second_truncated)
        self.assertEqual(buffer.pending_bytes, 0)

    def test_clear_discards_pending_output_after_interrupt(self) -> None:
        buffer = TickAudioBuffer(sample_rate=1000)
        buffer.append(PcmAudio(data=b"\x02\x00" * 300, sample_rate=1000))
        buffer.pop_tick(100)

        buffer.clear()
        fixed, captured, truncated = buffer.pop_tick(100)

        self.assertEqual(captured.data, b"")
        self.assertEqual(fixed.data, b"\x00" * 200)
        self.assertFalse(truncated)
        self.assertEqual(buffer.pending_bytes, 0)

    def test_rejects_tick_duration_that_is_not_a_whole_number_of_samples(self) -> None:
        buffer = TickAudioBuffer(sample_rate=11025)

        with self.assertRaisesRegex(ValueError, "whole number of samples"):
            buffer.pop_tick(1)


class TickResultTests(unittest.TestCase):
    def test_tick_result_fields_are_json_friendly(self) -> None:
        result = TickResult(
            tick_number=3,
            tick_duration_ms=200,
            audio={},
            captured_audio={},
            truncated={"human": False, "robot": True},
            interrupted_agents=("robot",),
        )

        self.assertEqual(result.simulation_time_ms, 600)
        self.assertEqual(
            tick_result_fields(result),
            {
                "tick_number": 3,
                "simulation_time_ms": 600,
                "tick_duration_ms": 200,
                "truncated": {"human": False, "robot": True},
                "interrupted_agents": ["robot"],
            },
        )


class ProviderInterruptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_interruption_clears_tick_and_audio_queues(self) -> None:
        class RecordingModel:
            def __init__(self) -> None:
                self.truncations: list[tuple[RealtimeInterruption, int]] = []

            async def truncate_response(
                self,
                event: RealtimeInterruption,
                *,
                audio_end_ms: int,
            ) -> None:
                self.truncations.append((event, audio_end_ms))

        models = {agent: RecordingModel() for agent in AGENT_KEYS}
        audio_queues = {agent: asyncio.Queue() for agent in AGENT_KEYS}
        event_queues = {agent: asyncio.Queue() for agent in AGENT_KEYS}
        buffers = {agent: TickAudioBuffer(sample_rate=1000) for agent in AGENT_KEYS}
        buffers["robot"].append(PcmAudio(data=b"\x01\x00" * 300, sample_rate=1000))
        buffers["robot"].pop_tick(100)
        audio_queues["robot"].put_nowait(PcmAudio(data=b"\x02\x00", sample_rate=1000))
        event = RealtimeInterruption(item_id="item-1")
        event_queues["robot"].put_nowait(event)
        interrupted_turn = {agent: False for agent in AGENT_KEYS}
        forwarded_audio_ms = {"human": 0.0, "robot": 125.5}
        dialogue_log = {"events": []}

        interrupted = await _consume_provider_interruptions(
            models=models,
            audio_queues=audio_queues,
            event_queues=event_queues,
            buffers=buffers,
            interrupted_turn=interrupted_turn,
            forwarded_audio_ms=forwarded_audio_ms,
            dialogue_log=dialogue_log,
            tick_number=4,
        )

        self.assertEqual(interrupted, {"robot"})
        self.assertEqual(buffers["robot"].pending_bytes, 0)
        self.assertTrue(audio_queues["robot"].empty())
        self.assertTrue(interrupted_turn["robot"])
        self.assertEqual(models["robot"].truncations, [(event, 126)])
        self.assertEqual(dialogue_log["events"][0]["type"], "provider_interruption")


if __name__ == "__main__":
    unittest.main()
