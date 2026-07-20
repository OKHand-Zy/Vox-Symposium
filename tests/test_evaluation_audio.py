from __future__ import annotations

import asyncio
import unittest

from vox_symposium.evaluation_audio import collect_text_after_audio


class EvaluationAudioTests(unittest.IsolatedAsyncioTestCase):
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
