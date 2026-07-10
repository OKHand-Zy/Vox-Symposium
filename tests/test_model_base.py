from __future__ import annotations

import asyncio
import unittest

from vox_symposium.models.base import cancel_task


class ModelBaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_task_waits_for_task_cleanup(self) -> None:
        cleaned_up = asyncio.Event()

        async def worker() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                cleaned_up.set()

        task = asyncio.create_task(worker())
        await asyncio.sleep(0)

        await cancel_task(task)

        self.assertTrue(task.cancelled())
        self.assertTrue(cleaned_up.is_set())


if __name__ == "__main__":
    unittest.main()
