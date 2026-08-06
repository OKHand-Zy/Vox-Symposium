from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, TypeVar

from vox_symposium.audio import PcmAudio

T = TypeVar("T")


@dataclass(frozen=True)
class RealtimeInterruption:
    """Provider-native signal that an in-progress model response was interrupted."""

    item_id: str | None = None
    response_id: str | None = None
    reason: str = "barge_in"


class RealtimeAudioModel(ABC):
    input_sample_rate: int
    output_sample_rate: int

    @abstractmethod
    async def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def send_audio(self, audio: PcmAudio) -> None:
        raise NotImplementedError

    async def start_audio_turn(self) -> None:
        return None

    async def end_audio_turn(self) -> None:
        return None

    async def flush_input_stream(self) -> None:
        """Flush one finite prerecorded input segment without closing the session."""
        await self.end_audio_turn()

    @abstractmethod
    def receive_audio(self) -> AsyncIterator[PcmAudio]:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError

    async def receive_text(self) -> AsyncIterator[str]:
        if False:
            yield ""

    async def receive_events(self) -> AsyncIterator[RealtimeInterruption]:
        if False:
            yield RealtimeInterruption()

    async def truncate_response(
        self,
        event: RealtimeInterruption,
        *,
        audio_end_ms: int,
    ) -> None:
        """Optionally truncate provider conversation state after an interruption."""
        del event, audio_end_ms


class QueueBackedRealtimeAudioModel(RealtimeAudioModel):
    def __init__(self, *, output_queue_size: int = 100) -> None:
        self._audio_out: asyncio.Queue[PcmAudio | None] = asyncio.Queue(maxsize=output_queue_size)
        self._text_out: asyncio.Queue[str | None] = asyncio.Queue(maxsize=output_queue_size)
        self._event_out: asyncio.Queue[RealtimeInterruption | None] = asyncio.Queue(
            maxsize=output_queue_size
        )
        self._output_streams_closed = False

    def receive_audio(self) -> AsyncIterator[PcmAudio]:
        return _receive_until_closed(self._audio_out)

    def receive_text(self) -> AsyncIterator[str]:
        return _receive_until_closed(self._text_out)

    def receive_events(self) -> AsyncIterator[RealtimeInterruption]:
        return _receive_until_closed(self._event_out)

    async def _emit_event(self, event: RealtimeInterruption) -> None:
        await self._event_out.put(event)

    def close_output_streams(self) -> None:
        if self._output_streams_closed:
            return
        self._output_streams_closed = True
        _close_queue(self._audio_out)
        _close_queue(self._text_out)
        _close_queue(self._event_out)


async def cancel_task(task: asyncio.Task[Any] | None) -> None:
    """Cancel a background task and consume its terminal result."""
    if task is None or task is asyncio.current_task():
        return
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _receive_until_closed(queue: asyncio.Queue[T | None]) -> AsyncIterator[T]:
    while True:
        item = await queue.get()
        if item is None:
            return
        yield item


def _close_queue(queue: asyncio.Queue[T | None]) -> None:
    try:
        queue.put_nowait(None)
    except asyncio.QueueFull:
        queue.get_nowait()
        queue.put_nowait(None)
