from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TypeVar

from vox_symposium.audio import PcmAudio

T = TypeVar("T")


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

    @abstractmethod
    def receive_audio(self) -> AsyncIterator[PcmAudio]:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError

    async def receive_text(self) -> AsyncIterator[str]:
        if False:
            yield ""


class QueueBackedRealtimeAudioModel(RealtimeAudioModel):
    def __init__(self, *, output_queue_size: int = 100) -> None:
        self._audio_out: asyncio.Queue[PcmAudio | None] = asyncio.Queue(maxsize=output_queue_size)
        self._text_out: asyncio.Queue[str | None] = asyncio.Queue(maxsize=output_queue_size)

    def receive_audio(self) -> AsyncIterator[PcmAudio]:
        return _receive_until_closed(self._audio_out)

    def receive_text(self) -> AsyncIterator[str]:
        return _receive_until_closed(self._text_out)

    def close_output_streams(self) -> None:
        _close_queue(self._audio_out)
        _close_queue(self._text_out)


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
