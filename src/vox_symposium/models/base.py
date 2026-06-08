from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from vox_symposium.audio import PcmAudio


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
