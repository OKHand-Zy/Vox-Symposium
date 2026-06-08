from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from vox_symposium.audio import PcmAudio, normalize_audio
from vox_symposium.models.base import RealtimeAudioModel


class GeminiLiveModel(RealtimeAudioModel):
    input_sample_rate = 16_000
    output_sample_rate = 24_000

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        instructions: str,
        manual_activity: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.instructions = instructions
        self.manual_activity = manual_activity
        self._client = genai.Client(api_key=api_key)
        self._session_cm = None
        self._session = None
        self._audio_out: asyncio.Queue[PcmAudio | None] = asyncio.Queue(maxsize=100)
        self._text_out: asyncio.Queue[str | None] = asyncio.Queue(maxsize=100)
        self._reader_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": self.instructions,
            "output_audio_transcription": {},
        }
        if self.manual_activity:
            config["realtime_input_config"] = {
                "automatic_activity_detection": {
                    "disabled": True,
                },
                "activity_handling": "NO_INTERRUPTION",
                "turn_coverage": "TURN_INCLUDES_ALL_INPUT",
            }
        self._session_cm = self._client.aio.live.connect(model=self.model, config=config)
        self._session = await self._session_cm.__aenter__()
        self._reader_task = asyncio.create_task(self._read_loop(), name=f"gemini-{self.model}-reader")

    async def send_audio(self, audio: PcmAudio) -> None:
        if self._session is None:
            raise RuntimeError("Gemini Live session is not connected")
        pcm = normalize_audio(
            audio.data,
            from_rate=audio.sample_rate,
            to_rate=self.input_sample_rate,
            channels=audio.channels,
        )
        if not pcm:
            return
        await self._session.send_realtime_input(
            audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={self.input_sample_rate}")
        )

    async def start_audio_turn(self) -> None:
        if self._session is None:
            raise RuntimeError("Gemini Live session is not connected")
        if self.manual_activity:
            await self._session.send_realtime_input(activity_start=types.ActivityStart())

    async def end_audio_turn(self) -> None:
        if self._session is None:
            raise RuntimeError("Gemini Live session is not connected")
        if self.manual_activity:
            await self._session.send_realtime_input(activity_end=types.ActivityEnd())

    async def receive_audio(self) -> AsyncIterator[PcmAudio]:
        while True:
            item = await self._audio_out.get()
            if item is None:
                return
            yield item

    async def receive_text(self) -> AsyncIterator[str]:
        while True:
            item = await self._text_out.get()
            if item is None:
                return
            yield item

    async def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
        await self._audio_out.put(None)
        await self._text_out.put(None)

    async def _read_loop(self) -> None:
        if self._session is None:
            raise RuntimeError("Gemini Live session is not connected")
        try:
            while True:
                async for response in self._session.receive():
                    content = response.server_content
                    if not content:
                        continue
                    if content.output_transcription and content.output_transcription.text:
                        await self._text_out.put(content.output_transcription.text)
                    if not content.model_turn:
                        continue
                    for part in content.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            await self._audio_out.put(
                                PcmAudio(
                                    data=part.inline_data.data,
                                    sample_rate=self.output_sample_rate,
                                    channels=1,
                                )
                            )
        finally:
            await self._audio_out.put(None)
            await self._text_out.put(None)
