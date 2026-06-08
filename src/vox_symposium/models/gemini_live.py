from __future__ import annotations

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
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.instructions = instructions
        self._client = genai.Client(api_key=api_key)
        self._session_cm = None
        self._session = None

    async def connect(self) -> None:
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": self.instructions,
        }
        self._session_cm = self._client.aio.live.connect(model=self.model, config=config)
        self._session = await self._session_cm.__aenter__()

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

    async def receive_audio(self) -> AsyncIterator[PcmAudio]:
        if self._session is None:
            raise RuntimeError("Gemini Live session is not connected")
        async for response in self._session.receive():
            content = response.server_content
            if not content or not content.model_turn:
                continue
            for part in content.model_turn.parts:
                if part.inline_data and part.inline_data.data:
                    yield PcmAudio(
                        data=part.inline_data.data,
                        sample_rate=self.output_sample_rate,
                        channels=1,
                    )

    async def close(self) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
