from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from google import genai
from google.genai import types
from google.oauth2 import service_account

from vox_symposium.audio import PcmAudio, normalize_audio
from vox_symposium.config import gemini_uses_thinking_level
from vox_symposium.models.base import QueueBackedRealtimeAudioModel

VERTEX_AI_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


class GeminiLiveModel(QueueBackedRealtimeAudioModel):
    input_sample_rate = 16_000
    output_sample_rate = 24_000

    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend: str = "ai_studio",
        vertex_project: str | None = None,
        vertex_location: str | None = None,
        credentials_file: str | None = None,
        model: str,
        instructions: str,
        thinking_level: str = "minimal",
        thinking_budget: int | None = None,
        enable_affective_dialog: bool = False,
        initial_history: Sequence[dict[str, Any]] = (),
        manual_activity: bool = False,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.instructions = instructions
        self.thinking_level = thinking_level
        self.thinking_budget = thinking_budget
        self.enable_affective_dialog = enable_affective_dialog
        self.initial_history = tuple(initial_history)
        self.manual_activity = manual_activity
        if self.enable_affective_dialog and gemini_uses_thinking_level(self.model):
            raise ValueError(
                "Affective dialog is supported only by Gemini 2.5 Flash Live, not Gemini 3 Live models"
            )
        if self.initial_history and not gemini_uses_thinking_level(self.model):
            raise ValueError(
                "Initial history via send_client_content is supported only by Gemini 3 Live models; "
                "Gemini 2.5 client content remains a normal turn-based input flow"
            )
        if backend == "vertex":
            if not vertex_project or not vertex_location or not credentials_file:
                raise ValueError(
                    "Vertex Gemini requires project, location, and a service account credentials file"
                )
            credentials = service_account.Credentials.from_service_account_file(
                credentials_file,
                scopes=VERTEX_AI_SCOPES,
            )
            self._client = genai.Client(
                vertexai=True,
                project=vertex_project,
                location=vertex_location,
                credentials=credentials,
                **self._client_http_options(),
            )
        elif backend == "ai_studio":
            if not api_key:
                raise ValueError("AI Studio Gemini requires an API key")
            self._client = genai.Client(api_key=api_key, **self._client_http_options())
        else:
            raise ValueError(f"Unsupported Gemini backend: {backend!r}")
        self._session_cm = None
        self._session = None
        self._reader_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        uses_thinking_level = gemini_uses_thinking_level(self.model)
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=self.instructions,
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )
        if uses_thinking_level:
            config.thinking_config = types.ThinkingConfig(thinking_level=self.thinking_level)
        elif self.thinking_budget is not None:
            config.thinking_config = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        if self.enable_affective_dialog:
            config.enable_affective_dialog = True
        if self.initial_history:
            config.history_config = types.HistoryConfig(initial_history_in_client_content=True)
        if self.manual_activity:
            config.realtime_input_config = types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
                activity_handling="NO_INTERRUPTION",
                **(
                    {"turn_coverage": "TURN_INCLUDES_ALL_INPUT"}
                    if not uses_thinking_level
                    else {}
                ),
            )
        self._session_cm = self._client.aio.live.connect(model=self.model, config=config)
        self._session = await self._session_cm.__aenter__()
        if self.initial_history:
            await self._session.send_client_content(
                turns=list(self.initial_history),
                turn_complete=True,
            )
        self._reader_task = asyncio.create_task(self._read_loop(), name=f"gemini-{self.model}-reader")

    def _client_http_options(self) -> dict[str, types.HttpOptions]:
        if not self.enable_affective_dialog:
            return {}
        return {"http_options": types.HttpOptions(api_version="v1alpha")}

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

    async def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
        self.close_output_streams()

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
            self.close_output_streams()
