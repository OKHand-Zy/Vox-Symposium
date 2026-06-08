from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator

from websockets.asyncio.client import ClientConnection, connect

from vox_symposium.audio import PcmAudio, normalize_audio
from vox_symposium.models.base import RealtimeAudioModel


class OpenAIRealtimeModel(RealtimeAudioModel):
    input_sample_rate = 24_000
    output_sample_rate = 24_000

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        voice: str,
        instructions: str,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self._ws: ClientConnection | None = None
        self._audio_out: asyncio.Queue[PcmAudio | None] = asyncio.Queue(maxsize=100)
        self._text_out: asyncio.Queue[str | None] = asyncio.Queue(maxsize=100)
        self._reader_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        self._ws = await connect(
            url,
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            max_size=None,
        )
        await self._send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": self.model,
                    "instructions": self.instructions,
                    "output_modalities": ["audio"],
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": self.input_sample_rate,
                            },
                            "turn_detection": {
                                "type": "semantic_vad",
                            },
                        },
                        "output": {
                            "voice": self.voice,
                            "format": {
                                "type": "audio/pcm",
                                "rate": self.output_sample_rate,
                            },
                        }
                    },
                },
            }
        )
        self._reader_task = asyncio.create_task(self._read_loop(), name=f"openai-{self.model}-reader")

    async def send_audio(self, audio: PcmAudio) -> None:
        pcm = normalize_audio(
            audio.data,
            from_rate=audio.sample_rate,
            to_rate=self.input_sample_rate,
            channels=audio.channels,
        )
        if not pcm:
            return
        await self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )

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
        if self._ws:
            await self._ws.close()
        await self._audio_out.put(None)
        await self._text_out.put(None)

    async def _send(self, event: dict) -> None:
        if self._ws is None:
            raise RuntimeError("OpenAI realtime websocket is not connected")
        await self._ws.send(json.dumps(event))

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                event = json.loads(raw)
                event_type = event.get("type")
                if event_type in {"response.output_audio.delta", "response.audio.delta"}:
                    delta = event.get("delta")
                    if delta:
                        await self._audio_out.put(
                            PcmAudio(
                                data=base64.b64decode(delta),
                                sample_rate=self.output_sample_rate,
                                channels=1,
                            )
                        )
                elif event_type in {
                    "response.audio_transcript.delta",
                    "response.output_audio_transcript.delta",
                    "response.text.delta",
                    "response.output_text.delta",
                }:
                    delta = event.get("delta")
                    if delta:
                        await self._text_out.put(delta)
                elif event_type in {
                    "response.audio_transcript.done",
                    "response.output_audio_transcript.done",
                    "response.text.done",
                    "response.output_text.done",
                }:
                    text = event.get("transcript") or event.get("text")
                    if text:
                        await self._text_out.put(text)
                elif event_type == "error":
                    raise RuntimeError(f"OpenAI realtime error: {event}")
        finally:
            await self._audio_out.put(None)
            await self._text_out.put(None)
