from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from urllib.parse import urlencode, urlsplit, urlunsplit

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
        backend: str = "openai",
        endpoint: str | None = None,
        api_version: str | None = None,
        manual_activity: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.backend = backend
        self.endpoint = endpoint
        self.api_version = api_version
        self.manual_activity = manual_activity
        self._ws: ClientConnection | None = None
        self._audio_out: asyncio.Queue[PcmAudio | None] = asyncio.Queue(maxsize=100)
        self._text_out: asyncio.Queue[str | None] = asyncio.Queue(maxsize=100)
        self._reader_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        url, headers = self._connection_config()
        self._ws = await connect(
            url,
            additional_headers=headers,
            max_size=None,
        )
        session = {
            "type": "realtime",
            "instructions": self.instructions,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.input_sample_rate,
                    },
                    "turn_detection": (
                        None
                        if self.manual_activity
                        else {
                            "type": "semantic_vad",
                        }
                    ),
                },
                "output": {
                    "voice": self.voice,
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.output_sample_rate,
                    },
                }
            },
        }
        # Azure selects the deployment in the URL and doesn't accept a deployment
        # alias as session.model. The official endpoint keeps the existing behavior.
        if self.backend == "openai":
            session["model"] = self.model
        await self._send(
            {
                "type": "session.update",
                "session": session,
            }
        )
        self._reader_task = asyncio.create_task(self._read_loop(), name=f"openai-{self.model}-reader")

    async def end_audio_turn(self) -> None:
        if not self.manual_activity:
            return
        await self._send({"type": "input_audio_buffer.commit"})
        await self._send({"type": "response.create"})

    def _connection_config(self) -> tuple[str, dict[str, str]]:
        if self.backend == "openai":
            query = urlencode({"model": self.model})
            return (
                f"wss://api.openai.com/v1/realtime?{query}",
                {"Authorization": f"Bearer {self.api_key}"},
            )
        if self.backend != "azure":
            raise RuntimeError(f"Unsupported OpenAI backend: {self.backend}")
        if not self.endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is required for the Azure OpenAI backend")

        parsed = urlsplit(self.endpoint.rstrip("/"))
        if parsed.scheme not in {"https", "wss"} or not parsed.netloc:
            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT must be an HTTPS URL, for example "
                "https://your-resource.openai.azure.com"
            )
        if parsed.query or parsed.fragment:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT must not contain a query string or fragment")
        scheme = "wss"
        base_path = parsed.path.rstrip("/")
        if self.api_version:
            path = f"{base_path}/openai/realtime"
            query = urlencode({"api-version": self.api_version, "deployment": self.model})
        else:
            path = f"{base_path}/openai/v1/realtime"
            query = urlencode({"model": self.model})
        url = urlunsplit((scheme, parsed.netloc, path, query, ""))
        return url, {"api-key": self.api_key}

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
