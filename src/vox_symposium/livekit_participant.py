from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from array import array
from dataclasses import dataclass
from pathlib import Path

from livekit import api, rtc

from vox_symposium.audio import PcmAudio, normalize_audio, rechunk_pcm16
from vox_symposium.config import AgentConfig, Settings
from vox_symposium.models.base import RealtimeAudioModel
from vox_symposium.models.gemini_live import GeminiLiveModel
from vox_symposium.models.openai_realtime import OpenAIRealtimeModel
from vox_symposium.recording import ConversationRecorder, audio_event_fields, write_wav


logger = logging.getLogger(__name__)
RECORDING_TURN_IDLE_SECONDS = 1.0


@dataclass(frozen=True)
class PeerRoute:
    local_identity: str
    remote_identity: str


class ProgrammableParticipant:
    def __init__(
        self,
        *,
        settings: Settings,
        agent: AgentConfig,
        remote_identity: str,
        recorder: ConversationRecorder | None = None,
        recording_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self.agent = agent
        self.route = PeerRoute(agent.identity, remote_identity)
        self.room = rtc.Room()
        self.model = build_model(settings, agent)
        self.source = rtc.AudioSource(settings.publish_sample_rate, 1)
        self.recorder = recorder
        self.recording_dir = recording_dir
        self._recording_turn_index = 0
        self._recording_text_parts: list[str] = []
        self._tasks: set[asyncio.Task] = set()
        self._closed = asyncio.Event()

    async def run(self) -> None:
        await self.model.connect()
        logger.info("%s connected to %s model", self.agent.identity, self.agent.provider)
        token = self._build_token()
        self.room.on("track_subscribed", self._on_track_subscribed)
        await self.room.connect(self.settings.livekit_url, token)
        logger.info("%s joined LiveKit room %s", self.agent.identity, self.settings.livekit_room)
        await self._publish_model_track()
        logger.info("%s published model audio track", self.agent.identity)
        self._attach_existing_remote_tracks()

        self._tasks.add(asyncio.create_task(self._publish_model_audio(), name=f"{self.agent.identity}-publisher"))
        self._tasks.add(asyncio.create_task(self._consume_model_text(), name=f"{self.agent.identity}-text"))
        await self._closed.wait()

    async def close(self) -> None:
        self._closed.set()
        for task in self._tasks:
            task.cancel()
        await self.model.close()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._flush_recorded_text_only()
        with contextlib.suppress(Exception):
            await self.room.disconnect()

    def _build_token(self) -> str:
        grant = api.VideoGrants(
            room_join=True,
            room=self.settings.livekit_room,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        )
        return (
            api.AccessToken(self.settings.livekit_api_key, self.settings.livekit_api_secret)
            .with_identity(self.agent.identity)
            .with_name(self.agent.identity)
            .with_grants(grant)
            .to_jwt()
        )

    async def _publish_model_track(self) -> None:
        track = rtc.LocalAudioTrack.create_audio_track(f"{self.agent.identity}-model-audio", self.source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self.room.local_participant.publish_track(track, options)

    def _on_track_subscribed(self, track, publication, participant) -> None:
        if participant.identity != self.route.remote_identity:
            return
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        logger.info("%s subscribed to %s audio track", self.agent.identity, participant.identity)
        task = asyncio.create_task(
            self._forward_livekit_audio_to_model(track),
            name=f"{self.agent.identity}-from-{participant.identity}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _attach_existing_remote_tracks(self) -> None:
        for participant in self.room.remote_participants.values():
            if participant.identity != self.route.remote_identity:
                continue
            for publication in participant.track_publications.values():
                track = publication.track
                if track is not None:
                    self._on_track_subscribed(track, publication, participant)

    async def _forward_livekit_audio_to_model(self, track) -> None:
        stream = rtc.AudioStream.from_track(
            track=track,
            sample_rate=self.model.input_sample_rate,
            num_channels=1,
            frame_size_ms=self.settings.frame_ms,
        )
        frame_count = 0
        byte_count = 0
        last_log_at = time.monotonic()
        try:
            async for event in stream:
                frame = event.frame
                frame_count += 1
                byte_count += len(bytes(frame.data))
                await self.model.send_audio(
                    PcmAudio(
                        data=bytes(frame.data),
                        sample_rate=frame.sample_rate,
                        channels=frame.num_channels,
                    )
                )
                now = time.monotonic()
                if now - last_log_at >= 5:
                    logger.info(
                        "%s forwarded LiveKit audio from %s to model: %s frames, %s bytes",
                        self.agent.identity,
                        self.route.remote_identity,
                        frame_count,
                        byte_count,
                    )
                    last_log_at = now
        finally:
            await stream.aclose()

    async def _publish_model_audio(self) -> None:
        chunk_count = 0
        byte_count = 0
        last_log_at = time.monotonic()
        turn_chunks: list[PcmAudio] = []
        audio_iter = self.model.receive_audio().__aiter__()
        pending_audio = asyncio.create_task(anext(audio_iter))
        try:
            while True:
                recording_enabled = self.recorder is not None and self.recording_dir is not None
                timeout = RECORDING_TURN_IDLE_SECONDS if recording_enabled and turn_chunks else None
                done, _ = await asyncio.wait({pending_audio}, timeout=timeout)
                if not done:
                    self._save_recorded_audio_turn(turn_chunks)
                    turn_chunks = []
                    continue

                try:
                    audio = pending_audio.result()
                except StopAsyncIteration:
                    break
                pending_audio = asyncio.create_task(anext(audio_iter))

                chunk_count += 1
                byte_count += len(audio.data)
                if recording_enabled:
                    turn_chunks.append(audio)
                await self._publish_audio_chunk(audio)
                now = time.monotonic()
                if now - last_log_at >= 5:
                    logger.info(
                        "%s published model audio to LiveKit: %s chunks, %s bytes",
                        self.agent.identity,
                        chunk_count,
                        byte_count,
                    )
                    last_log_at = now
        finally:
            if not pending_audio.done():
                pending_audio.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pending_audio
            self._save_recorded_audio_turn(turn_chunks)

    async def _publish_audio_chunk(self, audio: PcmAudio) -> None:
        pcm48 = normalize_audio(
            audio.data,
            from_rate=audio.sample_rate,
            to_rate=self.settings.publish_sample_rate,
            channels=audio.channels,
        )
        for chunk in rechunk_pcm16(pcm48, self.settings.publish_sample_rate, self.settings.frame_ms):
            frame = rtc.AudioFrame.create(
                self.settings.publish_sample_rate,
                1,
                int(self.settings.publish_sample_rate * self.settings.frame_ms / 1000),
            )
            try:
                frame.data[:] = chunk
            except TypeError:
                samples = array("h")
                samples.frombytes(chunk)
                frame.data[:] = samples
            await self.source.capture_frame(frame)

    async def _consume_model_text(self) -> None:
        async for text in self.model.receive_text():
            if not text:
                continue
            self._recording_text_parts.append(text)

    def _save_recorded_audio_turn(self, chunks: list[PcmAudio]) -> None:
        if not chunks or self.recorder is None or self.recording_dir is None:
            return

        first = chunks[0]
        data = b"".join(chunk.data for chunk in chunks)
        audio = PcmAudio(data=data, sample_rate=first.sample_rate, channels=first.channels)
        self._recording_turn_index += 1
        path = self.recording_dir / f"{self.agent.identity}-{self._recording_turn_index:04d}.wav"
        recording = write_wav(path, audio)
        text = self._take_recorded_text()
        self.recorder.append(
            {
                "type": "model_output_turn",
                "agent": self.agent.identity,
                "text": text,
                **audio_event_fields(recording),
            }
        )

    def _flush_recorded_text_only(self) -> None:
        if self.recorder is None:
            self._recording_text_parts.clear()
            return
        text = self._take_recorded_text()
        if not text:
            return
        self.recorder.append(
            {
                "type": "model_output_text",
                "agent": self.agent.identity,
                "text": text,
                "audio": None,
            }
        )

    def _take_recorded_text(self) -> str:
        text = "".join(self._recording_text_parts).strip()
        self._recording_text_parts.clear()
        return text


def build_model(settings: Settings, agent: AgentConfig) -> RealtimeAudioModel:
    provider = agent.provider.lower()
    if provider == "openai":
        if settings.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY is required when a participant uses provider=openai")
        return OpenAIRealtimeModel(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            voice=settings.openai_voice,
            instructions=agent.instructions,
        )
    if provider == "gemini":
        return GeminiLiveModel(
            api_key=settings.gemini_api_key,
            backend=settings.gemini_backend,
            vertex_project=settings.gemini_vertex_project,
            vertex_location=settings.gemini_vertex_location,
            credentials_file=settings.gemini_credentials_file,
            model=settings.gemini_model,
            instructions=agent.instructions,
        )
    raise RuntimeError(f"Unsupported provider for {agent.identity}: {agent.provider}")
