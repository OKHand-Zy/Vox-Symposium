from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from array import array
from dataclasses import dataclass

from livekit import api, rtc

from vox_symposium.audio import PcmAudio, normalize_audio, rechunk_pcm16
from vox_symposium.config import AgentConfig, Settings
from vox_symposium.models.base import RealtimeAudioModel
from vox_symposium.models.gemini_live import GeminiLiveModel
from vox_symposium.models.local_voice import LocalVoiceConfig, LocalVoiceModel
from vox_symposium.models.openai_realtime import OpenAIRealtimeModel


logger = logging.getLogger(__name__)


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
    ) -> None:
        self.settings = settings
        self.agent = agent
        self.route = PeerRoute(agent.identity, remote_identity)
        self.room = rtc.Room()
        self.model = build_model(settings, agent)
        self.source = rtc.AudioSource(settings.publish_sample_rate, 1)
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
        await self._closed.wait()

    async def close(self) -> None:
        self._closed.set()
        for task in self._tasks:
            task.cancel()
        await self.model.close()
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
        async for audio in self.model.receive_audio():
            chunk_count += 1
            byte_count += len(audio.data)
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
            now = time.monotonic()
            if now - last_log_at >= 5:
                logger.info(
                    "%s published model audio to LiveKit: %s chunks, %s bytes",
                    self.agent.identity,
                    chunk_count,
                    byte_count,
                )
                last_log_at = now


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
        if settings.gemini_api_key is None:
            raise RuntimeError("GEMINI_API_KEY is required when a participant uses provider=gemini")
        return GeminiLiveModel(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            instructions=agent.instructions,
        )
    if provider in {"local", "hf"}:
        return LocalVoiceModel(
            LocalVoiceConfig(
                provider=settings.local_model_provider,
                model=settings.local_model,
                instructions=agent.instructions,
                device=settings.local_model_device,
                dtype=settings.local_model_dtype,
                attn_implementation=settings.local_model_attn_implementation,
                ref_audio_path=settings.local_model_ref_audio,
                language=settings.local_model_language,
                turn_detection=settings.local_model_turn_detection,
                silero_threshold=settings.local_model_silero_threshold,
                turn_silence_ms=settings.local_model_turn_silence_ms,
                min_turn_ms=settings.local_model_min_turn_ms,
                chunk_ms=settings.local_model_chunk_ms,
                max_new_tokens=settings.local_model_max_new_tokens,
                qwen_speaker=settings.local_model_qwen_speaker,
            )
        )
    raise RuntimeError(f"Unsupported provider for {agent.identity}: {agent.provider}")
