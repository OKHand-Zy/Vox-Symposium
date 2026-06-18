from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from vox_symposium.config import load_settings
from vox_symposium.recording import ConversationRecorder


async def run() -> None:
    parser = argparse.ArgumentParser(description="Run programmable LiveKit realtime audio participants.")
    parser.add_argument(
        "--participant",
        choices=["agent-citizen", "agent-scholar", "both"],
        default="both",
        help="Run one participant or both in this process.",
    )
    parser.add_argument(
        "--record-dir",
        default=os.getenv("VOX_RECORD_DIR", "data/recordings"),
        help="Directory where conversation audio files and conversation-log.json are saved.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable saving model output audio and transcript text.",
    )
    args = parser.parse_args()

    settings = load_settings()
    from vox_symposium.livekit_participant import ProgrammableParticipant

    participants: list[ProgrammableParticipant] = []
    recorder: ConversationRecorder | None = None
    recording_dir: Path | None = None
    if not args.no_record:
        run_id = _recording_run_id(settings.livekit_room, args.participant)
        recording_dir = Path(args.record_dir) / run_id
        recorder = ConversationRecorder(
            recording_dir / "conversation-log.json",
            run_id=run_id,
            metadata={
                "livekit_room": settings.livekit_room,
                "participant": args.participant,
                "agents": {
                    "citizen": settings.agent_citizen.identity,
                    "scholar": settings.agent_scholar.identity,
                },
            },
        )
        recorder.write()
        logging.getLogger(__name__).info("Saving conversation recording to %s", recording_dir)

    if args.participant in {"agent-citizen", "both"}:
        participants.append(
            ProgrammableParticipant(
                settings=settings,
                agent=settings.agent_citizen,
                remote_identity=settings.agent_scholar.identity,
                recorder=recorder,
                recording_dir=recording_dir,
            )
        )
    if args.participant in {"agent-scholar", "both"}:
        participants.append(
            ProgrammableParticipant(
                settings=settings,
                agent=settings.agent_scholar,
                remote_identity=settings.agent_citizen.identity,
                recorder=recorder,
                recording_dir=recording_dir,
            )
        )

    tasks = [asyncio.create_task(participant.run(), name=participant.agent.identity) for participant in participants]
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        pass
    finally:
        for participant in participants:
            await participant.close()
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run())


def _recording_run_id(room: str, participant: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_room = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in room)
    return f"{stamp}-{safe_room}-{participant}"


if __name__ == "__main__":
    main()
