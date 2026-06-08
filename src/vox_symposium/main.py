from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging

from vox_symposium.config import load_settings


async def run() -> None:
    parser = argparse.ArgumentParser(description="Run programmable LiveKit realtime audio participants.")
    parser.add_argument(
        "--participant",
        choices=["agent-citizen", "agent-scholar", "both"],
        default="both",
        help="Run one participant or both in this process.",
    )
    args = parser.parse_args()

    settings = load_settings()
    from vox_symposium.livekit_participant import ProgrammableParticipant

    participants: list[ProgrammableParticipant] = []

    if args.participant in {"agent-citizen", "both"}:
        participants.append(
            ProgrammableParticipant(
                settings=settings,
                agent=settings.agent_citizen,
                remote_identity=settings.agent_scholar.identity,
            )
        )
    if args.participant in {"agent-scholar", "both"}:
        participants.append(
            ProgrammableParticipant(
                settings=settings,
                agent=settings.agent_scholar,
                remote_identity=settings.agent_citizen.identity,
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


if __name__ == "__main__":
    main()
