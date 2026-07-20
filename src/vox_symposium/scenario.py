from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from vox_symposium.json_io import read_json, write_json
from vox_symposium.providers import normalize_provider

AgentKey = Literal["citizen", "scholar"]

DIALOGUE_BEHAVIOR = (
    "Stay in character and respond with the speaking style, perspective, emotions, and reasoning that fit "
    "your assigned role. Use the conversation history as context, and continue the same conversation while "
    "preserving details from earlier turns.\n\n"
    "When the other speaker shows interest, agreement, or reduced hesitation, continue by exploring "
    "practical next steps, preferences, concerns, constraints, trade-offs, examples, or conditions for "
    "trying the recommendation.\n\n"
    "Avoid closing the conversation or shifting into farewell-style responses. Each response should leave a "
    "natural opening for the other speaker to continue, grounded in the existing conversation."
)

SHORT_REPLY_DIALOGUE_BEHAVIOR = (
    "Keep each reply under 2 sentences. Ask at most one question. Do not summarize repeatedly."
)
# Provider-specific aliases keep the generated prompt and documentation explicit while
# sharing the same conservative turn-taking policy.
MINICPM_DIALOGUE_BEHAVIOR = SHORT_REPLY_DIALOGUE_BEHAVIOR
FREEZE_OMNI_DIALOGUE_BEHAVIOR = SHORT_REPLY_DIALOGUE_BEHAVIOR

_SOURCE_ROLE_TO_AGENT: dict[str, AgentKey] = {
    "human": "citizen",
    "user": "citizen",
    "gpt": "scholar",
    "assistant": "scholar",
}


@dataclass(frozen=True)
class AgentPrompt:
    instructions: str
    initial_history: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class LoadedScenario:
    data: dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.data["id"])

    def build_instructions(
        self,
        agent: AgentKey,
        *,
        dialogue_behavior_extra: str | None = None,
        include_history: bool = True,
    ) -> str:
        return build_agent_instructions(
            self.data,
            agent,
            dialogue_behavior_extra=dialogue_behavior_extra,
            include_history=include_history,
        )

    def build_initial_history(self, agent: AgentKey) -> tuple[dict[str, Any], ...]:
        return build_agent_initial_history(self.data, agent)

    def build_prompt(
        self,
        agent: AgentKey,
        *,
        provider: str,
        use_structured_history: bool,
    ) -> AgentPrompt:
        dialogue_behavior_extra = (
            SHORT_REPLY_DIALOGUE_BEHAVIOR
            if normalize_provider(provider) in {"minicpm", "freeze_omni"}
            else None
        )
        return AgentPrompt(
            instructions=self.build_instructions(
                agent,
                dialogue_behavior_extra=dialogue_behavior_extra,
                include_history=not use_structured_history,
            ),
            initial_history=(self.build_initial_history(agent) if use_structured_history else ()),
        )


def load_scenario(
    path: str | Path,
    *,
    scenario_id: str | None = None,
    scenario_index: int | None = None,
    audio_dir: str | Path | None = None,
    question_audio_dir: str | Path | None = None,
    dialogue_turns: int = 5,
) -> LoadedScenario:
    scenarios = load_scenarios(
        path,
        scenario_id=scenario_id,
        scenario_index=scenario_index,
        audio_dir=audio_dir,
        question_audio_dir=question_audio_dir,
        dialogue_turns=dialogue_turns,
    )
    if len(scenarios) != 1:
        raise RuntimeError(f"Expected exactly one scenario, got {len(scenarios)}")
    return LoadedScenario(scenarios[0])


def load_scenarios(
    path: str | Path,
    *,
    scenario_id: str | None = None,
    scenario_index: int | None = None,
    audio_dir: str | Path | None = None,
    question_audio_dir: str | Path | None = None,
    dialogue_turns: int = 5,
) -> list[dict[str, Any]]:
    source_path = Path(path)
    payload = read_json(source_path)

    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        invalid_index = next(
            (index for index, record in enumerate(payload) if not isinstance(record, dict)),
            None,
        )
        if invalid_index is not None:
            raise RuntimeError(f"Scenario dataset entry {invalid_index} must be a JSON object")
        records = payload
    else:
        raise RuntimeError("Scenario JSON must be an object or an array of objects")
    selected = _select_records(records, scenario_id=scenario_id, scenario_index=scenario_index)
    question_audio_root = _question_audio_root(source_path, question_audio_dir)
    return [
        normalize_scenario(
            record,
            audio_dir=audio_dir,
            question_audio_dir=question_audio_root,
            dialogue_turns=dialogue_turns,
        )
        for record in selected
    ]


def normalize_scenario(
    record: dict[str, Any],
    *,
    audio_dir: str | Path | None = None,
    question_audio_dir: str | Path | None = None,
    dialogue_turns: int = 5,
) -> dict[str, Any]:
    if _is_normalized(record):
        return record

    scenario_id = str(record["id"])
    row_id = str(record.get("row_id") or scenario_id)
    human_name = str(record.get("human") or _name_from_profile(record.get("character_1", "")))
    gpt_name = str(record.get("gpt") or _name_from_profile(record.get("system", "")))
    profiles = [
        record.get("system", ""),
        record.get("character_1", ""),
        record.get("character_2", ""),
    ]

    turns = _normalize_turns(
        record.get("conversations") or [],
        human_name=human_name,
        gpt_name=gpt_name,
        speech=record.get("speech") or [],
        audio_dir=audio_dir,
    )
    history = turns[:-1]
    opening = turns[-1] if turns else None

    scenario: dict[str, Any] = {
        "id": scenario_id,
        "source": {
            "format": "two_test",
            "raw_id": scenario_id,
            "row_id": row_id,
        },
        "agents": {
            "citizen": {
                "name": human_name,
                "source_role": "human",
                "profile": _profile_for(human_name, profiles),
            },
            "scholar": {
                "name": gpt_name,
                "source_role": "gpt",
                "profile": _profile_for(gpt_name, [record.get("system", ""), *profiles]),
            },
        },
        "scene": {
            "type": record.get("type"),
            "subtype": record.get("subtype"),
            "topic": record.get("topic"),
            "goal": record.get("goal"),
        },
        "history": history,
        "opening": opening,
        "run": {
            "dialogue_turns": dialogue_turns,
            "turn_definition": "one citizen response plus one scholar response",
        },
        "evaluation": {
            "ask_after_turns": dialogue_turns,
            "target_agent": "scholar",
            "question": record.get("question"),
            "choices": record.get("multichoice") or [],
            "correct_answer": record.get("correct_answer"),
            "question_audio": _question_audio_path(scenario_id, question_audio_dir),
        },
        "assets": {
            "speech": _audio_paths(record.get("speech") or [], audio_dir=audio_dir),
        },
    }
    return scenario


def build_agent_instructions(
    scenario: dict[str, Any],
    agent: AgentKey,
    *,
    dialogue_behavior_extra: str | None = None,
    include_history: bool = True,
) -> str:
    if agent not in {"citizen", "scholar"}:
        raise ValueError(f"Unsupported scenario agent: {agent}")

    agents = scenario["agents"]
    self_agent = agents[agent]
    other_key = "scholar" if agent == "citizen" else "citizen"
    other_agent = agents[other_key]
    scene = scenario.get("scene") or {}

    lines = [
        f"You are {self_agent['name']}.",
        "",
        "Role profile:",
        str(self_agent.get("profile") or self_agent["name"]),
        "",
        "Conversation partner:",
        f"{other_agent['name']}: {other_agent.get('profile') or other_agent['name']}",
        "",
        "Scene:",
        f"- Type: {_empty_to_unknown(scene.get('type'))}",
        f"- Subtype: {_empty_to_unknown(scene.get('subtype'))}",
        f"- Topic: {_empty_to_unknown(scene.get('topic'))}",
        f"- Goal: {_empty_to_unknown(scene.get('goal'))}",
        "",
        "Dialogue behavior:",
        DIALOGUE_BEHAVIOR,
        *([dialogue_behavior_extra] if dialogue_behavior_extra else []),
    ]
    if include_history:
        lines.extend(_history_lines_for_agent(scenario, agent))

    return "\n".join(lines)


def build_agent_initial_history(
    scenario: dict[str, Any],
    agent: AgentKey,
) -> tuple[dict[str, Any], ...]:
    """Build Gemini Live Content[] history from a scenario's completed turns.

    The opening turn remains realtime input for its recipient, so it is seeded
    only for the speaker that already produced it.
    """
    if agent not in {"citizen", "scholar"}:
        raise ValueError(f"Unsupported scenario agent: {agent}")

    turns = list(scenario.get("history") or [])
    opening = scenario.get("opening")
    if opening and opening.get("agent") == agent:
        turns.append(opening)

    history: list[dict[str, Any]] = []
    for turn in turns:
        speaker = turn.get("agent")
        if speaker not in {"citizen", "scholar"}:
            raise ValueError(f"Scenario history turn has unsupported agent: {speaker!r}")
        history.append(
            {
                "role": "model" if speaker == agent else "user",
                "parts": [{"text": str(turn.get("text", ""))}],
            }
        )
    return tuple(history)


def _history_lines_for_agent(scenario: dict[str, Any], agent: AgentKey) -> list[str]:
    lines = ["", "Prior conversation history:"]
    turns = list(scenario.get("history") or [])
    opening = scenario.get("opening")
    if opening and opening.get("agent") == agent:
        turns.append(opening)
    if turns:
        lines.extend(_format_turn(turn) for turn in turns)
    else:
        lines.append("- No prior turns are available.")
    return lines


def write_scenarios(path: str | Path, scenarios: list[dict[str, Any]]) -> None:
    payload: dict[str, Any] | list[dict[str, Any]]
    payload = scenarios[0] if len(scenarios) == 1 else scenarios
    write_json(path, payload)


def build_evaluation_result(
    scenario: dict[str, Any],
    *,
    response_text: str,
    response_audio: str | None = None,
    dialogue_log: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    evaluation = scenario.get("evaluation") or {}
    choice = extract_answer_choice(response_text, evaluation.get("choices") or [])
    correct_answer = evaluation.get("correct_answer")
    is_correct = choice == correct_answer if choice and correct_answer else None
    return {
        "scenario_id": scenario.get("id"),
        "run_id": run_id or _default_run_id(str(scenario.get("id", "scenario"))),
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "evaluation": {
            "target_agent": evaluation.get("target_agent"),
            "question": evaluation.get("question"),
            "choices": evaluation.get("choices") or [],
            "correct_answer": correct_answer,
            "question_audio": evaluation.get("question_audio"),
        },
        "response": {
            "text": response_text,
            "audio": response_audio,
            "choice": choice,
            "is_correct": is_correct,
        },
        "artifacts": {
            "dialogue_log": dialogue_log,
        },
    }


def extract_answer_choice(response_text: str, choices: list[str]) -> str | None:
    text = response_text.strip()
    match = re.search(r"(?:^|[^A-Za-z])([A-D])(?:[\s\.\):：]|$)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    normalized_text = _normalize_choice_text(text)
    for choice in choices:
        choice_match = re.match(r"\s*([A-D])[\s\.\):：-]*(.*)", choice, flags=re.IGNORECASE)
        if not choice_match:
            continue
        label = choice_match.group(1).upper()
        choice_text = _normalize_choice_text(choice_match.group(2))
        if choice_text and choice_text in normalized_text:
            return label
    return None


def write_evaluation_result(
    path: str | Path, result: dict[str, Any] | list[dict[str, Any]]
) -> None:
    write_json(path, result)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "save-result":
        _save_result_main(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(
        description="Convert Vox Symposium dataset records into normalized scenarios."
    )
    parser.add_argument("input", help="Source JSON dataset, for example data/two_test.json.")
    parser.add_argument("output", help="Output normalized scenario JSON path.")
    parser.add_argument("--id", dest="scenario_id", help="Convert a single record by id.")
    parser.add_argument(
        "--index",
        dest="scenario_index",
        type=int,
        help="Convert a single record by zero-based index.",
    )
    parser.add_argument("--audio-dir", help="Directory containing speech wav files.")
    parser.add_argument(
        "--question-audio-dir",
        help="Directory containing question_{id}.wav files. Defaults to data/question_audio/<dataset>.",
    )
    parser.add_argument(
        "--dialogue-turns", type=int, default=5, help="Dialogue turns before evaluation."
    )
    args = parser.parse_args()

    scenarios = load_scenarios(
        args.input,
        scenario_id=args.scenario_id,
        scenario_index=args.scenario_index,
        audio_dir=args.audio_dir,
        question_audio_dir=args.question_audio_dir,
        dialogue_turns=args.dialogue_turns,
    )
    write_scenarios(args.output, scenarios)


def _save_result_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Save a scenario evaluation response result.")
    parser.add_argument(
        "scenario", help="Normalized scenario JSON path, or source dataset with --id/--index."
    )
    parser.add_argument("output", help="Output evaluation result JSON path.")
    parser.add_argument(
        "--id", dest="scenario_id", help="Select a scenario by id when scenario is a dataset."
    )
    parser.add_argument(
        "--index", dest="scenario_index", type=int, help="Select a scenario by zero-based index."
    )
    parser.add_argument("--audio-dir", help="Directory containing original speech wav files.")
    parser.add_argument(
        "--dialogue-turns", type=int, default=5, help="Dialogue turns used by the scenario."
    )
    parser.add_argument("--response", help="Final answer text from the evaluated model.")
    parser.add_argument("--response-file", help="Text file containing the final answer.")
    parser.add_argument(
        "--response-audio", help="Audio file for the evaluated model's final answer."
    )
    parser.add_argument("--dialogue-log", help="Optional saved dialogue transcript/log path.")
    parser.add_argument("--run-id", help="Stable run id for this evaluation.")
    args = parser.parse_args(argv)

    if bool(args.response) == bool(args.response_file):
        raise RuntimeError("Provide exactly one of --response or --response-file")

    response_text = args.response
    if args.response_file:
        response_text = Path(args.response_file).read_text(encoding="utf-8").strip()

    scenario = load_scenario(
        args.scenario,
        scenario_id=args.scenario_id,
        scenario_index=args.scenario_index,
        audio_dir=args.audio_dir,
        dialogue_turns=args.dialogue_turns,
    )
    result = build_evaluation_result(
        scenario.data,
        response_text=response_text or "",
        response_audio=args.response_audio,
        dialogue_log=args.dialogue_log,
        run_id=args.run_id,
    )
    write_evaluation_result(args.output, result)
    response = result["response"]
    print(
        "Saved evaluation result: "
        f"{args.output} "
        f"(choice={response['choice'] or 'unknown'}, "
        f"is_correct={response['is_correct']})"
    )


def _select_records(
    records: list[dict[str, Any]],
    *,
    scenario_id: str | None,
    scenario_index: int | None,
) -> list[dict[str, Any]]:
    if scenario_id is not None and scenario_index is not None:
        raise RuntimeError("Use either scenario_id or scenario_index, not both")
    if scenario_id is not None:
        matches = [record for record in records if str(record.get("id")) == scenario_id]
        if not matches:
            raise RuntimeError(f"Scenario id not found: {scenario_id}")
        return matches
    if scenario_index is not None:
        if scenario_index < 0:
            raise RuntimeError(f"Scenario index must be at least 0, got {scenario_index}")
        try:
            return [records[scenario_index]]
        except IndexError as exc:
            raise RuntimeError(f"Scenario index out of range: {scenario_index}") from exc
    return records


def _normalize_turns(
    conversations: list[dict[str, Any]],
    *,
    human_name: str,
    gpt_name: str,
    speech: list[str],
    audio_dir: str | Path | None,
) -> list[dict[str, Any]]:
    turns = []
    audio_paths = _audio_paths(speech, audio_dir=audio_dir)
    for index, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            raise RuntimeError(f"Conversation turn {index} must be a JSON object")
        source_role = str(turn.get("from") or "").strip().lower()
        try:
            agent = _SOURCE_ROLE_TO_AGENT[source_role]
        except KeyError as exc:
            supported = ", ".join(sorted(_SOURCE_ROLE_TO_AGENT))
            raise RuntimeError(
                f"Conversation turn {index} has unsupported role {source_role!r}; "
                f"expected one of: {supported}"
            ) from exc
        speaker = human_name if agent == "citizen" else gpt_name
        normalized = {
            "index": index,
            "agent": agent,
            "source_role": source_role,
            "speaker": speaker,
            "text": turn.get("value", ""),
        }
        if index < len(audio_paths):
            normalized["audio"] = audio_paths[index]
        turns.append(normalized)
    return turns


def _audio_paths(speech: list[str], *, audio_dir: str | Path | None) -> list[str]:
    if audio_dir is None:
        return list(speech)
    root = Path(audio_dir)
    return [str(root / filename) for filename in speech]


def _question_audio_root(source_path: Path, question_audio_dir: str | Path | None) -> Path | None:
    if question_audio_dir is not None:
        return Path(question_audio_dir)
    conventional_root = source_path.parent / "question_audio" / source_path.stem
    return conventional_root if conventional_root.is_dir() else None


def _question_audio_path(scenario_id: str, question_audio_dir: str | Path | None) -> str | None:
    if question_audio_dir is None:
        return None
    path = Path(question_audio_dir) / f"question_{scenario_id}.wav"
    if not path.is_file():
        raise RuntimeError(f"Question audio does not exist: {path}")
    return str(path)


def _is_normalized(record: dict[str, Any]) -> bool:
    return "agents" in record and "history" in record and "evaluation" in record


def _profile_for(name: str, profiles: list[str]) -> str:
    for profile in profiles:
        if _name_from_profile(profile) == name:
            return profile
    for profile in profiles:
        if profile.startswith(f"{name};") or profile.startswith(f"{name},") or profile == name:
            return profile
    for profile in profiles:
        if name and name in profile[:80]:
            return profile
    return name


def _name_from_profile(profile: str) -> str:
    if not profile:
        return ""
    return profile.split(";", 1)[0].strip()


def _format_turn(turn: dict[str, Any]) -> str:
    return f"- {turn.get('speaker', 'Unknown')}: {turn.get('text', '')}"


def _empty_to_unknown(value: Any) -> str:
    return str(value) if value else "Unknown"


def _normalize_choice_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace(" - ", "-")).strip(" .,:;()[]{}")


def _default_run_id(scenario_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{scenario_id}-{timestamp}"


if __name__ == "__main__":
    main()
