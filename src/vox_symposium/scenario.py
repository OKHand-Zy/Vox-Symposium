from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AgentKey = str


@dataclass(frozen=True)
class LoadedScenario:
    data: dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.data["id"])

    def build_instructions(self, agent: AgentKey) -> str:
        return build_agent_instructions(self.data, agent)


def load_scenario(
    path: str | Path,
    *,
    scenario_id: str | None = None,
    scenario_index: int | None = None,
    audio_dir: str | Path | None = None,
    dialogue_turns: int = 5,
) -> LoadedScenario:
    scenarios = load_scenarios(
        path,
        scenario_id=scenario_id,
        scenario_index=scenario_index,
        audio_dir=audio_dir,
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
    dialogue_turns: int = 5,
) -> list[dict[str, Any]]:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    records = payload if isinstance(payload, list) else [payload]
    selected = _select_records(records, scenario_id=scenario_id, scenario_index=scenario_index)
    return [
        normalize_scenario(record, audio_dir=audio_dir, dialogue_turns=dialogue_turns)
        for record in selected
    ]


def normalize_scenario(
    record: dict[str, Any],
    *,
    audio_dir: str | Path | None = None,
    dialogue_turns: int = 5,
) -> dict[str, Any]:
    if _is_normalized(record):
        return record

    scenario_id = str(record["id"])
    human_name = str(record.get("human") or _name_from_profile(record.get("character_1", "")))
    gpt_name = str(record.get("gpt") or _name_from_profile(record.get("system", "")))
    profiles = [record.get("system", ""), record.get("character_1", ""), record.get("character_2", "")]

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
            "question_audio": None,
        },
        "assets": {
            "speech": _audio_paths(record.get("speech") or [], audio_dir=audio_dir),
        },
    }
    return scenario


def build_agent_instructions(scenario: dict[str, Any], agent: AgentKey) -> str:
    if agent not in {"citizen", "scholar"}:
        raise ValueError(f"Unsupported scenario agent: {agent}")

    agents = scenario["agents"]
    self_agent = agents[agent]
    other_key = "scholar" if agent == "citizen" else "citizen"
    other_agent = agents[other_key]
    scene = scenario.get("scene") or {}
    run = scenario.get("run") or {}

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
        "Prior conversation history:",
    ]

    history = scenario.get("history") or []
    if history:
        lines.extend(_format_turn(turn) for turn in history)
    else:
        lines.append("- No prior turns are available.")

    opening = scenario.get("opening")
    if opening:
        lines.extend(
            [
                "",
                "Opening turn reserved for playback:",
                _format_turn(opening),
            ]
        )

    lines.extend(
        [
            "",
            "Runtime instructions:",
            "- Continue from the prior history and opening turn; do not restart the conversation.",
            "- Stay in character and preserve the established speaking style.",
            "- Keep each reply to 1-2 short spoken sentences.",
            f"- Continue for about {run.get('dialogue_turns', 5)} dialogue turns before evaluation.",
            "- Do not mention the evaluation question, answer choices, or correct answer during the dialogue.",
        ]
    )
    if agent == "citizen":
        lines.append("- You are the simulated conversation partner, not the model being evaluated.")
    else:
        lines.append("- You are the voice model being evaluated; answer later evaluation questions with the best choice only when asked.")

    return "\n".join(lines)


def write_scenarios(path: str | Path, scenarios: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] | list[dict[str, Any]]
    payload = scenarios[0] if len(scenarios) == 1 else scenarios
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


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
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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


def write_evaluation_result(path: str | Path, result: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "save-result":
        _save_result_main(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description="Convert Vox Symposium dataset records into normalized scenarios.")
    parser.add_argument("input", help="Source JSON dataset, for example data/two_test.json.")
    parser.add_argument("output", help="Output normalized scenario JSON path.")
    parser.add_argument("--id", dest="scenario_id", help="Convert a single record by id.")
    parser.add_argument("--index", dest="scenario_index", type=int, help="Convert a single record by zero-based index.")
    parser.add_argument("--audio-dir", help="Directory containing speech wav files.")
    parser.add_argument("--dialogue-turns", type=int, default=5, help="Dialogue turns before evaluation.")
    args = parser.parse_args()

    scenarios = load_scenarios(
        args.input,
        scenario_id=args.scenario_id,
        scenario_index=args.scenario_index,
        audio_dir=args.audio_dir,
        dialogue_turns=args.dialogue_turns,
    )
    write_scenarios(args.output, scenarios)


def _save_result_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Save a scenario evaluation response result.")
    parser.add_argument("scenario", help="Normalized scenario JSON path, or source dataset with --id/--index.")
    parser.add_argument("output", help="Output evaluation result JSON path.")
    parser.add_argument("--id", dest="scenario_id", help="Select a scenario by id when scenario is a dataset.")
    parser.add_argument("--index", dest="scenario_index", type=int, help="Select a scenario by zero-based index.")
    parser.add_argument("--audio-dir", help="Directory containing original speech wav files.")
    parser.add_argument("--dialogue-turns", type=int, default=5, help="Dialogue turns used by the scenario.")
    parser.add_argument("--response", help="Final answer text from the evaluated model.")
    parser.add_argument("--response-file", help="Text file containing the final answer.")
    parser.add_argument("--response-audio", help="Audio file for the evaluated model's final answer.")
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
        source_role = turn.get("from")
        agent = "citizen" if source_role == "human" else "scholar"
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
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{scenario_id}-{timestamp}"


if __name__ == "__main__":
    main()
