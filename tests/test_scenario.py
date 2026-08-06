from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vox_symposium.scenario import (
    FREEZE_OMNI_DIALOGUE_BEHAVIOR,
    LoadedScenario,
    MINICPM_DIALOGUE_BEHAVIOR,
    build_agent_initial_history,
    build_agent_instructions,
    extract_answer_choice,
    load_scenarios,
    normalize_scenario,
    other_agent,
    validate_agent,
)


class ScenarioTests(unittest.TestCase):
    def test_load_scenarios_rejects_non_object_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenarios.json"
            path.write_text(json.dumps([{"id": "valid"}, "invalid"]), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "entry 1 must be a JSON object"):
                load_scenarios(path)

    def test_load_scenarios_rejects_negative_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenarios.json"
            path.write_text(json.dumps([{"id": "case-1"}]), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "index must be at least 0"):
                load_scenarios(path, scenario_index=-1)

    def test_extract_answer_choice_from_label(self) -> None:
        self.assertEqual(extract_answer_choice("I choose C.", ["A. One", "C. Three"]), "C")

    def test_extract_answer_choice_from_choice_text(self) -> None:
        choices = [
            "A. Fully opposed",
            "B. Neutral",
            "C. Skeptical but willing to listen",
            "D. Enthusiastic",
        ]

        self.assertEqual(extract_answer_choice("Skeptical but willing to listen", choices), "C")

    def test_agent_helpers_validate_and_swap_two_roles(self) -> None:
        self.assertEqual(validate_agent("human"), "human")
        self.assertEqual(validate_agent("robot"), "robot")
        self.assertEqual(other_agent("human"), "robot")
        self.assertEqual(other_agent("robot"), "human")
        with self.assertRaisesRegex(ValueError, "Unsupported scenario agent: observer"):
            validate_agent("observer")

    def test_normalize_scenario_uses_last_turn_as_opening(self) -> None:
        scenario = normalize_scenario(
            {
                "id": "case-1",
                "row_id": "row-1",
                "human": "Human",
                "gpt": "Robot",
                "system": "Robot; expert profile",
                "character_1": "Human; practical profile",
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi there"},
                ],
                "speech": ["hello.wav", "opening.wav"],
                "question": "Question?",
                "multichoice": ["A. Yes", "B. No"],
                "correct_answer": "A",
            },
            audio_dir="audio",
            dialogue_turns=3,
        )

        self.assertEqual(scenario["history"][0]["audio"], "audio/hello.wav")
        self.assertEqual(scenario["opening"]["audio"], "audio/opening.wav")
        self.assertEqual(scenario["run"]["dialogue_turns"], 3)
        self.assertEqual(set(scenario["agents"]), {"human", "robot"})
        self.assertEqual(scenario["opening"]["agent"], "robot")
        self.assertEqual(scenario["evaluation"]["target_agent"], "robot")
        self.assertEqual(scenario["source"]["row_id"], "row-1")

    def test_normalize_scenario_rejects_unknown_conversation_roles(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unsupported role 'tool'"):
            normalize_scenario(
                {
                    "id": "case-1",
                    "conversations": [{"from": "tool", "value": "result"}],
                }
            )

    def test_normalize_scenario_rejects_unknown_normalized_agent_keys(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "must define exactly these agents: human, robot",
        ):
            normalize_scenario(
                {
                    "id": "case-1",
                    "agents": {"left": {}, "right": {}},
                    "history": [],
                    "evaluation": {},
                }
            )

    def test_agent_initial_history_uses_relative_roles_and_preserves_opening_playback(self) -> None:
        scenario = normalize_scenario(
            {
                "id": "case-1",
                "human": "Human",
                "gpt": "Robot",
                "system": "Robot; expert profile",
                "character_1": "Human; practical profile",
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Opening that is played as audio"},
                ],
                "question": "Question?",
                "multichoice": ["A. Yes", "B. No"],
                "correct_answer": "A",
            },
            dialogue_turns=3,
        )

        robot_history = build_agent_initial_history(scenario, "robot")
        human_history = build_agent_initial_history(scenario, "human")
        robot_instructions = build_agent_instructions(scenario, "robot", include_history=False)
        human_instructions = build_agent_instructions(scenario, "human", include_history=False)

        self.assertIn("You are Robot.", robot_instructions)
        self.assertIn("You are Human.", human_instructions)
        self.assertNotIn("Prior conversation history", robot_instructions)
        self.assertNotIn("Hello", robot_instructions)
        self.assertNotIn("Opening that is played as audio", robot_instructions)
        self.assertEqual(
            robot_history,
            (
                {"role": "user", "parts": [{"text": "Hello"}]},
                {"role": "model", "parts": [{"text": "Opening that is played as audio"}]},
            ),
        )
        self.assertEqual(
            human_history,
            ({"role": "model", "parts": [{"text": "Hello"}]},),
        )
        for instructions in (robot_instructions, human_instructions):
            self.assertIn("Dialogue behavior:", instructions)
            self.assertIn("Stay in character and respond with the speaking style", instructions)
            self.assertIn("Each response should leave a natural opening", instructions)
            self.assertNotIn("Opening turn reserved for playback", instructions)
            self.assertNotIn("Runtime instructions", instructions)
            self.assertNotIn("evaluation", instructions.lower())
            self.assertNotIn("voice model being evaluated", instructions)

    def test_agent_initial_history_includes_a_human_opening_only_for_human(self) -> None:
        scenario = normalize_scenario(
            {
                "id": "case-1",
                "human": "Human",
                "gpt": "Robot",
                "system": "Robot; expert profile",
                "character_1": "Human; practical profile",
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi there"},
                    {"from": "human", "value": "Human opening that is played as audio"},
                ],
                "question": "Question?",
                "multichoice": ["A. Yes", "B. No"],
                "correct_answer": "A",
            },
            dialogue_turns=3,
        )

        human_history = build_agent_initial_history(scenario, "human")
        robot_history = build_agent_initial_history(scenario, "robot")

        self.assertEqual(
            human_history,
            (
                {"role": "model", "parts": [{"text": "Hello"}]},
                {"role": "user", "parts": [{"text": "Hi there"}]},
                {"role": "model", "parts": [{"text": "Human opening that is played as audio"}]},
            ),
        )
        self.assertEqual(
            robot_history,
            (
                {"role": "user", "parts": [{"text": "Hello"}]},
                {"role": "model", "parts": [{"text": "Hi there"}]},
            ),
        )

    def test_agent_instructions_keep_history_when_structured_history_is_not_used(self) -> None:
        scenario = normalize_scenario(
            {
                "id": "case-1",
                "human": "Human",
                "gpt": "Robot",
                "system": "Robot; expert profile",
                "character_1": "Human; practical profile",
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Opening"},
                ],
                "question": "Question?",
                "multichoice": ["A. Yes", "B. No"],
                "correct_answer": "A",
            },
            dialogue_turns=3,
        )

        robot_instructions = build_agent_instructions(scenario, "robot")
        human_instructions = build_agent_instructions(scenario, "human")

        self.assertIn("Prior conversation history:", robot_instructions)
        self.assertIn("- Human: Hello", robot_instructions)
        self.assertIn("- Robot: Opening", robot_instructions)
        self.assertIn("- Human: Hello", human_instructions)
        self.assertNotIn("- Robot: Opening", human_instructions)

    def test_agent_instructions_can_append_provider_dialogue_behavior(self) -> None:
        scenario = normalize_scenario(
            {
                "id": "case-1",
                "human": "Human",
                "gpt": "Robot",
                "system": "Robot; expert profile",
                "character_1": "Human; practical profile",
                "conversations": [{"from": "gpt", "value": "Opening"}],
                "question": "Question?",
                "multichoice": ["A. Yes", "B. No"],
                "correct_answer": "A",
            },
            dialogue_turns=3,
        )

        robot_instructions = build_agent_instructions(
            scenario,
            "robot",
            dialogue_behavior_extra=MINICPM_DIALOGUE_BEHAVIOR,
        )
        human_instructions = build_agent_instructions(scenario, "human")

        self.assertIn("Dialogue behavior:", robot_instructions)
        self.assertIn(MINICPM_DIALOGUE_BEHAVIOR, robot_instructions)
        self.assertLess(
            robot_instructions.index("Dialogue behavior:"),
            robot_instructions.index(MINICPM_DIALOGUE_BEHAVIOR),
        )
        self.assertNotIn(MINICPM_DIALOGUE_BEHAVIOR, human_instructions)

    def test_freeze_omni_prompt_adds_short_reply_behavior(self) -> None:
        scenario = normalize_scenario(
            {
                "id": "case-1",
                "human": "Human",
                "gpt": "Robot",
                "system": "Robot; expert profile",
                "character_1": "Human; practical profile",
                "conversations": [{"from": "gpt", "value": "Opening"}],
                "question": "Question?",
                "multichoice": ["A. Yes", "B. No"],
                "correct_answer": "A",
            },
            dialogue_turns=3,
        )

        loaded = LoadedScenario(scenario)
        robot_prompt = loaded.build_prompt(
            "robot",
            provider="freeze-omni",
            use_structured_history=False,
        )
        human_prompt = loaded.build_prompt(
            "human",
            provider="gemini",
            use_structured_history=False,
        )

        self.assertIn(FREEZE_OMNI_DIALOGUE_BEHAVIOR, robot_prompt.instructions)
        self.assertNotIn(FREEZE_OMNI_DIALOGUE_BEHAVIOR, human_prompt.instructions)


if __name__ == "__main__":
    unittest.main()
