from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vox_symposium.evaluation import (
    _agent_provider,
    _format_dialogue_capture,
    _limit_scenarios,
    _load_resume_results,
    _parse_args,
    _remove_failed_case_artifacts,
    _result_payload,
    _scenario_artifact_dir,
    _scenario_run_id,
    _tee_console,
    _validate_args,
)
from vox_symposium.evaluation_artifacts import (
    EvaluationArtifacts,
    build_summary_report,
)
from vox_symposium.scenario import LoadedScenario


class EvaluationTests(unittest.TestCase):
    def test_parse_args_uses_200ms_tick_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys, "argv", ["vox-symposium-evaluate", "scenario.json", "result.json"]
        ):
            self.assertEqual(_parse_args().tick_duration_ms, 200)

    def test_parse_args_allows_tick_duration_environment_override(self) -> None:
        with patch.dict(os.environ, {"EVALUATION_TICK_DURATION_MS": "125"}, clear=True), patch.object(
            sys, "argv", ["vox-symposium-evaluate", "scenario.json", "result.json"]
        ):
            self.assertEqual(_parse_args().tick_duration_ms, 125)

    def test_agent_provider_uses_human_and_robot_environment_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AGENT_HUMAN_PROVIDER": "gemini",
                "AGENT_ROBOT_PROVIDER": "openai",
            },
            clear=True,
        ):
            self.assertEqual(_agent_provider("human"), "gemini")
            self.assertEqual(_agent_provider("robot"), "openai")

    def test_validate_args_rejects_non_positive_frame_duration(self) -> None:
        args = argparse.Namespace(
            start_index=0,
            limit=None,
            dialogue_turns=5,
            frame_ms=0,
            tick_duration_ms=200,
            audio_speed=1.0,
            idle_timeout=1.5,
            max_utterance_seconds=30.0,
            text_idle_timeout=0.7,
            text_max_wait=5.0,
            case_retries=3,
            case_delay=0.0,
            case_retry_delay=30.0,
        )

        with self.assertRaisesRegex(RuntimeError, "--frame-ms must be greater than 0"):
            _validate_args(args)

    def test_format_dialogue_capture_labels_role_turn_counts(self) -> None:
        self.assertEqual(
            _format_dialogue_capture("human", 1),
            "Captured human turn 1",
        )
        self.assertEqual(
            _format_dialogue_capture("human", 1, "hello"),
            "Captured human turn 1: hello",
        )
        self.assertEqual(
            _format_dialogue_capture("robot", 1, "hi"),
            "Captured robot turns 1: hi",
        )

    def test_scenario_artifact_dir_uses_source_row_id(self) -> None:
        self.assertEqual(
            _scenario_artifact_dir(
                Path("data/results/run-artifacts"),
                {"id": "case-1", "source": {"row_id": "row/001"}},
            ),
            Path("data/results/run-artifacts/row-001"),
        )

    def test_scenario_artifact_dir_falls_back_to_id(self) -> None:
        self.assertEqual(
            _scenario_artifact_dir(Path("artifacts"), {"id": "00000000"}),
            Path("artifacts/00000000"),
        )

    def test_remove_failed_case_artifacts_deletes_case_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_root = Path(tmpdir) / "artifacts"
            case_dir = artifact_root / "case-1"
            case_dir.mkdir(parents=True)
            (case_dir / "dialogue-log.json").write_text("{}", encoding="utf-8")

            _remove_failed_case_artifacts(artifact_root, {"id": "case-1"})

            self.assertFalse(case_dir.exists())
            self.assertTrue(artifact_root.exists())

    def test_tee_console_appends_existing_log_with_separator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "console-log.txt"
            path.write_text("previous", encoding="utf-8")

            with _tee_console(path):
                print("next")

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "previous\n###################################\nnext\n",
            )

    def test_scenario_run_id_keeps_single_case_run_id(self) -> None:
        self.assertEqual(
            _scenario_run_id("smoke", {"id": "case-1"}, index=0, total=1),
            "smoke",
        )

    def test_scenario_run_id_disambiguates_batch_cases(self) -> None:
        self.assertEqual(
            _scenario_run_id("smoke", {"source": {"row_id": "row/001"}}, index=0, total=2),
            "smoke-0001-row-001",
        )

    def test_result_payload_preserves_single_result_shape(self) -> None:
        result = {"scenario_id": "case-1"}

        self.assertIs(_result_payload([result], total=1), result)

    def test_result_payload_uses_list_for_batch(self) -> None:
        results = [{"scenario_id": "case-1"}, {"scenario_id": "case-2"}]

        self.assertEqual(_result_payload(results, total=2), results)

    def test_limit_scenarios_keeps_first_items(self) -> None:
        scenarios = [LoadedScenario({"id": str(index)}) for index in range(3)]

        self.assertEqual(
            [
                scenario.id
                for scenario in _limit_scenarios(
                    scenarios, limit=2, scenario_id=None, scenario_index=None
                )
            ],
            ["0", "1"],
        )

    def test_limit_scenarios_starts_at_index(self) -> None:
        scenarios = [LoadedScenario({"id": str(index)}) for index in range(5)]

        self.assertEqual(
            [
                scenario.id
                for scenario in _limit_scenarios(
                    scenarios,
                    limit=None,
                    scenario_id=None,
                    scenario_index=None,
                    start_index=2,
                )
            ],
            ["2", "3", "4"],
        )

    def test_limit_scenarios_combines_start_index_and_limit(self) -> None:
        scenarios = [LoadedScenario({"id": str(index)}) for index in range(5)]

        self.assertEqual(
            [
                scenario.id
                for scenario in _limit_scenarios(
                    scenarios,
                    limit=2,
                    scenario_id=None,
                    scenario_index=None,
                    start_index=2,
                )
            ],
            ["2", "3"],
        )

    def test_limit_scenarios_rejects_negative_start_index(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "--start-index must be at least 0"):
            _limit_scenarios([], limit=None, scenario_id=None, scenario_index=None, start_index=-1)

    def test_limit_scenarios_rejects_out_of_range_start_index(self) -> None:
        scenarios = [LoadedScenario({"id": str(index)}) for index in range(3)]

        with self.assertRaisesRegex(RuntimeError, "--start-index out of range: 3"):
            _limit_scenarios(
                scenarios, limit=None, scenario_id=None, scenario_index=None, start_index=3
            )

    def test_limit_scenarios_rejects_zero(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "--limit must be at least 1"):
            _limit_scenarios([], limit=0, scenario_id=None, scenario_index=None)

    def test_limit_scenarios_rejects_single_case_selection(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "--limit cannot be used with --id or --index"):
            _limit_scenarios([], limit=1, scenario_id=None, scenario_index=0)

    def test_limit_scenarios_rejects_start_index_with_single_case_selection(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError, "--start-index cannot be used with --id or --index"
        ):
            _limit_scenarios([], limit=None, scenario_id=None, scenario_index=0, start_index=2)

    def test_load_resume_results_keeps_completed_cases_before_start_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            with path.open("w", encoding="utf-8") as file:
                json.dump(
                    [
                        {"scenario_id": "case-1"},
                        {"scenario_id": "case-2", "case_index": 1},
                        {"scenario_id": "case-3", "case_index": 2},
                    ],
                    file,
                )

            results = _load_resume_results(path, start_index=2)

        self.assertEqual([result["scenario_id"] for result in results], ["case-1", "case-2"])
        self.assertEqual([result["case_index"] for result in results], [0, 1])
        self.assertEqual([result["case_number"] for result in results], [1, 2])

    def test_summary_report_counts_case_statuses(self) -> None:
        report = build_summary_report(
            [
                {
                    "scenario_id": "case-1",
                    "run_id": "run-1",
                    "evaluation": {"correct_answer": "A"},
                    "response": {
                        "choice": "A",
                        "is_correct": True,
                        "text": "A",
                        "audio": "case-1-answer.wav",
                    },
                    "artifacts": {"dialogue_log": "case-1-log.json"},
                },
                {
                    "scenario_id": "case-2",
                    "case_index": 6,
                    "case_number": 7,
                    "run_id": "run-2",
                    "evaluation": {"correct_answer": "B"},
                    "response": {"choice": "C", "is_correct": False, "text": "C"},
                    "artifacts": {},
                },
                {
                    "scenario_id": "case-3",
                    "run_id": "run-3",
                    "evaluation": {"correct_answer": "D"},
                    "response": {"choice": None, "is_correct": None, "text": ""},
                    "artifacts": {},
                },
            ],
            run_id="batch",
            scenario_path="scenarios.json",
            result_path=Path("results.json"),
            artifacts=EvaluationArtifacts(
                root=Path("results-artifacts"),
                env_snapshot=Path("results-artifacts/run-env.txt"),
                console_log=Path("results-artifacts/console-log.txt"),
                summary=Path("results-artifacts/summary.json"),
            ),
            total_cases=4,
        )

        self.assertEqual(report["total_cases"], 4)
        self.assertEqual(report["completed_cases"], 3)
        self.assertEqual(report["passed_cases"], 1)
        self.assertEqual(report["failed_cases"], 1)
        self.assertEqual(report["unknown_cases"], 1)
        self.assertAlmostEqual(report["pass_rate"], 1 / 4)
        self.assertAlmostEqual(report["completed_pass_rate"], 1 / 3)
        self.assertEqual(
            [case["status"] for case in report["cases"]], ["passed", "failed", "unknown"]
        )
        self.assertEqual(report["cases"][1]["index"], 6)
        self.assertEqual(report["cases"][1]["case_number"], 7)


if __name__ == "__main__":
    unittest.main()
