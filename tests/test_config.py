from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vox_symposium.config import (
    gemini_live_initial_history,
    gemini_thinking_budget,
    gemini_thinking_level,
    gemini_uses_thinking_level,
    load_moshi_settings,
    load_settings,
)
from vox_symposium.env import bool_env


class ConfigTests(unittest.TestCase):
    def test_gemini_thinking_level_defaults_to_minimal(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(gemini_thinking_level(), "minimal")

    def test_gemini_thinking_level_rejects_thinking_budget_values(self) -> None:
        with patch.dict(os.environ, {"GEMINI_LIVE_THINKING_LEVEL": "2048"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_LIVE_THINKING_LEVEL"):
                gemini_thinking_level()

    def test_gemini_thinking_budget_accepts_2_5_dynamic_and_fixed_values(self) -> None:
        for value, expected in {"-1": -1, "0": 0, "24576": 24_576}.items():
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ,
                    {"GEMINI_LIVE_THINKING_BUDGET": value},
                    clear=True,
                ),
            ):
                self.assertEqual(gemini_thinking_budget(), expected)

    def test_gemini_thinking_budget_rejects_out_of_range_values(self) -> None:
        with patch.dict(os.environ, {"GEMINI_LIVE_THINKING_BUDGET": "24577"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_LIVE_THINKING_BUDGET"):
                gemini_thinking_budget()

    def test_gemini_thinking_parameter_is_selected_by_model_generation(self) -> None:
        self.assertTrue(gemini_uses_thinking_level("gemini-3.1-flash-live-preview"))
        self.assertFalse(gemini_uses_thinking_level("gemini-live-2.5-flash-native-audio"))

    def test_gemini_affective_dialog_defaults_to_disabled(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(bool_env("GEMINI_LIVE_ENABLE_AFFECTIVE_DIALOG", False))

    def test_gemini_affective_dialog_can_be_enabled(self) -> None:
        with patch.dict(os.environ, {"GEMINI_LIVE_ENABLE_AFFECTIVE_DIALOG": "true"}, clear=True):
            self.assertTrue(bool_env("GEMINI_LIVE_ENABLE_AFFECTIVE_DIALOG", False))

    def test_gemini_initial_history_requires_content_turns(self) -> None:
        history = '[{"role":"user","parts":[{"text":"hello"}]}]'
        with patch.dict(os.environ, {"GEMINI_LIVE_INITIAL_HISTORY_JSON": history}, clear=True):
            self.assertEqual(
                gemini_live_initial_history(),
                ({"role": "user", "parts": [{"text": "hello"}]},),
            )

    def test_gemini_initial_history_rejects_invalid_roles(self) -> None:
        history = '[{"role":"system","parts":[{"text":"hello"}]}]'
        with patch.dict(os.environ, {"GEMINI_LIVE_INITIAL_HISTORY_JSON": history}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "role 'user' or 'model'"):
                gemini_live_initial_history()

    def test_gemini_2_5_does_not_parse_3_1_initial_history_setting(self) -> None:
        env = {
            "LIVEKIT_URL": "ws://localhost:7880",
            "LIVEKIT_API_KEY": "key",
            "LIVEKIT_API_SECRET": "secret",
            "AGENT_CITIZEN_PROVIDER": "gemini",
            "AGENT_SCHOLAR_PROVIDER": "gemini",
            "GEMINI_API_KEY": "key",
            "GEMINI_LIVE_MODEL": "gemini-live-2.5-flash-native-audio",
            "GEMINI_LIVE_INITIAL_HISTORY_JSON": "not-json",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings()

        assert settings.gemini is not None
        self.assertEqual(settings.gemini.initial_history, ())

    def test_scenario_history_uses_initial_content_only_for_gemini_3(self) -> None:
        scenario = {
            "id": "case-1",
            "human": "Citizen",
            "gpt": "Scholar",
            "system": "Scholar; expert profile",
            "character_1": "Citizen; practical profile",
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Opening"},
            ],
        }
        base_env = {
            "LIVEKIT_URL": "ws://localhost:7880",
            "LIVEKIT_API_KEY": "key",
            "LIVEKIT_API_SECRET": "secret",
            "AGENT_CITIZEN_PROVIDER": "gemini",
            "AGENT_SCHOLAR_PROVIDER": "gemini",
            "GEMINI_API_KEY": "key",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir) / "scenario.json"
            scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
            base_env["SCENARIO_FILE"] = str(scenario_path)

            with patch.dict(
                os.environ,
                {**base_env, "GEMINI_LIVE_MODEL": "gemini-live-2.5-flash-native-audio"},
                clear=True,
            ):
                settings_2_5 = load_settings()
            with patch.dict(
                os.environ,
                {**base_env, "GEMINI_LIVE_MODEL": "gemini-3.1-flash-live-preview"},
                clear=True,
            ):
                settings_3_1 = load_settings()

        self.assertIn("Prior conversation history:", settings_2_5.agent_scholar.instructions)
        self.assertEqual(settings_2_5.agent_scholar.initial_history, ())
        self.assertNotIn("Prior conversation history:", settings_3_1.agent_scholar.instructions)
        self.assertEqual(
            settings_3_1.agent_scholar.initial_history,
            (
                {"role": "user", "parts": [{"text": "Hello"}]},
                {"role": "model", "parts": [{"text": "Opening"}]},
            ),
        )

    def test_moshi_settings_adds_default_chat_path(self) -> None:
        with patch.dict(os.environ, {"MOSHI_REALTIME_URL": "http://localhost:8998"}, clear=True):
            settings = load_moshi_settings(required=True)

        assert settings is not None
        self.assertEqual(settings.url, "ws://localhost:8998/api/chat")

    def test_personaplex_settings_use_moshi_chat_protocol(self) -> None:
        env = {
            "PERSONAPLEX_REALTIME_URL": "http://localhost:8998?voice_prompt=NATF2.pt",
            "PERSONAPLEX_API_KEY": "token",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = load_moshi_settings("personaplex", required=True)

        assert settings is not None
        self.assertEqual(settings.url, "ws://localhost:8998/api/chat?voice_prompt=NATF2.pt")
        self.assertEqual(settings.api_key, "token")


if __name__ == "__main__":
    unittest.main()
