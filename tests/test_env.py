from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from vox_symposium.env import first_env, float_env, int_env, optional_int_env, required_env


class EnvTests(unittest.TestCase):
    def test_first_env_returns_first_non_empty_value(self) -> None:
        with patch.dict(os.environ, {"SECOND": "value"}, clear=True):
            self.assertEqual(first_env(["FIRST", "SECOND"]), "value")

    def test_int_env_reports_variable_name(self) -> None:
        with patch.dict(os.environ, {"FRAME_MS": "fast"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "FRAME_MS must be an integer"):
                int_env("FRAME_MS", 20)

    def test_optional_int_env_treats_empty_as_none(self) -> None:
        with patch.dict(os.environ, {"SCENARIO_INDEX": ""}, clear=True):
            self.assertIsNone(optional_int_env("SCENARIO_INDEX"))

    def test_required_env_rejects_empty_values(self) -> None:
        with patch.dict(os.environ, {"API_KEY": ""}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError, "Missing required environment variable: API_KEY"
            ):
                required_env("API_KEY")

    def test_float_env_rejects_non_finite_values(self) -> None:
        with patch.dict(os.environ, {"TIMEOUT": "nan"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "TIMEOUT must be a finite number"):
                float_env("TIMEOUT", 1.0)


if __name__ == "__main__":
    unittest.main()
