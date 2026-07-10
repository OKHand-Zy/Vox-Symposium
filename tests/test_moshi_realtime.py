from __future__ import annotations

import unittest
from urllib.parse import parse_qs, urlsplit

from vox_symposium.models.moshi_realtime import moshi_url_with_text_prompt


class MoshiRealtimeTests(unittest.TestCase):
    def test_text_prompt_is_added_without_losing_voice_prompt(self) -> None:
        url = moshi_url_with_text_prompt(
            "ws://127.0.0.1:8998/api/chat?voice_prompt=NATF2.pt",
            "You are Dr. Lin.\nStay concise.",
        )

        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        self.assertEqual(parsed.path, "/api/chat")
        self.assertEqual(query["voice_prompt"], ["NATF2.pt"])
        self.assertEqual(query["text_prompt"], ["You are Dr. Lin.\nStay concise."])

    def test_text_prompt_overrides_static_prompt(self) -> None:
        url = moshi_url_with_text_prompt(
            "ws://127.0.0.1:8998/api/chat?voice_prompt=NATF2.pt&text_prompt=static",
            "scenario prompt",
        )

        query = parse_qs(urlsplit(url).query)
        self.assertEqual(query["voice_prompt"], ["NATF2.pt"])
        self.assertEqual(query["text_prompt"], ["scenario prompt"])


if __name__ == "__main__":
    unittest.main()
