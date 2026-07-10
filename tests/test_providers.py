from __future__ import annotations

import unittest

from vox_symposium.providers import (
    MOSHI_PROTOCOL_PROVIDERS,
    normalize_provider,
    provider_env_prefix,
    provider_label,
)


class ProviderTests(unittest.TestCase):
    def test_normalize_provider_accepts_aliases(self) -> None:
        self.assertEqual(normalize_provider("MiniCPM-O-4_5"), "minicpm")
        self.assertEqual(normalize_provider("azure-openai"), "openai")

    def test_provider_metadata_uses_normalized_names(self) -> None:
        self.assertEqual(provider_env_prefix("personaplex"), "PERSONAPLEX")
        self.assertEqual(provider_label("personaplex"), "PersonaPlex")

    def test_covo_is_no_longer_a_supported_alias(self) -> None:
        self.assertEqual(normalize_provider("covo-audio-chat"), "covo_audio_chat")
        with self.assertRaisesRegex(ValueError, "Unsupported provider"):
            provider_env_prefix("covo")

    def test_personaplex_uses_moshi_protocol(self) -> None:
        self.assertIn("personaplex", MOSHI_PROTOCOL_PROVIDERS)


if __name__ == "__main__":
    unittest.main()
