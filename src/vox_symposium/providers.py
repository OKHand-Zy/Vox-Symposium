from __future__ import annotations


SUPPORTED_PROVIDERS = frozenset(
    {
        "openai",
        "gemini",
        "minicpm",
        "moshi",
        "personaplex",
        "covo_audio_chat_fd",
    }
)

MOSHI_PROTOCOL_PROVIDERS = frozenset({"moshi", "personaplex"})

PCM_GATEWAY_PROVIDERS = frozenset({"covo_audio_chat_fd"})

_PROVIDER_ALIASES = {
    "azure_openai": "openai",
    "minicpm_o_4_5": "minicpm",
    "minicpm_o": "minicpm",
    "covo": "covo_audio_chat_fd",
    "covo_audio": "covo_audio_chat_fd",
    "covo_audio_chat": "covo_audio_chat_fd",
    "covo_audio_chat_fd": "covo_audio_chat_fd",
}

_PROVIDER_ENV_PREFIXES = {
    "openai": "OPENAI",
    "gemini": "GEMINI",
    "minicpm": "MINICPM",
    "moshi": "MOSHI",
    "personaplex": "PERSONAPLEX",
    "covo_audio_chat_fd": "COVO_AUDIO_CHAT_FD",
}

_PROVIDER_LABELS = {
    "openai": "OpenAI Realtime",
    "gemini": "Gemini Live",
    "minicpm": "MiniCPM-o 4.5",
    "moshi": "Moshi",
    "personaplex": "PersonaPlex",
    "covo_audio_chat_fd": "Covo-Audio-Chat-FD",
}


def normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower().replace("-", "_")
    return _PROVIDER_ALIASES.get(normalized, normalized)


def provider_env_prefix(provider: str) -> str:
    normalized = normalize_provider(provider)
    try:
        return _PROVIDER_ENV_PREFIXES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {provider!r}") from exc


def provider_label(provider: str) -> str:
    normalized = normalize_provider(provider)
    return _PROVIDER_LABELS.get(normalized, normalized)
