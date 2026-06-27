from __future__ import annotations


SUPPORTED_PROVIDERS = frozenset(
    {
        "openai",
        "gemini",
        "minicpm",
        "moshi",
        "personaplex",
    }
)

MOSHI_PROTOCOL_PROVIDERS = frozenset({"moshi", "personaplex"})

_PROVIDER_ALIASES = {
    "azure_openai": "openai",
    "minicpm_o_4_5": "minicpm",
    "minicpm_o": "minicpm",
}

_PROVIDER_ENV_PREFIXES = {
    "openai": "OPENAI",
    "gemini": "GEMINI",
    "minicpm": "MINICPM",
    "moshi": "MOSHI",
    "personaplex": "PERSONAPLEX",
}

_PROVIDER_LABELS = {
    "openai": "OpenAI Realtime",
    "gemini": "Gemini Live",
    "minicpm": "MiniCPM-o 4.5",
    "moshi": "Moshi",
    "personaplex": "PersonaPlex",
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
