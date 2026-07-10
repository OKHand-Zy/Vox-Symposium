from __future__ import annotations

import os
from collections.abc import Iterable


def load_environment() -> bool:
    """Load a local ``.env`` file when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return False
    return load_dotenv()


def env_with_legacy(primary: str, legacy: str, *, default: str) -> str:
    return os.getenv(primary) or os.getenv(legacy) or default


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def first_env(names: Iterable[str]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def normalized_env(name: str, default: str) -> str:
    return os.getenv(name, default).strip().lower().replace("-", "_")


def int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return parse_int_env(name, raw)


def optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    return parse_int_env(name, raw)


def parse_int_env(name: str, raw: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from exc


def optional_float_env(name: str, default: float | None = None) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"", "none", "null", "off", "disabled"}:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number or 'none', got {raw!r}") from exc


def bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean, got {raw!r}")
