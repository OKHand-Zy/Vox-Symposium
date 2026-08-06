from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON document from *path*."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: Any, *, atomic: bool = True) -> None:
    """Write indented UTF-8 JSON, replacing an existing file atomically by default."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_path: Path | None = None

    try:
        if atomic:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                write_path = Path(file.name)
                json.dump(payload, file, ensure_ascii=False, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
        else:
            write_path = output_path
            with write_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
                file.write("\n")
        if atomic:
            assert write_path is not None
            os.replace(write_path, output_path)
    except BaseException:
        if atomic and write_path is not None:
            write_path.unlink(missing_ok=True)
        raise
