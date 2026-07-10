from __future__ import annotations

import json
import os
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
    write_path = output_path.with_name(f".{output_path.name}.tmp") if atomic else output_path

    try:
        with write_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            if atomic:
                file.flush()
                os.fsync(file.fileno())
        if atomic:
            write_path.replace(output_path)
    except BaseException:
        if atomic:
            write_path.unlink(missing_ok=True)
        raise
