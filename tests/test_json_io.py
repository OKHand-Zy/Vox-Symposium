from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vox_symposium.json_io import read_json, write_json


class JsonIoTests(unittest.TestCase):
    def test_atomic_write_preserves_existing_file_when_serialization_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            write_json(path, {"status": "complete"})

            with self.assertRaises(TypeError):
                write_json(path, {"invalid": object()})

            self.assertEqual(read_json(path), {"status": "complete"})
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
