from __future__ import annotations

import os
from pathlib import Path


_loaded_env_paths: set[Path] = set()


def load_env_file(env_path: Path | None = None) -> None:
    path = (env_path or (Path(__file__).resolve().parent.parent / ".env")).resolve()
    if path in _loaded_env_paths or not path.exists():
        _loaded_env_paths.add(path)
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        if not key:
            continue

        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)

    _loaded_env_paths.add(path)
