from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class JsonCache:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, namespace: str, key_parts: list[str]) -> Any | None:
        path = self._path(namespace, key_parts)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def set(self, namespace: str, key_parts: list[str], payload: Any) -> Path:
        path = self._path(namespace, key_parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        return path

    def _path(self, namespace: str, key_parts: list[str]) -> Path:
        digest = hashlib.sha256("::".join(key_parts).encode()).hexdigest()
        return self.root / namespace / f"{digest}.json"
