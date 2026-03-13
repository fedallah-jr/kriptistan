from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from typing import Any
from urllib.request import urlopen

from .cache import JsonCache


@dataclass(slots=True, frozen=True)
class FearGreedPoint:
    timestamp: datetime
    value: int


@dataclass(slots=True)
class FearGreedClient:
    cache: JsonCache
    base_url: str = "https://api.alternative.me"

    def history(self, *, limit: int = 0) -> list[FearGreedPoint]:
        payload = self._get_json(f"/fng/?limit={limit}&format=json")
        return [
            FearGreedPoint(
                timestamp=datetime.fromtimestamp(int(item["timestamp"]), tz=UTC),
                value=int(item["value"]),
            )
            for item in payload["data"]
        ]

    def latest_as_of(self, *, points: list[FearGreedPoint], as_of: datetime, use_previous_day: bool) -> FearGreedPoint | None:
        if not points:
            return None
        candidates = sorted(points, key=lambda item: item.timestamp)
        if use_previous_day:
            cutoff = as_of.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            eligible = [point for point in candidates if point.timestamp < cutoff]
            return eligible[-1] if eligible else None
        eligible = [point for point in candidates if point.timestamp <= as_of]
        return eligible[-1] if eligible else None

    def _get_json(self, path: str) -> dict[str, Any]:
        cached = self.cache.get("fng", [path])
        if cached is not None:
            return cached
        with urlopen(f"{self.base_url}{path}") as response:
            payload = json.loads(response.read())
        self.cache.set("fng", [path], payload)
        return payload
