from datetime import UTC, datetime

from kriptistan.gates import resolve_collisions
from kriptistan.models import CollisionPolicy, EntryClaim, Side


def test_signal_priority_collision_policy_picks_stronger_claim() -> None:
    now = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    claims = [
        EntryClaim("bot_a", "TREND_MOM", "ETHUSDT", Side.LONG, now, "a", 2, 0.4, 10),
        EntryClaim("bot_b", "CYCLE_REV", "ETHUSDT", Side.LONG, now, "b", 1, 0.9, 5),
    ]
    winners, rejections = resolve_collisions(
        claims,
        policy=CollisionPolicy.SIGNAL_PRIORITY,
        timestamp=now,
    )
    assert winners[0].bot_name == "bot_b"
    assert rejections[0].reason == "collision_lost"
