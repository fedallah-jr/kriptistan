from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
import random

from .indicators import close_position_in_range
from .models import Candle, CollisionPolicy, EntryClaim, RejectedClaim, Side


def due_date_in_range(pump_due: float | None, dump_due: float | None, minimum: float, maximum: float) -> bool:
    values = [value for value in (pump_due, dump_due) if value is not None]
    return any(minimum <= value <= maximum for value in values)


def chase_filter_passes(
    side: Side,
    candle: Candle,
    *,
    tp_percent: float,
    close_position_limit: float,
    range_multiple_tp: float,
) -> bool:
    if candle.close <= 0:
        return False
    range_pct = (candle.range / candle.close) * 100
    close_position = close_position_in_range(candle)
    if side is Side.LONG:
        if close_position >= close_position_limit and range_pct >= tp_percent * range_multiple_tp:
            return False
        return True
    short_limit = 1 - close_position_limit
    if close_position <= short_limit:
        return False
    return True


def btc_vol_guard_triggered(
    btc_5m_candles: list[Candle],
    *,
    max_range_pct: float,
    max_candle_pct: float,
) -> bool:
    if len(btc_5m_candles) < 12:
        return False
    recent = btc_5m_candles[-12:]
    highest = max(candle.high for candle in recent)
    lowest = min(candle.low for candle in recent)
    start_price = recent[0].open
    if start_price > 0 and ((highest - lowest) / start_price) * 100 >= max_range_pct:
        return True
    latest = recent[-1]
    if latest.open > 0 and (abs(latest.close - latest.open) / latest.open) * 100 >= max_candle_pct:
        return True
    return False


def symbol_on_cooldown(closed_times: list[datetime], *, now: datetime, cooldown_hours: int) -> bool:
    if not closed_times:
        return False
    cutoff = now - timedelta(hours=cooldown_hours)
    return closed_times[-1] >= cutoff


def dead_end_blacklisted(
    recent_loss_times: list[datetime],
    *,
    now: datetime,
    max_consecutive_losses: int,
    lookback_days: int,
    blacklist_days: int,
    all_close_times: list[datetime],
) -> bool:
    if len(recent_loss_times) < max_consecutive_losses:
        return False
    lookback_cutoff = now - timedelta(days=lookback_days)
    relevant = [value for value in recent_loss_times if value >= lookback_cutoff]
    if len(relevant) < max_consecutive_losses:
        return False
    loss_set = set(relevant)
    recent_closes = sorted(
        (t for t in all_close_times if t >= lookback_cutoff), reverse=True,
    )
    streak = 0
    for t in recent_closes:
        if t in loss_set:
            streak += 1
        else:
            break
    if streak < max_consecutive_losses:
        return False
    return recent_closes[0] + timedelta(days=blacklist_days) >= now


def anti_repetition_guard(symbol: str, recent_symbols: list[str], *, lookback: int = 4) -> bool:
    return symbol not in recent_symbols[-lookback:]


def resolve_collisions(
    claims: list[EntryClaim],
    *,
    policy: CollisionPolicy,
    timestamp: datetime,
    shuffle_seed: int = 0,
) -> tuple[list[EntryClaim], list[RejectedClaim]]:
    by_symbol: dict[str, list[EntryClaim]] = defaultdict(list)
    for claim in claims:
        by_symbol[claim.symbol].append(claim)

    winners: list[EntryClaim] = []
    rejections: list[RejectedClaim] = []
    for symbol, symbol_claims in by_symbol.items():
        if len(symbol_claims) == 1:
            winners.extend(symbol_claims)
            continue
        ranked = _rank_claims(symbol_claims, policy=policy, timestamp=timestamp, shuffle_seed=shuffle_seed)
        winners.append(ranked[0])
        rejections.extend(RejectedClaim(claim=item, reason="collision_lost") for item in ranked[1:])
    return winners, rejections


def _rank_claims(
    claims: list[EntryClaim],
    *,
    policy: CollisionPolicy,
    timestamp: datetime,
    shuffle_seed: int,
) -> list[EntryClaim]:
    if policy is CollisionPolicy.FIXED_PRIORITY:
        return sorted(claims, key=lambda item: (-item.fixed_priority, -item.signal_strength, -item.technical_score, item.bot_name))
    if policy is CollisionPolicy.SIGNAL_PRIORITY:
        return sorted(claims, key=lambda item: (-item.signal_strength, -item.technical_score, -item.fixed_priority, item.bot_name))
    rng = random.Random(f"{shuffle_seed}:{timestamp.isoformat()}")
    ranked = list(claims)
    rng.shuffle(ranked)
    return ranked
