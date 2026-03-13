from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .config import AppConfig, BotConfig
from .gates import resolve_collisions
from .models import CollisionPolicy, CycleStats, EntryClaim, RejectedClaim, StrategyContext
from .strategies import evaluate_strategy


@dataclass(slots=True)
class EngineDecisionBatch:
    timestamp: datetime
    winners: list[EntryClaim]
    rejections: list[RejectedClaim]


def collect_entry_claims(
    *,
    timestamp: datetime,
    bots: list[BotConfig],
    symbol_candles: dict[str, list],
    cycle_stats: dict[str, CycleStats],
) -> list[EntryClaim]:
    claims: list[EntryClaim] = []
    for bot in bots:
        for symbol, candles in symbol_candles.items():
            decision = evaluate_strategy(
                bot.strategy,
                StrategyContext(symbol=symbol, candles=candles, cycle_stats=cycle_stats.get(symbol), tp_percent=bot.tp_percent),
            )
            if decision is None:
                continue
            claims.append(
                EntryClaim(
                    bot_name=bot.name,
                    strategy=bot.strategy,
                    symbol=symbol,
                    side=decision.side,
                    signal_time=timestamp,
                    reason=decision.reason,
                    technical_score=decision.technical_score,
                    signal_strength=decision.signal_strength,
                    fixed_priority=bot.fixed_priority,
                )
            )
    return claims


def resolve_entry_batch(
    *,
    timestamp: datetime,
    claims: list[EntryClaim],
    collision_policy: CollisionPolicy,
    shuffle_seed: int,
) -> EngineDecisionBatch:
    winners, rejections = resolve_collisions(
        claims,
        policy=collision_policy,
        timestamp=timestamp,
        shuffle_seed=shuffle_seed,
    )
    return EngineDecisionBatch(timestamp=timestamp, winners=winners, rejections=rejections)


class BacktestEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build_hourly_batch(
        self,
        *,
        timestamp: datetime,
        symbol_candles: dict[str, list],
        cycle_stats: dict[str, CycleStats],
    ) -> EngineDecisionBatch:
        claims = collect_entry_claims(
            timestamp=timestamp,
            bots=list(self.config.bots),
            symbol_candles=symbol_candles,
            cycle_stats=cycle_stats,
        )
        return resolve_entry_batch(
            timestamp=timestamp,
            claims=claims,
            collision_policy=self.config.backtest.collision_policy,
            shuffle_seed=self.config.backtest.shuffle_seed,
        )
