from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class Side(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class ExitReason(StrEnum):
    TP = "TP"
    SL = "SL"
    OPEN = "OPEN"


class CollisionPolicy(StrEnum):
    FIXED_PRIORITY = "fixed_priority"
    SIGNAL_PRIORITY = "signal_priority"
    SEEDED_SHUFFLE = "seeded_shuffle"


class CapitalMode(StrEnum):
    PER_BOT = "per_bot"
    SHARED = "shared"


class ResolutionLevel(StrEnum):
    DAY = "1d"
    HOUR = "1h"
    MINUTE = "1m"
    TRADE = "trade"


@dataclass(slots=True, frozen=True)
class Candle:
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    quote_volume: float = 0.0

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass(slots=True, frozen=True)
class ExchangeSymbol:
    symbol: str
    base_asset: str
    quote_asset: str
    quantity_precision: int
    step_size: float
    min_qty: float
    min_notional: float
    status: str = "TRADING"


@dataclass(slots=True, frozen=True)
class AggTrade:
    trade_id: int
    timestamp: datetime
    price: float
    quantity: float


@dataclass(slots=True, frozen=True)
class CycleStats:
    as_of: datetime
    symbol: str
    pump_mean_days: float | None
    pump_median_days: float | None
    pump_stdev_days: float | None
    pump_last_interval_days: float | None
    pump_date_due_days: float | None
    dump_mean_days: float | None
    dump_median_days: float | None
    dump_stdev_days: float | None
    dump_last_interval_days: float | None
    dump_date_due_days: float | None
    passes_stdev_filter: bool


@dataclass(slots=True, frozen=True)
class Candidate:
    symbol: str
    cycle_stats: CycleStats
    distance: float


@dataclass(slots=True, frozen=True)
class StrategyDecision:
    side: Side
    reason: str
    technical_score: int
    signal_strength: float


@dataclass(slots=True, frozen=True)
class EntryClaim:
    bot_name: str
    strategy: str
    symbol: str
    side: Side
    signal_time: datetime
    reason: str
    technical_score: int
    signal_strength: float
    fixed_priority: int = 0


@dataclass(slots=True, frozen=True)
class RejectedClaim:
    claim: EntryClaim
    reason: str


@dataclass(slots=True, frozen=True)
class EntryPriceBand:
    signal_time: datetime
    base_time: datetime
    base_price: float
    best_price: float
    worst_price: float
    window_end: datetime


@dataclass(slots=True, frozen=True)
class FillResult:
    margin_used: float
    notional_value: float
    quantity: float


@dataclass(slots=True, frozen=True)
class Position:
    trade_id: str
    bot_name: str
    strategy: str
    symbol: str
    side: Side
    entry_time: datetime
    entry_price: float
    quantity: float
    leverage: float
    tp_percent: float
    sl_percent: float
    tp_price: float
    sl_price: float
    margin_used: float = 0.0
    notional_value: float = 0.0
    notes: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ExitResolution:
    reason: ExitReason
    exit_time: datetime
    exit_price: float
    resolution_level: ResolutionLevel


@dataclass(slots=True, frozen=True)
class CompletedTrade:
    position: Position
    exit: ExitResolution
    pnl_percent: float


@dataclass(slots=True, frozen=True)
class ScheduledTrade:
    position: Position
    entry_band: EntryPriceBand
    exit: ExitResolution
    pnl_percent: float
    signal_time: datetime


@dataclass(slots=True, frozen=True)
class StrategyContext:
    symbol: str
    candles: list[Candle]
    cycle_stats: CycleStats | None
    tp_percent: float


@dataclass(slots=True, frozen=True)
class BacktestMetrics:
    net_profit_pct: float
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    expectancy_pct: float
    profit_factor: float
    max_drawdown_pct: float
    final_balance: float


@dataclass(slots=True, frozen=True)
class BotBacktestResult:
    bot_name: str
    starting_balance: float
    metrics: BacktestMetrics
    trades: tuple[ScheduledTrade, ...]
    rejection_counts: dict[str, int]


@dataclass(slots=True, frozen=True)
class PortfolioBacktestResult:
    start: datetime
    end: datetime
    metrics: BacktestMetrics
    bots: tuple[BotBacktestResult, ...]
    collision_policy: CollisionPolicy


@dataclass(slots=True, frozen=True)
class WalkForwardGrid:
    bot_name: str
    parameters: dict[str, tuple[object, ...]]


@dataclass(slots=True, frozen=True)
class WalkForwardWindow:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass(slots=True, frozen=True)
class WalkForwardWindowResult:
    window: WalkForwardWindow
    selected_parameters: dict[str, dict[str, object]]
    train_result: PortfolioBacktestResult
    test_result: PortfolioBacktestResult


@dataclass(slots=True, frozen=True)
class WalkForwardReport:
    windows: tuple[WalkForwardWindowResult, ...]
    aggregate_test_metrics: BacktestMetrics
