from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import tomllib

from .models import CapitalMode, CollisionPolicy, WalkForwardGrid


@dataclass(slots=True, frozen=True)
class ScannerParamSet:
    percent_limit: float
    stdev_limit: float
    warmup_days: int


_DEFAULT_SCANNER_PARAM_SETS: tuple[ScannerParamSet, ...] = (
    ScannerParamSet(percent_limit=10.0, stdev_limit=12.0, warmup_days=365),
    ScannerParamSet(percent_limit=8.0, stdev_limit=12.0, warmup_days=500),
)


@dataclass(slots=True, frozen=True)
class BacktestConfig:
    start: datetime
    end: datetime
    signal_timeframe: str = "1h"
    scanner_timeframe: str = "1d"
    entry_delay_seconds: int = 5
    entry_window_seconds: int = 15
    warmup_days: int = 365
    starting_balance: float = 1_000.0
    capital_mode: CapitalMode = CapitalMode.PER_BOT
    top_candidates: int = 50
    collision_policy: CollisionPolicy = CollisionPolicy.SEEDED_SHUFFLE
    shuffle_seed: int = 7
    exact_mode: bool = True
    exact_mode_max_history_days: int = 365
    use_closed_candles_only: bool = True
    use_previous_day_fng: bool = True
    reverse_mode: bool = False
    scanner_param_sets: tuple[ScannerParamSet, ...] = _DEFAULT_SCANNER_PARAM_SETS

    @property
    def scanner_warmup_days(self) -> int:
        return max(s.warmup_days for s in self.scanner_param_sets)


@dataclass(slots=True, frozen=True)
class ExecutionConfig:
    taker_fee_rate: float = 0.0005
    btc_vol_guard_enabled: bool = True
    btc_vol_max_range_pct: float = 2.0
    btc_vol_max_candle_pct: float = 0.5
    btc_vol_cooldown_seconds: int = 600
    chase_filter_enabled: bool = True
    chase_close_pos: float = 0.85
    chase_range_mult_tp: float = 1.0
    btc_confirm_entry_enabled: bool = True


@dataclass(slots=True, frozen=True)
class BotConfig:
    name: str
    strategy: str
    max_open_trades: int = 3
    leverage: float = 1.0
    tp_percent: float = 1.5
    sl_percent: float = 1.0
    min_24h_volume: float = 2_000_000
    due_date_min: float = -5.0
    due_date_max: float = 5.0
    same_ticker_cooldown_hours: int = 1
    dead_end_max_consecutive_losses: int = 2
    dead_end_lookback_days: int = 7
    dead_end_blacklist_days: int = 3
    fng_enabled: bool = True
    risk_aversion: float = 1.0
    reverse_mode: bool = False
    fixed_priority: int = 0
    symbol_whitelist: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class WalkForwardConfig:
    enabled: bool = False
    train_days: int = 180
    test_days: int = 30
    step_days: int = 30
    warmup_days: int = 365
    objective_metric: str = "net_profit_pct"
    max_combinations: int = 64
    bot_grids: tuple[WalkForwardGrid, ...] = ()


@dataclass(slots=True, frozen=True)
class AppConfig:
    backtest: BacktestConfig
    execution: ExecutionConfig
    bots: tuple[BotConfig, ...]
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)


def load_config(path: str | Path) -> AppConfig:
    raw = tomllib.loads(Path(path).read_text())
    backtest_raw = raw["backtest"]
    execution_raw = raw.get("execution", {})
    bots_raw = raw.get("bots", [])
    walk_forward_raw = raw.get("walk_forward", {})

    scanner_sets_raw = backtest_raw.get("scanner_param_sets")
    if scanner_sets_raw is not None:
        scanner_param_sets = tuple(
            ScannerParamSet(
                percent_limit=item["percent_limit"],
                stdev_limit=item["stdev_limit"],
                warmup_days=item["warmup_days"],
            )
            for item in scanner_sets_raw
        )
    else:
        scanner_param_sets = _DEFAULT_SCANNER_PARAM_SETS
    backtest = BacktestConfig(
        start=_normalize_datetime(backtest_raw["start"]),
        end=_normalize_datetime(backtest_raw["end"]),
        signal_timeframe=backtest_raw.get("signal_timeframe", "1h"),
        scanner_timeframe=backtest_raw.get("scanner_timeframe", "1d"),
        entry_delay_seconds=backtest_raw.get("entry_delay_seconds", 5),
        entry_window_seconds=backtest_raw.get("entry_window_seconds", 15),
        warmup_days=backtest_raw.get("warmup_days", 250),
        starting_balance=backtest_raw.get("starting_balance", 1_000.0),
        capital_mode=CapitalMode(backtest_raw.get("capital_mode", "per_bot")),
        top_candidates=backtest_raw.get("top_candidates", 50),
        collision_policy=CollisionPolicy(backtest_raw.get("collision_policy", "seeded_shuffle")),
        shuffle_seed=backtest_raw.get("shuffle_seed", 7),
        exact_mode=backtest_raw.get("exact_mode", True),
        exact_mode_max_history_days=backtest_raw.get("exact_mode_max_history_days", 365),
        use_closed_candles_only=backtest_raw.get("use_closed_candles_only", True),
        use_previous_day_fng=backtest_raw.get("use_previous_day_fng", True),
        reverse_mode=backtest_raw.get("reverse_mode", False),
        scanner_param_sets=scanner_param_sets,
    )
    execution = ExecutionConfig(
        taker_fee_rate=execution_raw.get("taker_fee_rate", 0.0005),
        btc_vol_guard_enabled=execution_raw.get("btc_vol_guard_enabled", True),
        btc_vol_max_range_pct=execution_raw.get("btc_vol_max_range_pct", 2.0),
        btc_vol_max_candle_pct=execution_raw.get("btc_vol_max_candle_pct", 0.5),
        btc_vol_cooldown_seconds=execution_raw.get("btc_vol_cooldown_seconds", 600),
        chase_filter_enabled=execution_raw.get("chase_filter_enabled", True),
        chase_close_pos=execution_raw.get("chase_close_pos", 0.85),
        chase_range_mult_tp=execution_raw.get("chase_range_mult_tp", 1.0),
        btc_confirm_entry_enabled=execution_raw.get("btc_confirm_entry_enabled", True),
    )
    bots = tuple(
        BotConfig(
            name=item["name"],
            strategy=item["strategy"],
            max_open_trades=item.get("max_open_trades", 3),
            leverage=item.get("leverage", 1.0),
            tp_percent=item.get("tp_percent", 1.5),
            sl_percent=item.get("sl_percent", 1.0),
            min_24h_volume=item.get("min_24h_volume", 2_000_000),
            due_date_min=item.get("due_date_min", -5.0),
            due_date_max=item.get("due_date_max", 5.0),
            same_ticker_cooldown_hours=item.get("same_ticker_cooldown_hours", 1),
            dead_end_max_consecutive_losses=item.get("dead_end_max_consecutive_losses", 2),
            dead_end_lookback_days=item.get("dead_end_lookback_days", 7),
            dead_end_blacklist_days=item.get("dead_end_blacklist_days", 3),
            fng_enabled=item.get("fng_enabled", True),
            risk_aversion=item.get("risk_aversion", 1.0),
            reverse_mode=item.get("reverse_mode", False),
            fixed_priority=item.get("fixed_priority", 0),
            symbol_whitelist=tuple(item.get("symbol_whitelist", [])),
        )
        for item in bots_raw
    )
    if backtest.reverse_mode and any(bot.reverse_mode for bot in bots):
        raise ValueError(
            "Config mixes global [backtest].reverse_mode=true with per-bot reverse_mode=true. "
            "Choose either the global switch or per-bot switches."
        )
    grids = tuple(
        WalkForwardGrid(
            bot_name=item["bot_name"],
            parameters={key: tuple(value) for key, value in item.get("parameters", {}).items()},
        )
        for item in walk_forward_raw.get("bot_grids", [])
    )
    walk_forward = WalkForwardConfig(
        enabled=walk_forward_raw.get("enabled", False),
        train_days=walk_forward_raw.get("train_days", 180),
        test_days=walk_forward_raw.get("test_days", 30),
        step_days=walk_forward_raw.get("step_days", 30),
        warmup_days=walk_forward_raw.get("warmup_days", backtest.warmup_days),
        objective_metric=walk_forward_raw.get("objective_metric", "net_profit_pct"),
        max_combinations=walk_forward_raw.get("max_combinations", 64),
        bot_grids=grids,
    )
    return AppConfig(backtest=backtest, execution=execution, bots=bots, walk_forward=walk_forward)


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
