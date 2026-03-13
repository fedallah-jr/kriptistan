from datetime import UTC, datetime

from kriptistan.config import AppConfig, BacktestConfig, BotConfig, ExecutionConfig, WalkForwardConfig
from kriptistan.models import WalkForwardGrid
from kriptistan.walkforward import build_override_variants, iterate_windows


def test_iterate_windows_builds_time_ordered_train_test_slices() -> None:
    config = AppConfig(
        backtest=BacktestConfig(
            start=datetime(2025, 1, 1, tzinfo=UTC),
            end=datetime(2025, 7, 1, tzinfo=UTC),
        ),
        execution=ExecutionConfig(),
        bots=(BotConfig(name="trend", strategy="TREND_MOM"),),
        walk_forward=WalkForwardConfig(enabled=True, train_days=60, test_days=30, step_days=30),
    )

    windows = iterate_windows(config)

    assert windows
    assert windows[0].train_start == datetime(2025, 1, 1, tzinfo=UTC)
    assert windows[0].test_start > windows[0].train_start
    assert windows[-1].test_end <= datetime(2025, 7, 1, tzinfo=UTC)


def test_build_override_variants_expands_cartesian_product_per_bot_grid() -> None:
    config = AppConfig(
        backtest=BacktestConfig(
            start=datetime(2025, 1, 1, tzinfo=UTC),
            end=datetime(2025, 7, 1, tzinfo=UTC),
        ),
        execution=ExecutionConfig(),
        bots=(BotConfig(name="trend", strategy="TREND_MOM"),),
        walk_forward=WalkForwardConfig(
            enabled=True,
            max_combinations=16,
            bot_grids=(
                WalkForwardGrid(
                    bot_name="trend",
                    parameters={
                        "tp_percent": (1.0, 1.5),
                        "sl_percent": (0.8, 1.0),
                    },
                ),
            ),
        ),
    )

    variants = build_override_variants(config)

    assert len(variants) == 4
    assert {"tp_percent": 1.0, "sl_percent": 0.8} in variants[0].values() or {"tp_percent": 1.0, "sl_percent": 0.8} in variants[-1].values()
