from __future__ import annotations

from datetime import timedelta
from itertools import islice, product

from .backtest import Backtester, apply_bot_overrides
from .config import AppConfig
from .models import BacktestMetrics, WalkForwardReport, WalkForwardWindow, WalkForwardWindowResult
from .reports import metric_value


def iterate_windows(config: AppConfig) -> list[WalkForwardWindow]:
    wf = config.walk_forward
    train_delta = timedelta(days=wf.train_days)
    test_delta = timedelta(days=wf.test_days)
    step_delta = timedelta(days=wf.step_days)
    cursor = config.backtest.start
    windows: list[WalkForwardWindow] = []
    while True:
        train_start = cursor
        train_end = train_start + train_delta
        test_start = train_end
        test_end = test_start + test_delta
        if test_end > config.backtest.end:
            break
        windows.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        cursor += step_delta
    return windows


def build_override_variants(config: AppConfig) -> list[dict[str, dict[str, object]]]:
    grids = config.walk_forward.bot_grids
    if not grids:
        return [{}]
    per_bot_variants: list[list[tuple[str, dict[str, object]]]] = []
    for grid in grids:
        keys = sorted(grid.parameters)
        value_lists = [grid.parameters[key] for key in keys]
        overrides = []
        for combo in product(*value_lists):
            overrides.append((grid.bot_name, {key: value for key, value in zip(keys, combo)}))
        per_bot_variants.append(overrides or [(grid.bot_name, {})])

    combinations = []
    for combo in islice(product(*per_bot_variants), config.walk_forward.max_combinations):
        merged: dict[str, dict[str, object]] = {}
        for bot_name, override in combo:
            merged[bot_name] = override
        combinations.append(merged)
    return combinations or [{}]


def run_walk_forward(config: AppConfig, bundle) -> WalkForwardReport:
    backtester = Backtester(config, bundle)
    windows = iterate_windows(config)
    override_variants = build_override_variants(config)
    results: list[WalkForwardWindowResult] = []

    for window in windows:
        best_train = None
        best_overrides: dict[str, dict[str, object]] = {}
        for overrides in override_variants:
            bots = apply_bot_overrides(config.bots, overrides)
            train_result = backtester.run(start=window.train_start, end=window.train_end, bots=bots)
            score = metric_value(train_result, config.walk_forward.objective_metric)
            if best_train is None or score > best_train[0]:
                best_train = (score, train_result)
                best_overrides = overrides
        selected_bots = apply_bot_overrides(config.bots, best_overrides)
        test_result = backtester.run(start=window.test_start, end=window.test_end, bots=selected_bots)
        assert best_train is not None
        results.append(
            WalkForwardWindowResult(
                window=window,
                selected_parameters=best_overrides,
                train_result=best_train[1],
                test_result=test_result,
            )
        )

    aggregate = _aggregate_test_metrics(results, config.backtest.starting_balance, len(config.bots))
    return WalkForwardReport(windows=tuple(results), aggregate_test_metrics=aggregate)


def _aggregate_test_metrics(results: list[WalkForwardWindowResult], starting_balance: float, bot_count: int) -> BacktestMetrics:
    base_balance = starting_balance * bot_count
    if not results:
        return BacktestMetrics(
            net_profit_pct=0.0,
            total_trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            expectancy_pct=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            final_balance=base_balance,
        )
    final_balance = base_balance
    for window in results:
        final_balance *= 1 + (window.test_result.metrics.net_profit_pct / 100)
    total_trades = sum(window.test_result.metrics.total_trades for window in results)
    wins = sum(window.test_result.metrics.wins for window in results)
    losses = sum(window.test_result.metrics.losses for window in results)
    gross_expectancy = sum(window.test_result.metrics.expectancy_pct * window.test_result.metrics.total_trades for window in results)
    expectancy = gross_expectancy / total_trades if total_trades else 0.0
    gross_profit = sum(
        sum(trade.pnl_percent for bot in window.test_result.bots for trade in bot.trades if trade.pnl_percent > 0)
        for window in results
    )
    gross_loss = abs(
        sum(
            sum(trade.pnl_percent for bot in window.test_result.bots for trade in bot.trades if trade.pnl_percent < 0)
            for window in results
        )
    )
    profit_factor = (gross_profit / gross_loss) if gross_loss else float("inf") if gross_profit > 0 else 0.0
    win_rate = (wins / total_trades) * 100 if total_trades else 0.0
    net_profit_pct = ((final_balance - base_balance) / base_balance) * 100 if bot_count else 0.0
    return BacktestMetrics(
        net_profit_pct=net_profit_pct,
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        expectancy_pct=expectancy,
        profit_factor=profit_factor,
        max_drawdown_pct=max(window.test_result.metrics.max_drawdown_pct for window in results),
        final_balance=final_balance,
    )
