from datetime import UTC, datetime
from pathlib import Path

from kriptistan.models import BacktestMetrics, BotBacktestResult, CollisionPolicy, EntryPriceBand, ExitReason, ExitResolution, PortfolioBacktestResult, Position, ResolutionLevel, ScheduledTrade, Side, WalkForwardReport, WalkForwardWindow, WalkForwardWindowResult
from kriptistan.outputs import write_backtest_outputs, write_walk_forward_outputs


def _trade() -> ScheduledTrade:
    entry_time = datetime(2026, 3, 1, 0, 0, 5, tzinfo=UTC)
    position = Position(
        trade_id="t1",
        bot_name="trend_mom",
        strategy="TREND_MOM",
        symbol="ETHUSDT",
        side=Side.LONG,
        entry_time=entry_time,
        entry_price=100.0,
        quantity=1.0,
        leverage=2.0,
        tp_percent=1.5,
        sl_percent=1.0,
        tp_price=101.6,
        sl_price=99.1,
        margin_used=50.0,
        notional_value=100.0,
    )
    return ScheduledTrade(
        position=position,
        entry_band=EntryPriceBand(
            signal_time=datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC),
            base_time=entry_time,
            base_price=100.0,
            best_price=99.9,
            worst_price=100.1,
            window_end=datetime(2026, 3, 1, 0, 0, 15, tzinfo=UTC),
        ),
        exit=ExitResolution(
            reason=ExitReason.TP,
            exit_time=datetime(2026, 3, 1, 2, 0, 0, tzinfo=UTC),
            exit_price=101.6,
            resolution_level=ResolutionLevel.TRADE,
        ),
        pnl_percent=2.9,
        signal_time=datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC),
    )


def _portfolio() -> PortfolioBacktestResult:
    metrics = BacktestMetrics(
        net_profit_pct=2.9,
        total_trades=1,
        wins=1,
        losses=0,
        win_rate=100.0,
        expectancy_pct=2.9,
        profit_factor=float("inf"),
        max_drawdown_pct=0.0,
        final_balance=1029.0,
    )
    bot = BotBacktestResult(
        bot_name="trend_mom",
        starting_balance=1000.0,
        metrics=metrics,
        trades=(_trade(),),
        rejection_counts={"btc_confirm": 3},
    )
    return PortfolioBacktestResult(
        start=datetime(2026, 3, 1, tzinfo=UTC),
        end=datetime(2026, 3, 8, tzinfo=UTC),
        metrics=metrics,
        bots=(bot,),
        collision_policy=CollisionPolicy.SEEDED_SHUFFLE,
    )


def test_write_backtest_outputs_persists_summary_and_trades(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[backtest]\nstart = 2026-03-01T00:00:00Z\nend = 2026-03-08T00:00:00Z\n")

    run_dir = write_backtest_outputs(result=_portfolio(), config_path=config_path, root=tmp_path / "outputs")

    assert (run_dir / "summary.json").exists()
    assert (run_dir / "trades.csv").exists()
    assert (run_dir / "config.toml").exists()


def test_write_walk_forward_outputs_persists_windows_and_trades(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[backtest]\nstart = 2026-03-01T00:00:00Z\nend = 2026-03-08T00:00:00Z\n")
    portfolio = _portfolio()
    report = WalkForwardReport(
        windows=(
            WalkForwardWindowResult(
                window=WalkForwardWindow(
                    train_start=datetime(2026, 2, 15, tzinfo=UTC),
                    train_end=datetime(2026, 3, 1, tzinfo=UTC),
                    test_start=datetime(2026, 3, 1, tzinfo=UTC),
                    test_end=datetime(2026, 3, 8, tzinfo=UTC),
                ),
                selected_parameters={"trend_mom": {"tp_percent": 1.5}},
                train_result=portfolio,
                test_result=portfolio,
            ),
        ),
        aggregate_test_metrics=portfolio.metrics,
    )

    run_dir = write_walk_forward_outputs(report=report, config_path=config_path, root=tmp_path / "outputs")

    assert (run_dir / "summary.json").exists()
    assert (run_dir / "windows.json").exists()
    assert (run_dir / "trades.csv").exists()
