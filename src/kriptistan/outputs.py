from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
import math
from pathlib import Path
import shutil

from .models import BacktestMetrics, BotBacktestResult, PortfolioBacktestResult, ScheduledTrade, WalkForwardReport


def write_backtest_outputs(
    *,
    result: PortfolioBacktestResult,
    config_path: Path,
    root: Path = Path("outputs"),
) -> Path:
    run_dir = _create_run_dir(root=root, category="backtests", config_path=config_path)
    _copy_config(config_path, run_dir)
    _write_json(run_dir / "summary.json", _serialize_portfolio_result(result))
    _write_trades_csv(run_dir / "trades.csv", _iter_portfolio_trades(result))
    return run_dir


def write_walk_forward_outputs(
    *,
    report: WalkForwardReport,
    config_path: Path,
    root: Path = Path("outputs"),
) -> Path:
    run_dir = _create_run_dir(root=root, category="walk_forward", config_path=config_path)
    _copy_config(config_path, run_dir)
    summary = {
        "window_count": len(report.windows),
        "aggregate_test_metrics": _serialize_metrics(report.aggregate_test_metrics),
    }
    windows = []
    trades = []
    for index, window in enumerate(report.windows):
        windows.append(
            {
                "index": index,
                "train_start": window.window.train_start.isoformat(),
                "train_end": window.window.train_end.isoformat(),
                "test_start": window.window.test_start.isoformat(),
                "test_end": window.window.test_end.isoformat(),
                "selected_parameters": window.selected_parameters,
                "train_metrics": _serialize_metrics(window.train_result.metrics),
                "test_metrics": _serialize_metrics(window.test_result.metrics),
            }
        )
        trades.extend(_iter_portfolio_trades(window.train_result, phase="train", window_index=index))
        trades.extend(_iter_portfolio_trades(window.test_result, phase="test", window_index=index))
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "windows.json", windows)
    _write_trades_csv(run_dir / "trades.csv", trades)
    return run_dir


def _serialize_portfolio_result(result: PortfolioBacktestResult) -> dict[str, object]:
    return {
        "start": result.start.isoformat(),
        "end": result.end.isoformat(),
        "collision_policy": result.collision_policy.value,
        "metrics": _serialize_metrics(result.metrics),
        "bots": [_serialize_bot_result(bot) for bot in result.bots],
    }


def _serialize_bot_result(result: BotBacktestResult) -> dict[str, object]:
    return {
        "bot_name": result.bot_name,
        "starting_balance": result.starting_balance,
        "metrics": _serialize_metrics(result.metrics),
        "rejection_counts": result.rejection_counts,
        "trade_count": len(result.trades),
    }


def _serialize_metrics(metrics: BacktestMetrics) -> dict[str, float | int]:
    return {
        "net_profit_pct": _json_number(metrics.net_profit_pct),
        "total_trades": metrics.total_trades,
        "wins": metrics.wins,
        "losses": metrics.losses,
        "win_rate": _json_number(metrics.win_rate),
        "expectancy_pct": _json_number(metrics.expectancy_pct),
        "profit_factor": _json_number(metrics.profit_factor),
        "max_drawdown_pct": _json_number(metrics.max_drawdown_pct),
        "final_balance": _json_number(metrics.final_balance),
    }


def _iter_portfolio_trades(
    result: PortfolioBacktestResult,
    *,
    phase: str = "backtest",
    window_index: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bot in result.bots:
        for trade in bot.trades:
            rows.append(_trade_row(trade, phase=phase, window_index=window_index))
    return rows


def _trade_row(trade: ScheduledTrade, *, phase: str, window_index: int | None) -> dict[str, object]:
    position = trade.position
    exit_ = trade.exit
    return {
        "phase": phase,
        "window_index": "" if window_index is None else window_index,
        "bot_name": position.bot_name,
        "strategy": position.strategy,
        "symbol": position.symbol,
        "side": position.side.value,
        "signal_time": trade.signal_time.isoformat(),
        "entry_time": position.entry_time.isoformat(),
        "entry_price": position.entry_price,
        "entry_best_price": trade.entry_band.best_price,
        "entry_worst_price": trade.entry_band.worst_price,
        "quantity": position.quantity,
        "leverage": position.leverage,
        "tp_percent": position.tp_percent,
        "sl_percent": position.sl_percent,
        "tp_price": position.tp_price,
        "sl_price": position.sl_price,
        "exit_time": exit_.exit_time.isoformat(),
        "exit_price": exit_.exit_price,
        "exit_reason": exit_.reason.value,
        "resolution_level": exit_.resolution_level.value,
        "margin_used": position.margin_used,
        "notional_value": position.notional_value,
        "pnl_percent": trade.pnl_percent,
    }


def _write_trades_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "phase",
        "window_index",
        "bot_name",
        "strategy",
        "symbol",
        "side",
        "signal_time",
        "entry_time",
        "entry_price",
        "entry_best_price",
        "entry_worst_price",
        "quantity",
        "leverage",
        "tp_percent",
        "sl_percent",
        "tp_price",
        "sl_price",
        "exit_time",
        "exit_price",
        "exit_reason",
        "resolution_level",
        "margin_used",
        "notional_value",
        "pnl_percent",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _copy_config(config_path: Path, run_dir: Path) -> None:
    shutil.copy2(config_path, run_dir / config_path.name)


def _create_run_dir(*, root: Path, category: str, config_path: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = root / category / f"{stamp}_{config_path.stem}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _json_number(value: float | int) -> float | int | None:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
