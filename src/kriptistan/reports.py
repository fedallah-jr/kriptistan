from __future__ import annotations

from dataclasses import dataclass, field

from .models import BacktestMetrics, BotBacktestResult, PortfolioBacktestResult, ScheduledTrade


@dataclass(slots=True)
class BotLedger:
    bot_name: str
    starting_balance: float
    current_balance: float
    trades: list[ScheduledTrade] = field(default_factory=list)
    rejection_counts: dict[str, int] = field(default_factory=dict)
    equity_points: list[float] = field(default_factory=list)

    def reject(self, reason: str) -> None:
        self.rejection_counts[reason] = self.rejection_counts.get(reason, 0) + 1


def build_bot_result(ledger: BotLedger) -> BotBacktestResult:
    metrics = compute_metrics(ledger.starting_balance, ledger.current_balance, ledger.trades, ledger.equity_points)
    return BotBacktestResult(
        bot_name=ledger.bot_name,
        starting_balance=ledger.starting_balance,
        metrics=metrics,
        trades=tuple(ledger.trades),
        rejection_counts=dict(sorted(ledger.rejection_counts.items())),
    )


def build_portfolio_result(
    *,
    start,
    end,
    ledgers: list[BotLedger],
    collision_policy,
) -> PortfolioBacktestResult:
    bot_results = tuple(build_bot_result(ledger) for ledger in ledgers)
    starting_balance = sum(result.starting_balance for result in bot_results)
    final_balance = sum(result.metrics.final_balance for result in bot_results)
    all_trades = [trade for result in bot_results for trade in result.trades]
    combined_equity = [sum(values) for values in zip(*_normalize_equity_points([ledger.equity_points for ledger in ledgers], starting_balance))]
    metrics = compute_metrics(starting_balance, final_balance, all_trades, combined_equity)
    return PortfolioBacktestResult(
        start=start,
        end=end,
        metrics=metrics,
        bots=bot_results,
        collision_policy=collision_policy,
    )


def compute_metrics(
    starting_balance: float,
    final_balance: float,
    trades: list[ScheduledTrade],
    equity_points: list[float],
) -> BacktestMetrics:
    wins = sum(1 for trade in trades if trade.pnl_percent > 0)
    losses = sum(1 for trade in trades if trade.pnl_percent <= 0)
    total_trades = len(trades)
    win_rate = (wins / total_trades) * 100 if total_trades else 0.0
    expectancy = sum(trade.pnl_percent for trade in trades) / total_trades if total_trades else 0.0
    gross_profit = sum(trade.pnl_percent for trade in trades if trade.pnl_percent > 0)
    gross_loss = abs(sum(trade.pnl_percent for trade in trades if trade.pnl_percent < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss else float("inf") if gross_profit > 0 else 0.0
    net_profit_pct = ((final_balance - starting_balance) / starting_balance) * 100 if starting_balance else 0.0
    return BacktestMetrics(
        net_profit_pct=net_profit_pct,
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        expectancy_pct=expectancy,
        profit_factor=profit_factor,
        max_drawdown_pct=_max_drawdown(equity_points or [starting_balance, final_balance]),
        final_balance=final_balance,
    )


def metric_value(result: PortfolioBacktestResult, objective: str) -> float:
    metrics = result.metrics
    if objective == "expectancy_pct":
        return metrics.expectancy_pct
    if objective == "profit_factor":
        return metrics.profit_factor
    if objective == "win_rate":
        return metrics.win_rate
    return metrics.net_profit_pct


def _max_drawdown(equity_points: list[float]) -> float:
    peak = None
    drawdown = 0.0
    for value in equity_points:
        peak = value if peak is None else max(peak, value)
        if peak:
            drawdown = max(drawdown, ((peak - value) / peak) * 100)
    return drawdown


def _normalize_equity_points(series_list: list[list[float]], starting_balance: float) -> list[list[float]]:
    max_length = max((len(series) for series in series_list), default=0)
    if max_length == 0:
        return [[starting_balance] for _ in series_list]
    normalized: list[list[float]] = []
    for series in series_list:
        if not series:
            normalized.append([starting_balance] * max_length)
            continue
        extension = [series[-1]] * (max_length - len(series))
        normalized.append(series + extension)
    return normalized
