from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import Backtester
from .config import load_config
from .market_data import InsufficientMarketDataError, MarketDataRepository
from .outputs import write_backtest_outputs, write_walk_forward_outputs
from .walkforward import run_walk_forward


def main() -> None:
    parser = argparse.ArgumentParser(prog="kriptistan-backtest")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-config", help="Load and print a config summary.")
    inspect_parser.add_argument("config", type=Path)

    backtest_parser = subparsers.add_parser("backtest", help="Run a single backtest.")
    backtest_parser.add_argument("config", type=Path)
    backtest_parser.add_argument("--symbols", nargs="*")

    wf_parser = subparsers.add_parser("walk-forward", help="Run walk-forward research.")
    wf_parser.add_argument("config", type=Path)
    wf_parser.add_argument("--symbols", nargs="*")

    args = parser.parse_args()
    if args.command == "inspect-config":
        config = load_config(args.config)
        print(f"period={config.backtest.start.isoformat()}..{config.backtest.end.isoformat()}")
        print(f"bots={len(config.bots)} collision_policy={config.backtest.collision_policy}")
        return
    if args.command == "backtest":
        config = load_config(args.config)
        repo = MarketDataRepository(config)
        try:
            bundle = repo.build_bundle(symbols=args.symbols)
        except InsufficientMarketDataError as exc:
            raise SystemExit(str(exc)) from exc
        result = Backtester(config, bundle).run()
        run_dir = write_backtest_outputs(result=result, config_path=args.config)
        print(f"net_profit_pct={result.metrics.net_profit_pct:.2f}")
        print(f"trades={result.metrics.total_trades} win_rate={result.metrics.win_rate:.2f}")
        print(f"saved_to={run_dir}")
        return
    if args.command == "walk-forward":
        config = load_config(args.config)
        repo = MarketDataRepository(config)
        warmup_days = max(config.backtest.warmup_days, config.walk_forward.warmup_days)
        try:
            bundle = repo.build_bundle(symbols=args.symbols, warmup_days=warmup_days)
        except InsufficientMarketDataError as exc:
            raise SystemExit(str(exc)) from exc
        report = run_walk_forward(config, bundle)
        run_dir = write_walk_forward_outputs(report=report, config_path=args.config)
        print(f"windows={len(report.windows)} net_profit_pct={report.aggregate_test_metrics.net_profit_pct:.2f}")
        print(f"trades={report.aggregate_test_metrics.total_trades} win_rate={report.aggregate_test_metrics.win_rate:.2f}")
        print(f"saved_to={run_dir}")
        return
    raise SystemExit(2)


if __name__ == "__main__":
    main()
