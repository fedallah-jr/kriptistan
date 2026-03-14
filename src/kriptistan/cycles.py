from __future__ import annotations

from statistics import mean, median, pstdev

from .models import Candle, CycleStats


def scan_symbol_cycles(
    symbol: str,
    daily_candles: list[Candle],
    *,
    percent_limit: float = 8.0,
    stdev_limit: float = 8.0,
) -> CycleStats | None:
    candles = daily_candles
    if len(candles) < 3:
        return None
    pump_events = [candle.open_time for candle in candles if _pump_change_pct(candle) > percent_limit]
    dump_events = [candle.open_time for candle in candles if _dump_change_pct(candle) < -percent_limit]
    as_of = candles[-1].close_time
    pump = _event_stats(pump_events, as_of)
    dump = _event_stats(dump_events, as_of)
    pump_stdev = pump["stdev"]
    dump_stdev = dump["stdev"]
    passes = any(
        value is not None and value <= stdev_limit
        for value in (pump_stdev, dump_stdev)
    )
    return CycleStats(
        as_of=as_of,
        symbol=symbol,
        pump_mean_days=pump["mean"],
        pump_median_days=pump["median"],
        pump_stdev_days=pump_stdev,
        pump_last_interval_days=pump["last_interval"],
        pump_date_due_days=pump["date_due"],
        dump_mean_days=dump["mean"],
        dump_median_days=dump["median"],
        dump_stdev_days=dump_stdev,
        dump_last_interval_days=dump["last_interval"],
        dump_date_due_days=dump["date_due"],
        passes_stdev_filter=passes,
    )


def shared_market_scan(
    daily_series: dict[str, list[Candle]],
    *,
    percent_limit: float = 8.0,
    stdev_limit: float = 8.0,
) -> list[CycleStats]:
    results: list[CycleStats] = []
    for symbol, candles in daily_series.items():
        stats = scan_symbol_cycles(
            symbol,
            candles,
            percent_limit=percent_limit,
            stdev_limit=stdev_limit,
        )
        if stats is not None and stats.passes_stdev_filter:
            results.append(stats)
    return sorted(
        results,
        key=lambda item: min(
            abs(item.pump_date_due_days) if item.pump_date_due_days is not None else float("inf"),
            abs(item.dump_date_due_days) if item.dump_date_due_days is not None else float("inf"),
        ),
    )


def _event_stats(event_times: list, as_of) -> dict[str, float | None]:
    if len(event_times) < 2:
        return {
            "mean": None,
            "median": None,
            "stdev": None,
            "last_interval": None,
            "date_due": None,
        }
    intervals = [
        (event_times[idx] - event_times[idx - 1]).days
        for idx in range(1, len(event_times))
    ]
    last_interval = (as_of - event_times[-1]).days
    mean_value = float(mean(intervals))
    return {
        "mean": mean_value,
        "median": float(median(intervals)),
        "stdev": float(pstdev(intervals)),
        "last_interval": float(last_interval),
        "date_due": mean_value - float(last_interval),
    }


def _pump_change_pct(candle: Candle) -> float:
    return ((candle.high - candle.open) / candle.open) * 100 if candle.open else 0.0


def _dump_change_pct(candle: Candle) -> float:
    return ((candle.low - candle.open) / candle.open) * 100 if candle.open else 0.0
