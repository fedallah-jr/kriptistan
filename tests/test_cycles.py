from datetime import UTC, datetime, timedelta

from kriptistan.cycles import scan_symbol_cycles
from kriptistan.models import Candle


def _daily(open_time: datetime, open_price: float, high: float, low: float, close: float) -> Candle:
    return Candle(
        open_time=open_time,
        close_time=open_time + timedelta(days=1),
        open=open_price,
        high=high,
        low=low,
        close=close,
    )


def test_cycle_scanner_computes_due_dates_from_closed_daily_candles() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles = [
        _daily(start + timedelta(days=0), 100, 110, 99, 108),
        _daily(start + timedelta(days=1), 100, 102, 96, 101),
        _daily(start + timedelta(days=2), 100, 109, 98, 108),
        _daily(start + timedelta(days=3), 100, 102, 95, 97),
        _daily(start + timedelta(days=4), 100, 111, 99, 110),
    ]
    stats = scan_symbol_cycles("ETHUSDT", candles, percent_limit=8.0, stdev_limit=8.0)

    assert stats is not None
    assert stats.pump_mean_days == 2.0
    assert stats.pump_last_interval_days == 1.0
    assert stats.pump_date_due_days == 1.0
    assert stats.passes_stdev_filter is True
