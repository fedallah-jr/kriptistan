from datetime import UTC, datetime, timedelta

from kriptistan.execution import compute_effective_tp_sl, resolve_exit_hierarchical, select_entry_price_band
from kriptistan.models import AggTrade, Candle, ExitReason, Position, ResolutionLevel, Side


def _trade(idx: int, second: int, price: float) -> AggTrade:
    start = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
    return AggTrade(trade_id=idx, timestamp=start + timedelta(seconds=second), price=price, quantity=1.0)


def _candle(start: datetime, minutes: int, open_: float, high: float, low: float, close: float) -> Candle:
    return Candle(
        open_time=start,
        close_time=start + timedelta(minutes=minutes),
        open=open_,
        high=high,
        low=low,
        close=close,
    )


def test_entry_band_uses_first_trade_after_delay_and_directional_extremes() -> None:
    signal_time = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
    band = select_entry_price_band(
        [_trade(1, 1, 100), _trade(2, 5, 101), _trade(3, 9, 99)],
        side=Side.LONG,
        signal_time=signal_time,
        entry_delay_seconds=5,
        entry_window_seconds=15,
    )
    assert band.base_price == 101
    assert band.best_price == 99
    assert band.worst_price == 101


def test_hierarchical_exit_descends_into_aggregate_trades_for_ambiguous_minute() -> None:
    entry_time = datetime(2026, 1, 1, 10, 0, 5, tzinfo=UTC)
    _, _, tp_price, sl_price = compute_effective_tp_sl(
        entry_price=100,
        side=Side.LONG,
        tp_percent=1.5,
        sl_percent=1.0,
        taker_fee_rate=0.0005,
        fng_value=None,
    )
    position = Position(
        trade_id="t1",
        bot_name="bot",
        strategy="TREND_MOM",
        symbol="ETHUSDT",
        side=Side.LONG,
        entry_time=entry_time,
        entry_price=100,
        quantity=1,
        leverage=2,
        tp_percent=1.5,
        sl_percent=1.0,
        tp_price=tp_price,
        sl_price=sl_price,
    )
    minute_start = datetime(2026, 1, 1, 10, 1, 0, tzinfo=UTC)
    minute_candles = [
        _candle(minute_start, 1, 100, tp_price + 0.2, sl_price - 0.2, 100),
    ]
    hour_candles = [
        Candle(
            open_time=datetime(2026, 1, 1, 11, 0, 0, tzinfo=UTC),
            close_time=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
            open=100,
            high=101,
            low=99,
            close=100,
        )
    ]

    def loader(start: datetime, end: datetime) -> list[AggTrade]:
        if start == entry_time:
            return []
        return [
            AggTrade(1, minute_start + timedelta(seconds=3), sl_price - 0.01, 1.0),
            AggTrade(2, minute_start + timedelta(seconds=20), tp_price + 0.01, 1.0),
        ]

    result = resolve_exit_hierarchical(
        position,
        minute_candles=minute_candles,
        hour_candles=hour_candles,
        day_candles=None,
        agg_trade_loader=loader,
    )
    assert result.reason is ExitReason.SL


def test_same_hour_single_sided_hit_exits_at_hour_close_without_extra_trade_fetches() -> None:
    entry_time = datetime(2026, 1, 1, 10, 0, 5, tzinfo=UTC)
    _, _, tp_price, sl_price = compute_effective_tp_sl(
        entry_price=100,
        side=Side.LONG,
        tp_percent=1.5,
        sl_percent=1.0,
        taker_fee_rate=0.0005,
        fng_value=None,
    )
    position = Position(
        trade_id="t1",
        bot_name="bot",
        strategy="TREND_MOM",
        symbol="ETHUSDT",
        side=Side.LONG,
        entry_time=entry_time,
        entry_price=100,
        quantity=1,
        leverage=2,
        tp_percent=1.5,
        sl_percent=1.0,
        tp_price=tp_price,
        sl_price=sl_price,
    )
    minute_start = datetime(2026, 1, 1, 10, 1, 0, tzinfo=UTC)
    minute_candles = [
        _candle(minute_start, 1, 100, tp_price + 0.2, 100.1, 100.5),
    ]
    loader_calls: list[tuple[datetime, datetime]] = []

    def loader(start: datetime, end: datetime) -> list[AggTrade]:
        loader_calls.append((start, end))
        return []

    result = resolve_exit_hierarchical(
        position,
        minute_candles=minute_candles,
        hour_candles=[],
        day_candles=None,
        agg_trade_loader=loader,
    )

    assert result.reason is ExitReason.TP
    assert result.exit_time == datetime(2026, 1, 1, 11, 0, 0, tzinfo=UTC)
    assert result.resolution_level is ResolutionLevel.HOUR
    assert loader_calls == [(entry_time, datetime(2026, 1, 1, 10, 1, 0, tzinfo=UTC))]


def test_fng_adjustment_clamps_tp_and_sl() -> None:
    adjusted_tp, adjusted_sl, _, _ = compute_effective_tp_sl(
        entry_price=100,
        side=Side.LONG,
        tp_percent=1.5,
        sl_percent=1.0,
        taker_fee_rate=0.0005,
        fng_value=0,
        risk_aversion=3,
    )
    assert 0.2 <= adjusted_tp <= 25
    assert 0.2 <= adjusted_sl <= 25
