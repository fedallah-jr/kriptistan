from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Callable
from datetime import datetime, timedelta

from .models import AggTrade, Candle, EntryPriceBand, ExitReason, ExitResolution, Position, ResolutionLevel, Side


AggTradeLoader = Callable[[datetime, datetime], list[AggTrade]]


def select_entry_price_band(
    trades: list[AggTrade],
    *,
    side: Side,
    signal_time: datetime,
    entry_delay_seconds: int,
    entry_window_seconds: int,
) -> EntryPriceBand:
    earliest = signal_time + timedelta(seconds=entry_delay_seconds)
    window_end = signal_time + timedelta(seconds=entry_window_seconds)
    window = [trade for trade in trades if signal_time <= trade.timestamp <= window_end]
    if not window:
        raise ValueError("no aggregate trades available inside the entry window")
    base_trade = next((trade for trade in window if trade.timestamp >= earliest), None)
    if base_trade is None:
        raise ValueError("no aggregate trades available at or after the configured entry delay")
    prices = [trade.price for trade in window]
    if side is Side.LONG:
        best_price = min(prices)
        worst_price = max(prices)
    else:
        best_price = max(prices)
        worst_price = min(prices)
    return EntryPriceBand(
        signal_time=signal_time,
        base_time=base_trade.timestamp,
        base_price=base_trade.price,
        best_price=best_price,
        worst_price=worst_price,
        window_end=window_end,
    )


def approximate_entry_price_band(
    candles: list[Candle],
    *,
    side: Side,
    signal_time: datetime,
    entry_delay_seconds: int,
    entry_window_seconds: int,
) -> EntryPriceBand:
    open_times = [c.open_time for c in candles]
    idx = bisect_right(open_times, signal_time) - 1
    minute = None
    if 0 <= idx < len(candles) and candles[idx].open_time <= signal_time < candles[idx].close_time:
        minute = candles[idx]
    if minute is None:
        idx = bisect_left(open_times, signal_time)
        if idx < len(candles):
            minute = candles[idx]
    if minute is None:
        raise ValueError("no minute candle available for approximate entry pricing")
    if side is Side.LONG:
        best_price = minute.low
        worst_price = minute.high
    else:
        best_price = minute.high
        worst_price = minute.low
    base_time = signal_time + timedelta(seconds=entry_delay_seconds)
    return EntryPriceBand(
        signal_time=signal_time,
        base_time=base_time,
        base_price=minute.open,
        best_price=best_price,
        worst_price=worst_price,
        window_end=signal_time + timedelta(seconds=entry_window_seconds),
    )


def compute_effective_tp_sl(
    *,
    entry_price: float,
    side: Side,
    tp_percent: float,
    sl_percent: float,
    taker_fee_rate: float,
    fng_value: float | None = None,
    risk_aversion: float = 1.0,
    reverse_mode: bool = False,
) -> tuple[float, float, float, float]:
    position_side = side
    target_tp = tp_percent
    target_sl = sl_percent
    if reverse_mode:
        position_side = Side.SHORT if side is Side.LONG else Side.LONG
        target_tp, target_sl = target_sl, target_tp

    adjusted_tp, adjusted_sl = _apply_fng_adjustment(
        tp_percent=target_tp,
        sl_percent=target_sl,
        fng_value=fng_value,
        risk_aversion=risk_aversion,
    )
    fee_offset = taker_fee_rate * 2 * 100
    tp_with_fees = adjusted_tp + fee_offset
    sl_with_fees = max(adjusted_sl - fee_offset, 0.01)

    if position_side is Side.LONG:
        tp_price = entry_price * (1 + (tp_with_fees / 100))
        sl_price = entry_price * (1 - (sl_with_fees / 100))
    else:
        tp_price = entry_price * (1 - (tp_with_fees / 100))
        sl_price = entry_price * (1 + (sl_with_fees / 100))
    return adjusted_tp, adjusted_sl, tp_price, sl_price


def resolve_exit_hierarchical(
    position: Position,
    *,
    minute_candles: list[Candle],
    hour_candles: list[Candle],
    day_candles: list[Candle] | None,
    agg_trade_loader: AggTradeLoader,
) -> ExitResolution:
    first_hour_hit = resolve_first_hour_exit(
        position,
        minute_candles=minute_candles,
        agg_trade_loader=agg_trade_loader,
    )
    if first_hour_hit is not None:
        return first_hour_hit

    minute_open_times = [c.open_time for c in minute_candles]
    first_hour_end = _first_hour_end(position.entry_time)
    later_hours = [candle for candle in hour_candles if candle.open_time >= first_hour_end]
    for candle in later_hours:
        hour_hit = resolve_hour_candle_exit(
            position,
            hour_candle=candle,
            minute_candles_loader=lambda candle=candle: _slice_by_open_time(
                minute_candles,
                minute_open_times,
                candle.open_time,
                candle.close_time,
            ),
            agg_trade_loader=agg_trade_loader,
        )
        if hour_hit is not None:
            return hour_hit

    return ExitResolution(
        reason=ExitReason.OPEN,
        exit_time=hour_candles[-1].close_time if hour_candles else position.entry_time,
        exit_price=hour_candles[-1].close if hour_candles else position.entry_price,
        resolution_level=ResolutionLevel.HOUR,
    )


def resolve_first_hour_exit(
    position: Position,
    *,
    minute_candles: list[Candle],
    agg_trade_loader: AggTradeLoader,
) -> ExitResolution | None:
    minute_open_times = [c.open_time for c in minute_candles]

    entry_minute_start = position.entry_time.replace(second=0, microsecond=0)
    entry_minute_end = entry_minute_start + timedelta(minutes=1)
    entry_trades = agg_trade_loader(position.entry_time, entry_minute_end)
    trade_hit = _resolve_with_agg_trades(
        entry_trades,
        side=position.side,
        tp_price=position.tp_price,
        sl_price=position.sl_price,
        start_time=position.entry_time,
    )
    if trade_hit is not None:
        return trade_hit

    # Entry can happen a few seconds after the hour boundary. After protecting
    # that first partial minute with exact trades, the simulator resolves exits
    # hour-first to match its hourly decision cadence.
    first_hour_end = _first_hour_end(position.entry_time)
    same_hour_minutes = _slice_by_open_time(minute_candles, minute_open_times, entry_minute_end, first_hour_end)
    return _resolve_hour_interval(
        candles=same_hour_minutes,
        side=position.side,
        tp_price=position.tp_price,
        sl_price=position.sl_price,
        close_time=first_hour_end,
        agg_trade_loader=agg_trade_loader,
    )


def resolve_hour_candle_exit(
    position: Position,
    *,
    hour_candle: Candle,
    minute_candles_loader: Callable[[], list[Candle]],
    agg_trade_loader: AggTradeLoader,
) -> ExitResolution | None:
    outcome = _barrier_outcome(hour_candle, position.side, position.tp_price, position.sl_price)
    if outcome == ExitReason.OPEN:
        return None
    if outcome in (ExitReason.TP, ExitReason.SL):
        return ExitResolution(
            outcome,
            hour_candle.close_time,
            position.tp_price if outcome is ExitReason.TP else position.sl_price,
            ResolutionLevel.HOUR,
        )
    hour_minutes = minute_candles_loader()
    nested = _resolve_candles(
        hour_minutes,
        side=position.side,
        tp_price=position.tp_price,
        sl_price=position.sl_price,
        resolution_level=ResolutionLevel.MINUTE,
        agg_trade_loader=agg_trade_loader,
    )
    return nested


def _first_hour_end(entry_time: datetime) -> datetime:
    return entry_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)


def compute_pnl_percent(position: Position, exit_price: float, *, taker_fee_rate: float) -> float:
    if position.side is Side.LONG:
        gross = ((exit_price - position.entry_price) / position.entry_price) * 100 * position.leverage
    else:
        gross = ((position.entry_price - exit_price) / position.entry_price) * 100 * position.leverage
    fee_drag = 2 * taker_fee_rate * position.leverage * 100
    return gross - fee_drag


def _resolve_hour_interval(
    candles: list[Candle],
    *,
    side: Side,
    tp_price: float,
    sl_price: float,
    close_time: datetime,
    agg_trade_loader: AggTradeLoader,
) -> ExitResolution | None:
    outcome = _barrier_outcome_for_candles(candles, side=side, tp_price=tp_price, sl_price=sl_price)
    if outcome == ExitReason.OPEN:
        return None
    if outcome in (ExitReason.TP, ExitReason.SL):
        return ExitResolution(outcome, close_time, tp_price if outcome is ExitReason.TP else sl_price, ResolutionLevel.HOUR)
    nested = _resolve_candles(
        candles,
        side=side,
        tp_price=tp_price,
        sl_price=sl_price,
        resolution_level=ResolutionLevel.MINUTE,
        agg_trade_loader=agg_trade_loader,
    )
    return nested if nested is not None else ExitResolution(ExitReason.SL, close_time, sl_price, ResolutionLevel.HOUR)


def _resolve_candles(
    candles: list[Candle],
    *,
    side: Side,
    tp_price: float,
    sl_price: float,
    resolution_level: ResolutionLevel,
    agg_trade_loader: AggTradeLoader,
    minute_candles: list[Candle] | None = None,
    minute_open_times: list[datetime] | None = None,
) -> ExitResolution | None:
    for candle in candles:
        outcome = _barrier_outcome(candle, side, tp_price, sl_price)
        if outcome == ExitReason.OPEN:
            continue
        if outcome in (ExitReason.TP, ExitReason.SL):
            return ExitResolution(outcome, candle.close_time, tp_price if outcome is ExitReason.TP else sl_price, resolution_level)
        if resolution_level is ResolutionLevel.MINUTE:
            nested = _resolve_with_agg_trades(
                agg_trade_loader(candle.open_time, candle.close_time),
                side=side,
                tp_price=tp_price,
                sl_price=sl_price,
                start_time=candle.open_time,
            )
            if nested is not None:
                return nested
            return ExitResolution(ExitReason.SL, candle.close_time, sl_price, ResolutionLevel.MINUTE)
        elif resolution_level is ResolutionLevel.HOUR and minute_candles is not None:
            hour_minutes = _slice_by_open_time(minute_candles, minute_open_times, candle.open_time, candle.close_time)
            nested = _resolve_candles(
                hour_minutes,
                side=side,
                tp_price=tp_price,
                sl_price=sl_price,
                resolution_level=ResolutionLevel.MINUTE,
                agg_trade_loader=agg_trade_loader,
            )
            if nested is not None:
                return nested
    return None


def _barrier_outcome_for_candles(
    candles: list[Candle],
    *,
    side: Side,
    tp_price: float,
    sl_price: float,
) -> ExitReason | None:
    if not candles:
        return ExitReason.OPEN
    tp_touched = False
    sl_touched = False
    for candle in candles:
        if side is Side.LONG:
            tp_touched = tp_touched or candle.high >= tp_price
            sl_touched = sl_touched or candle.low <= sl_price
        else:
            tp_touched = tp_touched or candle.low <= tp_price
            sl_touched = sl_touched or candle.high >= sl_price
        if tp_touched and sl_touched:
            return None
    if not tp_touched and not sl_touched:
        return ExitReason.OPEN
    return ExitReason.TP if tp_touched else ExitReason.SL


def _resolve_with_agg_trades(
    trades: list[AggTrade],
    *,
    side: Side,
    tp_price: float,
    sl_price: float,
    start_time: datetime,
) -> ExitResolution | None:
    for trade in trades:
        if trade.timestamp < start_time:
            continue
        if side is Side.LONG:
            if trade.price >= tp_price:
                return ExitResolution(ExitReason.TP, trade.timestamp, tp_price, ResolutionLevel.TRADE)
            if trade.price <= sl_price:
                return ExitResolution(ExitReason.SL, trade.timestamp, sl_price, ResolutionLevel.TRADE)
            continue
        if trade.price <= tp_price:
            return ExitResolution(ExitReason.TP, trade.timestamp, tp_price, ResolutionLevel.TRADE)
        if trade.price >= sl_price:
            return ExitResolution(ExitReason.SL, trade.timestamp, sl_price, ResolutionLevel.TRADE)
    return None


def _apply_fng_adjustment(
    *,
    tp_percent: float,
    sl_percent: float,
    fng_value: float | None,
    risk_aversion: float,
) -> tuple[float, float]:
    if fng_value is None:
        return tp_percent, sl_percent
    bias = (fng_value - 50) / 50
    tp_multiplier = max(0.40, min(1.0, 1.0 - (0.30 * abs(bias) * risk_aversion)))
    sl_multiplier = max(0.60, min(1.50, 1.0 - (0.20 * bias * risk_aversion)))
    adjusted_tp = _clamp(tp_percent * tp_multiplier, 0.20, 25.00)
    adjusted_sl = _clamp(sl_percent * sl_multiplier, 0.20, 25.00)
    return adjusted_tp, adjusted_sl


def _barrier_outcome(candle: Candle, side: Side, tp_price: float, sl_price: float) -> ExitReason | None:
    if side is Side.LONG:
        tp_touched = candle.high >= tp_price
        sl_touched = candle.low <= sl_price
    else:
        tp_touched = candle.low <= tp_price
        sl_touched = candle.high >= sl_price
    if not tp_touched and not sl_touched:
        return ExitReason.OPEN
    if tp_touched and sl_touched:
        return None
    return ExitReason.TP if tp_touched else ExitReason.SL


def _price_for_reason(position: Position, reason: ExitReason) -> float:
    return position.tp_price if reason is ExitReason.TP else position.sl_price


def _slice_by_open_time(
    candles: list[Candle],
    open_times: list[datetime] | None,
    start: datetime,
    end: datetime,
) -> list[Candle]:
    if open_times is not None:
        left = bisect_left(open_times, start)
        right = bisect_left(open_times, end)
        return candles[left:right]
    return [c for c in candles if start <= c.open_time < end]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)
