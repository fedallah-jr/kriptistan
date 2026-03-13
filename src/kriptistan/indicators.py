from __future__ import annotations

from collections.abc import Iterable

from .models import Candle, PrecomputedIndicators


def closes(candles: Iterable[Candle]) -> list[float]:
    return [candle.close for candle in candles]


def highs(candles: Iterable[Candle]) -> list[float]:
    return [candle.high for candle in candles]


def lows(candles: Iterable[Candle]) -> list[float]:
    return [candle.low for candle in candles]


def ema(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be positive")
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result
    multiplier = 2 / (period + 1)
    seed = sum(values[:period]) / period
    result[period - 1] = seed
    current = seed
    for idx in range(period, len(values)):
        current = ((values[idx] - current) * multiplier) + current
        result[idx] = current
    return result


def rolling_high(values: list[float], window: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    for idx in range(window - 1, len(values)):
        result[idx] = max(values[idx - window + 1 : idx + 1])
    return result


def rolling_low(values: list[float], window: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    for idx in range(window - 1, len(values)):
        result[idx] = min(values[idx - window + 1 : idx + 1])
    return result


def rsi(values: list[float], period: int = 14) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return result
    gains: list[float] = []
    losses: list[float] = []
    for idx in range(1, period + 1):
        change = values[idx] - values[idx - 1]
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    result[period] = _rsi_from_avgs(avg_gain, avg_loss)
    for idx in range(period + 1, len(values)):
        change = values[idx] - values[idx - 1]
        gain = max(change, 0.0)
        loss = abs(min(change, 0.0))
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        result[idx] = _rsi_from_avgs(avg_gain, avg_loss)
    return result


def atr(candles: list[Candle], period: int = 14) -> list[float | None]:
    result: list[float | None] = [None] * len(candles)
    if len(candles) <= period:
        return result
    true_ranges: list[float] = []
    for idx, candle in enumerate(candles):
        if idx == 0:
            true_ranges.append(candle.range)
            continue
        previous_close = candles[idx - 1].close
        true_ranges.append(
            max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        )
    seed = sum(true_ranges[1 : period + 1]) / period
    result[period] = seed
    current = seed
    for idx in range(period + 1, len(candles)):
        current = ((current * (period - 1)) + true_ranges[idx]) / period
        result[idx] = current
    return result


def vwap(candles: list[Candle]) -> float | None:
    if not candles:
        return None
    cumulative_price_volume = 0.0
    cumulative_volume = 0.0
    for candle in candles:
        typical_price = (candle.high + candle.low + candle.close) / 3
        cumulative_price_volume += typical_price * candle.volume
        cumulative_volume += candle.volume
    if cumulative_volume == 0:
        return None
    return cumulative_price_volume / cumulative_volume


def close_position_in_range(candle: Candle) -> float:
    if candle.range <= 0:
        return 0.5
    return (candle.close - candle.low) / candle.range


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def precompute_indicators(candles: list[Candle]) -> PrecomputedIndicators:
    close_values = closes(candles)
    high_values = highs(candles)
    low_values = lows(candles)
    return PrecomputedIndicators(
        closes=close_values,
        highs=high_values,
        lows=low_values,
        ema_20=ema(close_values, 20),
        ema_50=ema(close_values, 50),
        ema_200=ema(close_values, 200),
        rsi_14=rsi(close_values, 14),
        atr_14=atr(candles, 14),
        cumulative_vwap=_cumulative_vwap(candles),
        running_high=_running_max(high_values),
        running_low=_running_min(low_values),
    )


def _cumulative_vwap(candles: list[Candle]) -> list[float | None]:
    result: list[float | None] = []
    cum_pv = 0.0
    cum_v = 0.0
    for candle in candles:
        typical = (candle.high + candle.low + candle.close) / 3
        cum_pv += typical * candle.volume
        cum_v += candle.volume
        result.append(cum_pv / cum_v if cum_v > 0 else None)
    return result


def _running_max(values: list[float]) -> list[float]:
    result: list[float] = []
    current = float("-inf")
    for v in values:
        current = max(current, v)
        result.append(current)
    return result


def _running_min(values: list[float]) -> list[float]:
    result: list[float] = []
    current = float("inf")
    for v in values:
        current = min(current, v)
        result.append(current)
    return result
