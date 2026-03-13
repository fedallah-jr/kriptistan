from __future__ import annotations

from collections.abc import Iterable

from .models import Candle


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
