from __future__ import annotations

from collections.abc import Callable

from .indicators import atr, closes, ema, highs, lows, rsi, vwap
from .models import Side, StrategyContext, StrategyDecision

StrategyFn = Callable[[StrategyContext], StrategyDecision | None]


def technical_score(context: StrategyContext) -> int:
    if context.indicators is not None and context.candle_end_idx is not None:
        idx = context.candle_end_idx - 1
        ema200 = context.indicators.ema_200[idx]
        rsi14 = context.indicators.rsi_14[idx]
        atr14 = context.indicators.atr_14[idx]
        latest_close = context.indicators.closes[idx]
    else:
        candle_slice = context.candles
        close_values = closes(candle_slice)
        ema200 = ema(close_values, 200)[-1]
        rsi14 = rsi(close_values, 14)[-1]
        atr14 = atr(candle_slice, 14)[-1]
        latest_close = close_values[-1]

    score = 0
    if ema200 is not None and latest_close > ema200:
        score += 1
    if rsi14 is not None and 45 <= rsi14 <= 65:
        score += 1
    if atr14 is not None and latest_close > 0 and ((atr14 / latest_close) * 100) > (context.tp_percent / 2):
        score += 1
    return score


def balanced(context: StrategyContext) -> StrategyDecision | None:
    if len(context.candles) < 50:
        return None
    latest = context.candles[-1]
    recent = context.candles[-50:]
    highest = max(candle.high for candle in recent)
    lowest = min(candle.low for candle in recent)
    level = highest - ((highest - lowest) * 0.618)
    proximity = abs((latest.close - level) / level) * 100 if level else float("inf")
    score = technical_score(context)
    if proximity <= 2.5 and latest.is_bullish and score >= 1:
        return StrategyDecision(Side.LONG, "GP Touch + Bull Candle", score, 1 / (1 + proximity))
    return None


def scalper(context: StrategyContext) -> StrategyDecision | None:
    if len(context.candles) < 2:
        return None
    latest = context.candles[-1]
    previous = context.candles[-2]
    if context.indicators is not None and context.candle_end_idx is not None:
        current_vwap = context.indicators.cumulative_vwap[context.candle_end_idx - 1]
    else:
        current_vwap = vwap(context.candles)
    if current_vwap is None or current_vwap == 0:
        return None
    proximity = abs((latest.close - current_vwap) / current_vwap) * 100
    score = technical_score(context)
    if proximity <= 1.0 and latest.close > previous.high:
        return StrategyDecision(Side.LONG, "POC Bounce", score, 1 / (1 + proximity))
    return None


def sniper(context: StrategyContext) -> StrategyDecision | None:
    if len(context.candles) < 2:
        return None
    if context.indicators is not None and context.candle_end_idx is not None:
        idx = context.candle_end_idx
        close_cur = context.indicators.closes[idx - 1]
        close_prev = context.indicators.closes[idx - 2]
        lowest = context.indicators.running_low[idx - 1]
        highest = context.indicators.running_high[idx - 1]
    else:
        close_values = closes(context.candles)
        close_cur = close_values[-1]
        close_prev = close_values[-2]
        lowest = min(lows(context.candles))
        highest = max(highs(context.candles))
    fib382 = lowest + ((highest - lowest) * 0.382)
    score = technical_score(context)
    if score >= 2 and close_cur > fib382 and close_prev < fib382:
        distance = abs(close_cur - fib382) / fib382 if fib382 else 1.0
        return StrategyDecision(Side.LONG, "0.382 Reclaim Confirmed", score, 1 / (1 + distance))
    return None


def bounce(context: StrategyContext) -> StrategyDecision | None:
    if context.indicators is not None and context.candle_end_idx is not None:
        idx = context.candle_end_idx - 1
        rsi_cur = context.indicators.rsi_14[idx]
        rsi_prev = context.indicators.rsi_14[idx - 1] if idx > 0 else None
    else:
        close_values = closes(context.candles)
        rsi14 = rsi(close_values, 14)
        if len(rsi14) < 2:
            return None
        rsi_cur = rsi14[-1]
        rsi_prev = rsi14[-2]
    if rsi_prev is None or rsi_cur is None:
        return None
    score = technical_score(context)
    if rsi_prev < 35 <= rsi_cur:
        return StrategyDecision(Side.LONG, "RSI Oversold Bounce", score, (35 - rsi_prev) + (rsi_cur - 35))
    return None


def cycle_rev(context: StrategyContext) -> StrategyDecision | None:
    if context.cycle_stats is None:
        return None
    latest = context.candles[-1]
    if context.indicators is not None and context.candle_end_idx is not None:
        ema50 = context.indicators.ema_50[context.candle_end_idx - 1]
    else:
        ema50 = ema(closes(context.candles), 50)[-1]
    pump_due = context.cycle_stats.pump_date_due_days
    dump_due = context.cycle_stats.dump_date_due_days
    score = technical_score(context)

    if pump_due is not None and -4 <= pump_due <= 1 and ema50 is not None and latest.close > ema50:
        strength = 1 / (1 + abs(pump_due))
        return StrategyDecision(Side.LONG, "Long: Due + EMA50 Reclaim", score, strength)
    if latest.is_bearish and ((pump_due is not None and pump_due > 3) or (dump_due is not None and -2 <= dump_due <= 2)):
        pump_distance = abs(pump_due) if pump_due is not None else 10.0
        dump_distance = abs(dump_due) if dump_due is not None else 10.0
        strength = 1 / (1 + min(pump_distance, dump_distance))
        return StrategyDecision(Side.SHORT, "Short: Cycle Exhaustion", score, strength)
    return None


def trend_mom(context: StrategyContext) -> StrategyDecision | None:
    if context.cycle_stats is None or len(context.candles) < 21:
        return None
    if context.indicators is not None and context.candle_end_idx is not None:
        idx = context.candle_end_idx
        ema200 = context.indicators.ema_200[idx - 1]
        latest_close = context.indicators.closes[idx - 1]
        prev_20_high = max(context.indicators.highs[idx - 21:idx - 1])
        prev_20_low = min(context.indicators.lows[idx - 21:idx - 1])
    else:
        close_values = closes(context.candles)
        ema200 = ema(close_values, 200)[-1]
        prev_20_high = max(highs(context.candles[-21:-1]))
        prev_20_low = min(lows(context.candles[-21:-1]))
        latest_close = close_values[-1]
    pump_due = context.cycle_stats.pump_date_due_days
    dump_due = context.cycle_stats.dump_date_due_days
    score = technical_score(context)
    if ema200 is not None and latest_close > ema200 and pump_due is not None and -2 <= pump_due <= 2 and latest_close > prev_20_high:
        return StrategyDecision(Side.LONG, "Trend Breakout Long", score, 1 / (1 + abs(pump_due)))
    if ema200 is not None and latest_close < ema200 and dump_due is not None and -2 <= dump_due <= 2 and latest_close < prev_20_low:
        return StrategyDecision(Side.SHORT, "Trend Breakout Short", score, 1 / (1 + abs(dump_due)))
    return None


def breakout_retest(context: StrategyContext) -> StrategyDecision | None:
    if len(context.candles) < 60:
        return None
    if context.indicators is not None and context.candle_end_idx is not None:
        ema200 = context.indicators.ema_200[context.candle_end_idx - 1]
    else:
        ema200 = ema(closes(context.candles), 200)[-1]
    latest = context.candles[-1]
    if ema200 is None or latest.close <= ema200:
        return None
    breakout_window = context.candles[-26:-6]
    if len(breakout_window) < 20:
        return None
    breakout_level = max(candle.high for candle in breakout_window)
    recent_breakout_closes = [candle.close for candle in context.candles[-6:-1]]
    proximity = abs((latest.close - breakout_level) / breakout_level) * 100 if breakout_level else float("inf")
    score = technical_score(context)
    if recent_breakout_closes and all(value > breakout_level for value in recent_breakout_closes) and proximity <= 0.6 and latest.is_bullish:
        return StrategyDecision(Side.LONG, "Breakout Retest Long", score, 1 / (1 + proximity))
    return None


def pullback_reclaim(context: StrategyContext) -> StrategyDecision | None:
    if len(context.candles) < 80:
        return None
    latest = context.candles[-1]
    previous = context.candles[-2]
    if context.indicators is not None and context.candle_end_idx is not None:
        idx = context.candle_end_idx
        ema200 = context.indicators.ema_200[idx - 1]
        ema20_cur = context.indicators.ema_20[idx - 1]
        ema20_prev = context.indicators.ema_20[idx - 2]
    else:
        close_values = closes(context.candles)
        ema200 = ema(close_values, 200)[-1]
        ema20 = ema(close_values, 20)
        ema20_cur = ema20[-1]
        ema20_prev = ema20[-2]
    if ema200 is None or ema20_cur is None or ema20_prev is None or latest.close <= ema200:
        return None
    previous_ranges = [candle.range for candle in context.candles[-17:-2]]
    if not previous_ranges:
        return None
    average_range = sum(previous_ranges) / len(previous_ranges)
    score = technical_score(context)
    if previous.range > 1.8 * average_range and previous.close < ema20_prev and latest.close > ema20_cur and latest.is_bullish:
        strength = previous.range / average_range if average_range else 0.0
        return StrategyDecision(Side.LONG, "Pullback Reclaim Long", score, strength)
    return None


STRATEGIES: dict[str, StrategyFn] = {
    "BALANCED": balanced,
    "SCALPER": scalper,
    "SNIPER": sniper,
    "BOUNCE": bounce,
    "CYCLE_REV": cycle_rev,
    "TREND_MOM": trend_mom,
    "BREAKOUT_RETEST": breakout_retest,
    "PULLBACK_RECLAIM": pullback_reclaim,
}


def evaluate_strategy(strategy: str, context: StrategyContext) -> StrategyDecision | None:
    return STRATEGIES[strategy](context)
