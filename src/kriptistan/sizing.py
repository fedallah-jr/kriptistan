from __future__ import annotations

import math

from .config import BotConfig
from .models import ExchangeSymbol, FillResult


def compute_fill_result(
    *,
    wallet_balance: float,
    free_balance: float,
    bot: BotConfig,
    exchange_symbol: ExchangeSymbol,
    entry_price: float,
) -> FillResult | None:
    if wallet_balance < 6 or free_balance < 6 or entry_price <= 0:
        return None
    target_margin = (wallet_balance / bot.max_open_trades) * 0.98
    margin = target_margin if free_balance >= target_margin else free_balance * 0.98
    notional_value = margin * bot.leverage
    quantity = notional_value / entry_price
    quantity = _round_down(quantity, exchange_symbol.step_size, exchange_symbol.quantity_precision)
    if quantity < exchange_symbol.min_qty:
        return None
    notional_value = quantity * entry_price
    margin = notional_value / bot.leverage if bot.leverage else notional_value
    if notional_value < max(exchange_symbol.min_notional, 6):
        return None
    return FillResult(margin_used=margin, notional_value=notional_value, quantity=quantity)


def _round_down(value: float, step_size: float, precision: int) -> float:
    if step_size <= 0:
        return round(value, precision)
    steps = math.floor(value / step_size)
    rounded = steps * step_size
    return round(rounded, precision)
