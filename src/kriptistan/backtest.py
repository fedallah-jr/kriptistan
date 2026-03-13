from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from .config import AppConfig, BotConfig
from .execution import approximate_entry_price_band, compute_effective_tp_sl, compute_pnl_percent, resolve_exit_hierarchical, select_entry_price_band
from .gates import anti_repetition_guard, btc_vol_guard_triggered, chase_filter_passes, dead_end_blacklisted, due_date_in_range, resolve_collisions, symbol_on_cooldown
from .indicators import precompute_indicators
from .market_data import MarketDataBundle
from .models import Candidate, CollisionPolicy, CycleStats, EntryClaim, ExitReason, Position, ScheduledTrade, Side, StrategyContext
from .reports import BotLedger, build_portfolio_result
from .sizing import compute_fill_result
from .strategies import evaluate_strategy


@dataclass(slots=True)
class AccountState:
    wallet_balance: float
    free_balance: float


@dataclass(slots=True)
class BotRuntime:
    config: BotConfig
    ledger: BotLedger
    account: AccountState
    open_trades: list[ScheduledTrade]
    closed_symbols: list[str]
    closed_times_by_symbol: dict[str, list[datetime]]
    sl_times_by_symbol: dict[str, list[datetime]]

    @property
    def available_slots(self) -> int:
        return max(self.config.max_open_trades - len(self.open_trades), 0)


class Backtester:
    def __init__(self, config: AppConfig, bundle: MarketDataBundle) -> None:
        self.config = config
        self.bundle = bundle
        self._scanner_cache: dict[datetime, list[Candidate]] = {}
        self._btc_guard_cache: dict[datetime, bool] = {}
        self._indicator_cache: dict[str, object] = {}

    def run(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        bots: tuple[BotConfig, ...] | None = None,
        collision_policy: CollisionPolicy | None = None,
        shuffle_seed: int | None = None,
    ):
        start_time = start or self.config.backtest.start
        end_time = end or self.config.backtest.end
        active_bots = bots or self.config.bots
        policy = collision_policy or self.config.backtest.collision_policy
        seed = shuffle_seed if shuffle_seed is not None else self.config.backtest.shuffle_seed
        shared_account = None
        if self.config.backtest.capital_mode.value == "shared":
            shared_account = AccountState(
                wallet_balance=self.config.backtest.starting_balance,
                free_balance=self.config.backtest.starting_balance,
            )

        ledgers: list[BotLedger] = []
        runtimes: list[BotRuntime] = []
        per_bot_start = self.config.backtest.starting_balance
        if shared_account is not None:
            per_bot_start = self.config.backtest.starting_balance / max(len(active_bots), 1)
        for bot in active_bots:
            account = shared_account or AccountState(
                wallet_balance=self.config.backtest.starting_balance,
                free_balance=self.config.backtest.starting_balance,
            )
            ledger = BotLedger(
                bot_name=bot.name,
                starting_balance=per_bot_start,
                current_balance=per_bot_start,
                equity_points=[per_bot_start],
            )
            ledgers.append(ledger)
            runtimes.append(
                BotRuntime(
                    config=bot,
                    ledger=ledger,
                    account=account,
                    open_trades=[],
                    closed_symbols=[],
                    closed_times_by_symbol={},
                    sl_times_by_symbol={},
                )
            )

        timestamps = self.bundle.hourly_timestamps(start=start_time, end=end_time)
        for timestamp in timestamps:
            self._settle_due_trades(runtimes, timestamp)
            open_symbols = {
                trade.position.symbol
                for runtime in runtimes
                for trade in runtime.open_trades
            }
            claims: list[EntryClaim] = []
            btc_guard_active = self._btc_guard_active(timestamp)
            base_candidates = self._scanner_candidates(timestamp)
            for runtime in runtimes:
                if runtime.available_slots <= 0:
                    runtime.ledger.equity_points.append(runtime.ledger.current_balance)
                    continue
                for candidate in self._filter_candidates(runtime, base_candidates, timestamp, open_symbols):
                    symbol_data = self.bundle.symbols[candidate.symbol]
                    indicators = self._get_indicators(candidate.symbol)
                    end_idx = symbol_data.futures_1h.closed_until_idx(timestamp)
                    candles = symbol_data.futures_1h.last_n_closed(timestamp, 300)
                    decision = evaluate_strategy(
                        runtime.config.strategy,
                        StrategyContext(
                            symbol=candidate.symbol,
                            candles=candles,
                            cycle_stats=candidate.cycle_stats,
                            tp_percent=runtime.config.tp_percent,
                            indicators=indicators,
                            candle_end_idx=end_idx,
                        ),
                    )
                    if decision is None:
                        continue
                    latest = candles[-1]
                    raw_side = decision.side
                    if btc_guard_active:
                        runtime.ledger.reject("btc_vol_guard")
                        continue
                    if self.config.execution.chase_filter_enabled and not chase_filter_passes(
                        raw_side,
                        latest,
                        tp_percent=runtime.config.tp_percent,
                        close_position_limit=self.config.execution.chase_close_pos,
                        range_multiple_tp=self.config.execution.chase_range_mult_tp,
                    ):
                        runtime.ledger.reject("chase_filter")
                        continue
                    if self.config.execution.btc_confirm_entry_enabled and not self._btc_confirm(runtime, candidate.symbol, timestamp, raw_side):
                        runtime.ledger.reject("btc_confirm")
                        continue
                    actual_side, _, _ = _effective_side_and_risk(runtime.config, raw_side)
                    claims.append(
                        EntryClaim(
                            bot_name=runtime.config.name,
                            strategy=runtime.config.strategy,
                            symbol=candidate.symbol,
                            side=actual_side,
                            signal_time=timestamp,
                            reason=decision.reason,
                            technical_score=decision.technical_score,
                            signal_strength=decision.signal_strength,
                            fixed_priority=runtime.config.fixed_priority,
                        )
                    )
                runtime.ledger.equity_points.append(runtime.ledger.current_balance)

            winners, rejections = resolve_collisions(claims, policy=policy, timestamp=timestamp, shuffle_seed=seed)
            for rejected in rejections:
                runtime = _runtime_for_bot(runtimes, rejected.claim.bot_name)
                runtime.ledger.reject(rejected.reason)

            selected_by_bot: dict[str, list[EntryClaim]] = {}
            for claim in winners:
                selected_by_bot.setdefault(claim.bot_name, []).append(claim)
            accepted_claims: list[EntryClaim] = []
            for bot_name, bot_claims in selected_by_bot.items():
                runtime = _runtime_for_bot(runtimes, bot_name)
                ranked = sorted(bot_claims, key=lambda item: (-item.signal_strength, -item.technical_score, item.symbol))
                accepted = ranked[: runtime.available_slots]
                for rejected in ranked[runtime.available_slots :]:
                    runtime.ledger.reject("bot_capacity")
                accepted_claims.extend(accepted)

            accepted_claims = sorted(
                accepted_claims,
                key=lambda item: (-item.signal_strength, -item.technical_score, -item.fixed_priority, item.bot_name, item.symbol),
            )
            for claim in accepted_claims:
                runtime = _runtime_for_bot(runtimes, claim.bot_name)
                trade = self._open_trade(runtime, claim, end_time)
                if trade is None:
                    continue
                runtime.open_trades.append(trade)
                open_symbols.add(trade.position.symbol)

        self._settle_due_trades(runtimes, end_time + timedelta(days=3650))
        return build_portfolio_result(
            start=start_time,
            end=end_time,
            ledgers=ledgers,
            collision_policy=policy,
        )

    def _scanner_candidates(self, timestamp: datetime) -> list[Candidate]:
        day_candle = next(iter(self.bundle.symbols.values())).closed_daily(timestamp)
        if not day_candle:
            return []
        cache_key = day_candle[-1].close_time
        cached = self._scanner_cache.get(cache_key)
        if cached is not None:
            return cached
        candidates: list[Candidate] = []
        for symbol in self.bundle.symbols:
            stats = self.bundle.cycle_stats_as_of(symbol, timestamp)
            if stats is None or not stats.passes_stdev_filter:
                continue
            distance = min(
                abs(stats.pump_date_due_days) if stats.pump_date_due_days is not None else float("inf"),
                abs(stats.dump_date_due_days) if stats.dump_date_due_days is not None else float("inf"),
            )
            candidates.append(Candidate(symbol=symbol, cycle_stats=stats, distance=distance))
        ordered = sorted(candidates, key=lambda item: item.distance)
        self._scanner_cache[cache_key] = ordered
        return ordered

    def _filter_candidates(
        self,
        runtime: BotRuntime,
        candidates: list[Candidate],
        timestamp: datetime,
        open_symbols: set[str],
    ) -> list[Candidate]:
        filtered: list[Candidate] = []
        for candidate in candidates:
            symbol = candidate.symbol
            if runtime.config.symbol_whitelist and symbol not in runtime.config.symbol_whitelist:
                runtime.ledger.reject("not_whitelisted")
                continue
            if symbol in open_symbols:
                runtime.ledger.reject("symbol_open_elsewhere")
                continue
            if self.bundle.quote_volume_24h(symbol, timestamp) < runtime.config.min_24h_volume:
                runtime.ledger.reject("min_volume")
                continue
            if not due_date_in_range(
                candidate.cycle_stats.pump_date_due_days,
                candidate.cycle_stats.dump_date_due_days,
                runtime.config.due_date_min,
                runtime.config.due_date_max,
            ):
                runtime.ledger.reject("due_date_range")
                continue
            if symbol_on_cooldown(runtime.closed_times_by_symbol.get(symbol, []), now=timestamp, cooldown_hours=runtime.config.same_ticker_cooldown_hours):
                runtime.ledger.reject("same_ticker_cooldown")
                continue
            if dead_end_blacklisted(
                runtime.sl_times_by_symbol.get(symbol, []),
                now=timestamp,
                max_consecutive_losses=runtime.config.dead_end_max_consecutive_losses,
                lookback_days=runtime.config.dead_end_lookback_days,
                blacklist_days=runtime.config.dead_end_blacklist_days,
            ):
                runtime.ledger.reject("dead_end_blacklist")
                continue
            if not anti_repetition_guard(symbol, runtime.closed_symbols):
                runtime.ledger.reject("anti_repetition")
                continue
            filtered.append(candidate)
            if len(filtered) >= self.config.backtest.top_candidates:
                break
        return filtered

    def _btc_guard_active(self, timestamp: datetime) -> bool:
        cached = self._btc_guard_cache.get(timestamp)
        if cached is not None:
            return cached
        if not self.config.execution.btc_vol_guard_enabled:
            self._btc_guard_cache[timestamp] = False
            return False
        closed = self.bundle.btc_5m.last_n_closed(timestamp, 36)
        triggered = False
        cooldown = timedelta(seconds=self.config.execution.btc_vol_cooldown_seconds)
        for index in range(max(11, len(closed) - 24), len(closed)):
            window = closed[index - 11 : index + 1]
            if len(window) < 12:
                continue
            if not btc_vol_guard_triggered(
                window,
                max_range_pct=self.config.execution.btc_vol_max_range_pct,
                max_candle_pct=self.config.execution.btc_vol_max_candle_pct,
            ):
                continue
            if window[-1].close_time + cooldown >= timestamp:
                triggered = True
                break
        self._btc_guard_cache[timestamp] = triggered
        return triggered

    def _btc_confirm(self, runtime: BotRuntime, symbol: str, timestamp: datetime, raw_side: Side) -> bool:
        data = self.bundle.symbols[symbol]
        if data.confirm_symbol is None or data.confirm_1h is None:
            return True
        end_idx = data.confirm_1h.closed_until_idx(timestamp)
        if end_idx == 0:
            return True
        candles = data.confirm_1h.last_n_closed(timestamp, 300)
        confirm_key = f"confirm:{symbol}"
        indicators = self._indicator_cache.get(confirm_key)
        if indicators is None:
            indicators = precompute_indicators(data.confirm_1h.candles)
            self._indicator_cache[confirm_key] = indicators
        confirm_decision = evaluate_strategy(
            runtime.config.strategy,
            StrategyContext(
                symbol=data.confirm_symbol,
                candles=candles,
                cycle_stats=self.bundle.confirm_cycle_stats_as_of(symbol, timestamp),
                tp_percent=runtime.config.tp_percent,
                indicators=indicators,
                candle_end_idx=end_idx,
            ),
        )
        if confirm_decision is None:
            return False
        return confirm_decision.side is raw_side

    def _get_indicators(self, symbol: str):
        cached = self._indicator_cache.get(symbol)
        if cached is not None:
            return cached
        indicators = precompute_indicators(self.bundle.symbols[symbol].futures_1h.candles)
        self._indicator_cache[symbol] = indicators
        return indicators

    def _open_trade(self, runtime: BotRuntime, claim: EntryClaim, end_time: datetime) -> ScheduledTrade | None:
        symbol_data = self.bundle.symbols[claim.symbol]
        minute_candles = self.bundle.minute_candles(claim.symbol)
        exact_cutoff = datetime.now(UTC) - timedelta(days=self.config.backtest.exact_mode_max_history_days)
        use_exact = self.config.backtest.exact_mode and claim.signal_time >= exact_cutoff
        try:
            if use_exact:
                entry_band = select_entry_price_band(
                    self.bundle.agg_trade_loader(claim.symbol)(
                        claim.signal_time,
                        claim.signal_time + timedelta(seconds=self.config.backtest.entry_window_seconds),
                    ),
                    side=claim.side,
                    signal_time=claim.signal_time,
                    entry_delay_seconds=self.config.backtest.entry_delay_seconds,
                    entry_window_seconds=self.config.backtest.entry_window_seconds,
                )
            else:
                entry_band = approximate_entry_price_band(
                    minute_candles,
                    side=claim.side,
                    signal_time=claim.signal_time,
                    entry_delay_seconds=self.config.backtest.entry_delay_seconds,
                    entry_window_seconds=self.config.backtest.entry_window_seconds,
                )
        except ValueError:
            runtime.ledger.reject("missing_entry_data")
            return None

        fill = compute_fill_result(
            wallet_balance=runtime.account.wallet_balance,
            free_balance=runtime.account.free_balance,
            bot=runtime.config,
            exchange_symbol=symbol_data.exchange_symbol,
            entry_price=entry_band.base_price,
        )
        if fill is None:
            runtime.ledger.reject("sizing_failed")
            return None

        tp_percent, sl_percent = _effective_tp_sl(runtime.config)
        fng_value = self.bundle.latest_fng_value(as_of=claim.signal_time, use_previous_day=self.config.backtest.use_previous_day_fng) if runtime.config.fng_enabled else None
        adjusted_tp, adjusted_sl, tp_price, sl_price = compute_effective_tp_sl(
            entry_price=entry_band.base_price,
            side=claim.side,
            tp_percent=tp_percent,
            sl_percent=sl_percent,
            taker_fee_rate=self.config.execution.taker_fee_rate,
            fng_value=fng_value,
            risk_aversion=runtime.config.risk_aversion,
            reverse_mode=False,
        )

        runtime.account.free_balance -= fill.margin_used
        position = Position(
            trade_id=f"{runtime.config.name}:{claim.symbol}:{int(claim.signal_time.timestamp())}",
            bot_name=runtime.config.name,
            strategy=runtime.config.strategy,
            symbol=claim.symbol,
            side=claim.side,
            entry_time=entry_band.base_time,
            entry_price=entry_band.base_price,
            quantity=fill.quantity,
            leverage=runtime.config.leverage,
            tp_percent=adjusted_tp,
            sl_percent=adjusted_sl,
            tp_price=tp_price,
            sl_price=sl_price,
            margin_used=fill.margin_used,
            notional_value=fill.notional_value,
            notes={
                "TPP": f"{adjusted_tp:.3f}",
                "SLP": f"{adjusted_sl:.3f}",
            },
        )
        exit_resolution = resolve_exit_hierarchical(
            position,
            minute_candles=minute_candles,
            hour_candles=symbol_data.futures_1h.between_open(position.entry_time.replace(minute=0, second=0, microsecond=0), end_time + timedelta(hours=2)),
            day_candles=symbol_data.futures_1d.between_open(position.entry_time.replace(hour=0, minute=0, second=0, microsecond=0), end_time + timedelta(days=2)),
            agg_trade_loader=self.bundle.agg_trade_loader(claim.symbol) if use_exact else (lambda _start, _end: []),
        )
        pnl_percent = compute_pnl_percent(position, exit_resolution.exit_price, taker_fee_rate=self.config.execution.taker_fee_rate)
        return ScheduledTrade(
            position=position,
            entry_band=entry_band,
            exit=exit_resolution,
            pnl_percent=pnl_percent,
            signal_time=claim.signal_time,
        )

    def _settle_due_trades(self, runtimes: list[BotRuntime], timestamp: datetime) -> None:
        for runtime in runtimes:
            remaining: list[ScheduledTrade] = []
            for trade in runtime.open_trades:
                if trade.exit.exit_time > timestamp:
                    remaining.append(trade)
                    continue
                profit_amount = trade.position.margin_used * (trade.pnl_percent / 100)
                runtime.account.free_balance += trade.position.margin_used + profit_amount
                runtime.account.wallet_balance += profit_amount
                runtime.ledger.current_balance += profit_amount
                runtime.ledger.trades.append(trade)
                runtime.closed_symbols.append(trade.position.symbol)
                runtime.closed_times_by_symbol.setdefault(trade.position.symbol, []).append(trade.exit.exit_time)
                if trade.exit.reason is ExitReason.SL:
                    runtime.sl_times_by_symbol.setdefault(trade.position.symbol, []).append(trade.exit.exit_time)
            runtime.open_trades = remaining


def apply_bot_overrides(bots: tuple[BotConfig, ...], overrides: dict[str, dict[str, object]]) -> tuple[BotConfig, ...]:
    updated: list[BotConfig] = []
    for bot in bots:
        params = overrides.get(bot.name)
        updated.append(replace(bot, **params) if params else bot)
    return tuple(updated)


def _effective_side_and_risk(bot: BotConfig, side: Side) -> tuple[Side, float, float]:
    if not bot.reverse_mode:
        return side, bot.tp_percent, bot.sl_percent
    reversed_side = Side.SHORT if side is Side.LONG else Side.LONG
    return reversed_side, bot.sl_percent, bot.tp_percent


def _effective_tp_sl(bot: BotConfig) -> tuple[float, float]:
    if not bot.reverse_mode:
        return bot.tp_percent, bot.sl_percent
    return bot.sl_percent, bot.tp_percent


def _runtime_for_bot(runtimes: list[BotRuntime], bot_name: str) -> BotRuntime:
    return next(runtime for runtime in runtimes if runtime.config.name == bot_name)
