from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from .config import AppConfig, BacktestConfig, BotConfig
from .execution import approximate_entry_price_band, compute_effective_tp_sl, compute_pnl_percent, resolve_first_hour_exit, resolve_hour_candle_exit, select_entry_price_band
from .gates import anti_repetition_guard, btc_vol_guard_triggered, chase_filter_passes, dead_end_blacklisted, due_date_in_range, resolve_collisions, symbol_on_cooldown
from .indicators import precompute_indicators
from .market_data import MarketDataBundle
from .models import Candidate, CollisionPolicy, CycleStats, EntryClaim, EntryPriceBand, ExitReason, ExitResolution, Position, ResolutionLevel, ScheduledTrade, Side, StrategyContext
from .progress import ProgressPrinter
from .reports import BotLedger, build_portfolio_result
from .sizing import compute_fill_result
from .strategies import evaluate_strategy


@dataclass(slots=True)
class AccountState:
    wallet_balance: float
    free_balance: float


@dataclass(slots=True)
class OpenTrade:
    position: Position
    entry_band: EntryPriceBand
    signal_time: datetime
    use_exact: bool
    resolved_until: datetime


@dataclass(slots=True)
class BotRuntime:
    config: BotConfig
    ledger: BotLedger
    account: AccountState
    open_trades: list[OpenTrade]
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

        runtime_by_name: dict[str, BotRuntime] = {r.config.name: r for r in runtimes}
        timestamps = self.bundle.hourly_timestamps(start=start_time, end=end_time)
        total_timestamps = len(timestamps)
        progress = ProgressPrinter("backtest", total_timestamps)
        try:
            for index, timestamp in enumerate(timestamps, start=1):
                self._settle_due_trades(runtimes, timestamp)
                open_symbols = {
                    trade.position.symbol
                    for runtime in runtimes
                    for trade in runtime.open_trades
                }
                claims: list[EntryClaim] = []
                btc_guard_active = self._btc_guard_active(timestamp)
                base_candidates = self._scanner_candidates(timestamp)
                idx_cache: dict[str, int] = {}
                slice_cache: dict[str, list] = {}
                for runtime in runtimes:
                    effective_tp_percent, _ = _effective_tp_sl(self.config.backtest, runtime.config)
                    if runtime.available_slots <= 0:
                        runtime.ledger.equity_points.append(runtime.ledger.current_balance)
                        continue
                    for candidate in self._filter_candidates(runtime, base_candidates, timestamp, open_symbols):
                        symbol_data = self.bundle.symbols[candidate.symbol]
                        indicators = self._get_indicators(candidate.symbol)
                        if candidate.symbol not in idx_cache:
                            idx_cache[candidate.symbol] = symbol_data.futures_1h.closed_until_idx(timestamp)
                            slice_cache[candidate.symbol] = symbol_data.futures_1h.last_n_closed(timestamp, 300)
                        end_idx = idx_cache[candidate.symbol]
                        candles = slice_cache[candidate.symbol]
                        decision = evaluate_strategy(
                            runtime.config.strategy,
                            StrategyContext(
                                symbol=candidate.symbol,
                                candles=candles,
                                cycle_stats=candidate.cycle_stats,
                                tp_percent=effective_tp_percent,
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
                            _effective_side(self.config.backtest, runtime.config, raw_side),
                            latest,
                            tp_percent=effective_tp_percent,
                            close_position_limit=self.config.execution.chase_close_pos,
                            range_multiple_tp=self.config.execution.chase_range_mult_tp,
                        ):
                            runtime.ledger.reject("chase_filter")
                            continue
                        actual_side = _effective_side(self.config.backtest, runtime.config, raw_side)
                        if self.config.execution.btc_confirm_entry_enabled and not self._btc_confirm(
                            runtime,
                            candidate.symbol,
                            timestamp,
                            actual_side,
                            effective_tp_percent,
                        ):
                            runtime.ledger.reject("btc_confirm")
                            continue
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
                    runtime = runtime_by_name[rejected.claim.bot_name]
                    runtime.ledger.reject(rejected.reason)

                selected_by_bot: dict[str, list[EntryClaim]] = {}
                for claim in winners:
                    selected_by_bot.setdefault(claim.bot_name, []).append(claim)
                accepted_claims: list[EntryClaim] = []
                for bot_name, bot_claims in selected_by_bot.items():
                    runtime = runtime_by_name[bot_name]
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
                    runtime = runtime_by_name[claim.bot_name]
                    trade = self._open_trade(runtime, claim)
                    if trade is None:
                        continue
                    runtime.open_trades.append(trade)
                    open_symbols.add(trade.position.symbol)
                progress.update(index)
        finally:
            progress.close()

        self._settle_due_trades(runtimes, end_time + timedelta(days=3650), force_close=True)
        result = build_portfolio_result(
            start=start_time,
            end=end_time,
            ledgers=ledgers,
            collision_policy=policy,
        )
        return result

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
            stats = self.bundle.multi_set_cycle_stats_as_of(symbol, timestamp, self.config.backtest.scanner_param_sets)
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
                all_close_times=runtime.closed_times_by_symbol.get(symbol, []),
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

    def _btc_confirm(
        self,
        runtime: BotRuntime,
        symbol: str,
        timestamp: datetime,
        side: Side,
        tp_percent: float,
    ) -> bool:
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
                tp_percent=tp_percent,
                indicators=indicators,
                candle_end_idx=end_idx,
            ),
        )
        if confirm_decision is None:
            return False
        return confirm_decision.side is side

    def _get_indicators(self, symbol: str):
        cached = self._indicator_cache.get(symbol)
        if cached is not None:
            return cached
        indicators = precompute_indicators(self.bundle.symbols[symbol].futures_1h.candles)
        self._indicator_cache[symbol] = indicators
        return indicators

    def _open_trade(self, runtime: BotRuntime, claim: EntryClaim) -> OpenTrade | None:
        symbol_data = self.bundle.symbols[claim.symbol]
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
                entry_minute_start = claim.signal_time.replace(second=0, microsecond=0)
                minute_candles = self.bundle.minute_candles_between(
                    claim.symbol,
                    entry_minute_start,
                    entry_minute_start + timedelta(minutes=2),
                )
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

        tp_percent, sl_percent = _effective_tp_sl(self.config.backtest, runtime.config)
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
        return OpenTrade(
            position=position,
            entry_band=entry_band,
            signal_time=claim.signal_time,
            use_exact=use_exact,
            resolved_until=position.entry_time,
        )

    def _settle_due_trades(self, runtimes: list[BotRuntime], timestamp: datetime, *, force_close: bool = False) -> None:
        for runtime in runtimes:
            remaining: list[OpenTrade] = []
            for trade in runtime.open_trades:
                exit_resolution = self._resolve_trade_until(trade, timestamp, force_close=force_close)
                if exit_resolution is None:
                    remaining.append(trade)
                    continue
                pnl_percent = compute_pnl_percent(
                    trade.position,
                    exit_resolution.exit_price,
                    taker_fee_rate=self.config.execution.taker_fee_rate,
                )
                completed = ScheduledTrade(
                    position=trade.position,
                    entry_band=trade.entry_band,
                    exit=exit_resolution,
                    pnl_percent=pnl_percent,
                    signal_time=trade.signal_time,
                )
                profit_amount = trade.position.margin_used * (pnl_percent / 100)
                runtime.account.free_balance += trade.position.margin_used + profit_amount
                runtime.account.wallet_balance += profit_amount
                runtime.ledger.current_balance += profit_amount
                runtime.ledger.trades.append(completed)
                runtime.closed_symbols.append(trade.position.symbol)
                runtime.closed_times_by_symbol.setdefault(trade.position.symbol, []).append(exit_resolution.exit_time)
                if exit_resolution.reason is ExitReason.SL:
                    runtime.sl_times_by_symbol.setdefault(trade.position.symbol, []).append(exit_resolution.exit_time)
            runtime.open_trades = remaining

    def _resolve_trade_until(self, trade: OpenTrade, cutoff: datetime, *, force_close: bool = False) -> ExitResolution | None:
        position = trade.position
        symbol_data = self.bundle.symbols[position.symbol]
        agg_trade_loader = self.bundle.agg_trade_loader(position.symbol) if trade.use_exact else _empty_agg_trade_loader
        first_hour_end = _first_hour_end(position.entry_time)
        current = trade.resolved_until

        if current <= position.entry_time:
            same_hour_start = position.entry_time.replace(second=0, microsecond=0)
            same_hour_end = min(first_hour_end, self.bundle.end)
            minute_candles = self.bundle.minute_candles_between(position.symbol, same_hour_start, same_hour_end)
            resolution = resolve_first_hour_exit(
                position,
                minute_candles=minute_candles,
                agg_trade_loader=agg_trade_loader,
            )
            if resolution is not None:
                return resolution
            trade.resolved_until = first_hour_end
            current = first_hour_end

        while current < cutoff:
            hour_candles = symbol_data.futures_1h.between_open(current, current + timedelta(hours=1))
            if not hour_candles:
                break
            hour_candle = hour_candles[0]
            resolution = resolve_hour_candle_exit(
                position,
                hour_candle=hour_candle,
                minute_candles_loader=lambda start=hour_candle.open_time, end=hour_candle.close_time, symbol=position.symbol: self.bundle.minute_candles_between(
                    symbol,
                    start,
                    end,
                ),
                agg_trade_loader=agg_trade_loader,
            )
            if resolution is not None:
                return resolution
            trade.resolved_until = hour_candle.close_time
            current = trade.resolved_until

        if force_close:
            return self._force_close_resolution(position, cutoff)
        return None

    def _force_close_resolution(self, position: Position, cutoff: datetime) -> ExitResolution:
        final_as_of = min(cutoff, self.bundle.end)
        hour_candles = self.bundle.symbols[position.symbol].futures_1h.last_n_closed(final_as_of, 1)
        if hour_candles:
            candle = hour_candles[-1]
            return ExitResolution(
                reason=ExitReason.OPEN,
                exit_time=candle.close_time,
                exit_price=candle.close,
                resolution_level=ResolutionLevel.HOUR,
            )
        return ExitResolution(
            reason=ExitReason.OPEN,
            exit_time=position.entry_time,
            exit_price=position.entry_price,
            resolution_level=ResolutionLevel.HOUR,
        )


def apply_bot_overrides(bots: tuple[BotConfig, ...], overrides: dict[str, dict[str, object]]) -> tuple[BotConfig, ...]:
    updated: list[BotConfig] = []
    for bot in bots:
        params = overrides.get(bot.name)
        updated.append(replace(bot, **params) if params else bot)
    return tuple(updated)


def _inverted(backtest: BacktestConfig, bot: BotConfig) -> bool:
    return backtest.reverse_mode or bot.reverse_mode


def _effective_side(backtest: BacktestConfig, bot: BotConfig, side: Side) -> Side:
    if not _inverted(backtest, bot):
        return side
    reversed_side = Side.SHORT if side is Side.LONG else Side.LONG
    return reversed_side


def _effective_tp_sl(backtest: BacktestConfig, bot: BotConfig) -> tuple[float, float]:
    if not _inverted(backtest, bot):
        return bot.tp_percent, bot.sl_percent
    return bot.sl_percent, bot.tp_percent


def _first_hour_end(entry_time: datetime) -> datetime:
    return entry_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)


def _empty_agg_trade_loader(_start: datetime, _end: datetime):
    return []
