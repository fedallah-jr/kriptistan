from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .cache import JsonCache
from .config import AppConfig
from .data_binance import BinanceFuturesPublicClient, BinanceSpotPublicClient
from .data_fng import FearGreedClient, FearGreedPoint
from .models import AggTrade, Candle, CycleStats, ExchangeSymbol


@dataclass(slots=True)
class CandleSeries:
    candles: list[Candle]
    _close_times: list[datetime] = field(init=False, repr=False)
    _open_times: list[datetime] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.candles = sorted(self.candles, key=lambda item: item.open_time)
        self._close_times = [candle.close_time for candle in self.candles]
        self._open_times = [candle.open_time for candle in self.candles]

    def closed_until(self, as_of: datetime) -> list[Candle]:
        idx = bisect_right(self._close_times, as_of)
        return self.candles[:idx]

    def between_open(self, start: datetime, end: datetime) -> list[Candle]:
        left = bisect_left(self._open_times, start)
        right = bisect_left(self._open_times, end)
        return self.candles[left:right]


@dataclass(slots=True)
class SymbolMarketData:
    symbol: str
    exchange_symbol: ExchangeSymbol
    futures_1d: CandleSeries
    futures_1h: CandleSeries
    confirm_symbol: str | None = None
    confirm_1d: CandleSeries | None = None
    confirm_1h: CandleSeries | None = None
    _futures_1m: CandleSeries | None = None

    def closed_daily(self, as_of: datetime) -> list[Candle]:
        return self.futures_1d.closed_until(as_of)

    def closed_hourly(self, as_of: datetime) -> list[Candle]:
        return self.futures_1h.closed_until(as_of)

    def closed_confirm_daily(self, as_of: datetime) -> list[Candle]:
        if self.confirm_1d is None:
            return []
        return self.confirm_1d.closed_until(as_of)

    def closed_confirm_hourly(self, as_of: datetime) -> list[Candle]:
        if self.confirm_1h is None:
            return []
        return self.confirm_1h.closed_until(as_of)

    def minute_series(self) -> CandleSeries:
        if self._futures_1m is None:
            raise RuntimeError("minute series not loaded")
        return self._futures_1m


@dataclass(slots=True)
class MarketDataBundle:
    start: datetime
    end: datetime
    warmup_start: datetime
    symbols: dict[str, SymbolMarketData]
    btc_5m: CandleSeries
    fng_points: list[FearGreedPoint]
    repo: "MarketDataRepository"
    _cycle_cache: dict[tuple[str, datetime], CycleStats | None] = field(default_factory=dict)

    def hourly_timestamps(self, *, start: datetime, end: datetime) -> list[datetime]:
        reference = next(iter(self.symbols.values())).futures_1h
        candles = reference.between_open(start - timedelta(hours=2), end + timedelta(hours=2))
        return [candle.close_time for candle in candles if start <= candle.close_time <= end]

    def cycle_stats_as_of(self, symbol: str, as_of: datetime) -> CycleStats | None:
        daily = self.symbols[symbol].closed_daily(as_of)
        if not daily:
            return None
        key = (symbol, daily[-1].close_time)
        cached = self._cycle_cache.get(key)
        if cached is not None or key in self._cycle_cache:
            return cached
        from .cycles import scan_symbol_cycles

        stats = scan_symbol_cycles(symbol, daily)
        self._cycle_cache[key] = stats
        return stats

    def confirm_cycle_stats_as_of(self, symbol: str, as_of: datetime) -> CycleStats | None:
        data = self.symbols[symbol]
        if data.confirm_symbol is None:
            return None
        daily = data.closed_confirm_daily(as_of)
        if not daily:
            return None
        from .cycles import scan_symbol_cycles

        return scan_symbol_cycles(data.confirm_symbol, daily)

    def btc_guard_slice(self, as_of: datetime) -> list[Candle]:
        closed = self.btc_5m.closed_until(as_of)
        return closed[-12:]

    def latest_fng_value(self, *, as_of: datetime, use_previous_day: bool) -> int | None:
        point = self.repo.fng.latest_as_of(points=self.fng_points, as_of=as_of, use_previous_day=use_previous_day)
        return point.value if point is not None else None

    def quote_volume_24h(self, symbol: str, as_of: datetime) -> float:
        hourly = self.symbols[symbol].closed_hourly(as_of)
        recent = hourly[-24:]
        return sum(candle.quote_volume for candle in recent)

    def minute_candles(self, symbol: str) -> list[Candle]:
        data = self.symbols[symbol]
        if data._futures_1m is None:
            candles = self.repo.fetch_futures_klines(
                symbol=symbol,
                interval="1m",
                start=self.warmup_start,
                end=self.end,
            )
            data._futures_1m = CandleSeries(candles)
        return data._futures_1m.candles

    def agg_trade_loader(self, symbol: str):
        def load(start: datetime, end: datetime) -> list[AggTrade]:
            return self.repo.fetch_agg_trades(symbol=symbol, start=start, end=end)

        return load


@dataclass(slots=True)
class MarketDataRepository:
    config: AppConfig
    cache_dir: Path = Path("data/cache")
    futures: BinanceFuturesPublicClient = field(init=False, repr=False)
    spot: BinanceSpotPublicClient = field(init=False, repr=False)
    fng: FearGreedClient = field(init=False, repr=False)
    _futures_symbols: dict[str, ExchangeSymbol] | None = field(init=False, default=None, repr=False)
    _spot_symbols: dict[str, ExchangeSymbol] | None = field(init=False, default=None, repr=False)
    _agg_cache: dict[tuple[str, datetime], list[AggTrade]] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        cache = JsonCache(self.cache_dir)
        self.futures = BinanceFuturesPublicClient(cache)
        self.spot = BinanceSpotPublicClient(cache)
        self.fng = FearGreedClient(cache)

    def list_futures_symbols(self) -> dict[str, ExchangeSymbol]:
        if self._futures_symbols is None:
            payload = self.futures.exchange_symbols()
            self._futures_symbols = {
                symbol: item
                for symbol, item in payload.items()
                if item.quote_asset == "USDT" and item.status == "TRADING"
            }
        return self._futures_symbols

    def list_spot_symbols(self) -> dict[str, ExchangeSymbol]:
        if self._spot_symbols is None:
            payload = self.spot.exchange_symbols()
            self._spot_symbols = {symbol: item for symbol, item in payload.items() if item.status == "TRADING"}
        return self._spot_symbols

    def build_bundle(self, symbols: list[str] | None = None, *, warmup_days: int | None = None) -> MarketDataBundle:
        futures_symbols = self.list_futures_symbols()
        selected = symbols or sorted(futures_symbols)
        warmup = warmup_days if warmup_days is not None else self.config.backtest.warmup_days
        warmup_start = self.config.backtest.start - timedelta(days=warmup)
        spot_symbols = self.list_spot_symbols() if self.config.execution.btc_confirm_entry_enabled else {}
        symbol_data: dict[str, SymbolMarketData] = {}

        for symbol in selected:
            meta = futures_symbols[symbol]
            futures_1d = CandleSeries(self.fetch_futures_klines(symbol=symbol, interval="1d", start=warmup_start, end=self.config.backtest.end))
            futures_1h = CandleSeries(self.fetch_futures_klines(symbol=symbol, interval="1h", start=warmup_start, end=self.config.backtest.end))
            confirm_symbol = f"{meta.base_asset}BTC"
            confirm_1d = None
            confirm_1h = None
            if confirm_symbol in spot_symbols:
                confirm_1d = CandleSeries(self.fetch_spot_klines(symbol=confirm_symbol, interval="1d", start=warmup_start, end=self.config.backtest.end))
                confirm_1h = CandleSeries(self.fetch_spot_klines(symbol=confirm_symbol, interval="1h", start=warmup_start, end=self.config.backtest.end))
            symbol_data[symbol] = SymbolMarketData(
                symbol=symbol,
                exchange_symbol=meta,
                futures_1d=futures_1d,
                futures_1h=futures_1h,
                confirm_symbol=confirm_symbol if confirm_1h is not None else None,
                confirm_1d=confirm_1d,
                confirm_1h=confirm_1h,
            )

        btc_5m = CandleSeries(self.fetch_futures_klines(symbol="BTCUSDT", interval="5m", start=warmup_start, end=self.config.backtest.end))
        fng_points = self.fng.history(limit=0)
        return MarketDataBundle(
            start=self.config.backtest.start,
            end=self.config.backtest.end,
            warmup_start=warmup_start,
            symbols=symbol_data,
            btc_5m=btc_5m,
            fng_points=fng_points,
            repo=self,
        )

    def fetch_futures_klines(self, *, symbol: str, interval: str, start: datetime, end: datetime) -> list[Candle]:
        client = self.futures
        cursor = start
        results: list[Candle] = []
        while cursor < end:
            batch = client.klines(symbol=symbol, interval=interval, start_time=cursor, end_time=end)
            if not batch:
                break
            results.extend(candle for candle in batch if candle.open_time < end)
            next_cursor = batch[-1].close_time
            if next_cursor <= cursor:
                break
            cursor = next_cursor
        return _dedupe_candles(results)

    def fetch_spot_klines(self, *, symbol: str, interval: str, start: datetime, end: datetime) -> list[Candle]:
        client = self.spot
        cursor = start
        results: list[Candle] = []
        while cursor < end:
            batch = client.klines(symbol=symbol, interval=interval, start_time=cursor, end_time=end)
            if not batch:
                break
            results.extend(candle for candle in batch if candle.open_time < end)
            next_cursor = batch[-1].close_time
            if next_cursor <= cursor:
                break
            cursor = next_cursor
        return _dedupe_candles(results)

    def fetch_agg_trades(self, *, symbol: str, start: datetime, end: datetime) -> list[AggTrade]:
        hour_start = start.replace(minute=0, second=0, microsecond=0)
        cached = self._agg_cache.get((symbol, hour_start))
        if cached is None:
            hour_end = min(hour_start + timedelta(hours=1), end)
            cached = self._fetch_hour_agg_trades(symbol=symbol, start=hour_start, end=hour_end)
            self._agg_cache[(symbol, hour_start)] = cached
        return [trade for trade in cached if start <= trade.timestamp <= end]

    def _fetch_hour_agg_trades(self, *, symbol: str, start: datetime, end: datetime) -> list[AggTrade]:
        cursor = start
        results: list[AggTrade] = []
        while cursor < end:
            batch = self.futures.agg_trades(symbol=symbol, start_time=cursor, end_time=end)
            if not batch:
                break
            results.extend(batch)
            if len(batch) < 1000:
                break
            next_cursor = batch[-1].timestamp + timedelta(milliseconds=1)
            if next_cursor <= cursor:
                break
            cursor = next_cursor
        return sorted({trade.trade_id: trade for trade in results}.values(), key=lambda item: (item.timestamp, item.trade_id))


def _dedupe_candles(candles: list[Candle]) -> list[Candle]:
    deduped = {candle.open_time: candle for candle in candles}
    return sorted(deduped.values(), key=lambda item: item.open_time)
