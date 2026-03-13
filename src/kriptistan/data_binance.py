from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from .cache import JsonCache
from .models import AggTrade, Candle, ExchangeSymbol


@dataclass(slots=True)
class RateLimiter:
    limit_per_minute: int = 2400
    _tokens: float = field(init=False, repr=False)
    _updated_at: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.limit_per_minute)
        self._updated_at = time.monotonic()

    def acquire(self, weight: int) -> None:
        while True:
            self._refill()
            if self._tokens >= weight:
                self._tokens -= weight
                return
            time.sleep(0.1)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated_at
        self._updated_at = now
        refill = (self.limit_per_minute / 60) * elapsed
        self._tokens = min(float(self.limit_per_minute), self._tokens + refill)


@dataclass(slots=True)
class BinanceFuturesPublicClient:
    cache: JsonCache
    base_url: str = "https://fapi.binance.com"
    rate_limiter: RateLimiter = field(default_factory=RateLimiter)

    def exchange_info(self) -> dict[str, Any]:
        return self._get_json("/fapi/v1/exchangeInfo", {}, weight=1)

    def exchange_symbols(self) -> dict[str, ExchangeSymbol]:
        payload = self.exchange_info()
        symbols: dict[str, ExchangeSymbol] = {}
        for item in payload.get("symbols", []):
            if item.get("contractType") != "PERPETUAL":
                continue
            filters = {entry["filterType"]: entry for entry in item.get("filters", [])}
            step_size = float(filters.get("LOT_SIZE", {}).get("stepSize", item.get("stepSize", "1")))
            min_qty = float(filters.get("LOT_SIZE", {}).get("minQty", "0"))
            notional_filter = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL") or {}
            min_notional = float(notional_filter.get("notional", notional_filter.get("minNotional", "6")))
            symbols[item["symbol"]] = ExchangeSymbol(
                symbol=item["symbol"],
                base_asset=item["baseAsset"],
                quote_asset=item["quoteAsset"],
                quantity_precision=int(item.get("quantityPrecision", 0)),
                step_size=step_size,
                min_qty=min_qty,
                min_notional=min_notional,
                status=item.get("status", "TRADING"),
            )
        return symbols

    def klines(
        self,
        *,
        symbol: str,
        interval: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1500,
    ) -> list[Candle]:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = _to_millis(start_time)
        if end_time is not None:
            params["endTime"] = _to_millis(end_time)
        payload = self._get_json("/fapi/v1/klines", params, weight=2)
        return [_parse_kline(item) for item in payload]

    def agg_trades(
        self,
        *,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000,
        from_id: int | None = None,
    ) -> list[AggTrade]:
        params: dict[str, Any] = {
            "symbol": symbol,
            "startTime": _to_millis(start_time),
            "endTime": _to_millis(end_time),
            "limit": limit,
        }
        if from_id is not None:
            params["fromId"] = from_id
        payload = self._get_json("/fapi/v1/aggTrades", params, weight=20)
        return [_parse_agg_trade(item) for item in payload]

    def _get_json(self, path: str, params: dict[str, Any], *, weight: int) -> Any:
        cache_key = [path] + [f"{key}={params[key]}" for key in sorted(params)]
        cached = self.cache.get("binance", cache_key)
        if cached is not None:
            return cached
        self.rate_limiter.acquire(weight)
        query = urlencode(params)
        with urlopen(f"{self.base_url}{path}?{query}") as response:
            payload = json.loads(response.read())
        self.cache.set("binance", cache_key, payload)
        return payload


@dataclass(slots=True)
class BinanceSpotPublicClient:
    cache: JsonCache
    base_url: str = "https://api.binance.com"
    rate_limiter: RateLimiter = field(default_factory=lambda: RateLimiter(limit_per_minute=6000))

    def exchange_info(self) -> dict[str, Any]:
        return self._get_json("/api/v3/exchangeInfo", {}, weight=20)

    def exchange_symbols(self) -> dict[str, ExchangeSymbol]:
        payload = self.exchange_info()
        symbols: dict[str, ExchangeSymbol] = {}
        for item in payload.get("symbols", []):
            filters = {entry["filterType"]: entry for entry in item.get("filters", [])}
            step_size = float(filters.get("LOT_SIZE", {}).get("stepSize", "1"))
            min_qty = float(filters.get("LOT_SIZE", {}).get("minQty", "0"))
            min_notional = float(filters.get("MIN_NOTIONAL", {}).get("minNotional", "0"))
            symbols[item["symbol"]] = ExchangeSymbol(
                symbol=item["symbol"],
                base_asset=item["baseAsset"],
                quote_asset=item["quoteAsset"],
                quantity_precision=int(item.get("baseAssetPrecision", 0)),
                step_size=step_size,
                min_qty=min_qty,
                min_notional=min_notional,
                status=item.get("status", "TRADING"),
            )
        return symbols

    def klines(
        self,
        *,
        symbol: str,
        interval: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[Candle]:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = _to_millis(start_time)
        if end_time is not None:
            params["endTime"] = _to_millis(end_time)
        payload = self._get_json("/api/v3/klines", params, weight=2)
        return [_parse_kline(item) for item in payload]

    def _get_json(self, path: str, params: dict[str, Any], *, weight: int) -> Any:
        cache_key = [path] + [f"{key}={params[key]}" for key in sorted(params)]
        cached = self.cache.get("binance", cache_key)
        if cached is not None:
            return cached
        self.rate_limiter.acquire(weight)
        query = urlencode(params)
        with urlopen(f"{self.base_url}{path}?{query}") as response:
            payload = json.loads(response.read())
        self.cache.set("binance", cache_key, payload)
        return payload


def _parse_kline(row: list[Any]) -> Candle:
    return Candle(
        open_time=_from_millis(row[0]),
        close_time=_from_millis(row[6]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
        quote_volume=float(row[7]),
    )


def _parse_agg_trade(row: dict[str, Any]) -> AggTrade:
    return AggTrade(
        trade_id=int(row["a"]),
        timestamp=_from_millis(row["T"]),
        price=float(row["p"]),
        quantity=float(row["q"]),
    )


def _to_millis(value: datetime) -> int:
    normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return int(normalized.timestamp() * 1000)


def _from_millis(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=UTC)
