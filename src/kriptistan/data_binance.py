from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import sys
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

from .cache import JsonCache
from .models import AggTrade, Candle, ExchangeSymbol

_DEFAULT_HTTP_RETRY_DELAY_SECONDS = 15.0
_MAX_HTTP_RETRY_DELAY_SECONDS = 300.0
_RETRYABLE_HTTP_STATUS_CODES = {418, 429}


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

    def sync_from_server(self, used_weight: int) -> None:
        """Sync local token count from Binance ``X-MBX-USED-WEIGHT-1m`` header.

        Only adjusts *downward* so the client never races ahead of the server.
        """
        self._refill()
        server_remaining = max(0.0, float(self.limit_per_minute) - used_weight)
        if server_remaining < self._tokens:
            self._tokens = server_remaining

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
        return _get_json_with_backoff(
            cache=self.cache,
            rate_limiter=self.rate_limiter,
            base_url=self.base_url,
            path=path,
            params=params,
            weight=weight,
        )


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
        return _get_json_with_backoff(
            cache=self.cache,
            rate_limiter=self.rate_limiter,
            base_url=self.base_url,
            path=path,
            params=params,
            weight=weight,
        )


def _get_json_with_backoff(
    *,
    cache: JsonCache,
    rate_limiter: RateLimiter,
    base_url: str,
    path: str,
    params: dict[str, Any],
    weight: int,
) -> Any:
    cache_key = [path] + [f"{key}={params[key]}" for key in sorted(params)]
    cached = cache.get("binance", cache_key)
    if cached is not None:
        return cached

    query = urlencode(params)
    url = f"{base_url}{path}?{query}"
    next_delay = _DEFAULT_HTTP_RETRY_DELAY_SECONDS
    attempt = 1
    while True:
        rate_limiter.acquire(weight)
        try:
            with urlopen(url) as response:
                payload = json.loads(response.read())
                _sync_rate_limiter(rate_limiter, response.headers)
        except HTTPError as exc:
            if exc.code not in _RETRYABLE_HTTP_STATUS_CODES:
                raise
            _sync_rate_limiter(rate_limiter, exc.headers)
            delay = _retry_delay_seconds(exc, fallback_seconds=next_delay)
            _log_rate_limit_wait(path=path, params=params, status_code=exc.code, delay_seconds=delay, attempt=attempt)
            time.sleep(delay)
            next_delay = min(max(delay * 2, _DEFAULT_HTTP_RETRY_DELAY_SECONDS), _MAX_HTTP_RETRY_DELAY_SECONDS)
            attempt += 1
            continue
        cache.set("binance", cache_key, payload)
        return payload


def _sync_rate_limiter(rate_limiter: RateLimiter, headers: Any) -> None:
    if headers is None:
        return
    used = headers.get("X-MBX-USED-WEIGHT-1m")
    if used is not None:
        try:
            rate_limiter.sync_from_server(int(used))
        except (ValueError, TypeError):
            pass


def _retry_delay_seconds(exc: HTTPError, *, fallback_seconds: float) -> float:
    if exc.headers is not None:
        retry_after = exc.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass
    return fallback_seconds


def _log_rate_limit_wait(
    *,
    path: str,
    params: dict[str, Any],
    status_code: int,
    delay_seconds: float,
    attempt: int,
) -> None:
    symbol = params.get("symbol")
    target = f"{path} symbol={symbol}" if symbol is not None else path
    print(
        f"Binance throttled ({status_code}) on {target}; waiting {delay_seconds:.1f}s before retry {attempt + 1}.",
        file=sys.stderr,
    )


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
