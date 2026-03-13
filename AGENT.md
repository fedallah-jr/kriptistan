# kriptistan Agent Notes

## Purpose
- This repository is a futures-first Binance strategy research codebase.
- The primary target is offline backtesting and walk-forward optimization.
- Live trading should reuse the same scanner, strategy, gate, sizing, and execution rules with different I/O adapters.

## Core Approach
- Treat the simulator as event-driven on closed hourly candles.
- Use only data that is known as-of the evaluation timestamp.
- Run entry logic on `1h` candles and daily cycle timing on closed `1d` candles.
- Resolve exits hierarchically:
  - entry minute via futures `aggTrades` when exact mode is available
  - then `1m`
  - then `1h`
  - optionally `1d` as a skip layer for long holds
- If a lower-level ambiguity cannot be resolved because trade-level data is unavailable, use a conservative stop-loss-first fallback.

## Data Rules
- Daily cycle stats must use only the most recent fully closed daily candle.
- Fear & Greed must be joined `as-of`; the default conservative mode is previous-day value for the whole UTC day.
- Binance requests should be cached locally and the simulation should run from cached data.
- Futures `aggTrades` are only expected to be exact for roughly the most recent year; older periods should use approximate mode.

## Collision Policy
- Never let loop order implicitly decide symbol ownership.
- Evaluate all candidate claims for a timestamp first, then resolve same-symbol conflicts with an explicit collision policy.
- Supported policies:
  - `fixed_priority`
  - `signal_priority`
  - `seeded_shuffle`

## Optimization Policy
- Walk-forward is the default research mode for any tuned parameter.
- Tuning and testing must be separated in time.
- Candidate parameter grids are defined per bot in config and combined into bounded Cartesian products.
- The selected variant for a window is the one with the best configured objective on the training slice, then it is replayed unchanged on the following test slice.

## Implementation Preferences
- Prefer pure functions for indicators, cycle stats, strategy logic, gates, sizing, and PnL math.
- Keep network and filesystem access in thin adapters.
- Prefer lazy loading for `1m` candles and `aggTrades`; preload only `1d`, `1h`, and BTC guard data.
- Preserve deterministic behavior by fixing shuffle seeds and avoiding hidden ordering dependencies.
