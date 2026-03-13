# kriptistan

Greenfield backtester for a multi-bot Binance futures trading system.

The first implementation pass focuses on:

- deterministic config and domain models
- daily cycle scanning
- strategy and gate primitives
- explicit collision handling
- hierarchical TP/SL resolution with exact futures aggregate trades as the final arbiter

The simulation is intended to run offline from cached Binance data. Live execution adapters come later.
