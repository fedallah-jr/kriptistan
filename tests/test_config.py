from pathlib import Path

from kriptistan.config import load_config


def test_load_config_normalizes_naive_datetimes_to_utc() -> None:
    config = load_config(Path("configs/backtest.toml"))

    assert config.backtest.start.tzinfo is not None
    assert config.backtest.end.tzinfo is not None
