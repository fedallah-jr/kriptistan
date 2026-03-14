from pathlib import Path

import pytest

from kriptistan.config import load_config


def test_load_config_normalizes_naive_datetimes_to_utc() -> None:
    config = load_config(Path("configs/backtest.toml"))

    assert config.backtest.start.tzinfo is not None
    assert config.backtest.end.tzinfo is not None


def test_load_config_accepts_global_reverse_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[backtest]",
                "start = 2026-03-01T00:00:00Z",
                "end = 2026-03-02T00:00:00Z",
                "reverse_mode = true",
                "",
                "[[bots]]",
                'name = "trend"',
                'strategy = "TREND_MOM"',
                "reverse_mode = false",
            ]
        )
    )

    config = load_config(config_path)

    assert config.backtest.reverse_mode is True
    assert config.bots[0].reverse_mode is False


def test_load_config_rejects_mixed_global_and_per_bot_reverse_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[backtest]",
                "start = 2026-03-01T00:00:00Z",
                "end = 2026-03-02T00:00:00Z",
                "reverse_mode = true",
                "",
                "[[bots]]",
                'name = "trend"',
                'strategy = "TREND_MOM"',
                "reverse_mode = true",
            ]
        )
    )

    with pytest.raises(ValueError, match="global \\[backtest\\]\\.reverse_mode=true"):
        load_config(config_path)
