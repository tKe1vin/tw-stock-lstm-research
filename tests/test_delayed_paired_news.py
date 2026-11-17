"""Only current daily research may use a later real decision timestamp."""
import numpy as np
import pandas as pd
import pytest

from scripts.paired_research import _sentiment_table, ResearchConfig, TICKERS


def news():
    return pd.DataFrame({"Date": "2026-11-16", "ticker": TICKERS,
                         "Daily_Sentiment": np.zeros(5),
                         "Published_At": "2026-11-16T12:00:00+08:00",
                         "Available_At": "2026-11-17T07:00:00+08:00"})


def test_daily_research_keeps_actual_next_day_decision_and_checks_availability():
    day = pd.DatetimeIndex(["2026-11-16"])
    _scores, evidence, _raw = _sentiment_table(news(), day, TICKERS, ResearchConfig(),
        "2026-11-17T08:00:00+08:00", allow_delayed_decision=True)
    assert evidence["delayed_daily_research"]
    assert evidence["recorded_decision_times"]["2026-11-16"] == "2026-11-17T00:00:00+00:00"
    assert not evidence["news_point_in_time_verified"]
    with pytest.raises(ValueError, match="recorded decision"):
        _sentiment_table(news(), day, TICKERS, ResearchConfig(),
                         "2026-11-17T06:00:00+08:00", allow_delayed_decision=True)


def test_default_historical_decisions_cannot_use_next_day_news():
    with pytest.raises(ValueError, match="on its signal date"):
        _sentiment_table(news(), pd.DatetimeIndex(["2026-11-16"]), TICKERS,
                         ResearchConfig(), "2026-11-17T08:00:00+08:00")


def test_delayed_mode_cannot_relax_multi_day_backtest_or_use_pre_signal_time():
    with pytest.raises(ValueError, match="one current daily"):
        _sentiment_table(news(), pd.DatetimeIndex(["2026-11-13", "2026-11-16"]), TICKERS,
                         ResearchConfig(), allow_delayed_decision=True)
    with pytest.raises(ValueError, match="on its signal date"):
        _sentiment_table(news(), pd.DatetimeIndex(["2026-11-16"]), TICKERS, ResearchConfig(),
                         "2026-11-15T08:00:00+08:00", allow_delayed_decision=True)
