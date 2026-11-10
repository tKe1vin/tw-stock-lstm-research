"""V3 target alignment, paired data coverage, and train-only model state."""
import json

import numpy as np
import pandas as pd
import pytest

from research.v3 import models


def synthetic_data(days=1100):
    dates = pd.bdate_range("2020-01-01", periods=days)
    rows, news = [], []
    for number, ticker in enumerate(models.TICKERS + (models.BENCHMARK,)):
        # Nonconstant deterministic signal with sufficiently long warmup.
        returns = 0.0002 + 0.004 * np.sin(np.arange(days) / (8 + number)) + number * 0.00001
        tri = 100 * np.cumprod(1 + returns)
        for i, date in enumerate(dates):
            rows.append({"Date": date, "Ticker": ticker, "Total_Return_Index": tri[i],
                         "Volume": 10000 + i + 1000 * (1 + np.sin(i / 20))})
            if ticker != models.BENCHMARK:
                news.append({"Date": date, "Ticker": ticker, "Daily_Sentiment": float(np.sin(i / 12 + number)),
                             "News_Count": i % 7, "Availability_Verified": False})
    return pd.DataFrame(rows), pd.DataFrame(news), dates


def prepared_data(days=1100):
    bars, news, dates = synthetic_data(days)
    features, targets, metadata = models.prepare_features(bars, news)
    return features, targets, metadata, dates


def small_split(metadata, dates):
    plan = models.make_splits(metadata["eligible_dates"], session_dates=dates, train_samples=100,
                             validation_samples=20, test_samples=20, n_folds=3, min_train_samples=100)
    return plan["folds"][-1]


def test_target_is_next_close_to_sixth_session_excess_not_signal_close():
    bars, news, dates = synthetic_data(100)
    features, targets, metadata = models.prepare_features(bars, news)
    panel = bars.pivot(index="Date", columns="Ticker", values="Total_Return_Index")
    i, ticker = 50, models.TICKERS[0]
    wanted = 100 * ((panel.at[dates[i + 6], ticker] / panel.at[dates[i + 1], ticker] - 1)
                    - (panel.at[dates[i + 6], models.BENCHMARK] / panel.at[dates[i + 1], models.BENCHMARK] - 1))
    assert targets.at[(dates[i], ticker), "Target_Excess_Pct"] == pytest.approx(wanted)
    assert list(features.columns) == list(models.PRICE_FEATURES + models.NEWS_FEATURES)
    assert metadata["historical_news_point_in_time_proven"] is False
    assert targets.loc[dates[-6]:].Target_Excess_Pct.isna().all()


def test_full_calendar_purge_does_not_compress_missing_eligible_dates():
    dates = pd.bdate_range("2020-01-01", periods=500)
    eligible = dates[30:-6].delete(np.arange(0, len(dates[30:-6]), 5))
    split = models.make_splits(eligible, session_dates=dates, train_samples=100, validation_samples=20,
                              test_samples=20, n_folds=3, min_train_samples=100)
    for fold in split["folds"]:
        assert dates.get_loc(fold["train"][-1]) + 6 < dates.get_loc(fold["validation"][0])
    for first, second in zip(split["folds"], split["folds"][1:]):
        assert dates.get_loc(first["validation"][-1]) + 6 < dates.get_loc(second["validation"][0])
    assert dates.get_loc(split["folds"][-1]["validation"][-1]) + 6 < dates.get_loc(split["test"][0])
    assert dates.get_loc(split["final"]["train"][-1]) + 6 < dates.get_loc(split["test"][0])


def test_scaler_and_ridge_weights_ignore_future_feature_extremes():
    features, targets, metadata, dates = prepared_data(500)
    split = small_split(metadata, dates)
    predictions, artifact = models.fit_predict(features, targets, split, "ridge_news")
    shocked = features.copy()
    future = shocked.index.get_level_values("Date") >= split["validation"][0]
    shocked.loc[future] *= 1e7
    _, other = models.fit_predict(shocked, targets, split, "ridge_news")
    assert artifact["metadata"]["scaler"] == other["metadata"]["scaler"]
    np.testing.assert_array_equal(artifact["arrays"]["coefficient"], other["arrays"]["coefficient"])
    assert artifact["metadata"]["test_used_for_training"] is False
    assert artifact["metadata"]["outer_validation_used_for_training"] is False
    json.dumps(artifact["metadata"])


def test_news_and_price_share_all_dates_and_targets_and_news_changes_model():
    features, targets, metadata, dates = prepared_data(500)
    split = small_split(metadata, dates)
    price, _ = models.fit_predict(features, targets, split, "ridge_price")
    news, artifact = models.fit_predict(features, targets, split, "ridge_news")
    pd.testing.assert_frame_equal(price[["Date", "Ticker", "Actual", "Split"]], news[["Date", "Ticker", "Actual", "Split"]])
    assert not np.allclose(price.Predicted, news.Predicted)
    assert artifact["metadata"]["news_channels_used"] is True


def test_missing_news_has_explicit_flags_without_forward_fill():
    bars, news, dates = synthetic_data(100)
    ticker = models.TICKERS[0]
    news = news[~((news.Date == dates[50]) & (news.Ticker == ticker))]
    features, _, _ = models.prepare_features(bars, news)
    row = features.loc[(dates[50], ticker)]
    assert row.Sentiment == 0
    assert row.News_Score_Missing == 1
    assert row.News_Count_Missing == 1
    assert row.News_Availability_Verified == 0


def test_nonfinite_news_feature_rejected_by_price_control_too():
    features, targets, metadata, dates = prepared_data(500)
    split = small_split(metadata, dates)
    features.loc[(split["validation"][0], models.TICKERS[0]), "Sentiment"] = np.nan
    for candidate in ("ridge_price", "ridge_news"):
        with pytest.raises(ValueError, match="Non-finite paired"):
            models.fit_predict(features, targets, split, candidate)


def test_saved_ridge_reproduces_without_fit_or_targets(monkeypatch):
    features, targets, metadata, dates = prepared_data(500)
    split = small_split(metadata, dates)
    predictions, artifact = models.fit_predict(features, targets, split, "ridge_news")
    monkeypatch.setattr(models.Ridge, "fit", lambda *args, **kwargs: pytest.fail("frozen prediction must never fit"))
    frozen = models.predict_saved(features, split["validation"], artifact)
    np.testing.assert_allclose(frozen.Predicted, predictions.Predicted, atol=1e-10, rtol=1e-10)


def test_split_overlap_and_unregistered_candidates_rejected():
    features, targets, metadata, dates = prepared_data(500)
    split = small_split(metadata, dates)
    split["validation"] = split["train"][-20:]
    with pytest.raises(ValueError, match="crosses"):
        models.fit_predict(features, targets, split, "ridge_news")
    with pytest.raises(ValueError, match="Unregistered"):
        models.fit_predict(features, targets, split, "try_until_winner")


def test_bare_close_cannot_silently_become_total_return():
    bars, news, _ = synthetic_data(100)
    bars = bars.rename(columns={"Total_Return_Index": "Close"})
    with pytest.raises(ValueError, match="Total_Return_Index"):
        models.prepare_features(bars, news)


def test_duplicate_source_rows_rejected():
    bars, news, _ = synthetic_data(100)
    with pytest.raises(ValueError, match="Duplicate bars"):
        models.prepare_features(pd.concat([bars, bars.iloc[:1]]), news)
    with pytest.raises(ValueError, match="Duplicate sentiment"):
        models.prepare_features(bars, pd.concat([news, news.iloc[:1]]))


def test_halted_entry_or_exit_is_not_a_training_target_or_shorter_horizon():
    bars, news, dates = synthetic_data(120)
    bars["Tradable"] = True
    bars.loc[(bars.Date == dates[70]) & (bars.Ticker == models.BENCHMARK), "Tradable"] = False
    _, targets, metadata = models.prepare_features(bars, news)
    assert targets.loc[dates[69]].Target_Excess_Pct.isna().all()  # Entry would be halted.
    assert targets.loc[dates[64]].Target_Excess_Pct.isna().all()  # Exit would be halted.
    assert targets.loc[dates[63]].Target_Excess_Pct.notna().all()
    assert metadata["session_dates"][70] == str(dates[70].date())
