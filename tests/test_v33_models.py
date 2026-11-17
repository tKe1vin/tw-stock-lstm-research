"""Monthly horizon, purged training, paired news controls and portable HGB replay."""
import json

import numpy as np
import pandas as pd
import pytest

from research.v3_3 import models


def synthetic_data(days=600):
    dates = pd.bdate_range("2020-01-01", periods=days)
    bars, news = [], []
    for number, ticker in enumerate(models.TICKERS + (models.BENCHMARK,)):
        ret = 0.0003 + .003 * np.sin(np.arange(days)/(9+number)) + number*.00003
        tri = 100*np.cumprod(1+ret)
        for i, day in enumerate(dates):
            bars.append({"Date": day, "Ticker": ticker, "Total_Return_Index": tri[i],
                         "Volume": 10000+i+1000*np.sin(i/30+number), "Tradable": True})
            if ticker != models.BENCHMARK:
                news.append({"Date": day, "Ticker": ticker, "Daily_Sentiment": np.sin(i/14+number),
                             "News_Count": i % 4, "Availability_Verified": False})
    return pd.DataFrame(bars), pd.DataFrame(news), dates


def prepared(days=600):
    bars, news, dates = synthetic_data(days)
    features, target, meta = models.prepare_features(bars, news)
    plan = models.make_splits(meta["eligible_dates"], session_dates=dates, train_samples=100,
        validation_samples=20, test_samples=20, n_folds=3, min_train_samples=100)
    return features, target, meta, plan, bars, news, dates


def test_monthly_target_is_i_plus1_to_i_plus21_relative_to_0050():
    bars, news, dates = synthetic_data(100)
    features, targets, meta = models.prepare_features(bars, news)
    panel = bars.pivot(index="Date", columns="Ticker", values="Total_Return_Index")
    i, ticker = 45, models.TICKERS[0]
    expected = 100*((panel.at[dates[i+21], ticker]/panel.at[dates[i+1], ticker]-1)
                    -(panel.at[dates[i+21], models.BENCHMARK]/panel.at[dates[i+1], models.BENCHMARK]-1))
    assert targets.at[(dates[i], ticker), "Target_Excess_Pct"] == pytest.approx(expected)
    assert targets.loc[dates[-21]:].Target_Excess_Pct.isna().all()
    assert meta["label_gap_sessions"] == 21
    assert list(features.columns) == list(models.ALL_FEATURES)


def test_all_temporal_splits_purge_twenty_one_full_sessions_even_with_gaps():
    dates = pd.bdate_range("2020-01-01", periods=650)
    eligible = dates[30:-21].delete(np.arange(0, len(dates[30:-21]), 7))
    plan = models.make_splits(eligible, session_dates=dates, train_samples=100,
        validation_samples=20, test_samples=20, n_folds=3, min_train_samples=100)
    for fold in plan["folds"]:
        assert dates.get_loc(fold["train"][-1])+21 < dates.get_loc(fold["validation"][0])
    assert dates.get_loc(plan["folds"][-1]["validation"][-1])+21 < dates.get_loc(plan["test"][0])
    assert dates.get_loc(plan["final"]["train"][-1])+21 < dates.get_loc(plan["test"][0])
    with pytest.raises(ValueError, match="21-session"):
        models.make_splits(eligible, session_dates=dates, label_gap=6)


def test_untradable_endpoint_invalidates_label_without_compressing_calendar():
    bars, news, dates = synthetic_data(150)
    ticker = models.TICKERS[0]
    bars.loc[(bars.Ticker == models.BENCHMARK) & bars.Date.eq(dates[71]), "Tradable"] = False
    _, targets, meta = models.prepare_features(bars, news)
    assert pd.isna(targets.at[(dates[50], ticker), "Target_Excess_Pct"])
    assert str(dates[50].date()) not in meta["eligible_dates"]
    assert len(meta["session_dates"]) == 150


def test_missing_news_is_flagged_and_interactions_do_not_look_forward():
    bars, news, dates = synthetic_data(120)
    ticker = models.TICKERS[0]
    news = news[~(news.Ticker.eq(ticker) & news.Date.eq(dates[60]))]
    a, _, _ = models.prepare_features(bars, news)
    row = a.loc[(dates[60], ticker)]
    assert row.News_Score_Missing == 1 and row.Sentiment == 0
    assert row.Sentiment_X_Relative_Return_20 == 0
    future = news.copy()
    future.loc[future.Date.gt(dates[60]), "Daily_Sentiment"] *= -1
    b, _, _ = models.prepare_features(bars, future)
    pd.testing.assert_frame_equal(a.loc[:dates[60]], b.loc[:dates[60]])


def test_ridge_scaler_and_weights_ignore_extreme_outer_validation_features():
    features, targets, _, plan, *_ = prepared()
    split = plan["folds"][0]
    _, artifact = models.fit_predict(features, targets, split, "ridge_news")
    shocked = features.copy()
    shocked.loc[shocked.index.get_level_values("Date") >= split["validation"][0]] *= 1e8
    _, other = models.fit_predict(shocked, targets, split, "ridge_news")
    assert artifact["metadata"]["scaler"] == other["metadata"]["scaler"]
    np.testing.assert_array_equal(artifact["arrays"]["coefficient"], other["arrays"]["coefficient"])
    assert not artifact["metadata"]["outer_validation_used_for_training"]


@pytest.mark.parametrize("family", ["ridge", "hgb"])
def test_price_news_pair_memberships_equal_and_portable_predictions_match(family):
    features, targets, _, plan, *_ = prepared()
    split = plan["folds"][0]
    price, _ = models.fit_predict(features, targets, split, family+"_price")
    news, artifact = models.fit_predict(features, targets, split, family+"_news")
    pd.testing.assert_frame_equal(price[["Date", "Ticker", "Actual", "Split"]], news[["Date", "Ticker", "Actual", "Split"]])
    # JSON is safe to load; no pickle/model class is needed for HGB replay.
    safe = json.loads(json.dumps(artifact, default=lambda value: value.tolist(), allow_nan=False))
    frozen = models.predict_saved(features, split["validation"], safe)
    np.testing.assert_allclose(frozen.Predicted, news.Predicted, atol=1e-10, rtol=1e-10)
    assert artifact["metadata"]["news_channels_used"]
    assert not artifact["metadata"]["test_used_for_training"]
    if family == "hgb":
        assert artifact["metadata"]["tree_count"] == 150
        assert not artifact["metadata"]["model_parameters"]["early_stopping"]


def test_nonfinite_paired_input_rejected_even_by_price_control():
    features, targets, _, plan, *_ = prepared()
    split = plan["folds"][0]
    broken = features.copy()
    broken.loc[(split["train"][0], models.TICKERS[0]), "Sentiment"] = np.nan
    with pytest.raises(ValueError, match="Non-finite paired"):
        models.fit_predict(broken, targets, split, "ridge_price")


def test_saved_feature_order_or_scaler_corruption_rejected():
    features, targets, _, plan, *_ = prepared()
    split = plan["folds"][0]
    _, artifact = models.fit_predict(features, targets, split, "ridge_news")
    artifact["metadata"]["scaler"]["scale"][0] = 0
    with pytest.raises(ValueError, match="scaler"):
        models.predict_saved(features, split["validation"], artifact)
