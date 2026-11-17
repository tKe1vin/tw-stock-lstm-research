"""Finite monthly 0050-relative price/news models; all history is exploratory.

The V3.2 sources and saved models remain unchanged.  This version changes the
holding target to twenty sessions and compares Ridge with a small fixed HGB.
Outer validation selects the portfolio, never a scaler or training parameter.
Portable artifacts contain numerical arrays, not executable pickle payloads.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from research.v3 import models as legacy

BENCHMARK = legacy.BENCHMARK
TICKERS = legacy.TICKERS
CANDIDATES = ("ridge_price", "ridge_news", "hgb_price", "hgb_news")
HOLD_SESSIONS = 20
LABEL_GAP = 21
TICKER_FEATURES = tuple("Ticker_" + ticker.replace(".", "_") for ticker in TICKERS)
PRICE_FEATURES = legacy.PRICE_FEATURES + TICKER_FEATURES
INTERACTION_FEATURES = ("Sentiment_X_Relative_Return_20", "Negative_News_X_Volatility_20",
                        "Sentiment_X_Log_News_Count")
NEWS_FEATURES = legacy.NEWS_FEATURES + INTERACTION_FEATURES
ALL_FEATURES = PRICE_FEATURES + NEWS_FEATURES
MODEL_PARAMETERS = {
    "ridge": {"alpha": 10.0, "fit_intercept": True},
    "hgb": {"loss": "squared_error", "max_iter": 150, "learning_rate": 0.05,
            "max_leaf_nodes": 7, "min_samples_leaf": 40, "l2_regularization": 10.0,
            "early_stopping": False},
}
TREE_FIELDS = ("value", "feature_idx", "num_threshold", "left", "right", "is_leaf")
_normal_dates = legacy._normal_dates


def prepare_features(bars, sentiment, benchmark=BENCHMARK, tickers=None, time_step=10):
    """Keep causal V3 inputs, replacing ALL labels with next-close 20-session excess.

    No date intersection compresses the market calendar. Both stock and ETF
    must be tradable at i+1 and i+21. All four candidates share complete rows,
    including price controls when news values are represented by missing flags.
    """
    universe = tuple(tickers or TICKERS)
    if universe != TICKERS or benchmark != BENCHMARK:
        raise ValueError("V3.3 registers the original five stocks and 0050")
    raw = legacy._canonical_long(bars, "bars")
    if "Tradable" not in raw:
        raise ValueError("V3.3 requires audited Tradable endpoint evidence")
    if not raw.Tradable.isin([True, False, "True", "False", "true", "false", 0, 1]).all():
        raise ValueError("Tradable must contain explicit booleans")
    features, _unused_five_day_targets, previous = legacy.prepare_features(
        raw, sentiment, benchmark=benchmark, tickers=universe, time_step=time_step)
    calendar = _normal_dates(previous["session_dates"])
    calendar.name = "Date"
    panel = raw.pivot(index="Date", columns="Ticker", values="Total_Return_Index").reindex(calendar)
    volume = raw.pivot(index="Date", columns="Ticker", values="Volume").reindex(calendar)
    tradable = raw.pivot(index="Date", columns="Ticker", values="Tradable").reindex(calendar)
    tradable = tradable.map(lambda value: value is True or str(value).lower() in ("true", "1", "1.0"))
    tradable &= volume.gt(0)
    forward = panel.shift(-LABEL_GAP) / panel.shift(-1) - 1
    target = 100 * forward[list(universe)].sub(forward[benchmark], axis=0)
    endpoints = tradable.shift(-1, fill_value=False) & tradable.shift(-LABEL_GAP, fill_value=False)
    for ticker in universe:
        target.loc[~(endpoints[ticker] & endpoints[benchmark]), ticker] = np.nan
    targets = target.rename_axis(columns="Ticker").stack(future_stack=True).to_frame("Target_Excess_Pct")
    targets = targets.reindex(features.index)
    for ticker, name in zip(TICKERS, TICKER_FEATURES):
        features[name] = (features.index.get_level_values("Ticker") == ticker).astype(float)
    features[INTERACTION_FEATURES[0]] = features.Sentiment * features.Relative_Return_20
    features[INTERACTION_FEATURES[1]] = (-features.Sentiment).clip(lower=0) * features.Volatility_20
    features[INTERACTION_FEATURES[2]] = features.Sentiment * features.Log_News_Count
    features = features.loc[:, list(ALL_FEATURES)]
    finite = pd.Series(np.isfinite(features.to_numpy(float)).all(axis=1), index=features.index)
    common = finite.unstack("Ticker").all(axis=1).reindex(calendar)
    windows = common.rolling(time_step, min_periods=time_step).sum().eq(time_step)
    mature = np.isfinite(targets.Target_Excess_Pct.unstack("Ticker").reindex(calendar).to_numpy(float)).all(axis=1)
    eligible = calendar[windows.to_numpy() & mature]
    features.attrs.update(session_dates=[str(day.date()) for day in calendar], tickers=list(universe),
                          eligible_dates=[str(day.date()) for day in eligible], label_gap=LABEL_GAP)
    metadata = {**previous, "protocol_version": "sentiment_0050_v3_3_monthly",
                "eligible_dates": features.attrs["eligible_dates"], "label_gap_sessions": LABEL_GAP,
                "hold_sessions": HOLD_SESSIONS,
                "target": "100 * [(stock TRI[i+21]/TRI[i+1]-1) - (0050 TRI[i+21]/TRI[i+1]-1)]",
                "price_features": list(PRICE_FEATURES), "news_features": list(NEWS_FEATURES),
                "news_interactions": list(INTERACTION_FEATURES), "paired_common_coverage": True,
                "historical_news_point_in_time_proven": previous["historical_news_point_in_time_proven"],
                "historical_alpha_proven": False, "performance_claim": "none"}
    return features, targets, metadata


def make_splits(dates: Iterable, *, session_dates, train_samples=756, validation_samples=126,
                test_samples=126, n_folds=3, label_gap=LABEL_GAP, min_train_samples=252):
    """Temporal folds purge mature labels using full calendar positions, never row count."""
    calendar, eligible = _normal_dates(session_dates), _normal_dates(dates)
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("session_dates must be unique, increasing and nonempty")
    if eligible.empty or eligible.has_duplicates or not eligible.is_monotonic_increasing:
        raise ValueError("Eligible dates must be unique, increasing and nonempty")
    if label_gap != LABEL_GAP:
        raise ValueError("Twenty-session next-close labels require a 21-session gap")
    if min(train_samples, validation_samples, test_samples, n_folds, min_train_samples) < 1:
        raise ValueError("Split sizes must be positive")
    positions = calendar.get_indexer(eligible)
    if (positions < 0).any() or (positions + label_gap >= len(calendar)).any():
        raise ValueError("Eligible dates must have mature labels on the full calendar")
    if len(eligible) < test_samples:
        raise ValueError("Insufficient mature exploratory holdout")
    test = eligible[-test_samples:]
    boundary = calendar.get_loc(test[0])
    folds = []
    for _ in range(n_folds):
        before = eligible[positions + label_gap < boundary]
        if len(before) < validation_samples:
            raise ValueError("Insufficient purged validation signals")
        validation = before[-validation_samples:]
        first = calendar.get_loc(validation[0])
        train = eligible[positions + label_gap < first][-train_samples:]
        if len(train) < min_train_samples:
            raise ValueError("Insufficient purged training signals")
        folds.append({"train": train, "validation": validation, "test": pd.DatetimeIndex([])})
        boundary = first
    final_train = eligible[positions + label_gap < calendar.get_loc(test[0])][-train_samples:]
    if len(final_train) < min_train_samples:
        raise ValueError("Insufficient final training signals")
    return {"folds": list(reversed(folds)), "test": test,
            "final": {"train": final_train, "validation": pd.DatetimeIndex([]), "test": test},
            "session_dates": calendar, "label_gap": label_gap}


def _samples(features, target, split, columns, step):
    if features.index.names != ["Date", "Ticker"] or target.index.names != ["Date", "Ticker"]:
        raise ValueError("Features and targets require Date/Ticker MultiIndex")
    if features.index.has_duplicates or target.index.has_duplicates or not features.index.equals(target.index):
        raise ValueError("Feature/target memberships must be unique and identical")
    if list(features.columns) != list(ALL_FEATURES):
        raise ValueError("Feature order differs from the registered V3.3 contract")
    calendar = _normal_dates(features.attrs.get("session_dates", []))
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("The original full session calendar is required")
    tickers = list(TICKERS)
    if set(features.index.get_level_values("Ticker")) != set(tickers):
        raise ValueError("Incomplete registered stock membership")
    raw = {ticker: features.xs(ticker, level="Ticker").reindex(calendar) for ticker in tickers}
    labels = target.Target_Excess_Pct.unstack("Ticker").reindex(index=calendar, columns=tickers)
    arrays, keys = {}, {}
    for name in ("train", "validation", "test"):
        dates = _normal_dates(split.get(name, []))
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("Split dates must be unique and increasing")
        positions = calendar.get_indexer(dates)
        if (positions < step - 1).any() or (positions + LABEL_GAP >= len(calendar)).any():
            raise ValueError("Insufficient input history or immature selected label")
        x, y, index = [], [], []
        for day, position in zip(dates, positions):
            for ticker in tickers:
                full = raw[ticker].iloc[position-step+1:position+1]
                label = labels.at[day, ticker]
                if len(full) != step or not np.isfinite(full.to_numpy(float)).all() or not np.isfinite(label):
                    raise ValueError("Non-finite paired input or label")
                x.append(full.loc[:, list(columns)].to_numpy(float))
                y.append(label)
                index.append((day, ticker))
        arrays[name] = (np.asarray(x, dtype=np.float64).reshape(-1, step, len(columns)), np.asarray(y, dtype=np.float64))
        keys[name] = index
    if len(_normal_dates(split.get("train", []))) == 0:
        raise ValueError("Training split is empty")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        a, b = _normal_dates(split.get(left, [])), _normal_dates(split.get(right, []))
        if len(a) and len(b) and calendar.get_loc(a[-1]) + LABEL_GAP >= calendar.get_loc(b[0]):
            raise ValueError("A label crosses the next split boundary")
    return arrays, keys, calendar


def _fit_scaler(features, train_dates, columns, calendar, step):
    positions = calendar.get_indexer(_normal_dates(train_dates))
    union = np.unique(np.concatenate([np.arange(position-step+1, position+1) for position in positions]))
    rows = np.concatenate([features.xs(ticker, level="Ticker").reindex(calendar).iloc[union]
                           .loc[:, list(columns)].to_numpy(float) for ticker in TICKERS])
    return StandardScaler().fit(rows), union


def _scale(x, scaler):
    return scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape) if len(x) else x


def _saved_hgb_arrays(model):
    arrays = {"baseline": np.asarray(model._baseline_prediction, dtype=np.float64).ravel()}
    for i, predictors in enumerate(model._predictors):
        if len(predictors) != 1:
            raise ValueError("Only one-output regression HGB trees are supported")
        nodes = predictors[0].nodes
        if np.any(nodes["is_categorical"]):
            raise ValueError("Categorical tree serialization is not registered")
        for field in TREE_FIELDS:
            arrays[f"tree_{i}_{field}"] = nodes[field].copy()
    return arrays


def _predict_hgb(x, arrays, tree_count):
    arrays = {name: np.asarray(value) for name, value in arrays.items()}
    baseline = arrays["baseline"]
    if baseline.shape != (1,) or not np.isfinite(baseline).all():
        raise ValueError("Invalid saved HGB baseline")
    result = np.full(len(x), float(baseline[0]))
    for i in range(tree_count):
        node = {field: arrays[f"tree_{i}_{field}"] for field in TREE_FIELDS}
        lengths = {len(value) for value in node.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
            raise ValueError("Invalid saved HGB node arrays")
        ids = np.zeros(len(x), dtype=np.int64)
        for _ in range(len(node["value"]) + 1):
            active = np.flatnonzero(~node["is_leaf"][ids].astype(bool))
            if len(active) == 0:
                break
            current = ids[active]
            feature = node["feature_idx"][current].astype(int)
            if (feature < 0).any() or (feature >= x.shape[1]).any():
                raise ValueError("Saved tree feature index outside contract")
            go_left = x[active, feature] <= node["num_threshold"][current]
            ids[active] = np.where(go_left, node["left"][current], node["right"][current]).astype(int)
            if (ids < 0).any() or (ids >= len(node["value"])).any():
                raise ValueError("Saved tree child index outside contract")
        else:
            raise ValueError("Saved HGB tree contains a cycle")
        result += node["value"][ids]
    if not np.isfinite(result).all():
        raise ValueError("Non-finite saved HGB predictions")
    return result


def fit_predict(features, target, split, candidate="ridge_news", seed=42, time_step=10):
    """Fit one fixed registered candidate using only TRAIN, returning portable arrays."""
    if candidate not in CANDIDATES or seed != 42 or time_step != 10:
        raise ValueError("Unregistered candidate, seed or time step")
    columns = PRICE_FEATURES + NEWS_FEATURES if candidate.endswith("_news") else PRICE_FEATURES
    arrays, keys, calendar = _samples(features, target, split, columns, time_step)
    scaler, fit_positions = _fit_scaler(features, split["train"], columns, calendar, time_step)
    scaled = {part: _scale(x, scaler).reshape(len(x), -1) if len(x) else
              np.empty((0, time_step*len(columns))) for part, (x, _y) in arrays.items()}
    x, y = scaled["train"], arrays["train"][1]
    family = "ridge" if candidate.startswith("ridge") else "hgb"
    if family == "ridge":
        model = Ridge(**MODEL_PARAMETERS[family]).fit(x, y)
        weights = {"coefficient": np.asarray(model.coef_, dtype=np.float64),
                   "intercept": np.asarray(model.intercept_, dtype=np.float64)}
        tree_count = 0
    else:
        with threadpool_limits(limits=1, user_api="openmp"):
            model = HistGradientBoostingRegressor(**MODEL_PARAMETERS[family], random_state=seed).fit(x, y)
        weights = _saved_hgb_arrays(model)
        tree_count = len(model._predictors)
    rows = []
    for part in ("validation", "test"):
        if len(scaled[part]) == 0:
            continue
        with threadpool_limits(limits=1, user_api="openmp"):
            predictions = model.predict(scaled[part])
        if not np.isfinite(predictions).all():
            raise ValueError("Non-finite candidate prediction")
        if family == "hgb":
            replay = _predict_hgb(scaled[part], weights, tree_count)
            if not np.allclose(predictions, replay, rtol=1e-12, atol=1e-12):
                raise ValueError("Portable HGB replay differs from fitted prediction")
        for (date, ticker), predicted, actual in zip(keys[part], predictions, arrays[part][1]):
            position = calendar.get_loc(date)
            rows.append({"Date": date, "Ticker": ticker, "Predicted": float(predicted),
                         "Actual": float(actual), "Split": part, "Entry_Date": calendar[position+1],
                         "Label_End": calendar[position+LABEL_GAP], "Candidate": candidate})
    metadata = {"candidate": candidate, "model_family": family, "feature_order": list(columns),
                "all_feature_order": list(ALL_FEATURES), "model_parameters": MODEL_PARAMETERS[family],
                "time_step": time_step, "label_gap_sessions": LABEL_GAP, "hold_sessions": HOLD_SESSIONS,
                "tickers": list(TICKERS), "seed": seed, "tree_count": tree_count,
                "target_units": "percentage_points_excess_total_return_versus_0050",
                "splits": legacy._split_document(split), "training_signal_dates": len(split["train"]),
                "training_stock_samples": len(y), "scaler_train_only": True,
                "scaler_unique_input_rows_per_stock": len(fit_positions),
                "scaler_first_input_date": str(calendar[fit_positions[0]].date()),
                "scaler_last_input_date": str(calendar[fit_positions[-1]].date()),
                "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "var": scaler.var_.tolist()},
                "outer_validation_used_for_training": False, "test_used_for_training": False,
                "news_channels_used": candidate.endswith("_news"), "performance_claim": "none"}
    return pd.DataFrame(rows, columns=["Date", "Ticker", "Predicted", "Actual", "Split", "Entry_Date", "Label_End", "Candidate"]), {"metadata": metadata, "arrays": weights}


def predict_saved(features, dates, artifact):
    """Frozen numerical prediction, with no fitting or executable deserialization."""
    metadata, arrays = artifact["metadata"], artifact["arrays"]
    candidate = metadata["candidate"]
    if candidate not in CANDIDATES or metadata.get("label_gap_sessions") != LABEL_GAP:
        raise ValueError("Unregistered saved monthly model")
    columns = PRICE_FEATURES + NEWS_FEATURES if candidate.endswith("_news") else PRICE_FEATURES
    if list(features.columns) != list(ALL_FEATURES) or list(columns) != metadata["feature_order"]:
        raise ValueError("Saved feature order differs from registered contract")
    if metadata.get("tickers") != list(TICKERS) or metadata.get("time_step") != 10:
        raise ValueError("Saved ticker or sequence contract differs")
    calendar = _normal_dates(features.attrs.get("session_dates", []))
    selected = _normal_dates(dates)
    if selected.has_duplicates or not selected.is_monotonic_increasing:
        raise ValueError("Prediction dates must be unique and increasing")
    mean, scale = np.asarray(metadata["scaler"]["mean"]), np.asarray(metadata["scaler"]["scale"])
    if mean.shape != (len(columns),) or scale.shape != mean.shape or not np.isfinite([mean, scale]).all() or (scale <= 0).any():
        raise ValueError("Invalid saved scaler contract")
    samples, keys = [], []
    for date in selected:
        position = calendar.get_indexer([date])[0]
        if position < 9:
            raise ValueError("Unknown date or insufficient frozen input history")
        for ticker in TICKERS:
            full = features.xs(ticker, level="Ticker").reindex(calendar).iloc[position-9:position+1]
            if not np.isfinite(full.to_numpy(float)).all():
                raise ValueError("Non-finite paired frozen inputs")
            samples.append(((full.loc[:, list(columns)].to_numpy(float)-mean)/scale).ravel())
            keys.append((date, ticker))
    x = np.asarray(samples, dtype=np.float64).reshape(-1, 10*len(columns))
    if candidate.startswith("ridge"):
        coefficient = np.asarray(arrays["coefficient"], dtype=np.float64)
        if coefficient.shape != (x.shape[1],):
            raise ValueError("Invalid saved Ridge coefficient shape")
        prediction = x @ coefficient + np.asarray(arrays["intercept"])
    else:
        prediction = _predict_hgb(x, arrays, int(metadata["tree_count"]))
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite frozen predictions")
    return pd.DataFrame([{"Date": date, "Ticker": ticker, "Predicted": float(value), "Candidate": candidate}
                         for (date, ticker), value in zip(keys, prediction)],
                        columns=["Date", "Ticker", "Predicted", "Candidate"])
