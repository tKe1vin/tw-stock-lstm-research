"""Paired, purged benchmark-relative models for the exploratory V3 protocol.

Only this module's four registered candidates are supported.  Outer validation
and test observations never fit a scaler, choose a training epoch, or update a
weight.  Historical news availability remains an explicit limitation; a score
in a legacy CSV does not establish when that score was available.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BENCHMARK = "0050.TW"
TICKERS = ("2330.TW", "2317.TW", "2603.TW", "2002.TW", "2891.TW")
CANDIDATES = ("ridge_price", "ridge_news", "lstm_price", "lstm_news")
PRICE_FEATURES = (
    "Return_1", "Return_5", "Return_20", "Volatility_20", "Relative_MA20",
    "RSI_14", "Benchmark_Return_1", "Benchmark_Return_5", "Benchmark_Volatility_20",
    "Relative_Return_5", "Relative_Return_20", "Relative_Volume_20",
)
NEWS_FEATURES = (
    "Sentiment", "Sentiment_Change_1", "Sentiment_Surprise_20", "Log_News_Count",
    "News_Score_Missing", "News_Count_Missing", "News_Availability_Verified",
)


def _normal_dates(values):
    result = pd.DatetimeIndex(pd.to_datetime(values))
    if result.hasnans:
        raise ValueError("Dates must not be missing")
    if result.tz is not None:
        result = result.tz_convert("Asia/Taipei").tz_localize(None)
    return result.normalize()


def _canonical_long(frame, name):
    result = frame.copy()
    if "Ticker" not in result and "ticker" in result:
        result = result.rename(columns={"ticker": "Ticker"})
    if not {"Date", "Ticker"}.issubset(result.columns):
        raise ValueError(f"{name} needs Date and Ticker columns")
    result["Date"] = _normal_dates(result.Date)
    if result.duplicated(["Date", "Ticker"]).any():
        raise ValueError(f"Duplicate {name} Date/Ticker rows")
    return result.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def prepare_features(bars, sentiment, benchmark=BENCHMARK, tickers=None, time_step=10):
    """Return features, five-session excess targets, and a coverage document.

    ``Total_Return_Index`` must be created by the audited data adapter from its
    explicit raw/split convention.  A bare Close is deliberately insufficient:
    this prevents silently applying a split twice or omitting cash dividends.
    All shifts use the FULL union session calendar, before finite filtering.
    Missing news is represented by a missing flag and a zero imputation; the
    paired price control shares the exact same eligible stock/date samples.
    """
    if not isinstance(time_step, int) or time_step < 1:
        raise ValueError("time_step must be a positive integer")
    if benchmark != BENCHMARK:
        raise ValueError("The registered V3 benchmark is 0050.TW")
    raw = _canonical_long(bars, "bars")
    universe = tuple(tickers or TICKERS)
    if benchmark in universe or len(set(universe)) != len(universe):
        raise ValueError("The benchmark and unique satellite tickers must differ")
    required = {"Total_Return_Index", "Volume"}
    if not required.issubset(raw.columns):
        raise ValueError("Audited Total_Return_Index and Volume are required; Close is not a total-return index")
    raw = raw[raw.Ticker.isin(universe + (benchmark,))]
    if set(raw.Ticker) != set(universe + (benchmark,)):
        raise ValueError("Missing benchmark or satellite price series")
    calendar = pd.DatetimeIndex(sorted(raw.Date.unique()), name="Date")
    tri = raw.pivot(index="Date", columns="Ticker", values="Total_Return_Index").reindex(calendar)
    volume = raw.pivot(index="Date", columns="Ticker", values="Volume").reindex(calendar)
    if (tri <= 0).any().any():
        raise ValueError("Total_Return_Index must be positive")
    # Negative volume cannot be a legitimate missing observation.
    if (volume < 0).any().any():
        raise ValueError("Volume must be nonnegative")
    if np.isinf(tri.to_numpy(float)).any() or np.isinf(volume.to_numpy(float)).any():
        raise ValueError("Infinite price or volume input")
    news = _canonical_long(sentiment, "sentiment")
    score_name = next((key for key in ("Daily_Sentiment", "DailySentiment", "Sentiment") if key in news), None)
    if score_name is None:
        raise ValueError("Sentiment needs Daily_Sentiment, DailySentiment, or Sentiment")
    score = news.pivot(index="Date", columns="Ticker", values=score_name).reindex(index=calendar, columns=universe)
    score = score.apply(pd.to_numeric, errors="raise")
    if np.isinf(score.to_numpy(float)).any() or (score.abs() > 1).any().any():
        raise ValueError("Sentiment must be finite when observed and in [-1, 1]")
    counts = (news.pivot(index="Date", columns="Ticker", values="News_Count").reindex(index=calendar, columns=universe)
              if "News_Count" in news else pd.DataFrame(np.nan, index=calendar, columns=universe))
    counts = counts.apply(pd.to_numeric, errors="raise")
    if (counts < 0).any().any() or np.isinf(counts.to_numpy(float)).any():
        raise ValueError("News_Count must be nonnegative and finite when observed")
    # Verification is stronger than mere presence of a score or Source_Status.
    # It must be supplied by an upstream point-in-time provenance validation.
    verified = (news.pivot(index="Date", columns="Ticker", values="Availability_Verified")
                .reindex(index=calendar, columns=universe)
                if "Availability_Verified" in news else pd.DataFrame(False, index=calendar, columns=universe))
    verified = verified.map(lambda value: value is True or str(value).lower() in ("true", "1", "1.0"))
    market_r1 = tri[benchmark].pct_change(fill_method=None)
    market_r5 = tri[benchmark].pct_change(5, fill_method=None)
    market_vol = market_r1.rolling(20, min_periods=20).std(ddof=1)
    target = 100 * ((tri.shift(-6) / tri.shift(-1) - 1).sub(
        tri[benchmark].shift(-6) / tri[benchmark].shift(-1) - 1, axis=0))
    if "Tradable" in raw:
        tradable = (raw.pivot(index="Date", columns="Ticker", values="Tradable").reindex(calendar)
                    .map(lambda value: value is True or str(value).lower() in ("true", "1", "1.0")))
        tradable_endpoints = tradable.shift(-1, fill_value=False) & tradable.shift(-6, fill_value=False)
        for ticker in universe:
            valid_entry_exit = tradable_endpoints[ticker] & tradable_endpoints[benchmark]
            target.loc[~valid_entry_exit, ticker] = np.nan
    feature_blocks, target_blocks = [], []
    for ticker in universe:
        wealth = tri[ticker]
        r1 = wealth.pct_change(fill_method=None)
        delta = wealth.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        rsi.loc[(loss == 0) & (gain > 0)] = 100
        rsi.loc[(loss == 0) & (gain == 0)] = 50
        block = pd.DataFrame(index=calendar)
        block["Return_1"] = r1
        block["Return_5"] = wealth.pct_change(5, fill_method=None)
        block["Return_20"] = wealth.pct_change(20, fill_method=None)
        block["Volatility_20"] = r1.rolling(20, min_periods=20).std(ddof=1)
        block["Relative_MA20"] = wealth / wealth.rolling(20, min_periods=20).mean() - 1
        block["RSI_14"] = rsi / 100
        block["Benchmark_Return_1"] = market_r1
        block["Benchmark_Return_5"] = market_r5
        block["Benchmark_Volatility_20"] = market_vol
        block["Relative_Return_5"] = block.Return_5 - market_r5
        block["Relative_Return_20"] = block.Return_20 - tri[benchmark].pct_change(20, fill_method=None)
        block["Relative_Volume_20"] = volume[ticker] / volume[ticker].rolling(20, min_periods=20).mean().replace(0, np.nan) - 1
        observed = score[ticker].notna()
        observed_score = score[ticker]
        prior_mean = observed_score.shift(1).rolling(20, min_periods=5).mean()
        prior_std = observed_score.shift(1).rolling(20, min_periods=5).std(ddof=1)
        block["Sentiment"] = observed_score.fillna(0)
        # Never turn an unknown previous score into a measured change.
        block["Sentiment_Change_1"] = observed_score.diff().fillna(0)
        block["Sentiment_Surprise_20"] = ((observed_score - prior_mean) / prior_std.replace(0, np.nan)).clip(-10, 10).fillna(0)
        block["Log_News_Count"] = np.log1p(counts[ticker].fillna(0))
        block["News_Score_Missing"] = (~observed).astype(float)
        block["News_Count_Missing"] = counts[ticker].isna().astype(float)
        block["News_Availability_Verified"] = (verified[ticker] & observed).astype(float)
        block["Ticker"] = ticker
        feature_blocks.append(block.reset_index().set_index(["Date", "Ticker"]))
        label = pd.DataFrame({"Date": calendar, "Ticker": ticker, "Target_Excess_Pct": target[ticker].to_numpy()})
        target_blocks.append(label.set_index(["Date", "Ticker"]))
    features = pd.concat(feature_blocks).sort_index()
    targets = pd.concat(target_blocks).sort_index()
    finite = np.isfinite(features.to_numpy(float)).all(axis=1)
    per_day = pd.Series(finite, index=features.index).unstack("Ticker").all(axis=1)
    ready_windows = per_day.rolling(time_step, min_periods=time_step).sum().eq(time_step)
    mature = np.isfinite(targets.Target_Excess_Pct.unstack("Ticker").to_numpy(float)).all(axis=1)
    eligible = calendar[ready_windows.to_numpy() & mature]
    sessions = [str(value.date()) for value in calendar]
    features.attrs["session_dates"] = sessions
    features.attrs["tickers"] = list(universe)
    features.attrs["eligible_dates"] = [str(value.date()) for value in eligible]
    metadata = {"benchmark": benchmark, "tickers": list(universe), "session_dates": sessions,
                "eligible_dates": features.attrs["eligible_dates"], "time_step": time_step,
                "target": "100 * [(stock TRI[i+6]/TRI[i+1]-1) - (0050 TRI[i+6]/TRI[i+1]-1)]",
                "label_gap_sessions": 6, "price_features": list(PRICE_FEATURES), "news_features": list(NEWS_FEATURES),
                "news_observed_rows": int(score.notna().sum().sum()),
                "news_availability_verified_rows": int((verified & score.notna()).sum().sum()),
                "news_missing_policy": "zero value with explicit missing flags; no forward/backfill",
                "historical_news_point_in_time_proven": bool(score.notna().any().any() and (verified | score.isna()).all().all()),
                "paired_common_coverage": True, "performance_claim": "none"}
    return features, targets, metadata


def make_splits(dates: Iterable, *, session_dates, train_samples=756, validation_samples=60,
                test_samples=60, n_folds=3, label_gap=6, min_train_samples=252):
    """Lock three chronological validation folds and a later exploratory test.

    ``dates`` are eligible signal dates; ``session_dates`` are the original full
    exchange/bar calendar.  Purging uses calendar positions, so a dropped quote
    or non-finite sample can never shorten the six-session label horizon.
    """
    calendar = _normal_dates(session_dates)
    eligible = _normal_dates(dates)
    if calendar.has_duplicates or not calendar.is_monotonic_increasing or len(calendar) == 0:
        raise ValueError("session_dates must be unique, increasing, and nonempty")
    if eligible.has_duplicates or not eligible.is_monotonic_increasing or len(eligible) == 0:
        raise ValueError("Eligible dates must be unique, increasing, and nonempty")
    positions = calendar.get_indexer(eligible)
    if (positions < 0).any():
        raise ValueError("Eligible dates must belong to the full session calendar")
    if (positions + label_gap >= len(calendar)).any():
        raise ValueError("Eligible dates include an immature future label")
    if label_gap != 6:
        raise ValueError("The five-session next-close target requires label_gap=6")
    if min(train_samples, validation_samples, test_samples, n_folds, min_train_samples) < 1:
        raise ValueError("All split sizes must be positive")
    if len(eligible) < test_samples:
        raise ValueError("Insufficient mature test signals")
    test = eligible[-test_samples:]
    boundary = int(calendar.get_loc(test[0]))
    reverse_folds = []
    for _ in range(n_folds):
        before = eligible[positions + label_gap < boundary]
        if len(before) < validation_samples:
            raise ValueError("Insufficient purged validation signals")
        validation = before[-validation_samples:]
        first_val = int(calendar.get_loc(validation[0]))
        train = eligible[positions + label_gap < first_val][-train_samples:]
        if len(train) < min_train_samples:
            raise ValueError("Insufficient purged training signals")
        reverse_folds.append({"train": train, "validation": validation, "test": pd.DatetimeIndex([])})
        boundary = first_val
    final_train = eligible[positions + label_gap < calendar.get_loc(test[0])][-train_samples:]
    if len(final_train) < min_train_samples:
        raise ValueError("Insufficient final training signals")
    return {"folds": list(reversed(reverse_folds)), "test": test,
            "final": {"train": final_train, "validation": pd.DatetimeIndex([]), "test": test},
            "session_dates": calendar, "label_gap": label_gap}


def _split_document(split):
    return {name: [str(day.date()) for day in _normal_dates(split.get(name, []))]
            for name in ("train", "validation", "test")}


def _sample_arrays(features, target, split, columns, time_step):
    if features.index.names != ["Date", "Ticker"] or target.index.names != ["Date", "Ticker"]:
        raise ValueError("Features and targets need a Date/Ticker MultiIndex")
    if features.index.has_duplicates or target.index.has_duplicates:
        raise ValueError("Duplicate feature or target samples")
    if list(features.columns) != list(PRICE_FEATURES + NEWS_FEATURES):
        raise ValueError("Feature order differs from the registered contract")
    calendar = _normal_dates(features.attrs.get("session_dates", sorted(features.index.get_level_values("Date").unique())))
    tickers = sorted(features.index.get_level_values("Ticker").unique())
    if not features.index.equals(target.index):
        raise ValueError("Feature and target memberships must match exactly")
    raw = {ticker: features.xs(ticker, level="Ticker").reindex(calendar)[list(columns)].to_numpy(float)
           for ticker in tickers}
    # Coverage is checked against ALL columns even for the price-only control.
    coverage = {ticker: features.xs(ticker, level="Ticker").reindex(calendar).to_numpy(float)
                for ticker in tickers}
    labels = target.Target_Excess_Pct.unstack("Ticker").reindex(index=calendar, columns=tickers)
    arrays, indices = {}, {}
    for name in ("train", "validation", "test"):
        dates = _normal_dates(split.get(name, []))
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("Split dates must be unique and increasing")
        positions = calendar.get_indexer(dates)
        if (positions < time_step - 1).any():
            raise ValueError("Unknown signal date or insufficient sequence history")
        if (positions + 6 >= len(calendar)).any():
            raise ValueError("A selected signal has an immature future label")
        x, y, keys = [], [], []
        for day, position in zip(dates, positions):
            for ticker in tickers:
                complete = coverage[ticker][position - time_step + 1:position + 1]
                label = labels.at[day, ticker]
                if len(complete) != time_step or not np.isfinite(complete).all() or not np.isfinite(label):
                    raise ValueError(f"Non-finite paired sample or target: {day.date()} {ticker}")
                x.append(raw[ticker][position - time_step + 1:position + 1])
                y.append(label)
                keys.append((day, ticker))
        arrays[name] = (np.asarray(x, dtype=np.float64).reshape(-1, time_step, len(columns)), np.asarray(y, dtype=np.float64))
        indices[name] = keys
    train_dates = _normal_dates(split.get("train", []))
    if len(train_dates) == 0:
        raise ValueError("Training split is empty")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_dates, right_dates = _normal_dates(split.get(left, [])), _normal_dates(split.get(right, []))
        if len(left_dates) and len(right_dates):
            if calendar.get_loc(left_dates[-1]) + 6 >= calendar.get_loc(right_dates[0]):
                raise ValueError("A training or validation label crosses the next split boundary")
    return arrays, indices, calendar


def _scale_arrays(arrays):
    scaler = StandardScaler().fit(arrays["train"][0].reshape(-1, arrays["train"][0].shape[-1]))
    scaled = {}
    for part, (x, y) in arrays.items():
        scaled[part] = (scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)
                        if len(x) else x.astype(np.float32), y.astype(np.float32))
    return scaled, scaler


def _tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import tensorflow as tf
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass
    return tf


def _new_lstm(tf, shape, seed):
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=shape), tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(1),
    ])
    # Every probe and refit owns a fresh optimizer.  Never clear its variables:
    # Keras includes the learning-rate variable among optimizer variables.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.Huber(delta=1.0))
    return model


def _fit_epochs(model, x, y, epochs, *, seed, validation=None, patience=3):
    random = np.random.default_rng(seed)
    best_loss, best_epoch, stale, best_weights = np.inf, 0, 0, None
    for epoch in range(1, epochs + 1):
        order = random.permutation(len(x))
        for begin in range(0, len(order), 32):
            batch = order[begin:begin + 32]
            model.train_on_batch(x[batch], y[batch])
        if validation is None:
            continue
        val_x, val_y = validation
        prediction = np.asarray(model(val_x, training=False)).ravel()
        error = np.abs(prediction - val_y)
        loss = float(np.where(error <= 1, 0.5 * error ** 2, error - 0.5).mean())
        if not np.isfinite(loss):
            raise ValueError("Non-finite inner early-stopping loss")
        if loss < best_loss - 1e-8:
            best_loss, best_epoch, stale = loss, epoch, 0
            best_weights = model.get_weights()
        else:
            stale += 1
            if stale >= patience:
                break
    if validation is not None:
        if best_weights is None:
            raise ValueError("No finite early-stopping checkpoint")
        model.set_weights(best_weights)
        return best_epoch
    return epochs


def fit_predict(features, target, split, candidate="ridge_news", seed=42, time_step=10,
                max_epochs=20, inner_validation_samples=60):
    """Fit one registered model, returning predictions and portable NPZ/JSON data.

    Ridge has fixed alpha=10. LSTM has a fixed 32-unit/Huber/Adam architecture.
    LSTM epoch selection sees a purged chronological inner tail of TRAIN only;
    a fresh model/scaler then refits all outer training rows for that epoch count.
    No output is automatically registered as a production or winning model.
    """
    if candidate not in CANDIDATES:
        raise ValueError("Unregistered candidate")
    if max_epochs < 1 or max_epochs > 20:
        raise ValueError("The registered maximum is twenty epochs")
    columns = PRICE_FEATURES + NEWS_FEATURES if candidate.endswith("_news") else PRICE_FEATURES
    arrays, keys, calendar = _sample_arrays(features, target, split, columns, time_step)
    scaled, scaler = _scale_arrays(arrays)
    weights, epochs, inner_document = {}, None, None
    x, y = scaled["train"]
    if candidate.startswith("ridge"):
        model = Ridge(alpha=10.0, fit_intercept=True).fit(x.reshape(len(x), -1).astype(np.float64), arrays["train"][1])
        predict = lambda value: model.predict(value.reshape(len(value), -1).astype(np.float64))
        weights = {"coefficient": model.coef_.astype(np.float64), "intercept": np.asarray(model.intercept_, dtype=np.float64)}
    else:
        train_dates = _normal_dates(split["train"])
        if len(train_dates) <= inner_validation_samples + 6:
            raise ValueError("Insufficient training dates for purged inner early stopping")
        inner_validation = train_dates[-inner_validation_samples:]
        boundary = calendar.get_loc(inner_validation[0])
        inner_train = train_dates[calendar.get_indexer(train_dates) + 6 < boundary]
        if len(inner_train) < 30:
            raise ValueError("Insufficient inner training history")
        inner_split = {"train": inner_train, "validation": inner_validation, "test": []}
        inner_arrays, _, _ = _sample_arrays(features, target, inner_split, columns, time_step)
        inner_scaled, _ = _scale_arrays(inner_arrays)
        tf = _tensorflow()
        probe = _new_lstm(tf, x.shape[1:], seed)
        epochs = _fit_epochs(probe, *inner_scaled["train"], max_epochs, seed=seed,
                             validation=inner_scaled["validation"], patience=3)
        inner_document = _split_document(inner_split)
        del probe
        tf.keras.backend.clear_session()
        model = _new_lstm(tf, x.shape[1:], seed)
        before = [value.copy() for value in model.get_weights()]
        _fit_epochs(model, x, y, epochs, seed=seed)
        if not any(np.any(left != right) for left, right in zip(before, model.get_weights())):
            raise ValueError("LSTM refit failed to update any weights")
        if not np.isclose(float(model.optimizer.learning_rate.numpy()), 0.001):
            raise ValueError("LSTM optimizer learning rate changed from its registered value")
        weights = {f"weight_{i}": value.copy() for i, value in enumerate(model.get_weights())}
        predict = lambda value: np.asarray(model(value, training=False)).ravel()
    rows = []
    for part in ("validation", "test"):
        values, _ = scaled[part]
        actual = arrays[part][1]
        if len(values) == 0:
            continue
        prediction = np.asarray(predict(values), dtype=np.float64)
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite model prediction")
        for (date, ticker), predicted, label in zip(keys[part], prediction, actual):
            position = calendar.get_loc(date)
            rows.append({"Date": date, "Ticker": ticker, "Predicted": float(predicted), "Actual": float(label),
                         "Split": part, "Entry_Date": calendar[position + 1], "Label_End": calendar[position + 6],
                         "Candidate": candidate})
    metadata = {"candidate": candidate, "feature_order": list(columns), "time_step": time_step,
                "tickers": sorted(features.index.get_level_values("Ticker").unique()), "seed": seed,
                "architecture": "pooled_ridge_alpha10" if candidate.startswith("ridge") else "pooled_LSTM32_dropout0.2_Huber1",
                "target_units": "percentage_points_excess_total_return_versus_0050",
                "splits": _split_document(split), "inner_early_stopping_splits": inner_document,
                "training_signal_dates": len(_normal_dates(split["train"])), "training_stock_samples": len(x),
                "selected_epochs": epochs, "scaler_train_only": True,
                "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "var": scaler.var_.tolist()},
                "scaler_first_input_date": str(calendar[calendar.get_loc(_normal_dates(split["train"])[0]) - time_step + 1].date()),
                "scaler_last_input_date": str(_normal_dates(split["train"])[-1].date()),
                "outer_validation_used_for_training": False, "test_used_for_training": False,
                "news_channels_used": candidate.endswith("_news"), "performance_claim": "none"}
    artifact = {"metadata": metadata, "arrays": weights}
    return pd.DataFrame(rows, columns=["Date", "Ticker", "Predicted", "Actual", "Split", "Entry_Date", "Label_End", "Candidate"]), artifact


def predict_saved(features, dates, artifact):
    """Predict from retained arrays and scaler without fitting or selecting."""
    metadata, arrays = artifact["metadata"], artifact["arrays"]
    candidate = metadata["candidate"]
    if candidate not in CANDIDATES:
        raise ValueError("Unregistered saved candidate")
    columns = PRICE_FEATURES + NEWS_FEATURES if candidate.endswith("_news") else PRICE_FEATURES
    if list(columns) != metadata["feature_order"] or list(features.columns) != list(PRICE_FEATURES + NEWS_FEATURES):
        raise ValueError("Saved model feature order differs from runtime contract")
    calendar = _normal_dates(features.attrs.get("session_dates", sorted(features.index.get_level_values("Date").unique())))
    step = metadata["time_step"]
    mean, scale = np.asarray(metadata["scaler"]["mean"]), np.asarray(metadata["scaler"]["scale"])
    samples, keys = [], []
    for date in _normal_dates(dates):
        position = calendar.get_indexer([date])[0]
        if position < step - 1:
            raise ValueError("Unknown prediction date or insufficient history")
        for ticker in metadata["tickers"]:
            complete = features.xs(ticker, level="Ticker").reindex(calendar).iloc[position - step + 1:position + 1]
            if not np.isfinite(complete.to_numpy(float)).all():
                raise ValueError("Non-finite paired frozen forecast inputs")
            samples.append((complete[list(columns)].to_numpy(float) - mean) / scale)
            keys.append((date, ticker))
    x = np.asarray(samples, dtype=np.float32)
    if candidate.startswith("ridge"):
        prediction = x.reshape(len(x), -1).astype(np.float64) @ arrays["coefficient"] + arrays["intercept"]
    else:
        tf = _tensorflow()
        model = _new_lstm(tf, x.shape[1:], metadata["seed"])
        model.set_weights([arrays[f"weight_{i}"] for i in range(len(arrays))])
        prediction = np.asarray(model(x, training=False)).ravel()
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite saved forecast")
    return pd.DataFrame([{"Date": date, "Ticker": ticker, "Predicted": float(value), "Candidate": candidate}
                         for (date, ticker), value in zip(keys, prediction)])
