"""Bounded, validation-selected LSTM extension; historical results are exploratory.

Default check mode does not train, import TensorFlow, download, or create files.
Train mode writes only a new output directory. Existing production frozen weights
and published signals are never read as destinations or replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from experiments.risk_experiment import (
    FEATURES, MARKET, TICKERS, build_frames, load_panel, load_sentiment,
)
from scripts.common import ROOT, TZ

CONFIG_PATH = ROOT / "research/extension/config.json"
MARKET_DIR = ROOT / "experiments/results/v2/v211/market"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["label_gap"] != 6 or config["hold_sessions"] != 5:
        raise ValueError("The executable five-session target requires entry i+1 and exit i+6")
    if config.get("allow_test_selection") or config.get("allow_official_model_replacement"):
        raise ValueError("Test selection and automatic production replacement are prohibited")
    if config["candidate_order"] != ["price_lstm", "sentiment_risk_lstm"]:
        raise ValueError("This protocol registers exactly two candidates")
    if config["candidates"]["price_lstm"] != ["Close"] or config["candidates"]["sentiment_risk_lstm"] != FEATURES:
        raise ValueError("Candidate feature order differs from the registered design")
    if (config.get("broker_fee_rate") != 0.001425 or config.get("sell_tax_rate") != 0.003
            or config.get("slippage_bps_each_side") != 0):
        raise ValueError("The shared book supports only the disclosed fee/tax and zero-slippage protocol")
    return config


def split_positions(positions, config):
    """Use the same dated samples for both candidates and purge label overlap."""
    positions = np.asarray(positions, dtype=int)
    if len(positions) == 0 or np.any(np.diff(positions) <= 0):
        raise ValueError("Signal positions must be nonempty and strictly increasing")
    test_len, val_len = config["test_samples"], config["validation_samples"]
    gap = config["label_gap"]
    if len(positions) < test_len:
        raise ValueError("Insufficient mature test signals")
    test = positions[-test_len:]
    valid = positions[positions + gap < test[0]]
    if len(valid) < val_len:
        raise ValueError("Insufficient purged validation signals")
    val = valid[-val_len:]
    train = positions[positions + gap < val[0]][-config["train_samples"]:]
    if len(train) < config["min_train_samples"]:
        raise ValueError("Insufficient purged training signals")
    if train[-1] + gap >= val[0] or val[-1] + gap >= test[0]:
        raise AssertionError("A label crosses the next split boundary")
    return {"train": train, "validation": val, "test": test}


def prepare_data(price_path, sentiment_path, as_of, config):
    panel = load_panel(price_path)
    as_of = pd.Timestamp(as_of).normalize()
    panel = panel.loc[panel.index <= as_of]
    if panel.empty or panel.index[-1] != as_of:
        raise ValueError("The requested as-of close is not available")
    sentiment = load_sentiment(sentiment_path, panel.index)
    frames, risk = build_frames(panel, sentiment)
    five = (panel[TICKERS].shift(-6) / panel[TICKERS].shift(-1) - 1) * 100
    step = config["time_step"]
    good = np.ones(len(panel), bool)
    for ticker in TICKERS:
        finite = np.isfinite(frames[ticker][FEATURES].to_numpy(float)).all(axis=1)
        # The price-only control shares the multivariate candidate's coverage.
        windows = pd.Series(finite).rolling(step, min_periods=step).sum().eq(step).to_numpy()
        good &= windows
    good &= np.isfinite(five.to_numpy(float)).all(axis=1)
    positions = np.flatnonzero(good)
    split = split_positions(positions, config)
    _assert_trading_coverage(panel, panel.index[split["train"][0] - step + 1], as_of)
    return {"panel": panel, "sentiment": sentiment, "frames": frames, "risk": risk,
            "target": five, "split": split, "as_of": as_of}


def _assert_trading_coverage(panel, start, end):
    """A missing bar cannot silently shorten six-session labels or input windows.

    Legacy incomplete warm-up years before the retained training window do not
    change this run's dated windows. Every exchange session from the first
    retained training input to the as-of close is required.
    """
    from scripts.common import sessions

    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    expected = sessions(start, end)
    observed = panel.index[(panel.index >= start) & (panel.index <= end)]
    missing = expected.difference(observed)
    if len(missing):
        raise ValueError("Missing exchange-session prices in training/evaluation coverage: "
                         + ", ".join(str(day.date()) for day in missing[:10]))
    if not np.isfinite(panel.loc[observed, TICKERS + [MARKET]].to_numpy(float)).all():
        raise ValueError("Non-finite prices in training/evaluation coverage")


def scaled_windows(frame, features, split, step):
    """The scaler sees only rows contained in training input windows."""
    raw = frame[list(features)].to_numpy(float)
    fit_rows = np.unique(np.concatenate([np.arange(p - step + 1, p + 1) for p in split["train"]]))
    if fit_rows.max() >= split["validation"][0]:
        raise AssertionError("Scaler touched validation")
    scaler = MinMaxScaler().fit(raw[fit_rows])
    transformed = scaler.transform(raw)
    windows = {name: np.asarray([transformed[p-step+1:p+1] for p in positions], dtype=np.float32)
               for name, positions in split.items()}
    return windows, scaler, fit_rows


def _validation_rows(predictions):
    """Validate the full cross-section before correlation can drop any rows."""
    required = {"Split", "Signal_Date", "Ticker", "Predicted", "Actual_5d_Pct"}
    if not required.issubset(predictions.columns):
        raise ValueError("Missing validation prediction columns")
    part = predictions.loc[predictions.Split.eq("validation")].copy()
    if part.empty:
        raise ValueError("No validation observations")
    part["Signal_Date"] = pd.to_datetime(part.Signal_Date, errors="raise")
    if (part.Signal_Date.isna().any() or part.Signal_Date.dt.tz is not None
            or not part.Signal_Date.eq(part.Signal_Date.dt.normalize()).all()):
        raise ValueError("Validation signal dates must be nonmissing local dates")
    if part.duplicated(["Signal_Date", "Ticker"]).any():
        raise ValueError("Duplicate validation stock/signal observations")
    numeric = part[["Predicted", "Actual_5d_Pct"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Validation predictions and labels must be finite")
    part[["Predicted", "Actual_5d_Pct"]] = numeric
    for _date, day in part.groupby("Signal_Date", sort=True):
        if len(day) != len(TICKERS) or set(day.Ticker) != set(TICKERS):
            raise ValueError("Validation requires five distinct stocks per signal")
    return part.sort_values(["Signal_Date", "Ticker"]).reset_index(drop=True)


def validation_score(predictions):
    """Mean daily five-stock Spearman; finite ties use average ranks.

    A constant prediction or label cross-section contributes zero. Test rows,
    including invalid test values, never enter validation or selection.
    """
    part = _validation_rows(predictions)
    scores = []
    for _date, day in part.groupby("Signal_Date", sort=True):
        if day.Predicted.std() < 1e-12 or day.Actual_5d_Pct.std() < 1e-12:
            scores.append(0.0)
        else:
            scores.append(float(day.Predicted.corr(day.Actual_5d_Pct, method="spearman")))
    if not scores or not np.isfinite(scores).all():
        raise ValueError("No finite validation selection score")
    return float(np.mean(scores))


def choose_candidate(validation, config):
    if not {"Model", "Split"}.issubset(validation.columns):
        raise ValueError("Missing candidate model identities")
    names = config["candidate_order"]
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("Selection requires the two distinct registered candidates")
    margin = float(config["minimum_validation_improvement"])
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Validation improvement margin must be finite and nonnegative")
    observed = validation.loc[validation.Split.eq("validation"), "Model"]
    if set(observed) != set(names):
        raise ValueError("Validation must contain exactly the registered candidates")
    parts = {name: _validation_rows(validation[validation.Model.eq(name)]) for name in names}
    control_rows, fusion_rows = (parts[name] for name in names)
    if not control_rows[["Signal_Date", "Ticker"]].equals(fusion_rows[["Signal_Date", "Ticker"]]):
        raise ValueError("Candidate validation stock/signal dates differ")
    paired_columns = ["Actual_5d_Pct"] + [name for name in ("Fold", "Execution_Date", "Label_End")
                                           if name in validation.columns]
    if not control_rows[paired_columns].equals(fusion_rows[paired_columns]):
        raise ValueError("Candidate validation labels or target intervals differ")
    scores = {name: validation_score(parts[name]) for name in names}
    control, fusion = config["candidate_order"]
    selected = fusion if scores[fusion] > scores[control] + margin else control
    return {"selected": selected, "selection_metric": config["selection_metric"],
            "validation_scores": scores, "minimum_improvement": config["minimum_validation_improvement"],
            "test_used_for_selection": False,
            "selection_rule": "Select sentiment_risk_lstm only when its validation score exceeds price_lstm by more than the registered margin; ties retain control."}


def _prediction_rows(name, ticker, values, data, split_name):
    index, positions = data["panel"].index, data["split"][split_name]
    return [{"Model": name, "Split": split_name, "Fold": 1, "Ticker": ticker,
             "Signal_Date": str(index[p].date()), "Execution_Date": str(index[p+1].date()),
             "Label_End": str(index[p+6].date()), "Predicted": float(pred),
             "Actual_5d_Pct": float(data["target"].iloc[p][ticker])}
            for p, pred in zip(positions, values)]


def _scaler_document(scaler):
    return {name: getattr(scaler, name + "_").tolist()
            for name in ("data_min", "data_max", "scale", "min")}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def verify_run(output_dir):
    """Verify retained artifacts before recomputing a book or frozen forecast."""
    output = Path(output_dir).resolve()
    manifest = json.loads((output / "reproduction_manifest.json").read_text(encoding="utf-8"))
    for name, wanted in manifest["files_sha256"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or digest(path) != wanted:
            raise ValueError(f"Extension artifact hash mismatch: {name}")
    # Source snapshots verify provenance; replay additionally checks the current
    # implementation so a changed program cannot silently claim exact replay.
    return manifest


def _verify_runtime_sources(manifest):
    replay_sources = manifest.get("replay_code_sha256", manifest.get("code_sha256", {}))
    for name in ("scripts/research_extension.py", "experiments/risk_experiment.py",
                 "experiments/v2/books.py", "experiments/calendar_rules.py", "scripts/common.py"):
        if name in replay_sources and digest(ROOT / name) != replay_sources[name]:
            raise ValueError(f"Extension replay source differs from retained revision: {name}")


def reproduce_book(output_dir):
    """Recompute portfolio metrics from retained source inputs and predictions."""
    output = Path(output_dir)
    manifest = verify_run(output)
    _verify_runtime_sources(manifest)
    config = load_config(output / "registered_config.json")
    card = json.loads((output / "model_card.json").read_text(encoding="utf-8"))
    data = prepare_data(output / "inputs/prices.csv", output / "inputs/sentiment.csv", card["as_of"], config)
    predictions = pd.read_csv(output / "predictions.csv")
    actual, _books = evaluate_books(data, predictions, card["selected"])
    saved = pd.read_csv(output / "strategy_metrics.csv").set_index("Name")
    actual = actual.set_index("Name")
    deltas = {}
    for column in ("ROI_Pct", "Sharpe", "MDD_Pct", "Total_Transaction_Cost"):
        delta = float(np.max(np.abs(saved.loc[actual.index, column] - actual[column])))
        if delta > 1e-8:
            raise ValueError(f"Extension book reconstruction differs: {column} ({delta})")
        deltas[column] = delta
    return {"reproduced": True, "maximum_deltas": deltas, "tensorflow_used": False,
            "historical_alpha_proven": False}


def frozen_forecast(output_dir, price_path, sentiment_path, as_of):
    """Reuse saved candidate weights and scalers without fitting or selecting."""
    output = Path(output_dir)
    _verify_runtime_sources(verify_run(output))
    config = load_config(output / "registered_config.json")
    card = json.loads((output / "model_card.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    selected = card["selected"]
    data = prepare_data(price_path, sentiment_path, as_of, config)
    if pd.Timestamp(as_of) <= pd.Timestamp(card["splits"]["validation"][-1]):
        raise ValueError("Frozen forecast must follow the recorded selection period")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    import tensorflow as tf
    from experiments.v2.books import threshold_weights, validation_tau
    from scripts.common import sessions

    following = sessions(pd.Timestamp(as_of) + pd.Timedelta(days=1), pd.Timestamp(as_of) + pd.Timedelta(days=21))
    if len(following) < 6:
        raise ValueError("Insufficient exchange-calendar sessions for the five-session forecast")
    validation = pd.read_csv(output / "validation_predictions.csv")
    validation = validation[validation.Model == selected]
    tau = validation_tau(validation, fold=1, as_of=as_of)
    predictions = []
    with np.load(output / "weights.npz", allow_pickle=False) as store:
        for ticker in TICKERS:
            model_card = card["models"][selected][ticker]
            raw = data["frames"][ticker][model_card["features"]].iloc[-config["time_step"]:].to_numpy(float)
            if not np.isfinite(raw).all():
                raise ValueError(f"Missing current features: {ticker}")
            state = model_card["scaler"]
            matrix = (raw * np.asarray(state["scale"]) + np.asarray(state["min"])).astype(np.float32)[None]
            model = _network(tf, matrix.shape[1:], config)
            model.set_weights([store[f"{selected}__{ticker}__{i}"] for i in range(model_card["weight_count"])])
            predictions.append(float(np.asarray(model(matrix, training=False)).ravel()[0]))
    flag = bool(data["risk"].loc[pd.Timestamp(as_of), "Risk_Off"])
    weights = threshold_weights(predictions, tau, flag)
    return pd.DataFrame([{
        "Model": selected, "Ticker": ticker, "Signal_Date": str(pd.Timestamp(as_of).date()),
        "Forecast_Entry_Date": str(following[0].date()), "Label_End": str(following[5].date()),
        "Predicted_Return_Pct": predicted, "Target_Weight": float(weight), "Cash_Weight": float(1 - weights.sum()),
        "Risk_Off": flag, "Generated_At": datetime.now(TZ).isoformat(),
        "Model_Available_At": selection["selected_at"],
        "Forecast_Mode": "historical_frozen_replay" if pd.Timestamp(as_of).date() < pd.Timestamp(selection["selected_at"]).date()
                         else "forward_candidate_forecast_not_published",
        "Weight_Sha256": digest(output / "weights.npz"), "Weights_Reused": True,
        "Research_Signal_Published": False,
        "Note": "Frozen candidate forecast only; not a fill or a new independent live research sample.",
    } for ticker, predicted, weight in zip(TICKERS, predictions, weights)])


def _network(tf, shape, config):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=shape), tf.keras.layers.LSTM(config["units"]),
        tf.keras.layers.Dropout(config["dropout"]), tf.keras.layers.Dense(1),
    ])


def _fit_one(tf, windows, target, config):
    # A private one-thread dataset avoids accumulating TensorFlow input pools.
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    datasets = {}
    for name in ("train", "validation"):
        datasets[name] = tf.data.Dataset.from_tensor_slices((windows[name], target[name])).batch(config["batch_size"]).with_options(options)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["seed"])
    model = _network(tf, windows["train"].shape[1:], config)
    model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]), loss="mse")
    history = model.fit(datasets["train"], validation_data=datasets["validation"],
                        epochs=config["max_epochs"], verbose=0, shuffle=False,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config["patience"], restore_best_weights=True)])
    return model, len(history.history["loss"])


def fixed_holding_endpoint(panel, test, frequency=5):
    """Liquidate at the final scheduled trade's five-session exit.

    Daily forecasts between rebalance dates do not create extra holding periods.
    Ending at the final *daily* label maturity would extend the last trade past
    its registered holding period when later forecasts were never trade eligible.
    """
    entries = pd.to_datetime(test.Execution_Date).drop_duplicates().sort_values()
    if entries.empty:
        raise ValueError("No executable test predictions")
    start = pd.Timestamp(entries.iloc[0])
    anchor = int(panel.index.get_loc(start))
    eligible = []
    for signal, part in test.groupby("Signal_Date", sort=True):
        entry = pd.Timestamp(part.Execution_Date.iloc[0])
        if (int(panel.index.get_loc(entry)) - anchor) % frequency == 0:
            signal_position = int(panel.index.get_loc(pd.Timestamp(signal)))
            label_end = panel.index[signal_position + 6]
            if not pd.to_datetime(part.Label_End).eq(label_end).all():
                raise ValueError("Test label endpoint does not match the five-session hold")
            eligible.append((entry, label_end))
    if not eligible:
        raise ValueError("No scheduled test trades")
    return start, max(eligible, key=lambda pair: pair[0])[1]


def evaluate_books(data, predictions, selected):
    from experiments.v2.books import index_metrics, metrics, run_book

    test = predictions[predictions.Split == "test"]
    start, end = fixed_holding_endpoint(data["panel"], test)
    calendar = pd.to_datetime(test.Signal_Date.unique())
    rows, books = [], {}
    for name in ("price_lstm", "sentiment_risk_lstm"):
        block = predictions[predictions.Model == name]
        book = run_book(data["panel"], data["risk"], block[block.Split == "test"], data["sentiment"],
                        start, end, 5, "threshold", block[block.Split == "validation"], signal_calendar=calendar)
        row = metrics(book, name, "every_5_sessions")
        row["Selected_Before_Test_Inference"] = name == selected
        rows.append(row)
        books[name] = book
    for name, mode in (("matched_risk_equal", "equal"), ("full_exposure_buy_hold", "buy_hold")):
        book = run_book(data["panel"], data["risk"], None, data["sentiment"], start, end, 5, mode,
                        signal_calendar=calendar)
        row = metrics(book, name, "every_5_sessions" if mode == "equal" else "buy_and_hold")
        row["Selected_Before_Test_Inference"] = False
        rows.append(row)
        books[name] = book
    proxy = index_metrics(data["panel"], start, end)
    proxy["Name"] = "theoretical_twii_with_stock_cost_proxy_not_an_etf"
    proxy["Selected_Before_Test_Inference"] = False
    rows.append(proxy)
    frame = pd.DataFrame(rows)
    equal_roi = float(frame.loc[frame.Name == "matched_risk_equal", "ROI_Pct"].iloc[0])
    proxy_roi = float(proxy["ROI_Pct"])
    frame["Gap_Vs_Matched_Equal_Pp"] = frame.ROI_Pct - equal_roi
    frame["Gap_Vs_Theoretical_Index_Proxy_Pp"] = frame.ROI_Pct - proxy_roi
    return frame, books


def train(data, config, output_dir, price_path, sentiment_path, config_path):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; research selections and results are immutable")
    protected = (ROOT / "data", ROOT / "models", ROOT / "research/v211",
                 ROOT / "experiments/results/v2/v211")
    if output == ROOT.resolve() or any(output.is_relative_to(folder.resolve()) for folder in protected):
        raise ValueError("Research output must not replace production inputs or models")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import tensorflow as tf

    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    _write_json(output / "registered_config.json", config)
    sources = [Path(__file__), ROOT / "experiments/risk_experiment.py", ROOT / "experiments/v2/books.py",
               ROOT / "experiments/calendar_rules.py", Path(config_path)]
    for path in sources:
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path("external_config.json")
        target = output / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    # Retain the actual as-of inputs, including provenance columns. Nothing is
    # backfilled and no later source rows are silently included in this snapshot.
    for name, source in (("prices", price_path), ("sentiment", sentiment_path)):
        raw = pd.read_csv(source)
        raw = raw[pd.to_datetime(raw.Date).dt.normalize() <= data["as_of"]]
        raw.to_csv(output / "inputs" / f"{name}.csv", index=False)
    validation_rows, stored_models, arrays, cards = [], {}, {}, {}
    split = data["split"]
    for name in config["candidate_order"]:
        features = config["candidates"][name]
        cards[name] = {}
        for ticker in TICKERS:
            windows, scaler, fit_rows = scaled_windows(data["frames"][ticker], features, split, config["time_step"])
            target = {part: data["target"].iloc[pos][ticker].to_numpy(np.float32) for part, pos in split.items()}
            model, epochs = _fit_one(tf, windows, target, config)
            # Calling the model directly avoids model.predict input-pool creation.
            val = np.asarray(model(windows["validation"], training=False)).ravel()
            validation_rows.extend(_prediction_rows(name, ticker, val, data, "validation"))
            stored_models[(name, ticker)] = (model, windows)
            weights = model.get_weights()
            for i, array in enumerate(weights):
                arrays[f"{name}__{ticker}__{i}"] = array
            cards[name][ticker] = {"features": features, "scaler": _scaler_document(scaler), "weight_count": len(weights),
                                   "epochs_run": epochs, "scaler_first_date": str(data["panel"].index[fit_rows[0]].date()),
                                   "scaler_last_date": str(data["panel"].index[fit_rows[-1]].date())}
            print(f"[extension] fitted {name} {ticker}; epochs={epochs}", flush=True)
    validation = pd.DataFrame(validation_rows)
    selection = choose_candidate(validation, config)
    selection["selected_at"] = datetime.now(TZ).isoformat()
    selection["test_predictions_computed"] = False
    _write_json(output / "selection.json", selection)
    validation.to_csv(output / "validation_predictions.csv", index=False)
    # Persist weights and selection before looking at any candidate's test output.
    np.savez_compressed(output / "weights.npz", **arrays)
    card = {"protocol": config["protocol_version"], "as_of": str(data["as_of"].date()), "architecture": {
        key: config[key] for key in ("time_step", "units", "dropout", "learning_rate", "seed")},
        "selected": selection["selected"], "models": cards,
        "splits": {name: [str(data["panel"].index[p].date()) for p in positions] for name, positions in split.items()},
        "label": "100 * (close[i+6] / close[i+1] - 1)", "selected_without_test": True,
        "production_replacement": False, "study_status": config["study_status"]}
    _write_json(output / "model_card.json", card)
    test_rows = []
    for (name, ticker), (model, windows) in stored_models.items():
        values = np.asarray(model(windows["test"], training=False)).ravel()
        test_rows.extend(_prediction_rows(name, ticker, values, data, "test"))
    predictions = pd.concat([validation, pd.DataFrame(test_rows)], ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    portfolio, books = evaluate_books(data, predictions, selection["selected"])
    portfolio.to_csv(output / "strategy_metrics.csv", index=False)
    for name, book in books.items():
        book.to_csv(output / f"nav_{name}.csv", index=False)
    selected_row = portfolio[portfolio.Name == selection["selected"]].iloc[0]
    summary = {"study_status": config["study_status"], "selected": selection["selected"],
               "selection_used_test": False, "test_signal_days": len(split["test"]),
               "mature_test_labels_only": True, "label_gap_sessions": config["label_gap"],
               "portfolio_end_policy": "five_session_exit_of_last_scheduled_test_trade",
               "portfolio_start": str(books[selection["selected"]].Date.iloc[0].date()),
               "portfolio_end": str(books[selection["selected"]].Date.iloc[-1].date()),
               "selected_test_roi_pct": float(selected_row.ROI_Pct),
               "gap_vs_matched_equal_pp": float(selected_row.Gap_Vs_Matched_Equal_Pp),
               "gap_vs_theoretical_index_proxy_pp": float(selected_row.Gap_Vs_Theoretical_Index_Proxy_Pp),
               "proved_independent_alpha": False, "historical_news_point_in_time_proven": False,
               "official_model_replaced": False,
               "limitations": ["Previously studied historical test period; no new independent test.",
                               "Historical sentiment availability cannot be proven retroactively.",
                               "The index comparator is a theoretical price index with a stock-cost proxy, not an investable ETF.",
                               "Five-day labels overlap; 150 signals are not 150 independent observations.",
                               "Zero slippage; dividends are excluded because source closes are unadjusted."]}
    _write_json(output / "summary.json", summary)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    lines = ["# 新聞與風控 LSTM 延伸候選", "", f"採用候選：`{selection['selected']}`；採用只看驗證集。",
             "本結果為已觀察歷史區間的探索性研究，不是獨立樣本外 alpha 證據。", "",
             "| 策略 | ROI (%) | Sharpe | MDD (%) | 相對可比等權 (pp) |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in portfolio.itertuples():
        lines.append(f"| {row.Name} | {row.ROI_Pct:.3f} | {row.Sharpe:.3f} | {row.MDD_Pct:.3f} | {row.Gap_Vs_Matched_Equal_Pp:.3f} |")
    lines += ["", "大盤列為理論指數加股票成本代理，並非可交易 ETF。所有股票帳本使用相同次日收盤、五日排程、成本及訊號日風控。",
              "保留原始輸入、候選權重、Scaler、驗證採用決策、逐筆預測及帳本；沒有替換正式模型。",
              "歷史新聞可取得時間未獲證明，零滑價與不含股利也限制解讀；不可為了超過大盤而重選測試結果。", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    files = {path.relative_to(output).as_posix(): digest(path) for path in output.rglob("*") if path.is_file()}
    _write_json(output / "reproduction_manifest.json", {
        "schema": "extension-reproduction-v1", "source_commit": commit,
        "code_sha256": {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else "external_config.json": digest(path) for path in sources},
        "files_sha256": files, "versions": {name: importlib.metadata.version(name) for name in ("tensorflow_cpu", "numpy", "pandas", "scikit-learn")},
        "note": "Verify file hashes before inference or reconstruction; exact training sources are retained under sources/. Source hashes identify uncommitted code as well as committed code."})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "train", "reproduce", "forecast"), default="check")
    parser.add_argument("--prices", type=Path, default=MARKET_DIR / "prices.csv")
    parser.add_argument("--sentiment", type=Path, default=MARKET_DIR / "sentiment.csv")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.mode == "reproduce":
        if args.output_dir is None:
            parser.error("reproduce requires --output-dir containing a retained run")
        print(json.dumps(reproduce_book(args.output_dir), ensure_ascii=False, indent=2))
        return 0
    if args.mode == "forecast":
        if args.output_dir is None or args.as_of is None:
            parser.error("forecast requires --output-dir containing a retained run and --as-of")
        print(frozen_forecast(args.output_dir, args.prices, args.sentiment, args.as_of).to_csv(index=False))
        return 0
    config = load_config(args.config)
    as_of = args.as_of or str(pd.read_csv(args.prices).Date.max())
    data = prepare_data(args.prices, args.sentiment, as_of, config)
    if args.mode == "check":
        report = {"mode": "check", "ready": True, "read_only": True, "tensorflow_imported": False,
                  "as_of": str(data["as_of"].date()), "study_status": config["study_status"],
                  "splits": {name: {"samples": len(pos), "first": str(data["panel"].index[pos[0]].date()),
                                    "last": str(data["panel"].index[pos[-1]].date()),
                                    "last_label_end": str(data["panel"].index[pos[-1]+6].date())}
                             for name, pos in data["split"].items()}}
    else:
        output = args.output_dir or ROOT / "research/extension/runs" / str(data["as_of"].date())
        report = train(data, config, output, args.prices, args.sentiment, args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
