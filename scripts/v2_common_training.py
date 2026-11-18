"""Fresh, fixed V2 LSTM feature ablation on the 444/555 shared signal dates.

All previously observed history remains exploratory. The default CLI only
checks local inputs: no TensorFlow import, download or filesystem mutation.
Training requires an explicit fresh output directory and never copies raw data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from experiments.risk_experiment import FEATURES, MARKET, TICKERS, build_frames
from scripts.common import ROOT, TZ, sessions
from scripts.paired_research import market_risk, price_panel

CONFIG_PATH = ROOT / "research/v2_common/config.json"
PRICE_PATH = Path("experiments/results/v2/v211/market/prices.csv")
SENTIMENT_PATH = Path("experiments/results/v2/v211/market/sentiment.csv")
PAIRED_PROTOCOL_PATH = Path("reports/daily/2026-11-17/paired/protocol.json")
DEFAULT_CONFIG = {
    "protocol_version": "v2-common-training-v1", "study_status": "exploratory",
    "as_of": "2026-11-17", "candidate_order": ["price_lstm", "no_news_lstm", "news_lstm"],
    "candidates": {"price_lstm": ["Close"], "no_news_lstm": FEATURES[:-1], "news_lstm": FEATURES},
    "test_samples": 150, "folds": 3, "fold_test_samples": 50,
    "validation_samples": 60, "train_samples": 756,
    "time_step": 10, "label_gap": 6, "hold_sessions": 5,
    "layers": 2, "units": 50, "dropout": 0.2, "seed": 42,
    "learning_rate": 0.001, "max_epochs": 50, "patience": 5, "batch_size": 32,
    "allow_test_selection": False, "allow_official_model_replacement": False,
}
PREDICTION_COLUMNS = ["Model", "Fold", "Split", "Signal_Date", "Execution_Date", "Label_End",
                      "Label_State", "Ticker", "Predicted", "Actual_5d_Pct"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _validate_config(config):
    for key, expected in DEFAULT_CONFIG.items():
        if key not in config or config[key] != expected:
            raise ValueError(f"Fixed registered training setting differs: {key}")
    return config


def load_config(path=CONFIG_PATH):
    """Read the fixed protocol; extra reporting/book fields are permitted."""
    return _validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def _dates(values):
    result = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if result.tz is not None or result.isna().any() or not result.equals(result.normalize()):
        raise ValueError("Dates must be nonmissing local session-date labels")
    return result


def fold_positions(eligible, test_positions, config):
    """Purge on full-calendar positions, retaining exactly 756/60 per fold."""
    eligible, test_positions = np.asarray(eligible, int), np.asarray(test_positions, int)
    for values in (eligible, test_positions):
        if not len(values) or np.any(np.diff(values) <= 0):
            raise ValueError("Signal positions must be nonempty and strictly increasing")
    if len(test_positions) != config["test_samples"]:
        raise ValueError("Shared test coverage must retain every registered signal")
    folds = []
    for number, test in enumerate(np.split(test_positions, config["folds"]), 1):
        if len(test) != config["fold_test_samples"]:
            raise ValueError("Each registered test fold requires 50 signals")
        val = eligible[eligible + config["label_gap"] < test[0]][-config["validation_samples"]:]
        if len(val) != config["validation_samples"]:
            raise ValueError("Insufficient mature purged validation signals")
        train = eligible[eligible + config["label_gap"] < val[0]][-config["train_samples"]:]
        if len(train) != config["train_samples"]:
            raise ValueError("Insufficient mature purged training signals")
        if train[-1] + 6 >= val[0] or val[-1] + 6 >= test[0]:
            raise AssertionError("Label maturity crosses a split boundary")
        folds.append({"Fold": number, "train": train, "validation": val, "test": test})
    return folds


def _source_metadata(path, root, raw=None):
    path = Path(path).resolve()
    public_path = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
    result = {"path": public_path, "sha256": digest(path)}
    if raw is not None:
        dates = _dates(raw.Date)
        result.update(rows=len(raw), first_date=str(dates.min().date()), last_date=str(dates.max().date()))
    return result


def _assert_coverage(panel, start, end):
    expected = sessions(start, end)
    observed = panel.index[(panel.index >= start) & (panel.index <= end)]
    if len(observed.difference(expected)):
        raise ValueError("Non-XTAI price dates in retained coverage")
    part = panel.reindex(expected)[TICKERS + [MARKET]]
    bad = ~np.isfinite(part.to_numpy(float)).all(axis=1)
    if bad.any():
        raise ValueError("Missing/non-finite exchange-session prices in retained coverage: "
                         + ", ".join(str(day.date()) for day in expected[bad][:10]))


def prepare_data(root, config, price_path=None, sentiment_path=None, paired_protocol_path=None):
    """Read external immutable inputs; keep missing price rows instead of dropping them.

    ``calendar`` extends beyond as-of solely to date pending labels. ``panel``
    never contains future prices. Training eligibility uses finite mature labels;
    test coverage is fixed independently of target availability.
    """
    _validate_config(config)
    root = Path(root).resolve()
    price_path = Path(price_path) if price_path is not None else root / PRICE_PATH
    sentiment_path = Path(sentiment_path) if sentiment_path is not None else root / SENTIMENT_PATH
    paired_protocol_path = Path(paired_protocol_path) if paired_protocol_path is not None else root / PAIRED_PROTOCOL_PATH
    protocol = json.loads(paired_protocol_path.read_text(encoding="utf-8"))
    signal_dates, target_dates = _dates(protocol["shared_signal_dates"]), _dates(protocol["shared_target_dates"])
    if (len(signal_dates) != config["test_samples"] or len(target_dates) != len(signal_dates)
            or not signal_dates.is_monotonic_increasing or signal_dates.has_duplicates):
        raise ValueError("The paired protocol must identify exactly 150 distinct increasing shared signals")
    if list(protocol.get("tickers", TICKERS)) != TICKERS:
        raise ValueError("The shared protocol stock order differs")
    as_of = pd.Timestamp(config["as_of"])
    if target_dates.max() != as_of:
        raise ValueError("As-of must match the last shared execution date")
    raw_prices, raw_sentiment = pd.read_csv(price_path), pd.read_csv(sentiment_path)
    raw_prices["Date"], raw_sentiment["Date"] = _dates(raw_prices.Date), _dates(raw_sentiment.Date)
    observed = price_panel(raw_prices[raw_prices.Date <= as_of])
    if observed.empty or observed.index[-1] != as_of:
        raise ValueError("The as-of close is unavailable")
    # Preserve the canonical warm-up index, including four stock-missing rows.
    # Early exchange-calendar corrections are not allowed to rewrite the risk
    # threshold's history. Strict full-XTAI coverage is required below from the
    # first retained training input through as-of; no rows are dropped or filled.
    panel = observed.copy()
    calendar = panel.index.append(sessions(as_of + pd.Timedelta(days=1), as_of + pd.Timedelta(days=35)))
    panel.index.name = "Date"
    test_positions = panel.index.get_indexer(signal_dates)
    if (test_positions < 0).any():
        raise ValueError("A shared signal is not an available XTAI session")
    if not calendar[test_positions + 1].equals(target_dates):
        raise ValueError("Shared execution dates must be signal+1 on the full session calendar")
    historical_sentiment = raw_sentiment.loc[raw_sentiment.Date <= as_of]
    if historical_sentiment.duplicated(["Date", "ticker"]).any():
        raise ValueError("Duplicate sentiment date/ticker")
    sentiment = historical_sentiment.pivot(index="Date", columns="ticker", values="Daily_Sentiment")
    sentiment = sentiment.reindex(index=panel.index, columns=TICKERS).astype(float)
    if np.isinf(sentiment.to_numpy()).any() or (sentiment.abs() > 1).any().any():
        raise ValueError("Observed sentiment must be finite [-1, 1] or explicitly missing")
    frames, legacy_risk = build_frames(panel, sentiment)
    risk = market_risk(panel)
    # Both implementations currently use the same sample volatility definition.
    # Keep an explicit guard against silently changing the model's risk feature.
    if not np.allclose(legacy_risk.Market_Vol_20d, risk.Market_Vol_20d, equal_nan=True):
        raise ValueError("Training and paired Market_Vol_20d definitions differ; review required")
    for frame in frames.values():
        for column in ("Market_Vol_20d", "Risk_Threshold_Pct", "Risk_Off"):
            frame[column] = risk[column]
    target = 100 * (panel[TICKERS].shift(-6) / panel[TICKERS].shift(-1) - 1)
    good = np.ones(len(panel), bool)
    for ticker in TICKERS:
        finite = np.isfinite(frames[ticker][FEATURES].to_numpy(float)).all(axis=1)
        good &= pd.Series(finite).rolling(config["time_step"], min_periods=config["time_step"]).sum().eq(config["time_step"]).to_numpy()
    if not good[test_positions].all():
        raise ValueError("Missing features in a registered test input window")
    eligible = np.flatnonzero(good & np.isfinite(target.to_numpy(float)).all(axis=1))
    folds = fold_positions(eligible, test_positions, config)
    first_input = min(fold["train"][0] for fold in folds) - config["time_step"] + 1
    _assert_coverage(panel, panel.index[first_input], as_of)
    risk_block = risk.loc[signal_dates, ["Market_Vol_20d", "Risk_Threshold_Pct"]]
    if not np.isfinite(risk_block.to_numpy(float)).all():
        raise ValueError("Shared signals require mature finite paired risk information")
    sources = {"prices": _source_metadata(price_path, root, raw_prices),
               "sentiment": _source_metadata(sentiment_path, root, raw_sentiment),
               "paired_protocol": _source_metadata(paired_protocol_path, root)}
    for name, expected in config.get("source_sha256", {}).items():
        if name not in sources or sources[name]["sha256"] != expected:
            raise ValueError(f"Registered source hash differs: {name}")
    return {"root": root, "as_of": as_of, "panel": panel, "calendar": calendar,
            "sentiment": sentiment, "frames": frames, "risk": risk, "target": target,
            "folds": folds, "test_positions": test_positions, "source_metadata": sources,
            "source_paths": {"prices": price_path.resolve(), "sentiment": sentiment_path.resolve(),
                             "paired_protocol": paired_protocol_path.resolve()},
            "paired_protocol": protocol, "first_retained_input": panel.index[first_input]}


def scaled_windows(frame, features, split, step=10):
    """Fit once on the unique union of training input rows, never labels."""
    raw = frame[list(features)].to_numpy(float)
    fit_rows = np.unique(np.concatenate([np.arange(p - step + 1, p + 1) for p in split["train"]]))
    if fit_rows[0] < 0 or fit_rows[-1] >= split["validation"][0]:
        raise ValueError("Scaler fitting rows cross a split boundary")
    if not np.isfinite(raw[fit_rows]).all():
        raise ValueError("Non-finite scaler training inputs")
    scaler = MinMaxScaler().fit(raw[fit_rows])
    windows = {}
    for name in ("train", "validation", "test"):
        values = np.asarray([raw[p-step+1:p+1] for p in split[name]], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {name} inputs")
        windows[name] = scaler.transform(values.reshape(-1, len(features))).reshape(values.shape).astype(np.float32)
    return windows, scaler, fit_rows


def prediction_rows(model, fold, ticker, values, data, split):
    positions = fold[split]
    values = np.asarray(values, float).ravel()
    if len(values) != len(positions) or not np.isfinite(values).all():
        raise ValueError("Predictions must be finite and cover every registered signal")
    rows = []
    for p, value in zip(positions, values):
        label_end = data["calendar"][p + 6]
        matured = label_end <= data["as_of"]
        actual = float(data["target"].iloc[p][ticker]) if matured else np.nan
        if matured and not np.isfinite(actual):
            raise ValueError("A mature label lacks its real close prices")
        rows.append(dict(zip(PREDICTION_COLUMNS, [model, fold["Fold"], split,
            str(data["calendar"][p].date()), str(data["calendar"][p+1].date()),
            str(label_end.date()), "MATURED" if matured else "PENDING", ticker, float(value), actual])))
    return rows


def _network(tf, shape, config):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=shape),
        tf.keras.layers.LSTM(config["units"], return_sequences=True), tf.keras.layers.Dropout(config["dropout"]),
        tf.keras.layers.LSTM(config["units"]), tf.keras.layers.Dropout(config["dropout"]),
        tf.keras.layers.Dense(1),
    ])


def _fit_one(tf, windows, targets, config):
    """Fresh seeded weights and Adam for every fold/candidate/stock."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["seed"])
    model = _network(tf, windows["train"].shape[1:], config)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), loss="mse")
    initial_lr = float(model.optimizer.learning_rate.numpy())
    if not np.isclose(initial_lr, config["learning_rate"]):
        raise AssertionError("Fresh optimizer does not have the registered learning rate")
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    datasets = {name: tf.data.Dataset.from_tensor_slices((windows[name], targets[name])).batch(
        config["batch_size"]).with_options(options) for name in ("train", "validation")}
    started_at, clock = datetime.now(TZ).isoformat(), time.perf_counter()
    history = model.fit(datasets["train"], validation_data=datasets["validation"],
                        epochs=config["max_epochs"], shuffle=False, verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config["patience"],
                                                                   restore_best_weights=True)])
    metadata = {"fit_started_at": started_at, "fit_finished_at": datetime.now(TZ).isoformat(),
                "fit_seconds": time.perf_counter() - clock, "epochs_run": len(history.history["loss"]),
                "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
                "optimizer": "Adam", "optimizer_learning_rate_initial": initial_lr,
                "optimizer_learning_rate_final": float(model.optimizer.learning_rate.numpy()),
                "optimizer_iterations": int(model.optimizer.iterations.numpy()),
                "initialized_from_seed": config["seed"], "previous_fold_weights_carried": False,
                "previous_optimizer_carried": False,
                "history": {name: [float(v) for v in values] for name, values in history.history.items()}}
    return model, metadata


def _tensorflow():
    for key in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    return tf


def _scaler_document(scaler, features, fit_rows, index):
    return {"features": list(features), "feature_range": list(scaler.feature_range),
            "scale": scaler.scale_.tolist(), "min": scaler.min_.tolist(),
            "data_min": scaler.data_min_.tolist(), "data_max": scaler.data_max_.tolist(),
            "data_range": scaler.data_range_.tolist(), "n_samples_seen": int(scaler.n_samples_seen_),
            "fit_row_dates": [str(index[p].date()) for p in fit_rows]}


def _split_document(data):
    index = data["calendar"]
    return [{"Fold": fold["Fold"], **{name: [str(index[p].date()) for p in fold[name]]
             for name in ("train", "validation", "test")},
             "last_train_label_end": str(index[fold["train"][-1] + 6].date()),
             "last_validation_label_end": str(index[fold["validation"][-1] + 6].date())}
            for fold in data["folds"]]


def validate_output_dir(root, output_dir):
    root, output = Path(root).resolve(), Path(output_dir).resolve()
    protected = [root / name for name in ("data", "models", "research/v211", "experiments/results/v2/v211")]
    if output == root or any(output == p or output.is_relative_to(p) for p in protected):
        raise ValueError("Output must not replace protected raw data, frozen models or production ledgers")
    if output.exists():
        raise FileExistsError("Use a new output directory; retained research runs are immutable")
    return output


def _code_sources(root):
    paths = [Path(__file__).resolve(), root / "experiments/risk_experiment.py", root / "scripts/paired_research.py",
             root / "scripts/common.py", root / "experiments/v2/books.py"]
    return {str(path.relative_to(root)): digest(path) for path in paths}


def train(data, config, output_dir, config_path=None):
    """Fit all fixed candidates, seal validation/tau, then infer test without selecting."""
    from experiments.v2.books import validation_tau

    _validate_config(config)
    root = data["root"]
    output = validate_output_dir(root, output_dir)
    # Check source hashes again immediately before training, not merely at prepare.
    for name, source in data["source_metadata"].items():
        if digest(data["source_paths"][name]) != source["sha256"]:
            raise ValueError("An input source changed after data preparation")
    tf = _tensorflow()
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "registered_config.json", config)
    splits = _split_document(data)
    _write_json(output / "splits.json", splits)
    versions = {"tensorflow": tf.__version__, "keras": tf.keras.__version__,
                **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")}}
    card = {"protocol": config["protocol_version"], "study_status": "exploratory", "as_of": config["as_of"],
            "architecture": {key: config[key] for key in ("layers", "units", "dropout", "time_step", "seed", "learning_rate", "batch_size", "max_epochs", "patience")},
            "runtime": versions, "sources": data["source_metadata"], "splits": splits, "models": {},
            "label": "100 * (Close[i+6] / Close[i+1] - 1)", "selected_candidate": None,
            "test_used_for_selection": False, "production_replacement": False,
            "historical_news_point_in_time_proven": False, "raw_inputs_copied": False}
    validation_rows = []
    for fold in data["folds"]:
        fold_key = f"fold_{fold['Fold']}"
        card["models"][fold_key] = {}
        for name in config["candidate_order"]:
            card["models"][fold_key][name] = {}
            for ticker in TICKERS:
                windows, scaler, fit_rows = scaled_windows(data["frames"][ticker], config["candidates"][name], fold, config["time_step"])
                # Test labels never enter the fitter, callback or calibration.
                targets = {part: data["target"].iloc[fold[part]][ticker].to_numpy(np.float32)
                           for part in ("train", "validation")}
                if any(not np.isfinite(values).all() for values in targets.values()):
                    raise ValueError("Training/validation labels must be mature and finite")
                model, metadata = _fit_one(tf, windows, targets, config)
                values = np.asarray(model(windows["validation"], training=False)).ravel()
                validation_rows.extend(prediction_rows(name, fold, ticker, values, data, "validation"))
                weights_path = Path("models") / fold_key / name / f"{ticker}.weights.npz"
                (output / weights_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output / weights_path, **{f"weight_{i:02d}": value
                                                             for i, value in enumerate(model.get_weights())})
                scaler_path = weights_path.with_suffix(".scaler.json")
                _write_json(output / scaler_path, _scaler_document(scaler, config["candidates"][name], fit_rows, data["calendar"]))
                metadata.update(features=config["candidates"][name], scaler_path=scaler_path.as_posix(),
                                weights_path=weights_path.as_posix(), weights_sha256=digest(output / weights_path),
                                weight_count=len(model.get_weights()),
                                scaler_sha256=digest(output / scaler_path), source_sha256={k: v["sha256"] for k, v in data["source_metadata"].items()})
                card["models"][fold_key][name][ticker] = metadata
                del model, windows
                print(f"[v2-common] fitted {fold_key} {name} {ticker}; epochs={metadata['epochs_run']}", flush=True)
    validation = pd.DataFrame(validation_rows, columns=PREDICTION_COLUMNS)
    validation.to_csv(output / "validation_predictions.csv", index=False)
    taus = []
    for fold in data["folds"]:
        boundary = str(data["calendar"][fold["test"][0]].date())
        for name in config["candidate_order"]:
            tau = validation_tau(validation[validation.Model.eq(name)], fold=fold["Fold"], as_of=boundary)
            taus.append({"Fold": fold["Fold"], "Model": name, "as_of": boundary,
                         "tickers": TICKERS, "tau": tau.tolist(), "test_used": False})
    validation_seal = {"saved_at": datetime.now(TZ).isoformat(), "validation_sha256": digest(output / "validation_predictions.csv"),
                       "test_predictions_computed": False, "candidate_selection_performed": False, "thresholds": taus}
    _write_json(output / "validation_tau.json", validation_seal)
    _write_json(output / "model_card.json", card)
    # Every model, validation prediction and fold-specific tau is durable before
    # the first call involving test inputs. No test return chooses a winner.
    test_rows = []
    replay_deltas = []
    test_started_at = datetime.now(TZ).isoformat()
    for fold in data["folds"]:
        for name in config["candidate_order"]:
            for ticker in TICKERS:
                model_meta = card["models"][f"fold_{fold['Fold']}"][name][ticker]
                tf.keras.backend.clear_session()
                model = _load_weights_model(tf, output, model_meta, config)
                validation_values = np.asarray(model(_saved_windows(data, fold, ticker, output / model_meta["scaler_path"], "validation"), training=False)).ravel()
                saved_values = validation.loc[validation.Model.eq(name) & validation.Fold.eq(fold["Fold"])
                                              & validation.Ticker.eq(ticker), "Predicted"].to_numpy(float)
                replay_delta = float(np.max(np.abs(validation_values - saved_values)))
                if not np.isfinite(replay_delta) or replay_delta > 1e-6:
                    raise ValueError("Saved NPZ inference does not reproduce pre-test validation predictions")
                model_meta["saved_weights_validation_maximum_delta"] = replay_delta
                replay_deltas.append(replay_delta)
                matrix = _saved_windows(data, fold, ticker, output / model_meta["scaler_path"], "test")
                values = np.asarray(model(matrix, training=False)).ravel()
                test_rows.extend(prediction_rows(name, fold, ticker, values, data, "test"))
                del model, matrix
    test = pd.DataFrame(test_rows, columns=PREDICTION_COLUMNS)
    _write_json(output / "model_card.json", card)
    test.to_csv(output / "test_predictions.csv", index=False)
    pd.concat([validation, test], ignore_index=True).to_csv(output / "predictions.csv", index=False)
    counts = {name: {"test_rows": int(part.shape[0]), "matured_rows": int(part.Label_State.eq("MATURED").sum()),
                     "pending_rows": int(part.Label_State.eq("PENDING").sum()), "test_signal_days": int(part.Signal_Date.nunique())}
              for name, part in test.groupby("Model", sort=False)}
    summary = {"protocol": config["protocol_version"], "study_status": "exploratory", "as_of": config["as_of"],
               "candidate_counts": counts, "fresh_model_fits": len(data["folds"]) * len(config["candidate_order"]) * len(TICKERS),
               "selected_candidate": None, "test_used_for_selection": False, "official_model_replaced": False,
               "validation_saved_at": validation_seal["saved_at"], "test_inference_started_at": test_started_at,
               "saved_weights_validation_maximum_delta": max(replay_deltas), "prediction_replay_tolerance": 1e-6,
               "historical_alpha_proven": False, "historical_news_point_in_time_proven": False,
               "limitations": ["Previously observed history remains permanently exploratory.",
                               "Historical sentiment acquisition and scoring availability are unverified.",
                               "Overlapping five-session labels are not independent observations.",
                               "Unadjusted closes exclude dividends; no superiority claim or production adoption."]}
    _write_json(output / "summary.json", summary)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    manifest = {"schema": "v2-common-reproduction-v1", "source_commit": commit,
                "created_at": datetime.now(TZ).isoformat(), "code_sha256": _code_sources(root),
                "external_sources": data["source_metadata"], "runtime": versions,
                "config_source_sha256": digest(config_path) if config_path is not None else None,
                "files_sha256": {path.relative_to(output).as_posix(): digest(path) for path in output.rglob("*") if path.is_file()},
                "raw_inputs_copied": False, "exact_retraining_bytes_guaranteed": False,
                "replay": "Use retained lossless NPZ neural weights/scalers and exact external input hashes; never refit."}
    _write_json(output / "reproduction_manifest.json", manifest)
    return summary


def _saved_windows(data, fold, ticker, scaler_path, split):
    state = json.loads(Path(scaler_path).read_text(encoding="utf-8"))
    raw = data["frames"][ticker][state["features"]].to_numpy(float)
    step = 10
    values = np.asarray([raw[p-step+1:p+1] for p in fold[split]], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Replay inputs must be finite")
    return (values * np.asarray(state["scale"]) + np.asarray(state["min"])).astype(np.float32)


def _load_weights_model(tf, output, metadata, config):
    """Rebuild the public inference model without deserializing an optimizer."""
    model = _network(tf, (config["time_step"], len(metadata["features"])), config)
    with np.load(output / metadata["weights_path"], allow_pickle=False) as archive:
        model.set_weights([archive[f"weight_{i:02d}"] for i in range(metadata["weight_count"])])
    return model


def verify_run(output_dir):
    """Read-only artifact verification; does not import TensorFlow or need raw data."""
    output = Path(output_dir).resolve()
    manifest = json.loads((output / "reproduction_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "v2-common-reproduction-v1" or manifest.get("raw_inputs_copied") is not False:
        raise ValueError("Unexpected common-training artifact manifest")
    for name, expected in manifest["files_sha256"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Artifact missing or hash differs: {name}")
    return manifest


def replay_predictions(output_dir, root=ROOT, price_path=None, sentiment_path=None, paired_protocol_path=None, *, atol=1e-6):
    """Offline inference only: verify exact external inputs, predict, compare saved rows."""
    output, root = Path(output_dir).resolve(), Path(root).resolve()
    manifest = verify_run(output)
    if _code_sources(root) != manifest["code_sha256"]:
        raise ValueError("Runtime inference source differs from the retained training source")
    config = load_config(output / "registered_config.json")
    overrides = {"prices": price_path, "sentiment": sentiment_path, "paired_protocol": paired_protocol_path}
    paths = {name: Path(overrides[name]) if overrides[name] is not None else root / source["path"]
             for name, source in manifest["external_sources"].items()}
    for name, path in paths.items():
        if digest(path) != manifest["external_sources"][name]["sha256"]:
            raise ValueError(f"Replay external input hash differs: {name}")
    data = prepare_data(root, config, paths["prices"], paths["sentiment"], paths["paired_protocol"])
    tf = _tensorflow()
    card = json.loads((output / "model_card.json").read_text(encoding="utf-8"))
    rows = []
    for fold in data["folds"]:
        for name in config["candidate_order"]:
            for ticker in TICKERS:
                state = card["models"][f"fold_{fold['Fold']}"][name][ticker]
                tf.keras.backend.clear_session()
                model = _load_weights_model(tf, output, state, config)
                for split in ("validation", "test"):
                    values = np.asarray(model(_saved_windows(data, fold, ticker, output / state["scaler_path"], split), training=False)).ravel()
                    rows.extend(prediction_rows(name, fold, ticker, values, data, split))
                del model
    actual = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    saved = pd.read_csv(output / "predictions.csv")
    keys = ["Model", "Fold", "Split", "Signal_Date", "Ticker"]
    actual, saved = [part.sort_values(keys).reset_index(drop=True) for part in (actual, saved)]
    metadata = [name for name in PREDICTION_COLUMNS if name not in ("Predicted", "Actual_5d_Pct")]
    if not actual[metadata].equals(saved[metadata]):
        raise ValueError("Replay prediction intervals or label states differ")
    if not np.allclose(actual.Actual_5d_Pct, saved.Actual_5d_Pct, atol=1e-10, rtol=0, equal_nan=True):
        raise ValueError("Replay actual labels differ")
    delta = float(np.max(np.abs(actual.Predicted - saved.Predicted)))
    if not np.isfinite(delta) or delta > atol:
        raise ValueError(f"Replay prediction delta exceeds tolerance: {delta}")
    return {"reproduced": True, "model_fitting_executed": False, "tensorflow_used": True,
            "rows": len(actual), "maximum_prediction_delta": delta, "tolerance": atol,
            "historical_alpha_proven": False}


def check_report(data):
    mature = np.isfinite(data["target"].iloc[data["test_positions"]].to_numpy(float)).all(axis=1)
    return {"mode": "check", "ready": True, "read_only": True, "tensorflow_imported": False,
            "study_status": "exploratory", "as_of": str(data["as_of"].date()),
            "test_signal_days": len(mature), "matured_test_signal_days": int(mature.sum()),
            "pending_test_signal_days": int((~mature).sum()), "fresh_model_fits_required": 45,
            "first_retained_input": str(data["first_retained_input"].date()),
            "folds": _split_document(data), "sources": data["source_metadata"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "train", "verify", "replay"), default="check")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--sentiment", type=Path)
    parser.add_argument("--paired-protocol", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.mode in ("train", "verify", "replay") and args.output_dir is None:
        parser.error(f"{args.mode} requires an explicit --output-dir")
    if args.mode == "verify":
        verify_run(args.output_dir)
        report = {"verified": True, "tensorflow_imported": False, "read_only": True}
    elif args.mode == "replay":
        report = replay_predictions(args.output_dir, ROOT, args.prices, args.sentiment, args.paired_protocol)
    else:
        config = load_config(args.config)
        if args.mode == "train":
            validate_output_dir(ROOT, args.output_dir)
        data = prepare_data(ROOT, config, args.prices, args.sentiment, args.paired_protocol)
        report = check_report(data) if args.mode == "check" else train(data, config, args.output_dir, args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
