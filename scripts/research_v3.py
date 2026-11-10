"""Isolated 0050-relative sentiment research. Default mode checks without writes.

All existing history is exploratory. Validation can reject every news candidate;
the deterministic fallback is 0050, never a retrospective winning configuration.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import ROOT

CONFIG_PATH = ROOT / "research/v3/config_0050.json"
SOURCES = ("scripts/research_v3.py", "research/v3/data.py", "research/v3/models.py",
           "research/v3/portfolio.py", "research/v3/config_0050.json", "scripts/common.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _default(value):
    if isinstance(value, (np.ndarray, pd.Index)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=_default, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["benchmark"]["ticker"] != "0050.TW":
        raise ValueError("V3.2 registers 0050 as benchmark")
    if config.get("allow_test_selection") or config.get("allow_production_replacement"):
        raise ValueError("Holdout selection and production replacement are prohibited")
    if config["candidate_order"] != ["ridge_price", "ridge_news", "lstm_price", "lstm_news"]:
        raise ValueError("Candidate set differs from the registered finite comparison")
    if config["label_gap"] != 6 or config["hold_sessions"] != 5:
        raise ValueError("The target must mature at signal i+6 after entry i+1")
    supported = {"seed": 42, "time_step": 10, "units": 32, "dropout": 0.2,
                 "learning_rate": 0.001, "max_epochs": 20, "patience": 3,
                 "batch_size": 32, "ridge_alpha": 10.0}
    if any(config.get(name) != value for name, value in supported.items()):
        raise ValueError("Hyperparameters differ from the supported registered implementation")
    if config["paired_controls"] != {"ridge_news": "ridge_price", "lstm_news": "lstm_price"}:
        raise ValueError("Each news candidate must use its own architecture control")
    return config


def relative_metrics(nav, benchmark_nav, initial_capital=1_000_000):
    """Use identical complete dates; costs remain in each arm's own NAV."""
    a, b = nav.copy(), benchmark_nav.copy()
    for frame in (a, b):
        if frame.Date.duplicated().any():
            raise ValueError("Duplicate NAV dates")
    if list(a.Date.astype(str)) != list(b.Date.astype(str)):
        raise ValueError("Candidate and benchmark NAV dates differ")
    x, y = a.NAV.to_numpy(float), b.NAV.to_numpy(float)
    if len(x) < 2 or not np.isfinite([x, y]).all() or min(x.min(), y.min()) <= 0:
        raise ValueError("NAV must be finite and positive")
    start_x, start_y = np.r_[initial_capital, x], np.r_[initial_capital, y]
    active = np.diff(start_x) / start_x[:-1] - np.diff(start_y) / start_y[:-1]
    te = float(np.std(active, ddof=1) * np.sqrt(252)) if len(active) > 1 else 0.0
    ir = float(np.mean(active) * 252 / te) if te > 1e-12 else 0.0
    # Starting capital is common; entry costs must not be normalized away.
    return {"Relative_Wealth_Excess_Pct": float((x[-1] / y[-1] - 1) * 100),
            "Tracking_Error_Annualized": te, "Information_Ratio": ir}


def select_news_candidate(validation_records, config):
    """No holdout fields are accepted; compare each news arm with its own control."""
    records = pd.DataFrame(validation_records)
    required = {"Candidate", "Fold", "Relative_Wealth_Excess_Pct", "Tracking_Error_Annualized",
                "Stress_Relative_Wealth_Excess_Pct"}
    if not required.issubset(records.columns):
        raise ValueError("Incomplete validation records")
    if "Split" in records and not records.Split.eq("validation").all():
        raise ValueError("Only validation records may select candidates")
    if records.duplicated(["Candidate", "Fold"]).any():
        raise ValueError("Duplicate candidate/fold results")
    numeric = list(required - {"Candidate", "Fold"})
    if not np.isfinite(records[numeric].to_numpy(float)).all():
        raise ValueError("Non-finite validation results")
    expected = set(range(1, config["validation_folds"] + 1))
    if set(records.Candidate) != set(config["candidate_order"]):
        raise ValueError("Validation must retain every registered candidate")
    if any(set(part.Fold) != expected for _, part in records.groupby("Candidate")):
        raise ValueError("All candidates require the same registered folds")
    if (records.Relative_Wealth_Excess_Pct <= -100).any():
        raise ValueError("Invalid relative wealth")
    selection, assessments = config["selection"], []
    for candidate in config["news_candidate_order"]:
        part = records[records.Candidate.eq(candidate)].sort_values("Fold")
        control_name = config["paired_controls"][candidate]
        control = records[records.Candidate.eq(control_name)].sort_values("Fold")
        compound = float(np.prod(1 + part.Relative_Wealth_Excess_Pct / 100) - 1)
        control_compound = float(np.prod(1 + control.Relative_Wealth_Excess_Pct / 100) - 1)
        stress = float(np.prod(1 + part.Stress_Relative_Wealth_Excess_Pct / 100) - 1)
        reasons = []
        if int(part.Relative_Wealth_Excess_Pct.gt(0).sum()) < selection["minimum_positive_net_excess_folds"]:
            reasons.append("insufficient_positive_folds")
        if compound <= 0:
            reasons.append("nonpositive_validation_excess")
        if compound <= control_compound:
            reasons.append("news_does_not_beat_same_family_control")
        if part.Tracking_Error_Annualized.max() > selection["maximum_annualized_tracking_error"]:
            reasons.append("tracking_error_limit")
        if stress <= 0:
            reasons.append("nonpositive_slippage_stress_excess")
        assessments.append({"candidate": candidate, "paired_control": control_name,
                            "compound_validation_excess": compound,
                            "paired_control_excess": control_compound,
                            "compound_stress_excess": stress, "rejection_reasons": reasons})
    eligible = [row for row in assessments if not row["rejection_reasons"]]
    chosen = max(eligible, key=lambda row: row["compound_validation_excess"]) if eligible else None
    return {"selected": chosen["candidate"] if chosen else "B0_benchmark",
            "assessments": assessments, "holdout_used_for_selection": False,
            "prospective_deployment_allowed": False, "historical_alpha_proven": False}


def check(data_path=None, metadata_path=None, sentiment_path=None, config_path=CONFIG_PATH):
    config = load_config(config_path)
    result = {"protocol_version": config["protocol_version"], "benchmark": "0050.TW",
              "mode": "read_only_check", "trained": False, "performance_claim": "none",
              "production_replaced": False, "inputs_ready": False}
    if data_path is not None:
        from research.v3.data import load_dataset, validate_data
        bars = load_dataset(data_path, metadata_path=metadata_path,
                            expected_tickers=["0050.TW"] + config["satellite_tickers"])
        result["data_quality"] = validate_data(bars, metadata=bars.attrs.get("metadata"))
        if sentiment_path is not None:
            from research.v3.models import prepare_features
            features, target, meta = prepare_features(bars, pd.read_csv(sentiment_path),
                benchmark="0050.TW", tickers=config["satellite_tickers"], time_step=config["time_step"])
            result.update(inputs_ready=True, feature_rows=len(features), target_rows=len(target),
                          preparation=meta)
    return result


def _save_artifact(path, artifact):
    raw = json.dumps(artifact, default=_default, allow_nan=False, separators=(",", ":")).encode()
    Path(path).write_bytes(gzip.compress(raw, mtime=0))


def _book(bars, predictions, dates, config, mode="model", stress=False):
    from research.v3.portfolio import run_backtest
    options = {**config, "allocation_mode": mode}
    if stress:
        options = {**options, "execution": {**options["execution"], "slippage_bps_each_side":
                                             options["execution"]["stress_slippage_bps_each_side"]}}
    all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(bars.Date).unique()))
    # End at the last scheduled five-session hold, rather than extending by
    # the four unused daily forecasts after the final scheduled signal.
    positions = all_dates.get_indexer(pd.DatetimeIndex(dates))
    if (positions < 0).any() or len(positions) == 0:
        raise ValueError("Signals must belong to the full session calendar")
    scheduled = positions[(positions - positions[0]) % config["execution"]["rebalance_every_sessions"] == 0]
    last = scheduled[-1]
    if last < 0 or last + 6 >= len(all_dates):
        raise ValueError("The final signal has no mature five-session label")
    return run_backtest(bars, predictions, dates, options, end_date=all_dates[last + 6])


def _save_book(folder, book):
    folder.mkdir(parents=True, exist_ok=False)
    for name in ("nav", "fills", "weights"):
        book[name].to_csv(folder / (name + ".csv"), index=False)
    write_json(folder / "metrics.json", book["metrics"])


def train(data_path, metadata_path, sentiment_path, output_dir, config_path=CONFIG_PATH):
    output = Path(output_dir).resolve()
    protected = ("data", "models", "outputs", "experiments/results", "research/v211")
    if any(output.is_relative_to(ROOT / name) for name in protected):
        raise ValueError("V3 cannot overwrite production inputs, weights, outputs or ledgers")
    if output.exists():
        raise FileExistsError("Research runs are immutable; choose a new output directory")
    config = load_config(config_path)
    initial_sources = {name: digest(ROOT / name) for name in SOURCES}
    initial_inputs = {"bars.csv": digest(data_path), "bars.metadata.json": digest(metadata_path),
                      "sentiment.csv": digest(sentiment_path), "registered_config.json": digest(config_path)}
    from research.v3.data import load_dataset, validate_data
    from research.v3.models import prepare_features, make_splits, fit_predict
    bars = load_dataset(data_path, metadata_path=metadata_path,
                        expected_tickers=["0050.TW"] + config["satellite_tickers"])
    features, target, preparation = prepare_features(bars, pd.read_csv(sentiment_path),
        benchmark="0050.TW", tickers=config["satellite_tickers"], time_step=config["time_step"])
    dates = pd.DatetimeIndex(preparation["eligible_dates"]).sort_values()
    split = make_splits(dates, session_dates=pd.DatetimeIndex(sorted(pd.to_datetime(bars.Date).unique())),
        train_samples=config["train_samples"], validation_samples=config["validation_samples"],
        test_samples=config["exploratory_holdout_samples"], n_folds=config["validation_folds"],
        label_gap=config["label_gap"], min_train_samples=config["min_train_samples"])
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    for source, name in ((data_path, "bars.csv"), (metadata_path, "bars.metadata.json"),
                         (sentiment_path, "sentiment.csv")):
        if source is not None:
            shutil.copyfile(source, output / "inputs" / name)
    shutil.copyfile(config_path, output / "registered_config.json")
    for name, wanted in initial_inputs.items():
        retained_path = output / name if name == "registered_config.json" else output / "inputs" / name
        if digest(retained_path) != wanted:
            raise ValueError(f"Source input changed while preparing V3: {name}")
    write_json(output / "run_state.json", {"complete": False, "source_sha256_at_start": initial_sources,
                                           "input_sha256_at_start": initial_inputs})
    write_json(output / "splits.json", split)
    write_json(output / "data_quality.json", {"bars": validate_data(bars, metadata=bars.attrs.get("metadata")),
                                             "preparation": preparation})
    records, all_predictions = [], []
    for fold_id, fold in enumerate(split["folds"], 1):
        val_dates = fold["validation"]
        baseline = _book(bars, pd.DataFrame(), val_dates, config, mode="benchmark")
        stress_baseline = _book(bars, pd.DataFrame(), val_dates, config, mode="benchmark", stress=True)
        for candidate in config["candidate_order"]:
            print(json.dumps({"stage": "validation", "fold": fold_id, "candidate": candidate}), flush=True)
            predictions, artifact = fit_predict(features, target, fold, candidate=candidate,
                                                seed=config["seed"], time_step=config["time_step"],
                                                max_epochs=config["max_epochs"])
            part = predictions[predictions.Split.eq("validation")].copy()
            book = _book(bars, part, val_dates, config)
            stress_book = _book(bars, part, val_dates, config, stress=True)
            paired = relative_metrics(book["nav"], baseline["nav"])
            stressed = relative_metrics(stress_book["nav"], stress_baseline["nav"])
            records.append({"Candidate": candidate, "Fold": fold_id, "Split": "validation",
                            **paired, "Stress_Relative_Wealth_Excess_Pct": stressed["Relative_Wealth_Excess_Pct"],
                            "ROI_Pct": book["metrics"]["ROI_Pct"]})
            part["Candidate"], part["Fold"] = candidate, fold_id
            all_predictions.append(part)
            _save_artifact(output / f"fold{fold_id}_{candidate}.json.gz", artifact)
            _save_book(output / f"fold{fold_id}_{candidate}", book)
    pd.DataFrame(records).to_csv(output / "validation_metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(output / "validation_predictions.csv", index=False)
    selection = select_news_candidate(records, config)
    # Write selection before any exploratory holdout model is fitted or evaluated.
    write_json(output / "selection.json", selection)
    final = split["final"]
    family = selection["selected"] if selection["selected"] != "B0_benchmark" else "ridge_news"
    pair = [config["paired_controls"][family], family]
    test_dates = split["test"]
    baseline = _book(bars, pd.DataFrame(), test_dates, config, mode="benchmark")
    _save_book(output / "holdout_B0_benchmark", baseline)
    equal = _book(bars, pd.DataFrame(), test_dates, config, mode="equal")
    _save_book(output / "holdout_B1_equal", equal)
    holdout = [{"Arm": "B0_benchmark", **baseline["metrics"],
                **relative_metrics(baseline["nav"], baseline["nav"])},
               {"Arm": "B1_equal", **equal["metrics"], **relative_metrics(equal["nav"], baseline["nav"])}]
    for candidate in pair:
        print(json.dumps({"stage": "exploratory_holdout", "candidate": candidate}), flush=True)
        predictions, artifact = fit_predict(features, target, final, candidate=candidate,
                                            seed=config["seed"], time_step=config["time_step"],
                                            max_epochs=config["max_epochs"])
        part = predictions[predictions.Split.eq("test")].copy()
        part.to_csv(output / f"holdout_{candidate}_predictions.csv", index=False)
        _save_artifact(output / f"final_{candidate}.json.gz", artifact)
        book = _book(bars, part, test_dates, config)
        _save_book(output / f"holdout_{candidate}", book)
        holdout.append({"Arm": candidate, **book["metrics"],
                        **relative_metrics(book["nav"], baseline["nav"])})
    pd.DataFrame(holdout).to_csv(output / "holdout_metrics.csv", index=False)
    write_json(output / "limitations.json", {
        "historical_alpha_proven": False, "all_history_exploratory": True,
        "fallback_diagnostic_pair": selection["selected"] == "B0_benchmark",
        "news_point_in_time_evidence_complete": False, "prospective_deployment_allowed": False,
        "no_performance_guarantee": True,
        "notes": ["Historical last60 signals are exploratory holdout, not independent prospective evidence.",
                  "Payment dates, lot rights, intraday fills and quoted spreads require execution review.",
                  "A fallback diagnostic pair is reported even if all news candidates fail; it is never adopted."]})
    if {name: digest(ROOT / name) for name in SOURCES} != initial_sources:
        raise ValueError("V3 source changed during training; partial run cannot certify completion")
    write_json(output / "run_state.json", {"complete": True, "source_sha256_at_start": initial_sources,
                                           "input_sha256_at_start": initial_inputs})
    manifest = {"protocol_version": config["protocol_version"], "source_sha256":
                initial_sources, "python": platform.python_version(),
                "runtime_versions": {name: importlib.metadata.version(name) for name in
                                     ("numpy", "pandas", "scikit-learn", "tensorflow-cpu")},
                "files_sha256": {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*"))
                                 if path.is_file()}}
    write_json(output / "reproduction_manifest.json", manifest)
    return {"output": str(output), "selection": selection, "holdout": holdout,
            "performance_claim": "none", "production_replaced": False}


def verify(output_dir):
    output = Path(output_dir).resolve()
    manifest = json.loads((output / "reproduction_manifest.json").read_text())
    for name, wanted in manifest["files_sha256"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or digest(path) != wanted:
            raise ValueError(f"V3 retained artifact hash mismatch: {name}")
    for name, wanted in manifest["source_sha256"].items():
        if digest(ROOT / name) != wanted:
            raise ValueError(f"V3 runtime source differs: {name}")
    return {"verified": True, "historical_alpha_proven": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--sentiment", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify)
    elif args.train:
        if not all((args.data, args.metadata, args.sentiment, args.output_dir)):
            parser.error("--train requires --data, --metadata, --sentiment and --output-dir")
        result = train(args.data, args.metadata, args.sentiment, args.output_dir, args.config)
    else:
        result = check(args.data, args.metadata, args.sentiment, args.config)
    print(json.dumps(result, ensure_ascii=False, default=_default, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
