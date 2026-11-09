"""Adapt the shared-444-core research result to the durable 555 CSV contract.

The historical six-feature 555 notebook is a separate model. This exporter does
not train it, modify 444, publish trades, or append historical recommendations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import TZ

ARM = "dynamic_news_risk555"
PROTOCOL = "paired-444-core-allocation-v1"


def _single_metric(metrics, arm):
    rows = metrics.loc[metrics.Arm == arm]
    if len(rows) != 1:
        raise ValueError(f"Expected one paired metric row for {arm}")
    return rows.iloc[0]


def _direction_accuracy(predictions, model_metrics, tolerance):
    """Model-label accuracy only; never reinterpret it as trade profitability."""
    if "Direction_Correct" in predictions:
        values = predictions.Direction_Correct.astype(str).str.strip().str.lower()
        mapped = values.map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0,
                             "1.0": 1.0, "0.0": 0.0})
        if mapped.isna().any():
            raise ValueError("Invalid saved 444 direction correctness")
        return float(mapped.mean() * 100)
    if {"Actual_Return_Pct", "Predicted_Return_Pct"}.issubset(predictions.columns):
        actual = pd.to_numeric(predictions.Actual_Return_Pct).to_numpy(float)
        predicted = pd.to_numeric(predictions.Predicted_Return_Pct).to_numpy(float)
        if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise ValueError("Invalid saved 444 directional returns")
        actual[np.abs(actual) <= tolerance] = 0
        predicted[np.abs(predicted) <= tolerance] = 0
        return float((np.sign(actual) == np.sign(predicted)).mean() * 100)
    if "Direction_Accuracy_Pct" in model_metrics:
        accuracy = pd.to_numeric(model_metrics.Direction_Accuracy_Pct).to_numpy(float)
        if np.isfinite(accuracy).all() and ((0 <= accuracy) & (accuracy <= 100)).all():
            return float(accuracy.mean())
    raise ValueError("Saved 444 outputs have no verifiable direction accuracy")


def export_paired_native(result, recommendation, baseline_dir, destination, as_of, created_at):
    """Write the current 555 native export from the matched research result.

    ``destination`` is the native directory itself. Return paths and a metric
    summary for the runner; do not write a status or recommendation history.
    """
    baseline_dir, destination = Path(baseline_dir).resolve(), Path(destination).resolve()
    if destination == baseline_dir or destination.is_relative_to(baseline_dir):
        raise ValueError("Paired 555 destination must not modify the 444 source directory")
    created = pd.Timestamp(created_at)
    if pd.isna(created) or created.tzinfo is None:
        raise ValueError("Native export creation time must contain a timezone")
    created = created.tz_convert(TZ)
    day = str(pd.Timestamp(as_of).date())
    run_id = created.strftime("%Y%m%dT%H%M%S%f")
    source_paths = {name: baseline_dir / name for name in (
        "evaluation_protocol.json", "test_predictions.csv", "model_metrics.csv",
    )}
    if not all(path.is_file() for path in source_paths.values()):
        raise ValueError("Paired 555 requires complete saved 444 protocol and model outputs")
    baseline = json.loads(source_paths["evaluation_protocol.json"].read_text(encoding="utf-8"))
    if str(baseline.get("as_of_date", ""))[:10] != day:
        raise ValueError("Saved 444 protocol is not current for this paired export")
    protocol = dict(result["protocol"])
    if protocol.get("protocol_version") != PROTOCOL:
        raise ValueError("Unrecognized paired research protocol")
    selected = _single_metric(result["metrics"], ARM)
    market = _single_metric(result["metrics"], "market_price_reference")
    if str(selected.Last_Valuation_Date)[:10] != day:
        raise ValueError("Paired result valuation date differs from the requested as-of date")
    for name in ("ROI_Pct", "Sharpe", "MDD_Pct", "Total_Transaction_Cost"):
        if not np.isfinite(float(selected[name])) or not np.isfinite(float(market[name])):
            raise ValueError(f"Paired research metric is not finite: {name}")
    tickers = set(protocol["tickers"])
    rec = recommendation.loc[recommendation.Arm == ARM].copy()
    if len(rec) != len(tickers) or rec.Ticker.duplicated().any() or set(rec.Ticker) != tickers:
        raise ValueError("Paired 555 recommendation must contain each selected stock exactly once")
    signals = pd.to_datetime(rec.Signal_Date).dt.normalize()
    targets = pd.to_datetime(rec.Target_Date).dt.normalize()
    if (not signals.eq(pd.Timestamp(day)).all() or targets.nunique() != 1
            or not targets.gt(pd.Timestamp(day)).all()):
        raise ValueError("Paired recommendation is stale or has inconsistent future dates")
    for name in ("Predicted_Return_Pct", "Target_Weight", "Cash_Weight"):
        if not np.isfinite(pd.to_numeric(rec[name])).all():
            raise ValueError(f"Invalid paired recommendation: {name}")
    weights = pd.to_numeric(rec.Target_Weight).to_numpy(float)
    cash = pd.to_numeric(rec.Cash_Weight).to_numpy(float)
    if ((weights < 0).any() or (cash < 0).any() or weights.sum() > 1 + 1e-10
            or (weights > float(protocol["rules"]["max_stock_weight"]) + 1e-10).any()
            or not np.allclose(cash, cash[0], atol=1e-10, rtol=0)
            or not np.isclose(weights.sum() + cash[0], 1.0, atol=1e-10, rtol=0)):
        raise ValueError("Paired stock and cash weights do not conserve capital")
    predictions = pd.read_csv(source_paths["test_predictions.csv"])
    model_metrics = pd.read_csv(source_paths["model_metrics.csv"])
    # 444 also fits/exports ^TWII as a diagnostic model. Keep the original
    # files intact, but use exactly the registered five stocks in this overlay.
    excluded_source_tickers = sorted((set(predictions.Ticker) | set(model_metrics.Ticker)) - tickers)
    predictions = predictions.loc[predictions.Ticker.isin(tickers)].copy()
    model_metrics = model_metrics.loc[model_metrics.Ticker.isin(tickers)].copy()
    if (set(predictions.Ticker) != tickers or set(model_metrics.Ticker) != tickers
            or model_metrics.Ticker.duplicated().any()):
        raise ValueError("Saved 444 model outputs do not match the paired stocks")
    accuracy = _direction_accuracy(predictions, model_metrics, float(baseline.get("direction_zero_tolerance_pct", 1e-8)))
    predictions["Core_Model"] = "444"
    predictions["Prediction_Role"] = "shared_next_day_model_label_not_post_fill_profit"
    model_metrics["Core_Model"] = "444"
    model_metrics["Model_Comparison"] = "identical_core; allocation overlays differ"
    daily = result["nav"].loc[result["nav"].Arm == ARM].copy()
    reference = result["nav"].loc[result["nav"].Arm == "market_price_reference", ["Date", "NAV"]].copy()
    daily["Date"] = pd.to_datetime(daily.Date).dt.normalize()
    reference["Date"] = pd.to_datetime(reference.Date).dt.normalize()
    reference = reference.rename(columns={"NAV": "Market_Index"})
    nav = daily.merge(reference, on="Date", validate="one_to_one", how="left").sort_values("Date")
    if nav.Market_Index.isna().any() or len(nav) != int(selected.Observations):
        raise ValueError("Paired portfolio and market reference calendars differ")
    rules = protocol["rules"]
    first_signal = pd.Timestamp(daily.Signal_Date.min()).normalize()
    initial = {"Date": first_signal, "NAV": float(rules["initial_capital"]),
               "Market_Index": float(rules["initial_capital"]), "Cash": float(rules["initial_capital"]),
               "Equity_Weight": 0.0, "Transaction_Cost": 0.0, "Turnover": 0.0}
    nav = pd.concat([pd.DataFrame([initial]), nav], ignore_index=True)
    nav["Date"] = pd.to_datetime(nav.Date).dt.strftime("%Y-%m-%d")
    nav["Core_Model"] = "444"
    rec["Signal_Date"] = signals.dt.strftime("%Y-%m-%d")
    rec["Target_Date"] = targets.dt.strftime("%Y-%m-%d")
    rec["Created_At"] = created.isoformat()
    rec["Run_ID"] = run_id
    rec["Run_Mode"] = "daily" if str(created.date()) == day else "historical_replay"
    rec["Protocol_Version"] = PROTOCOL
    rec["Core_Model"] = "444"
    rec["Execution_Status"] = "research_target_only_no_order"
    rec["Execution_Price"] = np.nan
    rec["News_Time_Audit_Confirmed"] = bool(protocol.get("news_evidence", {}).get("news_point_in_time_verified", False))
    metric = {
        "Strategy_ID": "555_shared_444_core_dynamic_news_risk", "Strategy": ARM,
        "Run_ID": run_id, "As_Of_Date": day, "Protocol_Version": PROTOCOL,
        "ROI_Pct": float(selected.ROI_Pct), "Sharpe": float(selected.Sharpe),
        "MDD_Pct": float(selected.MDD_Pct), "Direction_Accuracy_Pct": accuracy,
        "Test_Days": int(selected.Observations), "Test_Start": str(selected.First_Execution_Date)[:10],
        "Test_End": str(selected.Last_Valuation_Date)[:10], "Core_Model": "444",
        "Total_Transaction_Cost": float(selected.Total_Transaction_Cost),
        "Direction_Definition": "same saved 444 next-day model-label accuracy; not trade win rate",
        "Result_Status": "exploratory",
    }
    source_sha256 = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_paths.items()}
    output_protocol = {
        **baseline, **protocol, "as_of_date": day, "strategy_id": metric["Strategy_ID"],
        "core_model": "444", "core_source_sha256": source_sha256,
        "excluded_nonportfolio_source_tickers": excluded_source_tickers,
        "test_target_dates": protocol["shared_target_dates"], "test_days": int(selected.Observations),
        "test_target_dates_sha256": hashlib.sha256("\n".join(protocol["shared_target_dates"]).encode()).hexdigest(),
        "execution_assumption": "signal-day known allocation; next-session close fill; first pre-fill return excluded",
        "direction_accuracy_definition": metric["Direction_Definition"],
        "model_training_executed": False, "result_status": "exploratory",
        "original_six_feature_555": "separate legacy model; not used in this shared-core export",
        "market_benchmark": "^TWII theoretical price reference, no costs or dividends; not investable net alpha",
        **{key: rules[key] for key in ("initial_capital", "broker_fee_rate", "sell_tax_rate", "annual_trading_days", "liquidate_at_end")},
    }
    market_metric = {"Strategy_ID": "market_price_reference", "ROI_Pct": float(market.ROI_Pct),
                     "Sharpe": float(market.Sharpe), "MDD_Pct": float(market.MDD_Pct),
                     "Total_Transaction_Cost": float(market.Total_Transaction_Cost),
                     "Return_Basis": market.Return_Basis, "As_Of_Date": day}
    frames = {"strategy_metrics.csv": pd.DataFrame([metric]), "next_day_recommendation.csv": rec,
              "backtest_nav.csv": nav, "model_metrics.csv": model_metrics,
              "test_predictions.csv": predictions, "market_metrics.csv": pd.DataFrame([market_metric])}
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(destination / name, index=False, encoding="utf-8-sig", float_format="%.17g")
    protocol_path = destination / "evaluation_protocol.json"
    protocol_path.write_text(json.dumps(output_protocol, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return {"protocol_version": PROTOCOL, "core_model": "444", "research_status": "exploratory",
            "run_id": run_id, "as_of_date": day, "training_executed": False,
            "native_output_dir": str(destination), "recommendation": str(destination / "next_day_recommendation.csv"),
            "files": {name: str(destination / name) for name in (*frames, "evaluation_protocol.json")},
            "strategy_metrics": metric, "core_source_sha256": source_sha256}
