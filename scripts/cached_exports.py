"""Read-only validation before reusing a completed daily export.

A successful status record is not evidence that its CSV files survived a
later interrupted write. Invalid or incomplete files are cache misses, so the
orchestrator can retry rather than indefinitely reusing a broken export.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import MARKET, TICKERS
from scripts.unified_csv import COLUMNS, NOT_APPLICABLE


def _dates(frame, column):
    if column not in frame or not frame[column].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError(f"Missing or invalid forecast date: {column}")
    result = pd.to_datetime(frame[column], format="%Y-%m-%d", errors="raise")
    if result.isna().any():
        raise ValueError(f"Missing forecast date: {column}")
    return result


def _finite(frame, columns):
    if not set(columns).issubset(frame):
        raise ValueError("Incomplete numeric export columns")
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Forecast or financial export contains nonfinite values")
    return values


def validate_forecast_frame(rows, day):
    """Validate either native or unified forecasts without filling missing data."""
    stock = "stock_id" if "stock_id" in rows else "Ticker"
    signal = "signal_date" if "signal_date" in rows else "Signal_Date"
    prediction = "predicted_return_pct" if "predicted_return_pct" in rows else "Predicted_Return_Pct"
    if (stock not in rows or len(rows) != len(TICKERS) or rows[stock].duplicated().any()
            or set(rows[stock].astype(str)) != set(TICKERS)
            or not _dates(rows, signal).eq(pd.Timestamp(day)).all()):
        raise ValueError("Incomplete or stale five-stock forecast")
    _finite(rows, [prediction])
    if "Target_Date" in rows and not _dates(rows, "Target_Date").gt(pd.Timestamp(day)).all():
        raise ValueError("Forecast target must be after its completed signal session")
    return rows


def validate_cached_export(root, model, session, status, export_path, native=None):
    """Return False for damaged caches; never write, train or repair evidence."""
    root = Path(root).resolve()
    observed = {}

    def read(path, *, csv=True):
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError("Unsafe or missing cached export")
        path = path.resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if path in observed and observed[path] != digest:
            raise ValueError("Cached files changed during validation")
        observed[path] = digest
        if csv:
            return pd.read_csv(BytesIO(raw), keep_default_na=False)
        value = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("Cached metadata must be an object")
        return value

    def native_files(folder):
        protocol = read(folder / "evaluation_protocol.json", csv=False)
        if protocol.get("as_of_date") != session:
            raise ValueError("Cached protocol has another data session")
        targets = protocol.get("test_target_dates")
        if (not isinstance(targets, list) or not targets or len(set(targets)) != len(targets)
                or targets != sorted(targets) or protocol.get("test_days") != len(targets)):
            raise ValueError("Cached protocol has incomplete evaluation dates")
        dates = _dates(pd.DataFrame({"Date": targets}), "Date")
        if dates.gt(pd.Timestamp(session)).any():
            raise ValueError("Cached evaluation contains a future target")
        recommendation = validate_forecast_frame(read(folder / "next_day_recommendation.csv"), session)
        unified = frame.set_index("stock_id").loc[TICKERS]
        native_forecast = recommendation.set_index("Ticker").loc[TICKERS]
        if not np.allclose(pd.to_numeric(unified.predicted_return_pct), pd.to_numeric(native_forecast.Predicted_Return_Pct), rtol=0, atol=1e-10):
            raise ValueError("Cached unified forecast differs from its native forecast")

        predictions = read(folder / "test_predictions.csv")
        symbols = [*TICKERS, MARKET] if model == "444" and MARKET in protocol.get("price_panel_columns", []) else TICKERS
        if (set(predictions.get("Ticker", [])) != set(symbols)
                or predictions.duplicated(["Date", "Ticker"]).any()
                or len(predictions) != len(targets) * len(symbols)):
            raise ValueError("Cached test predictions lack their complete cross-section")
        actual_dates = _dates(predictions, "Date")
        signal_dates = _dates(predictions, "Signal_Date")
        if not signal_dates.lt(actual_dates).all():
            raise ValueError("Cached test labels do not follow their signal date")
        for _, part in predictions.groupby("Ticker"):
            if sorted(part.Date.tolist()) != targets:
                raise ValueError("Cached test predictions disagree with protocol dates")
        _finite(predictions, ["Previous_Close", "Actual_Close", "Predicted_Close", "Actual_Return_Pct", "Predicted_Return_Pct"])
        if (predictions[["Previous_Close", "Actual_Close"]].apply(pd.to_numeric).to_numpy() <= 0).any():
            raise ValueError("Cached evaluation has invalid observed prices")

        metrics = read(folder / "strategy_metrics.csv")
        if len(metrics) != 1 or metrics.get("As_Of_Date", pd.Series(dtype=str)).tolist() != [session]:
            raise ValueError("Cached strategy metrics lack the completed session")
        _finite(metrics, ["ROI_Pct", "Sharpe", "MDD_Pct", "Direction_Accuracy_Pct", "Test_Days"])
        if float(metrics.Test_Days.iloc[0]) != len(targets):
            raise ValueError("Cached strategy sample size differs from its protocol")
        errors = read(folder / "model_metrics.csv")
        if "Ticker" not in errors or errors.Ticker.duplicated().any() or not set(TICKERS).issubset(errors.Ticker):
            raise ValueError("Cached model metrics lack five distinct stocks")
        _finite(errors, ["Price_MAE", "Price_RMSE", "Return_MAE_pp", "Return_RMSE_pp", "Direction_Accuracy_Pct"])
        nav = read(folder / "backtest_nav.csv")
        nav_dates = _dates(nav, "Date")
        wealth = "Portfolio_Net" if model == "444" else "NAV"
        if (len(nav) != len(targets) + 1 or not nav_dates.is_monotonic_increasing or nav_dates.duplicated().any()
                or nav.Date.iloc[1:].tolist() != targets or nav.Date.iloc[-1] != session):
            raise ValueError("Cached NAV lacks its complete evaluation timeline")
        if (_finite(nav, [wealth, "Market_Index"]) <= 0).any():
            raise ValueError("Cached NAV contains an invalid valuation")
        if model == "555":
            benchmark = read(folder / "market_metrics.csv")
            if len(benchmark) != 1:
                raise ValueError("Cached benchmark metrics are incomplete")
            _finite(benchmark, ["Sharpe", "MDD_Pct"])

    try:
        if model not in {"444", "555", "555_v2"}:
            return False
        frame = read(export_path)
        if frame.columns.tolist() != COLUMNS:
            return False
        # Native exports spell intentional missing values explicitly. A blank
        # field anywhere in the exchange schema can be a torn final CSV record,
        # even when the prediction near its beginning survived the write.
        if frame.astype(str).apply(lambda column: column.str.strip().eq("")).to_numpy().any():
            return False
        validate_forecast_frame(frame, session)
        if not frame.model_id.astype(str).eq(model).all() or not _dates(frame, "data_asof").eq(pd.Timestamp(session)).all():
            return False
        horizon = pd.to_numeric(frame.prediction_horizon_days, errors="coerce")
        if not horizon.eq(5 if model == "555_v2" else 1).all():
            return False
        allocations = {}
        for column in ("target_weight", "cash_weight"):
            applicable = ~frame[column].astype(str).eq(NOT_APPLICABLE)
            if applicable.any() and not applicable.all():
                return False
            if applicable.all():
                values = _finite(frame, [column])[:, 0]
                if (values < 0).any() or (values > 1).any():
                    return False
                allocations[column] = values
            elif model in {"555", "555_v2"}:
                return False
        if "target_weight" in allocations and allocations["target_weight"].sum() > 1 + 1e-8:
            return False
        if "cash_weight" in allocations and not np.allclose(allocations["cash_weight"], allocations["cash_weight"][0], rtol=0, atol=1e-8):
            return False
        if len(allocations) == 2 and not np.isclose(allocations["target_weight"].sum() + allocations["cash_weight"][0], 1, rtol=0, atol=1e-8):
            return False
        if model == "555_v2":
            ranks = _finite(frame, ["rank"])[:, 0]
            if sorted(ranks.tolist()) != list(range(1, len(TICKERS) + 1)):
                return False
        for column in ("intended_execution_date", "forecast_entry_date", "scheduled_rebalance_date"):
            applicable = ~frame[column].astype(str).eq(NOT_APPLICABLE)
            if applicable.any() and not _dates(frame.loc[applicable], column).gt(pd.Timestamp(session)).all():
                return False
        applicable = ~frame.execution_price.astype(str).eq(NOT_APPLICABLE)
        if applicable.any() and (_finite(frame.loc[applicable], ["execution_price"]) <= 0).any():
            return False
        eligible = frame.trade_eligible.astype(str)
        if not eligible.isin([NOT_APPLICABLE, "True", "False", "true", "false", "1", "0"]).all():
            return False
        if model in {"444", "555"}:
            folder = Path(native) if native is not None else Path(status.get("daily_native_dir") or Path(export_path).parent / "native")
            if not folder.is_absolute():
                folder = root / folder
            native_files(folder)
        if model == "555":
            # A repaired/recomputed 444 core invalidates the dependent 555 cache.
            core_status_path = root / "outputs/444" / session / "status.json"
            if not core_status_path.is_file():
                core_status_path = root / "outputs/444" / str(status.get("run_date", session)) / "status.json"
            core = read(core_status_path, csv=False)
            if core.get("status") != "success" or core.get("completed_session") != session:
                return False
            core_file = root / core["daily_native_dir"] / "test_predictions.csv"
            read(core_file)
            primary = status.get("primary_research")
            if not isinstance(primary, dict):
                return False
            expected = primary.get("source_prediction_sha256")
            if expected != observed[core_file.resolve()]:
                return False
        return all(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in observed.items())
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, OverflowError):
        return False
