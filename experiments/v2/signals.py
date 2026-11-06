"""Prediction diagnostics and validation-only candidate selection.

Portfolio ROI is not an input. A candidate replaces the phase-5 MSE control
only when its validation cross-sectional Spearman is higher on average and on
at least nine of the sixteen folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.horizon_experiment import TICKERS, rank_metrics
from experiments.risk_experiment import FEATURES

CONTROL = "mse_b"
CANDIDATES = ("mse_b_calibrated", "target_z", "huber", "no_sentiment")
# Equal validation skill keeps the smaller change. This order is not an ROI ranking.
TIE_ORDER = (CONTROL, "mse_b_calibrated", "target_z", "huber", "no_sentiment")
MIN_FOLD_WINS = 9
SLOPE_CAP = 5.0


def _matrices(frame):
    predicted = []
    actual = []
    folds = []
    for (fold_id, _), part in frame.groupby(["Fold", "Signal_Date"], sort=True):
        part = part.set_index("Ticker").loc[TICKERS]
        predicted.append(part.Predicted.to_numpy(dtype=float))
        actual.append(part.Actual_5d_Pct.to_numpy(dtype=float))
        folds.append(int(fold_id))
    return np.vstack(predicted), np.vstack(actual), np.asarray(folds)


def fold_spearman(frame):
    predicted, actual, folds = _matrices(frame)
    scores = {}
    for fold_id in np.unique(folds):
        mask = folds == fold_id
        scores[int(fold_id)] = rank_metrics(predicted[mask], actual[mask])["Cross_Section_Spearman"]
    return scores


def quality_row(name, frame):
    predicted, actual, folds = _matrices(frame)
    rank = rank_metrics(predicted, actual)
    error = predicted.ravel() - actual.ravel()
    pred = predicted.ravel()
    act = actual.ravel()
    direction = pred[np.abs(act) > 1e-8]
    actual_sign = act[np.abs(act) > 1e-8]
    pearson = float(pd.Series(pred).corr(pd.Series(act))) if np.std(pred) > 1e-12 else 0.0
    spearman = float(pd.Series(pred).corr(pd.Series(act), method="spearman")) if np.std(pred) > 1e-12 else 0.0
    if pearson != pearson:
        pearson = 0.0
    if spearman != spearman:
        spearman = 0.0
    return {
        "Model": name,
        "Direction_Accuracy": float(np.mean(np.sign(direction) == np.sign(actual_sign))),
        "Pearson": pearson,
        "Spearman": spearman,
        "Cross_Section_Spearman": rank["Cross_Section_Spearman"],
        "Top1_Hit_Rate": rank["Top1_Hit_Rate"],
        "Top2_Overlap": rank["Top2_Overlap"],
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "Pred_Mean": float(pred.mean()),
        "Pred_Std": float(pred.std(ddof=1)),
        "Actual_Mean": float(act.mean()),
        "Actual_Std": float(act.std(ddof=1)),
        "Shrinkage": float(pred.std(ddof=1) / act.std(ddof=1)),
        "Folds": int(len(np.unique(folds))),
    }


def ticker_bias(frame):
    rows = []
    for ticker, part in frame.groupby("Ticker"):
        rows.append({
            "Ticker": ticker,
            "Pred_Mean": float(part.Predicted.mean()),
            "Actual_Mean": float(part.Actual_5d_Pct.mean()),
            "Bias": float(part.Predicted.mean() - part.Actual_5d_Pct.mean()),
            "Pred_Std": float(part.Predicted.std(ddof=1)),
            "Actual_Std": float(part.Actual_5d_Pct.std(ddof=1)),
        })
    return pd.DataFrame(rows)


def _line(pred, actual):
    pearson = float(pd.Series(pred).corr(pd.Series(actual))) if np.std(pred) > 1e-8 else 0.0
    if pearson != pearson or pearson <= 0 or len(pred) < 5:
        return 0.0, 1.0, 0.0 if pearson != pearson else pearson, "identity_nonpositive_fit_window"
    slope, intercept = np.polyfit(pred, actual, 1)
    return float(intercept), float(np.clip(slope, 0.0, SLOPE_CAP)), float(pearson), "fit_window_line"


def calibrate(predictions, model="B"):
    """Fit each line on the earlier validation dates and score the later dates.

    Test labels are not used. The later validation block is the only block that
    may decide whether calibration replaces the control.
    """
    source = predictions[predictions.Model == model].copy()
    rows = []
    audits = []
    for (fold_id, ticker), part in source.groupby(["Fold", "Ticker"]):
        validation = part[part.Split == "validation"].sort_values("Signal_Date")
        test = part[part.Split == "test"].sort_values("Signal_Date")
        dates = validation.Signal_Date.drop_duplicates().sort_values()
        cut = max(len(dates) // 2, 1)
        fit_dates = set(pd.to_datetime(dates.iloc[:cut]))
        hold_dates = set(pd.to_datetime(dates.iloc[cut:]))
        fit = validation[pd.to_datetime(validation.Signal_Date).isin(fit_dates)]
        hold = validation[pd.to_datetime(validation.Signal_Date).isin(hold_dates)]
        intercept, slope, pearson, status = _line(
            fit.Predicted.to_numpy(dtype=float), fit.Actual_5d_Pct.to_numpy(dtype=float),
        )
        audits.append({
            "Fold": int(fold_id),
            "Ticker": ticker,
            "Fit_Pearson": pearson,
            "Intercept": intercept,
            "Slope": slope,
            "Status": status,
            "Fit_Days": int(len(fit_dates)),
            "Holdout_Days": int(len(hold_dates)),
        })
        for block, split in ((hold, "validation"), (test, "test")):
            adjusted = block.copy()
            adjusted["Predicted"] = intercept + slope * adjusted.Predicted.to_numpy(dtype=float)
            adjusted["Model"] = "mse_b_calibrated"
            adjusted["Split"] = split
            rows.append(adjusted)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audits)


def select_candidate(validation_scores):
    """Choose from per-fold validation Spearman lists. ROI cannot be passed in."""
    if CONTROL not in validation_scores:
        raise ValueError("缺少 MSE 控制組的驗證分數")
    control = np.asarray(validation_scores[CONTROL], dtype=float)
    if len(control) != 16:
        raise ValueError("驗證折數必須是既有的 16 折")
    best_name = CONTROL
    best_mean = float(np.mean(control))
    decisions = []
    for name in CANDIDATES:
        if name not in validation_scores:
            decisions.append({"Model": name, "Status": "not_trained"})
            continue
        values = np.asarray(validation_scores[name], dtype=float)
        if len(values) != len(control) or not np.isfinite(values).all():
            raise ValueError(f"{name} 的驗證分數不完整")
        wins = int((values > control).sum())
        mean = float(np.mean(values))
        accepted = mean > best_mean and wins >= MIN_FOLD_WINS and mean > float(np.mean(control))
        decisions.append({
            "Model": name,
            "Validation_Mean_Spearman": mean,
            "Folds_Beating_Control": wins,
            "Accepted_Against_Current_Best": bool(mean > best_mean and wins >= MIN_FOLD_WINS),
        })
        if mean > best_mean and wins >= MIN_FOLD_WINS:
            best_name = name
            best_mean = mean
        elif np.isclose(mean, best_mean) and wins >= MIN_FOLD_WINS and TIE_ORDER.index(name) < TIE_ORDER.index(best_name):
            best_name = name
    return best_name, decisions


def feature_alignment(frames, index, folds):
    """Train versus test correlation of each raw feature with the five-day label.

    This is a descriptive alignment check. It is not a new model and it is not
    used to drop a feature after seeing test ROI.
    """
    from experiments.horizon_experiment import executable_labels

    close = pd.DataFrame({ticker: frames[ticker]["Close"] for ticker in TICKERS}, index=index)
    _, five, _ = executable_labels(close)
    rows = []
    for feature in FEATURES:
        train_values = []
        test_values = []
        train_actual = []
        test_actual = []
        for fold in folds:
            for split, bucket_x, bucket_y in (
                ("train", train_values, train_actual),
                ("test", test_values, test_actual),
            ):
                for pos in fold[split]:
                    pos = int(pos)
                    for ticker in TICKERS:
                        bucket_x.append(float(frames[ticker][feature].iloc[pos]))
                        bucket_y.append(float(five.iloc[pos][ticker]))
        def corr(x, y):
            if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
                return 0.0
            value = float(pd.Series(x).corr(pd.Series(y)))
            return 0.0 if value != value else value
        rows.append({
            "Feature": feature,
            "Train_Pearson": corr(train_values, train_actual),
            "Test_Pearson": corr(test_values, test_actual),
        })
    return pd.DataFrame(rows)
