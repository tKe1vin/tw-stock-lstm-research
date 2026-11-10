"""Precommitted V2.8 rules. No TensorFlow and no test-set selection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.horizon_experiment import LABEL_GAP
from experiments.v2.guard import OUT

V28 = OUT / "v28"
MIN_WINS = 9
WINDOWS = (252, 504, 756)
MOMENTUM_LOOKBACK = 20
POS_RATE_MIN = 0.20
POS_RATE_MAX = 0.80
ES_LEN = 60
MIN_UPDATE_TRAIN = 120
RETRAIN_STEPS = (5, 1)
ARCHITECTURES = {
    "lstm1_50": {"layers": 1, "units": 50},
    "lstm1_32": {"layers": 1, "units": 32},
}
ADOPTABLE = ("rolling_selected", "lstm1_50", "lstm1_32", "retrain_5", "retrain_1")
# Equal skill keeps the earlier, smaller change. This is not an ROI ranking.
TIE_ORDER = ("mse_b",) + ADOPTABLE
CONTROL = "mse_b"


def assert_v28(path):
    resolved = Path(path).resolve()
    root = V28.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"V2.8 只能寫入 {root}：{resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def rolling_train(train_positions, window):
    """Keep the most recent `window` training signals. Shorter history is infeasible."""
    train = np.asarray(train_positions, dtype=int)
    if window not in WINDOWS:
        raise ValueError("訓練視窗不在事先清單")
    if len(train) < window:
        return None
    kept = train[-window:]
    if kept.max() > train.max() or kept.min() < train.min():
        raise AssertionError("滾動視窗超出該折訓練訊號")
    return kept


def select_windows(predictions):
    """Pick one window per fold from validation Spearman only. Ties keep the longer window."""
    rows = []
    choices = {}
    validation = predictions[predictions.Split == "validation"]
    for fold_id in range(1, 17):
        best_window = None
        best_score = None
        for window in WINDOWS:
            part = validation[(validation.Model == f"rolling_{window}") & (validation.Fold == fold_id)]
            feasible = len(part) > 0 and np.isfinite(part.Predicted.to_numpy(dtype=float)).all()
            score = np.nan
            if feasible:
                score = _cross_section(part)
                if best_window is None or score > best_score or (
                    np.isclose(score, best_score) and window > best_window
                ):
                    best_window = window
                    best_score = score
            rows.append({
                "Fold": fold_id,
                "Window": window,
                "Feasible": bool(feasible),
                "Validation_Spearman": score,
                "Selected": False,
            })
        if best_window is None:
            raise AssertionError(f"第 {fold_id} 折沒有可用的訓練視窗")
        choices[fold_id] = best_window
    table = pd.DataFrame(rows)
    for fold_id, window in choices.items():
        table.loc[(table.Fold == fold_id) & (table.Window == window), "Selected"] = True
    return choices, table


def stitch_windows(predictions, choices):
    rows = []
    for fold_id, window in choices.items():
        part = predictions[(predictions.Model == f"rolling_{window}") & (predictions.Fold == fold_id)].copy()
        if part.empty:
            raise AssertionError(f"第 {fold_id} 折缺少視窗 {window}")
        part["Model"] = "rolling_selected"
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def matured_positions(positions, asof):
    """Labels whose five-day hold has already ended at `asof`."""
    positions = np.asarray(positions, dtype=int)
    return positions[(positions + LABEL_GAP <= int(asof)) & (positions < int(asof))]


def split_update(eligible):
    """Train on older matured signals. The latest block is early stopping only."""
    eligible = np.asarray(sorted(np.unique(eligible)), dtype=int)
    if len(eligible) < MIN_UPDATE_TRAIN + ES_LEN:
        return None
    early_stop = eligible[-ES_LEN:]
    raw = eligible[:-ES_LEN]
    boundary = int(early_stop[0])
    train = raw[raw + LABEL_GAP < boundary]
    if len(train) < MIN_UPDATE_TRAIN:
        return None
    if int(train.max()) + LABEL_GAP >= boundary:
        raise AssertionError("更新訓練的標籤碰到 early stopping")
    return train, early_stop


def update_schedule(block, every):
    """Retrain at the first signal and then every `every` signals. The anchor does not move."""
    if every not in RETRAIN_STEPS:
        raise ValueError("重訓頻率不在事先清單")
    block = [int(pos) for pos in block]
    points = []
    for offset in range(0, len(block), every):
        points.append((block[offset], block[offset:offset + every]))
    return points


def classification_scores(predicted, actual):
    """Ternary direction accuracy; neutral predictions never count as down.

    Direction uses exact sign equality for every observation, including an
    actual zero. Balanced accuracy averages recalls on actual positive and
    negative observations only; a zero prediction is incorrect for either.
    Return errors use the unchanged input units (five-day percentage points).
    """
    pred = np.asarray(predicted, dtype=float)
    act = np.asarray(actual, dtype=float)
    if pred.ndim != 1 or act.ndim != 1 or pred.shape != act.shape:
        raise ValueError("Predictions and actual returns must be matching one-dimensional arrays")
    if not np.isfinite(pred).all() or not np.isfinite(act).all():
        raise ValueError("Predictions and actual returns must be finite")
    pred_up = pred > 0
    pred_down = pred < 0
    up = act > 0
    down = act < 0
    direction = float((np.sign(pred) == np.sign(act)).mean()) if len(pred) else np.nan
    tpr = float(pred_up[up].mean()) if up.any() else np.nan
    tnr = float(pred_down[down].mean()) if down.any() else np.nan
    balanced = float(0.5 * (tpr + tnr)) if np.isfinite(tpr) and np.isfinite(tnr) else np.nan
    return {
        "Direction_Accuracy": direction,
        "Balanced_Accuracy": balanced,
        "True_Positive_Rate": tpr,
        "True_Negative_Rate": tnr,
        "Positive_Rate": float(pred_up.mean()) if len(pred) else np.nan,
        "Up_Rate": float(up.mean()) if len(act) else np.nan,
        "Prediction_Zero_Rate": float((pred == 0).mean()) if len(pred) else np.nan,
        "Actual_Zero_Rate": float((act == 0).mean()) if len(act) else np.nan,
        "MAE": float(np.mean(np.abs(pred - act))) if len(pred) else np.nan,
        "RMSE": float(np.sqrt(np.mean((pred - act) ** 2))) if len(pred) else np.nan,
    }


def _cross_section(frame):
    from experiments.horizon_experiment import TICKERS, rank_metrics

    predicted = []
    actual = []
    for _, part in frame.groupby("Signal_Date", sort=True):
        part = part.set_index("Ticker").loc[list(TICKERS)]
        predicted.append(part.Predicted.to_numpy(dtype=float))
        actual.append(part.Actual_5d_Pct.to_numpy(dtype=float))
    if not predicted:
        return np.nan
    return rank_metrics(np.vstack(predicted), np.vstack(actual))["Cross_Section_Spearman"]


def fold_direction(frame):
    scores = {}
    for fold_id, part in frame.groupby("Fold"):
        scores[int(fold_id)] = classification_scores(part.Predicted, part.Actual_5d_Pct)["Direction_Accuracy"]
    return [scores[fold_id] for fold_id in range(1, 17)]


def fold_spearman_list(frame):
    from experiments.v2.signals import fold_spearman

    scores = fold_spearman(frame)
    missing = [fold_id for fold_id in range(1, 17) if fold_id not in scores]
    if missing:
        raise AssertionError(f"缺少折數 {missing}")
    return [scores[fold_id] for fold_id in range(1, 17)]


def passes_rank(control, candidate):
    control = np.asarray(control, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if len(control) != 16 or len(candidate) != 16 or not np.isfinite(candidate).all():
        raise ValueError("排序分數必須是 16 折的有限值")
    wins = int((candidate > control).sum())
    mean = float(candidate.mean())
    return bool(mean > float(control.mean()) and wins >= MIN_WINS), wins, mean


def passes_direction(stats, control_direction, candidate_direction, always_long_direction, balanced, positive_rate):
    """Direction adoption also has to beat always-long and stay away from a one-sided forecast."""
    control = np.asarray(stats["control"], dtype=float)
    candidate = np.asarray(stats["candidate"], dtype=float)
    if len(control) != 16 or len(candidate) != 16 or not np.isfinite(candidate).all():
        raise ValueError("方向分數必須是 16 折的有限值")
    wins = int((candidate > control).sum())
    band = np.isfinite(positive_rate) and POS_RATE_MIN <= float(positive_rate) <= POS_RATE_MAX
    better = (
        np.isfinite(candidate_direction)
        and np.isfinite(control_direction)
        and np.isfinite(always_long_direction)
        and np.isfinite(balanced)
        and float(candidate_direction) > float(control_direction)
        and float(candidate_direction) > float(always_long_direction)
        and float(balanced) > 0.5
        and wins >= MIN_WINS
        and band
    )
    return bool(better), wins


def choose(validation):
    """Validation dictionaries only. Test ROI cannot be passed in."""
    if CONTROL not in validation:
        raise ValueError("缺少 MSE 控制組")
    control = validation[CONTROL]
    accepted = []
    decisions = []
    for name in ADOPTABLE:
        if name not in validation:
            decisions.append({"Model": name, "Status": "not_trained", "Accepted": False})
            continue
        item = validation[name]
        rank_ok, rank_wins, rank_mean = passes_rank(control["spearman"], item["spearman"])
        direction_ok, direction_wins = passes_direction(
            {"control": control["direction"], "candidate": item["direction"]},
            control["direction_mean"],
            item["direction_mean"],
            item["always_long_direction"],
            item["balanced_accuracy"],
            item["positive_rate"],
        )
        passed = bool(rank_ok or direction_ok)
        decisions.append({
            "Model": name,
            "Validation_Mean_Spearman": rank_mean,
            "Control_Validation_Mean_Spearman": float(np.mean(control["spearman"])),
            "Rank_Folds_Beating_Control": rank_wins,
            "Rank_Accepted": rank_ok,
            "Direction_Accuracy": item["direction_mean"],
            "Control_Direction_Accuracy": control["direction_mean"],
            "Always_Long_Direction": item["always_long_direction"],
            "Balanced_Accuracy": item["balanced_accuracy"],
            "Positive_Rate": item["positive_rate"],
            "Direction_Folds_Beating_Control": direction_wins,
            "Direction_Accepted": direction_ok,
            "Accepted": passed,
        })
        if passed:
            accepted.append(name)
    if not accepted:
        return CONTROL, decisions
    accepted.sort(key=lambda name: (
        -validation[name]["balanced_accuracy"],
        -float(np.mean(validation[name]["spearman"])),
        TIE_ORDER.index(name),
    ))
    return accepted[0], decisions


class ModelRegistry:
    """Keep the last version that passed validation. A failed version does not replace it."""

    def __init__(self):
        self.accepted = None
        self.history = []

    def propose(self, version, validation_passed):
        active = version if validation_passed else self.accepted
        if validation_passed:
            self.accepted = version
        self.history.append({"version": version, "passed": bool(validation_passed), "active": self.accepted})
        return self.accepted


def research_prediction_dir():
    """Immutable daily files live under V2.8 only. This study does not write historical stand-ins."""
    return assert_v28(V28 / "daily_protocol")


def write_prediction_day(directory, day, frame):
    """First file for a day wins. A later run must not overwrite it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{day}.csv"
    if path.resolve().parent != directory.resolve():
        raise RuntimeError("當日預測不能寫到指定目錄之外")
    if path.exists():
        raise FileExistsError(f"當日預測已存在，拒絕覆蓋：{path.name}")
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return path
