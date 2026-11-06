"""V2.9 audit of the V2.8 direction denominator and the frozen Adam learning rate.

Saved V2.8 files are read only. Corrected fits write under experiments/results/v2/v29.
Nothing here promotes a model, and test ROI is not an input.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.horizon_experiment import LABEL_GAP, TICKERS
from experiments.v2.guard import OUT
from experiments.v2.v28_rules import V28, classification_scores, rolling_train

V29 = OUT / "v29"
AUDIT_MODELS = ("mse_b", "rolling_selected", "lstm1_50", "lstm1_32", "retrain_5", "retrain_1")
# Locked before any training. Neither candidate is fit in this audit.
PROPOSALS = (
    {
        "id": "five_day_direction",
        "target": "五日報酬大於 0 為 1，小於 0 為 0，剛好為 0 不進損失",
        "loss": "binary_crossentropy",
        "output": "sigmoid",
        "adoption": "驗證集方向準確率與 Balanced Accuracy 同時高於 mse_b 與永遠看多，預測正值比例在 0.20 到 0.80，而且至少 9/16 折較高",
        "trained": False,
    },
    {
        "id": "train_class_balance",
        "target": "沿用五日百分比報酬",
        "loss": "mse",
        "sample_weight": "該折訓練標籤的上漲與下跌筆數倒數，零報酬權重為 0",
        "adoption": "與 V2.8 方向門檻相同，只使用驗證集",
        "trained": False,
    },
)


def assert_v29(path):
    resolved = Path(path).resolve()
    root = V29.resolve()
    saved = V28.resolve()
    if resolved == saved or saved in resolved.parents:
        raise RuntimeError(f"V2.9 不能改寫 V2.8 結果：{resolved}")
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"V2.9 只能寫入 {root}：{resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def reconcile_direction(predicted, actual):
    """Direction drops exact zeros. Up_Rate keeps them in the denominator."""
    pred = np.asarray(predicted, dtype=float)
    act = np.asarray(actual, dtype=float)
    if len(pred) != len(act):
        raise ValueError("預測與標籤長度不同")
    zero = np.isclose(act, 0.0)
    scored = act[~zero]
    scores = classification_scores(pred, act)
    nonzero_up = float((scored > 0).mean()) if len(scored) else np.nan
    return {
        **scores,
        "Rows": int(len(act)),
        "Zero_Returns": int(zero.sum()),
        "Scored_Rows": int((~zero).sum()),
        "Up_Rate_All_Rows": scores["Up_Rate"],
        "Up_Rate_Nonzero": nonzero_up,
        "Always_Long_Matches_Nonzero_Up": bool(
            np.isfinite(nonzero_up) and np.isclose(scores["Direction_Accuracy"], nonzero_up)
        ) if np.all(pred > 0) else False,
    }


def sample_index(frame):
    """One row per test stock-day. Direction uses this index and no other filter."""
    required = {"Signal_Date", "Ticker", "Fold", "Actual_5d_Pct", "Actual_1d_Pct"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(sorted(missing))
    index = frame.loc[:, ["Signal_Date", "Ticker", "Fold", "Actual_5d_Pct", "Actual_1d_Pct"]].copy()
    index["Signal_Date"] = pd.to_datetime(index["Signal_Date"])
    index = index.drop_duplicates(["Signal_Date", "Ticker", "Fold"])
    index["Zero_5d"] = np.isclose(index.Actual_5d_Pct.to_numpy(dtype=float), 0.0)
    index["Up_5d"] = index.Actual_5d_Pct > 0
    index["Down_5d"] = index.Actual_5d_Pct < 0
    return index.sort_values(["Signal_Date", "Ticker"]).reset_index(drop=True)


def same_sample(left, right):
    keys = ["Signal_Date", "Ticker", "Fold"]
    a = left.copy()
    b = right.copy()
    a["Signal_Date"] = pd.to_datetime(a["Signal_Date"])
    b["Signal_Date"] = pd.to_datetime(b["Signal_Date"])
    merged = a.merge(b, on=keys, suffixes=("_left", "_right"))
    if len(merged) != len(a) or len(merged) != len(b):
        raise AssertionError("股票、日期或折數對不齊")
    return {
        "Rows": int(len(merged)),
        "Actual_5d_Equal": bool(np.allclose(merged.Actual_5d_Pct_left, merged.Actual_5d_Pct_right)),
        "Actual_1d_Equal": bool(np.allclose(merged.Actual_1d_Pct_left, merged.Actual_1d_Pct_right)),
    }


def fold_prediction_distribution(frame):
    rows = []
    for (model, split, fold_id), part in frame.groupby(["Model", "Split", "Fold"], sort=True):
        pred = part.Predicted.to_numpy(dtype=float)
        act = part.Actual_5d_Pct.to_numpy(dtype=float)
        rows.append({
            "Model": model,
            "Split": split,
            "Fold": int(fold_id),
            "Rows": int(len(part)),
            "Label_Mean": float(act.mean()),
            "Label_Positive_Rate": float((act > 0).mean()),
            "Label_Zero_Rate": float(np.isclose(act, 0.0).mean()),
            "Pred_Mean": float(pred.mean()),
            "Pred_Median": float(np.median(pred)),
            "Pred_Std": float(pred.std(ddof=1)) if len(pred) > 1 else np.nan,
            "Pred_Positive_Rate": float((pred > 0).mean()),
        })
    return pd.DataFrame(rows)


def ticker_prediction_bias(frame):
    rows = []
    test = frame[frame.Split == "test"]
    for (model, ticker), part in test.groupby(["Model", "Ticker"], sort=True):
        pred = part.Predicted.to_numpy(dtype=float)
        act = part.Actual_5d_Pct.to_numpy(dtype=float)
        rows.append({
            "Model": model,
            "Ticker": ticker,
            "Rows": int(len(part)),
            "Pred_Mean": float(pred.mean()),
            "Pred_Positive_Rate": float((pred > 0).mean()),
            "Label_Mean": float(act.mean()),
            "Label_Positive_Rate": float((act > 0).mean()),
            "Bias": float(pred.mean() - act.mean()),
        })
    return pd.DataFrame(rows)


def retrain_gap(schedule):
    left = schedule[schedule.Model == "retrain_5"][
        ["Split", "Fold", "Signal_Date", "Ticker", "Predicted"]
    ].rename(columns={"Predicted": "Retrain_5"})
    right = schedule[schedule.Model == "retrain_1"][
        ["Split", "Fold", "Signal_Date", "Ticker", "Predicted"]
    ].rename(columns={"Predicted": "Retrain_1"})
    merged = left.merge(right, on=["Split", "Fold", "Signal_Date", "Ticker"])
    if len(merged) != len(left) or len(merged) != len(right):
        raise AssertionError("每日與五日重訓的列對不齊")
    gap = merged.Retrain_1.to_numpy(dtype=float) - merged.Retrain_5.to_numpy(dtype=float)
    merged["Absolute_Gap"] = np.abs(gap)
    return merged


def gap_summary(merged):
    rows = []
    groups = [("all", "all", merged)]
    groups.extend(
        (split, int(fold_id), part)
        for (split, fold_id), part in merged.groupby(["Split", "Fold"], sort=True)
    )
    for split, fold_id, part in groups:
        left = part.Retrain_5.to_numpy(dtype=float)
        right = part.Retrain_1.to_numpy(dtype=float)
        gap = np.abs(right - left)
        corr = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 0 and np.std(right) > 0 else np.nan
        rows.append({
            "Split": split,
            "Fold": fold_id,
            "Rows": int(len(part)),
            "Exact": int(np.sum(gap < 1e-12)),
            "Within_1e-4": int(np.sum(gap < 1e-4)),
            "Mean_Absolute_Gap": float(gap.mean()),
            "Max_Absolute_Gap": float(gap.max()),
            "Correlation": corr,
        })
    return pd.DataFrame(rows)


def label_distribution(folds, five):
    rows = []
    for fold_id, fold in enumerate(folds, start=1):
        blocks = {
            "train": np.asarray(fold["train"], dtype=int),
            "validation": np.asarray(fold["val"], dtype=int),
            "test": np.asarray(fold["test"], dtype=int),
        }
        for window in (252, 504, 756):
            kept = rolling_train(fold["train"], window)
            if kept is not None:
                blocks[f"train_{window}"] = kept
        for name, positions in blocks.items():
            values = five.iloc[positions][list(TICKERS)].to_numpy(dtype=float)
            flat = values.ravel()
            finite = flat[np.isfinite(flat)]
            rows.append({
                "Fold": fold_id,
                "Block": name,
                "Signals": int(len(positions)),
                "Rows": int(len(finite)),
                "Label_Mean": float(finite.mean()),
                "Label_Positive_Rate": float((finite > 0).mean()),
                "Label_Negative_Rate": float((finite < 0).mean()),
                "Label_Zero_Rate": float(np.isclose(finite, 0.0).mean()),
            })
    return pd.DataFrame(rows)


def embargo_breaches(folds):
    breaches = []
    for fold_id, fold in enumerate(folds, start=1):
        train = np.asarray(fold["train"], dtype=int)
        val = np.asarray(fold["val"], dtype=int)
        test = np.asarray(fold["test"], dtype=int)
        if int(train.max()) + LABEL_GAP >= int(val[0]):
            breaches.append({"Fold": fold_id, "Boundary": "train_val"})
        if int(val.max()) + LABEL_GAP >= int(test[0]):
            breaches.append({"Fold": fold_id, "Boundary": "val_test"})
    return breaches


def schedule_maturity(cost):
    """Training length must reset when a split starts and then only grow."""
    rows = []
    fit = cost[cost.Status == "fit"].copy()
    for (model, fold_id), part in fit.groupby(["Model", "Fold"], sort=True):
        for split, block in part.groupby("Split", sort=False):
            samples = block.Train_Samples.to_numpy(dtype=int)
            rows.append({
                "Model": model,
                "Fold": int(fold_id),
                "Split": split,
                "Updates": int(len(block)),
                "First_Train_Samples": int(samples[0]),
                "Last_Train_Samples": int(samples[-1]),
                "Nondecreasing": bool(np.all(np.diff(samples) >= 0)),
                "Epochs_Min": float(block.Epochs.min()),
                "Epochs_Max": float(block.Epochs.max()),
            })
    table = pd.DataFrame(rows)
    resets = []
    for (model, fold_id), part in table.groupby(["Model", "Fold"]):
        val = part[part.Split == "validation"]
        test = part[part.Split == "test"]
        if len(val) and len(test):
            resets.append(int(test.First_Train_Samples.iloc[0]) == int(val.First_Train_Samples.iloc[0]))
    return table, bool(resets) and all(resets)


class CorrectedFitter:
    """Same network as V2.8, but each fit gets a new Adam optimizer."""

    def __init__(self):
        from experiments.v2.v28 import Fitter

        self._fitter = Fitter()

    def fit(self, x_train, y_train, x_val, y_val, layers, units, previous=None):
        from experiments.risk_experiment import BATCH_SIZE, EPOCHS, PATIENCE

        key, model = self._fitter._model(layers, units, x_train.shape[-1])
        model.set_weights(self._fitter.initial[key] if previous is None else previous)
        model.compile(optimizer=self._fitter.tf.keras.optimizers.Adam(), loss="mse")
        if float(model.optimizer.learning_rate) <= 0:
            raise AssertionError("修正後的學習率仍是 0")
        started = time.perf_counter()
        history = model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=False,
            validation_data=(x_val, y_val),
            callbacks=[self._fitter.tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE, restore_best_weights=True,
            )],
            verbose=0,
        )
        return model, history, time.perf_counter() - started


def _weight_delta(before, after):
    return float(np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(before, after))))


def reproduce_lstm1_50(folds, frames, index, one, five, excess):
    """Refit the single LSTM(50) with a live learning rate. Does not adopt it."""
    from experiments.horizon_experiment import assert_scaler_does_not_see_future
    from experiments.v2.v28 import _carry_predict, _emit, _prepare

    fitter = CorrectedFitter()
    rows = []
    diagnostics = []
    for fold_id, fold in enumerate(folds, start=1):
        train = np.asarray(fold["train"], dtype=int)
        assert_scaler_does_not_see_future(train, fold["val"], fold["test"])
        print(f"[V29] corrected lstm1_50 fold {fold_id} train {len(train)}", flush=True)
        predictions = {}
        weights = {}
        for ticker in TICKERS:
            values, scaler = _prepare(frames, ticker, train, fold["val"], fold["val"])
            x_train = _matrix(values, scaler, train)
            y_train = five.iloc[train][ticker].to_numpy(dtype=np.float32)
            x_val = _matrix(values, scaler, fold["val"])
            y_val = five.iloc[fold["val"]][ticker].to_numpy(dtype=np.float32)
            key, model = fitter._fitter._model(1, 50, values.shape[1])
            before = [item.copy() for item in (fitter._fitter.initial[key])]
            model, history, elapsed = fitter.fit(x_train, y_train, x_val, y_val, 1, 50, None)
            after = model.get_weights()
            predicted = model.predict(_matrix(values, scaler, fold["val"]), verbose=0).ravel()
            losses = history.history["val_loss"]
            best = int(np.argmin(losses)) + 1
            diagnostics.append({
                "Fold": fold_id,
                "Ticker": ticker,
                "Epochs_Run": int(len(losses)),
                "Best_Epoch": best,
                "Train_Loss_First": float(history.history["loss"][0]),
                "Train_Loss_Last": float(history.history["loss"][-1]),
                "Val_Loss_First": float(losses[0]),
                "Val_Loss_Best": float(min(losses)),
                "Learning_Rate": float(model.optimizer.learning_rate),
                "Weight_L2_From_Initial": _weight_delta(before, after),
                "Val_Pred_Mean": float(predicted.mean()),
                "Fit_Seconds": float(elapsed),
            })
            predictions[ticker] = predicted
            weights[ticker] = after
        test_predicted, _elapsed = _carry_predict(fitter._fitter, frames, weights, train, fold["test"], 1, 50)
        for ticker in TICKERS:
            rows.extend(_emit(
                "lstm1_50_corrected", index, fold["val"], ticker, predictions[ticker],
                one, five, excess, "validation", fold_id,
            ))
            rows.extend(_emit(
                "lstm1_50_corrected", index, fold["test"], ticker, test_predicted[ticker],
                one, five, excess, "test", fold_id,
            ))
    frame = pd.DataFrame(rows)
    frame["Origin"] = "corrected_optimizer_v29"
    if pd.to_datetime(frame.Label_End).max() > pd.Timestamp("2026-03-25"):
        raise AssertionError("修正對照的標籤超過 2026-03-25")
    return frame, pd.DataFrame(diagnostics)


def _matrix(values, scaler, positions):
    from experiments.horizon_experiment import _matrix as matrix

    return matrix(values, scaler, positions)


def quality_change(saved, corrected):
    rows = []
    for name, frame in (("lstm1_50_v28", saved), ("lstm1_50_corrected", corrected)):
        test = frame[frame.Split == "test"]
        scores = reconcile_direction(test.Predicted, test.Actual_5d_Pct)
        pred = test.Predicted.to_numpy(dtype=float)
        act = test.Actual_5d_Pct.to_numpy(dtype=float)
        spearman = float(pd.Series(pred).corr(pd.Series(act), method="spearman")) if np.std(pred) > 1e-12 else 0.0
        rows.append({
            "Model": name,
            "Rows": scores["Rows"],
            "Direction_Accuracy": scores["Direction_Accuracy"],
            "Balanced_Accuracy": scores["Balanced_Accuracy"],
            "Positive_Rate": scores["Positive_Rate"],
            "Spearman": spearman,
            "Pred_Mean": float(pred.mean()),
            "Pred_Std": float(pred.std(ddof=1)),
            "Adopted": False,
        })
    return pd.DataFrame(rows)


def write_audit():
    from experiments.horizon_experiment import design_audit, load_context
    from experiments.v2.guard import snapshot

    before = snapshot()
    predictions = pd.read_csv(V28 / "predictions.csv")
    schedule = pd.read_csv(V28 / "schedule_predictions.csv")
    cost = pd.read_csv(V28 / "schedule_cost.csv")
    fixed = pd.read_csv(V28 / "fixed_predictions.csv")
    pieces = [predictions[predictions.Model.isin(AUDIT_MODELS[:4])]]
    pieces.append(schedule[schedule.Model.isin(("retrain_5", "retrain_1"))])
    combined = pd.concat(pieces, ignore_index=True)
    test = combined[combined.Split == "test"].copy()
    always = predictions[(predictions.Model == "always_long") & (predictions.Split == "test")]
    index = sample_index(always)
    denominator = reconcile_direction(np.ones(len(index)), index.Actual_5d_Pct)
    alignment = {
        name: same_sample(always, test[test.Model == name])
        for name in AUDIT_MODELS
    }
    labels = None
    panel = load_context()
    _panel, _frames, _risk, _one, five, _excess = panel
    _audit, _positions, folds, _market = design_audit()
    labels = label_distribution(folds, five)
    breaches = embargo_breaches(folds)
    maturity, reset_ok = schedule_maturity(cost)
    gap = retrain_gap(schedule)
    summary = {
        "adopted_signal": "mse_b",
        "promoted": False,
        "v28_outputs_rewritten": False,
        "denominator": denominator,
        "sample_alignment": alignment,
        "actual_1d_5d_correlation": float(pd.Series(index.Actual_1d_Pct).corr(pd.Series(index.Actual_5d_Pct))),
        "embargo_breaches": breaches,
        "schedule_resets_at_test": reset_ok,
        "schedule_nondecreasing": bool(maturity.Nondecreasing.all()),
        "retrain_test_mean_absolute_gap": float(
            gap.loc[gap.Split == "test", "Absolute_Gap"].mean()
        ),
        "proposals_trained": False,
        "proposal_ids": [item["id"] for item in PROPOSALS],
    }
    assert_v29(V29 / "sample_index.csv")
    index.to_csv(V29 / "sample_index.csv", index=False)
    fold_prediction_distribution(combined).to_csv(V29 / "fold_prediction_distribution.csv", index=False)
    ticker_prediction_bias(combined).to_csv(V29 / "ticker_prediction_bias.csv", index=False)
    gap_summary(gap).to_csv(V29 / "retrain_gap_summary.csv", index=False)
    gap.loc[gap.Split == "test", ["Signal_Date", "Ticker", "Fold", "Retrain_5", "Retrain_1", "Absolute_Gap"]].to_csv(
        V29 / "retrain_test_gap.csv", index=False,
    )
    labels.to_csv(V29 / "label_distribution.csv", index=False)
    maturity.to_csv(V29 / "schedule_maturity.csv", index=False)
    (V29 / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    from experiments.v2.guard import assert_unchanged

    assert_unchanged(before)
    return summary, folds, panel, fixed


def write_correction(folds, panel, fixed):
    from experiments.v2.guard import assert_unchanged, snapshot

    target = V29 / "corrected_lstm1_50_predictions.csv"
    if target.exists() and target.stat().st_size > 0:
        print("[V29] corrected predictions already saved", flush=True)
        return
    before = snapshot()
    _panel, frames, _risk, one, five, excess = panel
    _audit_unused, _positions, folds, index = __import__(
        "experiments.horizon_experiment", fromlist=["design_audit"]
    ).design_audit()
    predictions, diagnostics = reproduce_lstm1_50(folds, frames, index, one, five, excess)
    saved = fixed[(fixed.Model == "lstm1_50")].copy()
    change = quality_change(saved, predictions)
    assert_v29(target)
    predictions.to_csv(target, index=False)
    diagnostics.to_csv(V29 / "corrected_lstm1_50_diagnostics.csv", index=False)
    change.to_csv(V29 / "corrected_quality.csv", index=False)
    note = V29 / "修正紀錄.md"
    note.write_text(_correction_note(change, diagnostics), encoding="utf-8")
    assert_unchanged(before)


def _correction_note(change, diagnostics):
    broken = change[change.Model == "lstm1_50_v28"].iloc[0]
    fixed = change[change.Model == "lstm1_50_corrected"].iloc[0]
    return "\n".join([
        "# V2.9 修正紀錄",
        "",
        "V2.8 的結果檔沒有改寫。下面是同一套單層 LSTM(50)、同一 16 折與五日標籤，只把每次擬合的 Adam 重新建立。",
        "",
        f"- 原程式測試預測平均 {broken.Pred_Mean:.6f}，正值比例 {broken.Positive_Rate:.4f}，方向準確率 {broken.Direction_Accuracy:.4f}，Spearman {broken.Spearman:.4f}。",
        f"- 修正後測試預測平均 {fixed.Pred_Mean:.6f}，正值比例 {fixed.Positive_Rate:.4f}，方向準確率 {fixed.Direction_Accuracy:.4f}，Spearman {fixed.Spearman:.4f}。",
        f"- 修正後各股權重與初始值的 L2 距離中位數 {float(diagnostics.Weight_L2_From_Initial.median()):.6f}，最佳 epoch 中位數 {float(diagnostics.Best_Epoch.median()):.1f}。",
        "- 這組修正預測只用來確認程式影響，不採用，也不進入模型選擇。",
        "",
    ])


def main():
    summary, folds, panel, fixed = write_audit()
    print(json.dumps({"denominator_rows": summary["denominator"]["Rows"], "promoted": False}, ensure_ascii=False), flush=True)
    write_correction(folds, panel, fixed)
    print(json.dumps({"corrected": True, "adopted_signal": "mse_b"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
