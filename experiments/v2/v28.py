"""Training-window, depth, and retrain-frequency study for 555 V2.8.

The saved two-layer MSE model stays the control. Window length, depth, and
retrain frequency are the only changes, and none of them is chosen by test ROI.
Results stay under experiments/results/v2/v28.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from experiments.horizon_experiment import (
    LABEL_GAP,
    TICKERS,
    TIME_STEP,
    _matrix,
    _rows_for,
    design_audit,
    load_context,
    scaler_rows,
)
from experiments.risk_experiment import (
    BATCH_SIZE,
    EPOCHS,
    FEATURES,
    MARKET,
    PATIENCE,
    SEED,
    load_sentiment,
)
from experiments.v2.books import index_metrics, metrics, run_book
from experiments.v2.guard import OUT, PHASE5, SENTIMENT, snapshot
from experiments.v2.signals import quality_row
from experiments.v2.v28_rules import (
    ADOPTABLE,
    ARCHITECTURES,
    CONTROL,
    MOMENTUM_LOOKBACK,
    V28,
    WINDOWS,
    assert_v28,
    choose,
    classification_scores,
    fold_direction,
    fold_spearman_list,
    matured_positions,
    research_prediction_dir,
    rolling_train,
    select_windows,
    split_update,
    stitch_windows,
    update_schedule,
)

ORIGIN = "retrained_walk_forward_v28"
KEPT_ROOT_NAMES = (
    "prediction_quality.csv",
    "strategy_metrics.csv",
    "summary.json",
    "variant_predictions.csv",
    "v21_strategy_metrics.csv",
)


def _read(path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _append(path, frame):
    current = _read(path)
    combined = pd.concat([current, frame], ignore_index=True) if len(current) else frame
    combined.to_csv(assert_v28(path), index=False)
    return combined


class Fitter:
    """Reuse one compiled network per shape. From-scratch fits restore the seeded weights."""

    def __init__(self):
        import os

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        try:
            intra = int(os.environ.get("TW_LSTM_INTRA_THREADS", "4"))
            inter = int(os.environ.get("TW_LSTM_INTER_THREADS", "2"))
            tf.config.threading.set_intra_op_parallelism_threads(intra)
            tf.config.threading.set_inter_op_parallelism_threads(inter)
        except RuntimeError:
            pass
        self.tf = tf
        self.models = {}
        self.initial = {}

    def _model(self, layers, units, n_features):
        key = (layers, units, n_features)
        if key in self.models:
            return key, self.models[key]
        tf = self.tf
        tf.keras.utils.set_random_seed(SEED)
        sequence = [tf.keras.layers.Input(shape=(TIME_STEP, n_features))]
        if layers == 2:
            sequence.extend([
                tf.keras.layers.LSTM(units, return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(units),
                tf.keras.layers.Dropout(0.2),
            ])
        elif layers == 1:
            sequence.extend([
                tf.keras.layers.LSTM(units),
                tf.keras.layers.Dropout(0.2),
            ])
        else:
            raise ValueError(layers)
        sequence.append(tf.keras.layers.Dense(1))
        model = tf.keras.Sequential(sequence)
        model.compile(optimizer="adam", loss="mse")
        self.models[key] = model
        self.initial[key] = model.get_weights()
        return key, model

    def fit(self, x_train, y_train, x_val, y_val, layers, units, previous=None):
        key, model = self._model(layers, units, x_train.shape[-1])
        model.set_weights(self.initial[key] if previous is None else previous)
        # Kept so a rerun still matches the saved V2.8 files. zeros_like also
        # sets Adam's learning_rate to 0 after the first fit, so later fits do
        # not move. The correction is experiments/v2/v29.py and is not written here.
        if getattr(model.optimizer, "built", False):
            for variable in model.optimizer.variables:
                variable.assign(self.tf.zeros_like(variable))
        started = time.perf_counter()
        history = model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=False,
            validation_data=(x_val, y_val),
            callbacks=[self.tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE, restore_best_weights=True,
            )],
            verbose=0,
        )
        return model, history, time.perf_counter() - started


def _prepare(frames, ticker, train, early_stop, predict_at):
    values = frames[ticker][list(FEATURES)].to_numpy(dtype=float)
    rows = scaler_rows(train)
    if len(rows) == 0 or int(rows.max()) >= int(np.min(predict_at)) or int(np.max(train)) >= int(np.min(predict_at)):
        raise AssertionError("標準化器或訓練訊號碰到了決策日")
    scaler = MinMaxScaler().fit(values[rows])
    return values, scaler


def _fit_block(fitter, frames, five, train, early_stop, predict_at, layers, units, previous):
    predictions = {}
    weights = {}
    elapsed = 0.0
    predict_elapsed = 0.0
    epochs = []
    for ticker in TICKERS:
        values, scaler = _prepare(frames, ticker, train, early_stop, predict_at)
        model, history, fit_elapsed = fitter.fit(
            _matrix(values, scaler, train),
            five.iloc[train][ticker].to_numpy(dtype=np.float32),
            _matrix(values, scaler, early_stop),
            five.iloc[early_stop][ticker].to_numpy(dtype=np.float32),
            layers,
            units,
            None if previous is None else previous[ticker],
        )
        started = time.perf_counter()
        predicted = model.predict(_matrix(values, scaler, predict_at), verbose=0).ravel()
        predict_elapsed += time.perf_counter() - started
        if not np.isfinite(predicted).all():
            raise ValueError(f"{ticker} 產生非有限預測")
        predictions[ticker] = predicted
        weights[ticker] = model.get_weights()
        elapsed += fit_elapsed
        epochs.append(len(history.history["val_loss"]))
    return predictions, weights, elapsed, predict_elapsed, float(np.mean(epochs))


def _emit(name, index, positions, ticker, predicted, one, five, excess, split, fold_id):
    rows = _rows_for(name, index, positions, ticker, predicted, one, five, excess, split, fold_id)
    for row in rows:
        row["Fold"] = fold_id
        row["Origin"] = ORIGIN
    return rows


def _standard_fold(fitter, frames, index, one, five, excess, fold, fold_id, train, layers, units, name):
    from experiments.horizon_experiment import assert_scaler_does_not_see_future

    assert_scaler_does_not_see_future(train, fold["val"], fold["test"])
    val_predicted, weights, fit_elapsed, predict_val, epoch_mean = _fit_block(
        fitter, frames, five, train, fold["val"], fold["val"], layers, units, None,
    )
    test_predicted, predict_test = _carry_predict(fitter, frames, weights, train, fold["test"], layers, units)
    rows = []
    for ticker in TICKERS:
        rows.extend(_emit(name, index, fold["val"], ticker, val_predicted[ticker], one, five, excess, "validation", fold_id))
        rows.extend(_emit(name, index, fold["test"], ticker, test_predicted[ticker], one, five, excess, "test", fold_id))
    logs = [{
        "Model": name,
        "Fold": fold_id,
        "Split": "both",
        "Train_Samples": int(len(train)),
        "Fit_Seconds": fit_elapsed,
        "Predict_Seconds": predict_val + predict_test,
        "Epochs": epoch_mean,
        "Status": "fit",
    }]
    return rows, logs


def train_fixed_models(folds, frames, index, one, five, excess):
    fitter = Fitter()
    path = V28 / "fixed_predictions.csv"
    log_path = V28 / "fixed_cost.csv"
    done = set(_read(path).Model) if path.exists() else set()
    jobs = [(f"rolling_{window}", 2, 50, window) for window in WINDOWS]
    jobs.extend((name, spec["layers"], spec["units"], None) for name, spec in ARCHITECTURES.items())
    for name, layers, units, window in jobs:
        if name in done:
            print(f"[V28] {name} already saved", flush=True)
            continue
        print(f"[V28] {name}", flush=True)
        rows = []
        logs = []
        for fold_id, fold in enumerate(folds, start=1):
            train = fold["train"] if window is None else rolling_train(fold["train"], window)
            if train is None:
                logs.append({
                    "Model": name, "Fold": fold_id, "Split": "both", "Train_Samples": int(len(fold["train"])),
                    "Fit_Seconds": 0.0, "Predict_Seconds": 0.0, "Epochs": 0.0, "Status": "infeasible",
                })
                continue
            print(f"[V28] {name} fold {fold_id} train {len(train)}", flush=True)
            built, fold_logs = _standard_fold(
                fitter, frames, index, one, five, excess, fold, fold_id, train, layers, units, name,
            )
            rows.extend(built)
            logs.extend(fold_logs)
        if rows:
            _append(path, pd.DataFrame(rows))
        _append(log_path, pd.DataFrame(logs))
        done.add(name)


def _carry_predict(fitter, frames, weights, train, positions, layers, units):
    predicted = {}
    elapsed = 0.0
    for ticker in TICKERS:
        values, scaler = _prepare(frames, ticker, train, train, positions)
        key, model = fitter._model(layers, units, values.shape[1])
        model.set_weights(weights[ticker])
        started = time.perf_counter()
        predicted[ticker] = model.predict(_matrix(values, scaler, positions), verbose=0).ravel()
        elapsed += time.perf_counter() - started
        if not np.isfinite(predicted[ticker]).all():
            raise ValueError(f"{ticker} 沿用模型時產生非有限預測")
    return predicted, elapsed


def train_schedules(folds, frames, index, one, five, excess):
    fitter = Fitter()
    path = V28 / "schedule_predictions.csv"
    log_path = V28 / "schedule_cost.csv"
    existing = _read(path)
    done = set(zip(existing.Model, existing.Fold)) if len(existing) else set()
    for every in (5, 1):
        name = f"retrain_{every}"
        from_scratch = every == 5
        for fold_id, fold in enumerate(folds, start=1):
            if (name, fold_id) in {(model, int(fold)) for model, fold in done}:
                continue
            print(f"[V28] {name} fold {fold_id}", flush=True)
            rows = []
            logs = []
            failed = False
            for split, block in (("validation", fold["val"]), ("test", fold["test"])):
                weights = None
                train_used = None
                for asof, horizon in update_schedule(block, every):
                    earlier = matured_positions(block, asof)
                    eligible = np.concatenate([np.asarray(fold["train"], dtype=int), np.asarray(earlier, dtype=int)])
                    parts = split_update(eligible)
                    status = "fit"
                    try:
                        if parts is None:
                            if weights is None:
                                status = "failed"
                                failed = True
                                logs.append(_schedule_log(name, fold_id, split, asof, 0, 0, 0, 0, status))
                                continue
                            predicted, predict_elapsed = _carry_predict(
                                fitter, frames, weights, train_used, horizon, 2, 50,
                            )
                            logs.append(_schedule_log(name, fold_id, split, asof, len(train_used), 0, predict_elapsed, 0, "carried_forward"))
                        else:
                            train, early_stop = parts
                            if int(np.max(np.concatenate([train, early_stop]))) >= int(asof):
                                raise AssertionError("更新資料碰到決策日")
                            previous = None if from_scratch or weights is None else weights
                            predicted, weights, fit_elapsed, predict_elapsed, epoch_mean = _fit_block(
                                fitter, frames, five, train, early_stop, horizon, 2, 50, previous,
                            )
                            train_used = train
                            logs.append(_schedule_log(
                                name, fold_id, split, asof, len(train), fit_elapsed, predict_elapsed, epoch_mean, status,
                            ))
                        for ticker in TICKERS:
                            rows.extend(_emit(
                                name, index, horizon, ticker, predicted[ticker], one, five, excess, split, fold_id,
                            ))
                    except Exception:
                        logs.append(_schedule_log(name, fold_id, split, asof, 0, 0, 0, 0, "failed"))
                        _append(log_path, pd.DataFrame(logs))
                        raise
            if not failed and rows:
                frame = pd.DataFrame(rows)
                if pd.to_datetime(frame.Label_End).max() > pd.Timestamp("2026-03-25"):
                    raise AssertionError("V2.8 標籤超過 2026-03-25")
                _append(path, frame)
            _append(log_path, pd.DataFrame(logs))


def _schedule_log(name, fold_id, split, asof, samples, fit_seconds, predict_seconds, epochs, status):
    return {
        "Model": name,
        "Fold": fold_id,
        "Split": split,
        "AsOf_Position": int(asof),
        "Train_Samples": int(samples),
        "Fit_Seconds": float(fit_seconds),
        "Predict_Seconds": float(predict_seconds),
        "Epochs": float(epochs),
        "Status": status,
    }


def _load_control():
    predictions = pd.read_csv(
        PHASE5 / "predictions.csv", parse_dates=["Signal_Date", "Execution_Date", "Label_End"],
    )
    control = predictions[predictions.Model == "B"].copy()
    control["Model"] = CONTROL
    if control.Fold.nunique() != 16:
        raise AssertionError("控制組不是既有的 16 折")
    return control


def _baselines(control, panel):
    rows = []
    for row in control.itertuples(index=False):
        location = panel.index.get_loc(pd.Timestamp(row.Signal_Date))
        history = panel[row.Ticker].iloc[location - MOMENTUM_LOOKBACK:location + 1]
        if len(history) != MOMENTUM_LOOKBACK + 1 or (history <= 0).any():
            raise AssertionError("動能基準使用了訊號日之後或不完整的價格")
        momentum = float(history.iloc[-1] / history.iloc[0] - 1.0) * 100.0
        base = row._asdict() if hasattr(row, "_asdict") else None
        del base
        common = {
            "Split": row.Split,
            "Fold": int(row.Fold),
            "Signal_Date": row.Signal_Date,
            "Execution_Date": row.Execution_Date,
            "Label_End": row.Label_End,
            "Ticker": row.Ticker,
            "Actual_1d_Pct": row.Actual_1d_Pct,
            "Actual_5d_Pct": row.Actual_5d_Pct,
            "Actual_Excess_5d_Pct": row.Actual_Excess_5d_Pct,
            "Origin": "baseline_not_a_model",
        }
        rows.append({**common, "Model": "always_long", "Predicted": 1.0})
        rows.append({**common, "Model": "momentum_20", "Predicted": momentum})
    return pd.DataFrame(rows)


def _payload(frame):
    validation = frame[frame.Split == "validation"]
    pooled = classification_scores(validation.Predicted, validation.Actual_5d_Pct)
    directions = fold_direction(validation)
    always = []
    for fold_id, part in validation.groupby("Fold"):
        always.append(classification_scores(np.ones(len(part)), part.Actual_5d_Pct)["Direction_Accuracy"])
    return {
        "spearman": fold_spearman_list(validation),
        "direction": directions,
        "direction_mean": float(np.mean(directions)),
        "balanced_accuracy": pooled["Balanced_Accuracy"],
        "positive_rate": pooled["Positive_Rate"],
        "always_long_direction": float(np.mean(always)),
    }


def _quality_table(named):
    rows = []
    for name, frame in named.items():
        for split in ("validation", "test"):
            part = frame[frame.Split == split]
            if part.empty:
                continue
            row = quality_row(name, part)
            row.update(classification_scores(part.Predicted, part.Actual_5d_Pct))
            row["Split"] = split
            row["Fold_Mean_Direction"] = float(np.mean(fold_direction(part))) if part.Fold.nunique() == 16 else np.nan
            row["Fold_Mean_Spearman"] = float(np.mean(fold_spearman_list(part))) if part.Fold.nunique() == 16 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _fold_table(named):
    rows = []
    for name, frame in named.items():
        for split in ("validation", "test"):
            part = frame[frame.Split == split]
            for fold_id, chunk in part.groupby("Fold"):
                scores = classification_scores(chunk.Predicted, chunk.Actual_5d_Pct)
                from experiments.v2.signals import fold_spearman
                spearman = fold_spearman(chunk)
                scores.update({
                    "Model": name,
                    "Split": split,
                    "Fold": int(fold_id),
                    "Cross_Section_Spearman": float(spearman[int(fold_id)]),
                })
                rows.append(scores)
    return pd.DataFrame(rows)


def _books(panel, risk, named, adopted, start, end):
    sentiment = load_sentiment(SENTIMENT, panel.index)
    specs = [
        ("equal_weekly", None, "equal", None),
        ("buy_hold", None, "buy_hold", None),
    ]
    control_dates = set(pd.to_datetime(named[CONTROL].loc[named[CONTROL].Split == "test", "Signal_Date"]))
    for name, frame in named.items():
        if name in ("always_long", "momentum_20"):
            continue
        test = frame[frame.Split == "test"]
        if set(pd.to_datetime(test.Signal_Date)) != control_dates:
            continue
        specs.append((f"{name}_threshold_weekly", test, "threshold", frame[frame.Split == "validation"]))
    # `adopted` was chosen using all folds. Backfilling that global choice into
    # the early test history would be a post-hoc strategy selection. Retain the
    # candidate decision for research, but do not manufacture its historical ROI.
    rows = []
    for name, preds, weight, tau in specs:
        daily = run_book(panel, risk, preds, sentiment, start, end, 5, weight, tau,
                         signal_calendar=control_dates)
        result = metrics(daily, name, "every_5_sessions" if weight != "buy_hold" else "buy_and_hold")
        result["Risk_As_Of"] = "preceding_signal_session"
        result["Threshold_Protocol"] = "same_fold_validation_labels_before_signal" if weight == "threshold" else "not_used"
        result["Global_Adoption_Backfilled"] = False
        result["Comparable_With_Equal"] = weight != "buy_hold"
        rows.append(result)
    rows.append(index_metrics(panel, start, end))
    portfolios = pd.DataFrame(rows)
    equal = float(portfolios.loc[portfolios.Name == "equal_weekly", "ROI_Pct"].iloc[0])
    index_roi = float(portfolios.loc[portfolios.Name == "twii_price_index_with_one_round_trip", "ROI_Pct"].iloc[0])
    portfolios["Gap_Vs_Equal_ROI_Pct"] = portfolios.ROI_Pct - equal
    portfolios["Gap_Vs_Index_ROI_Pct"] = portfolios.ROI_Pct - index_roi
    return portfolios


def _kept_bytes():
    paths = [OUT / name for name in KEPT_ROOT_NAMES]
    for child in ("v26", "v27"):
        folder = OUT / child
        if folder.exists():
            paths.extend(path for path in folder.rglob("*") if path.is_file())
    return {path: path.read_bytes() for path in paths if path.exists()}


def _write_report(adopted, decisions, quality, portfolios, windows, costs, coverage):
    payload = {
        "study": "exploratory_not_a_new_independent_test",
        "selection_used_test_roi": False,
        "selection_scope": "posthoc_research_choice_across_all_validation_folds",
        "global_adoption_historical_roi_reported": False,
        "execution_protocol": "next_close_signal_day_risk_fold_local_matured_validation_v2",
        "adopted_signal": adopted,
        "windows": list(WINDOWS),
        "architectures": ARCHITECTURES,
        "retrain_steps": [5, 1],
        "momentum_lookback": MOMENTUM_LOOKBACK,
        "daily_prediction_directory": str(research_prediction_dir().relative_to(V28.parent.parent.parent)),
        "live_predictions_written": False,
        "decisions": decisions,
        "test_coverage": coverage,
    }
    assert_v28(V28 / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# V2.8 訓練視窗、模型深度與重訓頻率",
        "",
        "探索性研究，沿用到 2026-03-25 的下一筆收盤、五日持有帳本。採用規則在訓練前寫入 `v28_rules.py`。測試 ROI 沒有用來選模型。",
        "",
        f"採用結果：`{adopted}`。",
        "",
        "排序門檻：驗證折平均橫斷面 Spearman 高於 `mse_b`，而且至少 9/16 折更高。",
        "方向門檻：驗證折平均方向準確率同時高於 `mse_b` 與永遠看多，平衡準確率高於 0.5，正預測比例落在 20% 到 80%，而且至少 9/16 折的方向準確率更高。",
        "兩個門檻只要有一個通過就可以進入候選；同時通過時，先比驗證平衡準確率，再比驗證 Spearman。永遠看多和二十日動能只作預測對照，不取代 LSTM。",
        "",
        "## 驗證採用",
        "",
    ]
    for row in decisions:
        if row.get("Status") == "not_trained":
            lines.append(f"- {row['Model']}：尚未訓練。")
            continue
        lines.append(
            f"- {row['Model']}：驗證 Spearman {row['Validation_Mean_Spearman']:.3f}，"
            f"排序 {row['Rank_Folds_Beating_Control']}/16，方向 {row['Direction_Accuracy']:.1%}，"
            f"永遠看多 {row['Always_Long_Direction']:.1%}，平衡準確率 {row['Balanced_Accuracy']:.3f}，"
            f"正預測比例 {row['Positive_Rate']:.1%}，採用 {row['Accepted']}。"
        )
    lines.extend(["", "## 預測品質", ""])
    lines.append("| 模型 | 區間 | 方向 | 平衡準確率 | 正預測 | 上漲比例 | Pearson | 橫斷面 Spearman | Top-1 | MAE | RMSE | 預測標準差 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    show = quality.sort_values(["Model", "Split"])
    for row in show.itertuples(index=False):
        lines.append(
            f"| {row.Model} | {row.Split} | {row.Direction_Accuracy:.1%} | {row.Balanced_Accuracy:.3f} | "
            f"{row.Positive_Rate:.1%} | {row.Up_Rate:.1%} | {row.Pearson:.3f} | {row.Cross_Section_Spearman:.3f} | "
            f"{row.Top1_Hit_Rate:.1%} | {row.MAE:.2f} | {row.RMSE:.2f} | {row.Pred_Std:.2f} |"
        )
    if len(windows):
        lines.extend(["", "## 訓練視窗", ""])
        lines.append("| 折 | 視窗 | 可用 | 驗證 Spearman | 採用 |")
        lines.append("| --- | ---: | --- | ---: | --- |")
        for row in windows.sort_values(["Fold", "Window"]).itertuples(index=False):
            score = "" if row.Validation_Spearman != row.Validation_Spearman else f"{row.Validation_Spearman:.3f}"
            lines.append(f"| {row.Fold} | {row.Window} | {row.Feasible} | {score} | {row.Selected} |")
    if len(costs):
        lines.extend(["", "## 計算成本", ""])
        summary = costs.groupby("Model", as_index=False).agg(
            Fits=("Status", lambda s: int((s == "fit").sum())),
            Failures=("Status", lambda s: int((s == "failed").sum())),
            Carried=("Status", lambda s: int((s == "carried_forward").sum())),
            Fit_Seconds=("Fit_Seconds", "sum"),
            Predict_Seconds=("Predict_Seconds", "sum"),
        )
        lines.append("| 模型 | 成功擬合 | 失敗 | 沿用舊模型 | 訓練秒數 | 推論秒數 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in summary.itertuples(index=False):
            lines.append(
                f"| {row.Model} | {row.Fits} | {row.Failures} | {row.Carried} | {row.Fit_Seconds:,.0f} | {row.Predict_Seconds:,.1f} |"
            )
    lines.extend(["", "## 同一交易規則下的績效", ""])
    lines.append("| 策略 | ROI | 相對等權 | 相對指數 | Sharpe | MDD | 成本 | 平均股票 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in portfolios.itertuples(index=False):
        lines.append(
            f"| {row.Name} | {row.ROI_Pct:+.2f}% | {row.Gap_Vs_Equal_ROI_Pct:+.2f} | {row.Gap_Vs_Index_ROI_Pct:+.2f} | "
            f"{row.Sharpe:.3f} | {row.MDD_Pct:.2f}% | {row.Total_Transaction_Cost:,.0f} | {row.Average_Equity_Weight_Pct:.1f}% |"
        )
    lines.extend([
        "",
        "等權與 LSTM 門檻使用同一個五日再平衡和 0.50 風險上限。買入持有與加權指數是滿部位對照，相對指數的差額不能解讀成模型超额。",
        "",
        "## 每日流程",
        "",
        "這次沒有改 GitHub Actions，也沒有補寫 live 預測。可執行的規則在 `experiments/v2/v28_rules.py`。",
        "",
        "1. 股價、指數與新聞仍用既有研究載入器；正式 CSV 與 Workflow 不寫入。",
        "2. 特徵只用訊號收盤當時已經存在的列，標準化器只用該次訓練列。",
        "3. 當日預測使用登錄表裡目前採用的版本。",
        "4. 五日持有報酬要到標籤結束那根收盤才算成熟，成熟前不能進入訓練。",
        "5. 每次更新寫下版本與驗證分數。",
        "6. 驗證規則沒過就維持上一版；沒有上一版就維持 `mse_b`。",
        "7. 同一個交易日的預測檔已存在就拒絕覆蓋。",
        "8. 成交仍是下一筆收盤。這份回測是探索結果，不是新的獨立樣本外紀錄。",
        "",
        f"預測覆蓋檢查：{json.dumps(coverage, ensure_ascii=False)}。",
        "",
    ])
    assert_v28(V28 / "結果說明.md").write_text("\n".join(lines), encoding="utf-8")


def _coverage(named, control):
    expected = set(pd.to_datetime(control.loc[control.Split == "test", "Signal_Date"]))
    report = {}
    for name, frame in named.items():
        got = set(pd.to_datetime(frame.loc[frame.Split == "test", "Signal_Date"]))
        report[name] = {"test_days": len(got), "matches_control": got == expected}
    return report


def main(train=True):
    before = snapshot()
    kept = _kept_bytes()
    panel, frames, risk, one, five, excess = load_context()
    audit, _positions, folds, index = design_audit()
    if not audit["ok"] or len(folds) != 16:
        raise RuntimeError("既有 walk-forward 不是 16 折或檢查沒有通過")
    if train:
        train_fixed_models(folds, frames, index, one, five, excess)
        train_schedules(folds, frames, index, one, five, excess)
    fixed = _read(V28 / "fixed_predictions.csv")
    if fixed.empty:
        raise FileNotFoundError("還沒有 V2.8 固定模型預測")
    for column in ("Signal_Date", "Execution_Date", "Label_End"):
        fixed[column] = pd.to_datetime(fixed[column], format="mixed")
    choices, window_table = select_windows(fixed)
    selected = stitch_windows(fixed, choices)
    control = _load_control()
    baselines = _baselines(control, panel)
    named = {CONTROL: control, "rolling_selected": selected, "always_long": baselines[baselines.Model == "always_long"], "momentum_20": baselines[baselines.Model == "momentum_20"]}
    for name in [f"rolling_{window}" for window in WINDOWS]:
        part = fixed[fixed.Model == name]
        if len(part):
            named[name] = part
    for name in ARCHITECTURES:
        part = fixed[fixed.Model == name]
        if len(part):
            named[name] = part
    scheduled = _read(V28 / "schedule_predictions.csv")
    if len(scheduled):
        for column in ("Signal_Date", "Execution_Date", "Label_End"):
            scheduled[column] = pd.to_datetime(scheduled[column], format="mixed")
        for name in ("retrain_5", "retrain_1"):
            part = scheduled[scheduled.Model == name]
            if len(part):
                named[name] = part
    validation = {CONTROL: _payload(control)}
    for name in ADOPTABLE:
        if name in named and named[name].Fold.nunique() == 16 and named[name].Split.nunique() == 2:
            validation[name] = _payload(named[name])
    adopted, decisions = choose(validation)
    if adopted != CONTROL and adopted not in named:
        raise AssertionError("採用了沒有預測的模型")
    quality = _quality_table(named)
    fold_rows = _fold_table(named)
    test = control[control.Split == "test"]
    portfolios = _books(
        panel, risk, named, adopted,
        pd.Timestamp(test.Execution_Date.min()), pd.Timestamp(test.Label_End.max()),
    )
    # Saved legacy ROI used a different information timing and global tau.
    # It is not an expected value for this corrected causal execution protocol.
    costs = pd.concat([
        _read(V28 / "fixed_cost.csv"),
        _read(V28 / "schedule_cost.csv"),
    ], ignore_index=True)
    coverage = _coverage(named, control)
    fixed_out = pd.concat([control, fixed, selected, baselines], ignore_index=True)
    fixed_out.to_csv(assert_v28(V28 / "predictions.csv"), index=False)
    if len(scheduled):
        scheduled.to_csv(assert_v28(V28 / "schedule_predictions.csv"), index=False)
    quality.to_csv(assert_v28(V28 / "prediction_quality.csv"), index=False)
    fold_rows.to_csv(assert_v28(V28 / "fold_quality.csv"), index=False)
    window_table.to_csv(assert_v28(V28 / "window_choices.csv"), index=False)
    portfolios.to_csv(assert_v28(V28 / "strategy_metrics.csv"), index=False)
    costs.to_csv(assert_v28(V28 / "training_cost.csv"), index=False)
    _write_report(adopted, decisions, quality, portfolios, window_table, costs, coverage)
    if snapshot() != before:
        raise AssertionError("V2.8 改動了正式檔或第五階段預測")
    for path, content in kept.items():
        if not path.exists() or path.read_bytes() != content:
            raise AssertionError(f"V2.8 覆蓋了既有 V2 結果：{path}")
    print(json.dumps({"adopted_signal": adopted}, ensure_ascii=False), flush=True)
    return adopted, portfolios


if __name__ == "__main__":
    main(train=True)
