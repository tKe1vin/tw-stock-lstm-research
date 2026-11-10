"""Walk-forward LSTM horizons aligned with a five-day, next-close execution.

The official notebooks, the stored walk-forward predictions, and the historical
sentiment file are read-only. New predictions are written only under
experiments/results/signal_models and are tagged retrained_walk_forward.
The 2022-2026 path has already been studied, so every portfolio number from
it is exploratory and is not a new independent test.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from experiments.calendar_rules import apply_rule
from experiments.risk_experiment import (
    BATCH_SIZE,
    BROKER_FEE_RATE,
    EPOCHS,
    FEATURES,
    INITIAL_CAPITAL,
    MARKET,
    MIN_TRAIN_SAMPLES,
    PATIENCE,
    RESULT_DIR,
    RUN,
    SEED,
    SELECTION_END,
    SELL_TAX_RATE,
    TIME_STEP,
    TICKERS,
    VALIDATION_DAYS,
    build_frames,
    load_panel,
    load_sentiment,
    max_drawdown,
    rebalance,
    research_price_source,
    sharpe_ratio,
)

ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_PATH = Path(__file__).resolve().parent / "data" / "sentiment_experimental.csv"
OUT = RESULT_DIR / "signal_models"
STORED_PREDICTIONS = RESULT_DIR / "walk_forward_predictions.csv"
OFFICIAL_SENTIMENT = ROOT / "data" / "sentiment_features_2026.csv"

# Executable holding period: signal uses the close at position i, the first
# later close we can trade is i+1, and a five-session hold ends at i+6.
LABEL_GAP = 6
HOLD_SESSIONS = 5
MIN_FINAL_FOLD = 20
MODELS = ("A", "B", "C", "D")
# Tie break only. Higher validation rank correlation wins; this order is not a ROI search.
TIE_BREAK = ("D", "C", "B", "A")
ORIGIN = "retrained_walk_forward"


def assert_experiment_output(path):
    resolved = Path(path).resolve()
    if "signal_models" not in resolved.parts or resolved == STORED_PREDICTIONS.resolve():
        raise RuntimeError(f"拒絕寫入既有預測或正式資料：{resolved}")
    return resolved


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def executable_labels(close):
    """Percent returns a next-close fill can actually earn.

    one-day label at i uses close[i+1] / close[i] - 1. That move is finished
    before a next-close order is filled, so a new entry does not earn it.
    five-day label at i uses close[i+6] / close[i+1] - 1, the five sessions
    after the fill.
    """
    close = close[TICKERS].astype(float)
    one = (close.shift(-1) / close - 1.0) * 100.0
    five = (close.shift(-LABEL_GAP) / close.shift(-1) - 1.0) * 100.0
    excess = five.sub(five.mean(axis=1), axis=0)
    return one, five, excess


def feature_mask(frames):
    mask = np.ones(len(frames[TICKERS[0]]), dtype=bool)
    for ticker in TICKERS:
        finite = np.isfinite(frames[ticker][FEATURES].to_numpy(dtype=float)).all(axis=1)
        good = np.zeros(len(finite), dtype=bool)
        for end in range(TIME_STEP - 1, len(finite)):
            good[end] = bool(finite[end - TIME_STEP + 1:end + 1].all())
        mask &= good
    return mask


def usable_positions(index, mask, one, five, selection_end=SELECTION_END):
    """Signal positions whose five-day executable label ends on or before the cutoff."""
    positions = []
    for i in range(len(index) - LABEL_GAP):
        if index[i + LABEL_GAP] > selection_end:
            continue
        if not mask[i]:
            continue
        if not np.isfinite(one.iloc[i].to_numpy(dtype=float)).all():
            continue
        if not np.isfinite(five.iloc[i].to_numpy(dtype=float)).all():
            continue
        positions.append(i)
    return np.asarray(positions, dtype=int)


def build_folds(
    positions,
    label_gap=LABEL_GAP,
    min_train=MIN_TRAIN_SAMPLES,
    val_len=VALIDATION_DAYS,
    test_len=VALIDATION_DAYS,
    min_final=MIN_FINAL_FOLD,
):
    """Non-overlapping test blocks. Training and validation labels end before the next block."""
    pos = np.asarray(positions, dtype=int)
    folds = []
    cursor = 0
    started = False
    while cursor < len(pos):
        remaining = pos[cursor:]
        if len(remaining) >= test_len:
            take = test_len
        elif started and len(remaining) >= min_final:
            take = len(remaining)
        elif not started:
            cursor += 1
            continue
        else:
            break
        test = remaining[:take]
        test_start = int(test[0])
        prior = pos[pos < test_start]
        val_ok = prior[prior + label_gap < test_start]
        if len(val_ok) < val_len + min_train:
            if started:
                break
            cursor += 1
            continue
        val = val_ok[-val_len:]
        val_start = int(val[0])
        train = pos[pos + label_gap < val_start]
        if len(train) < min_train:
            if started:
                break
            cursor += 1
            continue
        if int(train[-1]) + label_gap >= val_start or int(val[-1]) + label_gap >= test_start:
            raise AssertionError("標籤間隔沒有把未來報酬擋在下一個區間之外")
        if set(train) & set(val) or set(val) & set(test) or set(train) & set(test):
            raise AssertionError("訓練、驗證、測試的訊號日重疊")
        folds.append({
            "train": train.astype(int),
            "val": val.astype(int),
            "test": test.astype(int),
        })
        started = True
        cursor += take
    return folds


def overlap_report(positions):
    pos = np.asarray(positions, dtype=int)
    if len(pos) < 2:
        steps = np.array([])
    else:
        steps = np.diff(pos)
    return {
        "usable_signals": int(len(pos)),
        "adjacent_signals_one_session_apart": int((steps == 1).sum()) if len(steps) else 0,
        "feature_window_overlap_sessions": TIME_STEP - 1,
        "executable_label_overlap_sessions": HOLD_SESSIONS,
        "nonoverlapping_feature_windows": int(np.ceil(len(pos) / TIME_STEP)),
        "nonoverlapping_five_day_labels": int(np.ceil(len(pos) / HOLD_SESSIONS)),
    }


def scaler_rows(positions):
    rows = [np.arange(int(p) - TIME_STEP + 1, int(p) + 1) for p in positions]
    if not rows:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(rows))


def assert_scaler_does_not_see_future(train_positions, val_positions, test_positions):
    rows = scaler_rows(train_positions)
    boundary = int(np.min(val_positions))
    if len(rows) and int(rows.max()) >= boundary:
        raise AssertionError("標準化器使用了驗證區間的列")
    future = set(val_positions).union(test_positions)
    if future.intersection(rows):
        raise AssertionError("標準化器的列落在驗證或測試訊號上")
    return rows


def choose_model(validation_spearman):
    """Pick a model with validation rank correlation only. Portfolio ROI is not an input."""
    if set(validation_spearman) != set(MODELS):
        raise ValueError("驗證排序指標必須恰好涵蓋 A 到 D")
    if not np.isfinite([float(validation_spearman[name]) for name in MODELS]).all():
        raise ValueError("Validation model-selection scores must be finite")
    ranked = sorted(
        MODELS,
        key=lambda name: (float(validation_spearman[name]), TIE_BREAK.index(name) * -1),
        reverse=True,
    )
    return ranked[0]


def rank_metrics(predicted, actual):
    """Mean daily five-stock Spearman on complete finite cross-sections.

    Ties use average ranks; a constant prediction/actual row contributes zero.
    Top-k diagnostics retain deterministic stable input-column tie ordering.
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if (predicted.ndim != 2 or actual.ndim != 2 or predicted.shape != actual.shape
            or predicted.shape[0] == 0 or predicted.shape[1] != len(TICKERS)):
        raise ValueError("Rank metrics require matching nonempty date-by-five-stock arrays")
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("Rank predictions and actual returns must be finite")
    spearmans = []
    top1 = []
    top2 = []
    for row_pred, row_actual in zip(predicted, actual):
        if np.std(row_pred) < 1e-12 or np.std(row_actual) < 1e-12:
            spearmans.append(0.0)
        else:
            spearmans.append(float(pd.Series(row_pred).corr(pd.Series(row_actual), method="spearman")))
        pred_order = np.argsort(row_pred, kind="mergesort")
        actual_order = np.argsort(row_actual, kind="mergesort")
        top1.append(float(pred_order[-1] == actual_order[-1]))
        top2.append(len(set(pred_order[-2:]).intersection(actual_order[-2:])) / 2.0)
    return {
        "Cross_Section_Spearman": float(np.mean(spearmans)),
        "Top1_Hit_Rate": float(np.mean(top1)),
        "Top2_Overlap": float(np.mean(top2)),
        "Days": int(len(predicted)),
    }


def flat_skill(predicted, actual):
    """Pooled diagnostics without dropping observations or treating zero as down.

    Direction compares exact negative/zero/positive signs for all observations.
    Pearson/Spearman here are pooled diagnostics, not daily cross-section ranks.
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if (predicted.shape != actual.shape or predicted.ndim not in (1, 2)
            or predicted.size == 0):
        raise ValueError("Pooled metrics require matching nonempty one- or two-dimensional arrays")
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("Pooled predictions and actual returns must be finite")
    pred, act = predicted.ravel(), actual.ravel()
    pearson = float(pd.Series(pred).corr(pd.Series(act), method="pearson")) if len(pred) > 2 else 0.0
    spearman = float(pd.Series(pred).corr(pd.Series(act), method="spearman")) if len(pred) > 2 else 0.0
    if pearson != pearson:
        pearson = 0.0
    if spearman != spearman:
        spearman = 0.0
    error = pred - act
    return {
        "Direction_Accuracy": float(np.mean(np.sign(pred) == np.sign(act))),
        "Pearson": pearson,
        "Spearman": spearman,
        "MAE": float(np.mean(np.abs(error))) if len(error) else 0.0,
        "RMSE": float(np.sqrt(np.mean(error ** 2))) if len(error) else 0.0,
        "Pred_Mean": float(pred.mean()) if len(pred) else 0.0,
        "Pred_Std": float(pred.std(ddof=1)) if len(pred) > 1 else 0.0,
        "Actual_Mean": float(act.mean()) if len(act) else 0.0,
        "Actual_Std": float(act.std(ddof=1)) if len(act) > 1 else 0.0,
        "N": int(len(pred)),
    }


def load_context():
    if not SENTIMENT_PATH.exists():
        raise FileNotFoundError("缺少第四階段的實驗情緒副本")
    panel = load_panel(research_price_source())
    sentiment = load_sentiment(SENTIMENT_PATH, panel.index)
    if sentiment.loc[pd.Timestamp("2025-08-01"), TICKERS].isna().any():
        raise ValueError("實驗情緒副本仍缺少 2025-08-01，拒絕用 0 補上")
    frames, risk = build_frames(panel, sentiment)
    one, five, excess = executable_labels(panel)
    return panel, frames, risk, one, five, excess


def design_audit():
    panel, frames, risk, one, five, excess = load_context()
    del risk
    mask = feature_mask(frames)
    positions = usable_positions(frames[TICKERS[0]].index, mask, one, five)
    folds = build_folds(positions)
    index = frames[TICKERS[0]].index
    stored = pd.read_csv(STORED_PREDICTIONS, parse_dates=["Signal_Date"])
    stored_signals = set(pd.to_datetime(stored.Signal_Date).dt.normalize())
    new_signals = []
    for fold in folds:
        for pos in fold["test"]:
            day = pd.Timestamp(index[int(pos)])
            if day not in stored_signals:
                new_signals.append(str(day.date()))
    label_ok = True
    probe = int(positions[len(positions) // 2])
    close = panel[TICKERS]
    expected = (close.iloc[probe + LABEL_GAP] / close.iloc[probe + 1] - 1.0) * 100.0
    if not np.allclose(five.iloc[probe].to_numpy(dtype=float), expected.to_numpy(dtype=float)):
        label_ok = False
    changed = close.copy()
    changed.iloc[probe + LABEL_GAP + 1:] = changed.iloc[probe + LABEL_GAP + 1:] * 1.5
    rebound = (changed.iloc[probe + LABEL_GAP] / changed.iloc[probe + 1] - 1.0) * 100.0
    if not np.allclose(rebound.to_numpy(dtype=float), expected.to_numpy(dtype=float)):
        label_ok = False
    for fold in folds:
        assert_scaler_does_not_see_future(fold["train"], fold["val"], fold["test"])
    report = {
        "ok": bool(label_ok and len(folds) >= 4 and len(positions) > MIN_TRAIN_SAMPLES + VALIDATION_DAYS),
        "study": "exploratory_not_a_new_independent_test",
        "prices_available": ["Close"],
        "execution": "next_session_close",
        "same_close_assumed": False,
        "why_next_day_target_does_not_match_five_day_rebalance": (
            "原始標籤是訊號收盤到下一收盤的報酬，新單要到下一筆收盤才買得到，"
            "因此賺不到這一段；策略實際持有的是成交後五個交易日。"
        ),
        "label_definitions": {
            "A": "close[i+1]/close[i]-1，原始隔日標籤，成交前已經實現",
            "B": "close[i+6]/close[i+1]-1，成交後五個交易日",
            "C": "B 減去五檔同期等權報酬",
            "D": "以 B 的橫斷面順序做配對排序損失，網路深度與 A 相同",
        },
        "embargo_sessions": LABEL_GAP,
        "shared_split": "四個模型使用同一組測試日；間隔依照最長的五日可成交標籤",
        "selection_metric": "validation_cross_section_spearman_versus_executable_five_day_return",
        "selection_does_not_use_portfolio_roi": True,
        "observed_window_2026_03_26_to_2026_11_04_used_for_selection": False,
        "label_end_capped_at": str(SELECTION_END.date()),
        "usable_signals": int(len(positions)),
        "folds": len(folds),
        "first_test_signal": str(index[int(folds[0]["test"][0])].date()) if folds else None,
        "last_test_signal": str(index[int(folds[-1]["test"][-1])].date()) if folds else None,
        "train_sizes": [int(len(fold["train"])) for fold in folds],
        "overlap": overlap_report(positions),
        "label_identity_ok": label_ok,
        "stored_prediction_signals": int(len(stored_signals)),
        "test_signals_absent_from_stored_file": sorted(set(new_signals)),
        "august_predictions_will_be_retrained_not_copied": True,
        "network_depth_unchanged": "LSTM(50)+Dropout(0.2)+LSTM(50)+Dropout(0.2)+Dense(1)",
    }
    return report, positions, folds, index


def _matrix(values, scaler, positions):
    windows = [scaler.transform(values[int(p) - TIME_STEP + 1:int(p) + 1]) for p in positions]
    stacked = np.stack(windows).astype(np.float32)
    if not np.isfinite(stacked).all():
        raise ValueError("視窗含有非有限值")
    return stacked


def _fit_regressor(x_train, y_train, x_val, y_val):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(TIME_STEP, len(FEATURES))),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        validation_data=(x_val, y_val),
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True,
        )],
        verbose=0,
    )
    return model, history


def _pairwise_loss(y_true, y_pred):
    import tensorflow as tf

    better = tf.cast(y_true[:, :, None] > y_true[:, None, :], tf.float32)
    penalty = tf.nn.softplus(-(y_pred[:, :, None] - y_pred[:, None, :]))
    return tf.reduce_sum(penalty * better) / (tf.reduce_sum(better) + 1e-6)


def _fit_rank_model(x_train, y_train, x_val, y_val):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    inputs = tf.keras.Input(shape=(len(TICKERS), TIME_STEP, len(FEATURES)))
    stock_net = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(TIME_STEP, len(FEATURES))),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    pieces = [
        stock_net(tf.keras.layers.Lambda(lambda batch, k=k: batch[:, k, :, :])(inputs))
        for k in range(len(TICKERS))
    ]
    outputs = tf.keras.layers.Concatenate()(pieces)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss=_pairwise_loss)
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        validation_data=(x_val, y_val),
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True,
        )],
        verbose=0,
    )
    return model, history


def _rows_for(model_name, index, positions, ticker, predicted, one, five, excess, split, fold_id):
    rows = []
    for pos, value in zip(positions, predicted):
        pos = int(pos)
        rows.append({
            "Model": model_name,
            "Split": split,
            "Fold": fold_id,
            "Signal_Date": index[pos],
            "Execution_Date": index[pos + 1],
            "Label_End": index[pos + LABEL_GAP],
            "Ticker": ticker,
            "Predicted": float(value),
            "Actual_1d_Pct": float(one.iloc[pos][ticker]),
            "Actual_5d_Pct": float(five.iloc[pos][ticker]),
            "Actual_Excess_5d_Pct": float(excess.iloc[pos][ticker]),
            "Origin": ORIGIN,
        })
    return rows


def train_models(folds, frames, index, one, five, excess):
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    rows = []
    fold_rows = []
    for fold_id, fold in enumerate(folds, start=1):
        print(
            f"[HZ] fold {fold_id} test {index[int(fold['test'][0])].date()} -> {index[int(fold['test'][-1])].date()}",
            flush=True,
        )
        scaled = {}
        for ticker in TICKERS:
            values = frames[ticker][FEATURES].to_numpy(dtype=float)
            scaler = MinMaxScaler().fit(values[assert_scaler_does_not_see_future(
                fold["train"], fold["val"], fold["test"],
            )])
            scaled[ticker] = {
                split: _matrix(values, scaler, fold[split]) for split in ("train", "val", "test")
            }
        targets = {
            "A": one,
            "B": five,
            "C": excess,
        }
        for model_name, labels in targets.items():
            for ticker in TICKERS:
                tf.keras.backend.clear_session()
                y_train = labels.iloc[fold["train"]][ticker].to_numpy(dtype=np.float32)
                y_val = labels.iloc[fold["val"]][ticker].to_numpy(dtype=np.float32)
                model, history = _fit_regressor(
                    scaled[ticker]["train"], y_train, scaled[ticker]["val"], y_val,
                )
                val_pred = model.predict(scaled[ticker]["val"], verbose=0).ravel()
                test_pred = model.predict(scaled[ticker]["test"], verbose=0).ravel()
                if not np.isfinite(val_pred).all() or not np.isfinite(test_pred).all():
                    raise ValueError(f"{model_name} {ticker} 產生非有限預測")
                rows.extend(_rows_for(
                    model_name, index, fold["val"], ticker, val_pred, one, five, excess, "validation", fold_id,
                ))
                rows.extend(_rows_for(
                    model_name, index, fold["test"], ticker, test_pred, one, five, excess, "test", fold_id,
                ))
                fold_rows.append({
                    "Model": model_name,
                    "Ticker": ticker,
                    "Fold": fold_id,
                    "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
                    "epochs_ran": len(history.history["val_loss"]),
                })
                del model
        tf.keras.backend.clear_session()
        rank_sets = {}
        for split in ("train", "val", "test"):
            rank_sets[split] = np.stack([scaled[ticker][split] for ticker in TICKERS], axis=1)
        y_train = five.iloc[fold["train"]][TICKERS].to_numpy(dtype=np.float32)
        y_val = five.iloc[fold["val"]][TICKERS].to_numpy(dtype=np.float32)
        rank_model, history = _fit_rank_model(rank_sets["train"], y_train, rank_sets["val"], y_val)
        for split, positions in (("validation", fold["val"]), ("test", fold["test"])):
            key = "val" if split == "validation" else "test"
            predicted = rank_model.predict(rank_sets[key], verbose=0)
            if predicted.shape != (len(positions), len(TICKERS)) or not np.isfinite(predicted).all():
                raise ValueError("排序模型的預測形狀不正確")
            for ticker_index, ticker in enumerate(TICKERS):
                rows.extend(_rows_for(
                    "D", index, positions, ticker, predicted[:, ticker_index], one, five, excess, split, fold_id,
                ))
        fold_rows.append({
            "Model": "D",
            "Ticker": "SHARED",
            "Fold": fold_id,
            "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
            "epochs_ran": len(history.history["val_loss"]),
        })
        del rank_model
    predictions = pd.DataFrame(rows)
    for column in ("Signal_Date", "Execution_Date", "Label_End"):
        predictions[column] = pd.to_datetime(predictions[column]).dt.normalize()
    if predictions.Split.eq("test").any() and predictions.loc[predictions.Split.eq("test"), "Signal_Date"].max() > SELECTION_END:
        raise AssertionError("測試訊號超過 2026-03-25")
    if set(predictions.Origin) != {ORIGIN}:
        raise AssertionError("新預測沒有和既有每日預測分開標記")
    return predictions, pd.DataFrame(fold_rows)


def _day_matrix(frame, column):
    days = []
    actual = []
    predicted = []
    folds = []
    for (fold_id, day), part in frame.groupby(["Fold", "Signal_Date"], sort=True):
        part = part.set_index("Ticker").loc[TICKERS]
        days.append(pd.Timestamp(day))
        folds.append(int(fold_id))
        predicted.append(part.Predicted.to_numpy(dtype=float))
        actual.append(part[column].to_numpy(dtype=float))
    return days, folds, np.vstack(predicted), np.vstack(actual)


def fold_model_choices(predictions):
    """Choose each fold's model from that fold's validation rank skill, never from its test ROI."""
    choices = {}
    validation = predictions[predictions.Split == "validation"]
    test = predictions[predictions.Split == "test"]
    for fold_id, part in validation.groupby("Fold"):
        fold_id = int(fold_id)
        scores = {}
        for model_name in MODELS:
            model_part = part[part.Model == model_name]
            _, _, predicted, actual = _day_matrix(model_part, "Actual_5d_Pct")
            scores[model_name] = rank_metrics(predicted, actual)["Cross_Section_Spearman"]
        test_start = pd.Timestamp(test.loc[test.Fold == fold_id, "Signal_Date"].min())
        if pd.Timestamp(part.Label_End.max()) >= test_start:
            raise AssertionError(f"第 {fold_id} 折的驗證標籤碰到測試訊號")
        choices[fold_id] = {"choice": choose_model(scores), "validation_spearman": scores}
    return choices


def stitch_selected(predictions, choices):
    parts = []
    test = predictions[predictions.Split == "test"]
    for fold_id, info in choices.items():
        parts.append(test[(test.Fold == fold_id) & (test.Model == info["choice"])])
    selected = pd.concat(parts, ignore_index=True)
    if selected.duplicated(["Signal_Date", "Ticker"]).any():
        raise AssertionError("選出的預測在同一天重複")
    return selected


def quality_table(predictions):
    rows = []
    stability = []
    test = predictions[predictions.Split == "test"]
    validation = predictions[predictions.Split == "validation"]
    selection_scores = {}
    for model_name in MODELS:
        part = test[test.Model == model_name]
        own_column = {"A": "Actual_1d_Pct", "B": "Actual_5d_Pct", "C": "Actual_Excess_5d_Pct", "D": "Actual_5d_Pct"}
        own = flat_skill(part.Predicted, part[own_column[model_name]])
        holding = flat_skill(part.Predicted, part.Actual_5d_Pct)
        _, fold_ids, pred_own, actual_own = _day_matrix(part, own_column[model_name])
        _, _, pred_hold, actual_hold = _day_matrix(part, "Actual_5d_Pct")
        rank_own = rank_metrics(pred_own, actual_own)
        rank_hold = rank_metrics(pred_hold, actual_hold)
        rows.append({
            "Model": model_name,
            "Split": "test",
            "Study": "exploratory_not_a_new_independent_test",
            **{f"Own_{key}": value for key, value in own.items()},
            **{f"OwnRank_{key}": value for key, value in rank_own.items()},
            **{f"Holding_{key}": value for key, value in holding.items()},
            **{f"HoldingRank_{key}": value for key, value in rank_hold.items()},
        })
        val_part = validation[validation.Model == model_name]
        _, _, val_pred, val_actual = _day_matrix(val_part, "Actual_5d_Pct")
        val_rank = rank_metrics(val_pred, val_actual)
        selection_scores[model_name] = val_rank["Cross_Section_Spearman"]
        rows[-1]["Validation_Holding_Spearman"] = val_rank["Cross_Section_Spearman"]
        by_fold = pd.DataFrame({"Fold": fold_ids, "Pred": list(pred_hold), "Actual": list(actual_hold)})
        for fold_id, chunk in by_fold.groupby("Fold"):
            fold_rank = rank_metrics(np.vstack(chunk.Pred), np.vstack(chunk.Actual))
            stability.append({
                "Model": model_name,
                "Fold": int(fold_id),
                "Holding_Spearman": fold_rank["Cross_Section_Spearman"],
                "Top1_Hit_Rate": fold_rank["Top1_Hit_Rate"],
            })
    return pd.DataFrame(rows), pd.DataFrame(stability), selection_scores


def run_executable(panel, risk, predictions, weight_name, start, end):
    """Trade at the close after the signal. A missing signal keeps the old shares."""
    index = panel.index[(panel.index >= start) & (panel.index <= end)]
    by_signal = {}
    if predictions is not None and len(predictions):
        for signal, day in predictions.groupby("Signal_Date", sort=True):
            by_signal[pd.Timestamp(signal)] = day.set_index("Ticker").loc[TICKERS]
    location = {pd.Timestamp(day): i for i, day in enumerate(panel.index)}
    shares = np.zeros(len(TICKERS))
    cash = INITIAL_CAPITAL
    previous = None
    rows = []
    anchor = location[pd.Timestamp(index[0])]
    for day in index:
        day = pd.Timestamp(day)
        price = panel.loc[day, TICKERS].to_numpy(dtype=float)
        held = shares.copy()
        change = np.zeros(len(TICKERS)) if previous is None else held * (price - previous)
        old_values = held * price
        pre_nav = float(old_values.sum() + cash)
        flag = bool(risk.loc[day, "Risk_Off"])
        due = (location[day] - anchor) % HOLD_SESSIONS == 0
        signal = panel.index[location[day] - 1] if location[day] > 0 else None
        info = by_signal.get(pd.Timestamp(signal)) if signal is not None else None
        traded = False
        skipped = False
        ledger = {"Total_Cost": 0.0}
        pred = np.full(len(TICKERS), np.nan)
        weights = old_values / pre_nav if pre_nav else np.zeros(len(TICKERS))
        if due and weight_name == "buy_hold" and previous is None:
            weights = np.full(len(TICKERS), 0.20)
            target_values, cash, ledger = rebalance(old_values, cash, weights)
            shares = target_values / price
            traded = True
        elif due and weight_name != "buy_hold":
            if info is None and weight_name in {"tilt", "stored_rule_a"}:
                skipped = True
            else:
                if info is not None:
                    pred = info.Predicted.to_numpy(dtype=float) if "Predicted" in info else info.Predicted_Return_Pct.to_numpy(dtype=float)
                if weight_name == "tilt":
                    weights = apply_rule("D", pred, np.zeros(len(TICKERS)), flag, np.ones(len(TICKERS)))
                elif weight_name == "equal_risk":
                    weights = apply_rule("B", pred, np.zeros(len(TICKERS)), flag, np.ones(len(TICKERS)))
                elif weight_name == "always_full":
                    weights = np.full(len(TICKERS), 0.20)
                elif weight_name == "stored_rule_a":
                    sent = info.Daily_Sentiment.to_numpy(dtype=float)
                    weights = apply_rule("A", pred, sent, flag, np.ones(len(TICKERS)))
                elif weight_name == "matched_tilt":
                    weights = apply_rule(
                        "D", pred, np.zeros(len(TICKERS)), flag, np.ones(len(TICKERS)), equalize_gross=True,
                    )
                else:
                    raise ValueError(weight_name)
                target_values, cash, ledger = rebalance(old_values, cash, weights)
                shares = target_values / price
                traded = True
        post_values = shares * price
        nav = float(post_values.sum() + cash)
        liquidation_cost = 0.0
        if day == index[-1]:
            liquidation_cost = float(post_values.sum() * (BROKER_FEE_RATE + SELL_TAX_RATE))
            nav -= liquidation_cost
        denominator = float(post_values.sum() + cash)
        rows.append({
            "Date": day,
            "Trade": traded,
            "Skipped_For_Missing_Prediction": skipped,
            "NAV_Reported": nav,
            "Rebalance_Cost": ledger["Total_Cost"],
            "Liquidation_Cost": liquidation_cost,
            "Equity_Weight": float(post_values.sum() / denominator) if denominator else 0.0,
            "Cash_Weight": float(cash / denominator) if denominator else 1.0,
            "Stock_PnL": float(change.sum()),
            **{f"Target_{ticker}": float(weights[j]) for j, ticker in enumerate(TICKERS)},
            **{f"Trade_Px_{ticker}": float(price[j]) for j, ticker in enumerate(TICKERS)},
        })
        previous = price
    daily = pd.DataFrame(rows)
    identity = (
        INITIAL_CAPITAL
        + float(daily.Stock_PnL.sum())
        - float(daily.Rebalance_Cost.sum())
        - float(daily.Liquidation_Cost.sum())
        - float(daily.NAV_Reported.iloc[-1])
    )
    if abs(identity) > 1e-4:
        raise AssertionError(f"下一收盤成交的淨值恆等式不成立，差額 {identity}")
    return daily


def index_path(panel, start, end):
    dates = panel.index[(panel.index >= start) & (panel.index <= end)]
    entry = float(panel.loc[dates[0], MARKET])
    shares = (INITIAL_CAPITAL / (1 + BROKER_FEE_RATE)) / entry
    cash = INITIAL_CAPITAL - shares * entry * (1 + BROKER_FEE_RATE)
    nav = []
    for i, day in enumerate(dates):
        value = shares * float(panel.loc[day, MARKET]) + cash
        if i == len(dates) - 1:
            value -= shares * float(panel.loc[day, MARKET]) * (BROKER_FEE_RATE + SELL_TAX_RATE)
        nav.append(value)
    values = np.concatenate([[INITIAL_CAPITAL], np.asarray(nav, dtype=float)])
    price_roi = float(panel.loc[dates[-1], MARKET] / entry - 1.0)
    return {
        "ROI_Pct": float((values[-1] / INITIAL_CAPITAL - 1) * 100),
        "Price_ROI_Pct": price_roi * 100,
        "Sharpe": sharpe_ratio(values),
        "MDD_Pct": max_drawdown(values),
        "Total_Transaction_Cost": float(INITIAL_CAPITAL - cash - shares * entry + shares * float(panel.loc[dates[-1], MARKET]) * (BROKER_FEE_RATE + SELL_TAX_RATE)),
    }


def evaluate_portfolios(panel, risk, predictions, stored, stitched):
    from experiments.calendar_rules import book_metrics

    test = predictions[predictions.Split == "test"]
    start = pd.Timestamp(test.Execution_Date.min())
    end = pd.Timestamp(test.Label_End.max())
    if end > SELECTION_END:
        raise AssertionError("回測區間超過選參截止日")
    rows = []
    books = {}
    for model_name in MODELS:
        daily = run_executable(panel, risk, test[test.Model == model_name], "tilt", start, end)
        books[model_name] = daily
        row = book_metrics(daily, model_name, "next_close_every_5_sessions", "through_2026-03-25")
        row["Same_Close_Execution"] = False
        row["Name"] = f"model_{model_name}"
        rows.append(row)
    selected_daily = run_executable(panel, risk, stitched, "tilt", start, end)
    selected_row = book_metrics(selected_daily, "validation_selected", "next_close_every_5_sessions", "through_2026-03-25")
    selected_row["Same_Close_Execution"] = False
    selected_row["Name"] = "validation_selected"
    rows.append(selected_row)
    matched = run_executable(panel, risk, stitched, "matched_tilt", start, end)
    matched_metrics = book_metrics(matched, "validation_selected", "equalized_gross", "through_2026-03-25")
    specs = {
        "equal_risk": "equal_risk",
        "always_full": "always_full",
        "buy_hold": "buy_hold",
    }
    for name, mode in specs.items():
        daily = run_executable(panel, risk, None, mode, start, end)
        row = book_metrics(daily, name, "next_close_every_5_sessions", "through_2026-03-25")
        row["Same_Close_Execution"] = False
        row["Name"] = name
        rows.append(row)
    stored_book = run_executable(panel, risk, stored, "stored_rule_a", start, end)
    stored_metrics = book_metrics(stored_book, "stored_555_signal", "next_close_every_5_sessions", "through_2026-03-25")
    stored_metrics["Same_Close_Execution"] = False
    stored_metrics["Name"] = "stored_555_next_close"
    rows.append(stored_metrics)
    market = index_path(panel, start, end)
    selected_metrics = next(row for row in rows if row["Name"] == "validation_selected")
    return {
        "start": str(start.date()),
        "end": str(end.date()),
        "selected_model": "per_fold_validation_spearman",
        "strategies": rows,
        "selected_minus_equal_risk_pp": selected_metrics["ROI_Pct"] - next(
            row["ROI_Pct"] for row in rows if row["Name"] == "equal_risk"
        ),
        "selected_minus_matched_exposure_pp": selected_metrics["ROI_Pct"] - matched_metrics["ROI_Pct"],
        "matched_exposure_roi_pct": matched_metrics["ROI_Pct"],
        "index": market,
        "published_scheme_g_same_close_roi_pct": 4.966394355279036,
    }, books


def stability_summary(stability):
    out = {}
    for model_name, part in stability.groupby("Model"):
        values = part.Holding_Spearman.to_numpy(dtype=float)
        out[model_name] = {
            "folds": int(len(values)),
            "positive_folds": int((values > 0).sum()),
            "median_spearman": float(np.median(values)),
            "min_spearman": float(np.min(values)),
            "max_spearman": float(np.max(values)),
        }
    return out


def main():
    protected = {
        "stored_predictions": file_sha256(STORED_PREDICTIONS),
        "official_sentiment": file_sha256(OFFICIAL_SENTIMENT),
    }
    audit, positions, folds, index = design_audit()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "design_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if not audit["ok"]:
        raise RuntimeError(f"實驗設計未通過洩漏與樣本檢查：{audit}")
    panel, frames, risk, one, five, excess = load_context()
    del positions, index
    predictions, fold_log = train_models(folds, frames, frames[TICKERS[0]].index, one, five, excess)
    quality, stability, scores = quality_table(predictions)
    choices = fold_model_choices(predictions)
    stitched = stitch_selected(predictions, choices)
    stored = pd.read_csv(STORED_PREDICTIONS, parse_dates=["Signal_Date", "Date"])
    stored["Signal_Date"] = pd.to_datetime(stored.Signal_Date).dt.normalize()
    stored = stored.rename(columns={"Predicted_Return_Pct": "Predicted"})
    portfolio, _books = evaluate_portfolios(panel, risk, predictions, stored, stitched)
    if file_sha256(STORED_PREDICTIONS) != protected["stored_predictions"]:
        raise AssertionError("既有 walk-forward 預測被改寫")
    if file_sha256(OFFICIAL_SENTIMENT) != protected["official_sentiment"]:
        raise AssertionError("原始情緒檔被改寫")
    predictions.to_csv(assert_experiment_output(OUT / "predictions.csv"), index=False)
    fold_log.to_csv(OUT / "training_folds.csv", index=False)
    quality.to_csv(OUT / "prediction_quality.csv", index=False)
    stability.to_csv(OUT / "fold_stability.csv", index=False)
    pd.DataFrame(portfolio["strategies"]).to_csv(OUT / "strategy_metrics.csv", index=False)
    summary = {
        "study": "exploratory_not_a_new_independent_test",
        "selected_model": "per_fold_validation_spearman",
        "fold_choices": {str(key): value for key, value in choices.items()},
        "pooled_validation_spearman_vs_five_day": scores,
        "selection_input": "validation cross-sectional Spearman versus the executable five-day return",
        "quality": quality.to_dict(orient="records"),
        "stability": stability_summary(stability),
        "portfolio": portfolio,
        "design": {
            "folds": audit["folds"],
            "usable_signals": audit["usable_signals"],
            "new_test_signals": audit["test_signals_absent_from_stored_file"],
            "execution": audit["execution"],
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "fold_choices": {str(key): value["choice"] for key, value in choices.items()},
        "validation_spearman": scores,
        "folds": audit["folds"],
    }, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    main()
