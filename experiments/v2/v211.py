"""Forward tracker for the frozen rolling_selected candidate.

It reads prices and sentiment that are already on disk and does not download
them. An immutable live file is published only for today's completed session.
Saved weights are not retrained. It does not backfill, reselect on test ROI,
or replace the official 555.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from experiments.horizon_experiment import (
    FEATURES,
    LABEL_GAP,
    TIME_STEP,
    _matrix,
    feature_mask,
    load_context,
)
from experiments.risk_experiment import SEED, TICKERS
from experiments.v2.books import RISK, threshold_weights
from experiments.v2.guard import OUT, ROOT, sha256
from experiments.v2.v28 import _prepare
from experiments.v2.v28_rules import ES_LEN, classification_scores
from experiments.v2.v29 import CorrectedFitter
from scripts.common import TZ, current_completed_session, sessions
from scripts.model_bundle import verify_bundle
from scripts.ledger_integrity import (
    append_plan, apply_append, record_ledger_sources, validate_ledger_sources,
)

V211 = OUT / "v211"
LIVE = V211 / "daily"
WEIGHT_DIR = V211 / "weights"
_ledger = os.environ.get("TW_LSTM_V211_DIR")
if _ledger:
    V211 = Path(_ledger)
    LIVE = V211 / "daily"
    WEIGHT_DIR = Path(os.environ["TW_LSTM_WEIGHT_DIR"]) if os.environ.get("TW_LSTM_WEIGHT_DIR") else (V211 / "weights")
MODEL_VERSION = "rolling_selected_fold16_window756_carried"
FROZEN_WINDOW = 756
FROZEN_WINDOW_SOURCE = "v29c window_choices fold 16 validation Spearman 0.3067"
TAIPEI = ZoneInfo("Asia/Taipei")
BLOCKED = ("v26", "v27", "v28", "v29", "v29b", "v29c", "v210")


def assert_v211(path):
    resolved = Path(path).resolve()
    root = V211.resolve()
    for name in BLOCKED:
        saved = (OUT / name).resolve()
        if resolved == saved or saved in resolved.parents:
            raise RuntimeError(f"V2.11 不能改寫既有結果：{resolved}")
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"V2.11 只能寫入 {root}：{resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def training_split(index, mask, five):
    """Last 756 matured signals before a 60-day early-stop block. The last bar is not a label."""
    last = len(index) - 1
    positions = []
    for i in range(last):
        if i + LABEL_GAP > last:
            break
        if not mask[i] or not np.isfinite(five.iloc[i].to_numpy(dtype=float)).all():
            continue
        positions.append(i)
    positions = np.asarray(positions, dtype=int)
    if len(positions) < FROZEN_WINDOW + ES_LEN:
        raise AssertionError("成熟標籤不足以重建 756 日滾動視窗")
    early = positions[-ES_LEN:]
    prior = positions[positions + LABEL_GAP < int(early[0])]
    if len(prior) < FROZEN_WINDOW:
        raise AssertionError("早期停止之前沒有 756 個可訓練訊號")
    train = prior[-FROZEN_WINDOW:]
    if int(train.max()) >= int(early.min()) or int(early.max()) >= last:
        raise AssertionError("訓練或早期停止碰到了最後一根價格")
    return train, early


def publication_decision(now, price_dates):
    """Publish only today's completed session, and only when its close is already stored."""
    if now.tzinfo is None:
        raise ValueError("時間必須帶時區")
    completed = pd.Timestamp(current_completed_session(now)).normalize()
    today = pd.Timestamp(now.astimezone(TAIPEI).date())
    dates = set(pd.Timestamp(day).normalize() for day in price_dates)
    if completed not in dates:
        return {"publish": False, "reason": "completed_session_missing_from_prices", "completed_session": str(completed.date())}
    if completed != today:
        return {"publish": False, "reason": "would_backfill_an_earlier_session", "completed_session": str(completed.date())}
    return {"publish": True, "reason": "today_close_is_stored", "completed_session": str(completed.date())}


def _scaler_state(scaler):
    return {
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        "scale": scaler.scale_.tolist(),
        "min": scaler.min_.tolist(),
    }


def _restore_scaler(state):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.data_min_ = np.asarray(state["data_min"], dtype=float)
    scaler.data_max_ = np.asarray(state["data_max"], dtype=float)
    scaler.scale_ = np.asarray(state["scale"], dtype=float)
    scaler.min_ = np.asarray(state["min"], dtype=float)
    scaler.n_features_in_ = len(state["data_min"])
    scaler.feature_range = (0.0, 1.0)
    return scaler


def fit_and_save(context=None):
    """Fit the frozen 756-day window once. Existing weight files are not replaced."""
    card = WEIGHT_DIR / "model_card.json"
    if card.exists():
        return json.loads(card.read_text(encoding="utf-8"))
    panel, frames, _risk, _one, five, _excess = context or load_context()
    index = frames[TICKERS[0]].index
    train, early = training_split(index, feature_mask(frames), five)
    fitter = CorrectedFitter()
    weights = {}
    scalers = {}
    tau = {}
    for ticker in TICKERS:
        values, scaler = _prepare(frames, ticker, train, early, early)
        model, _history, _elapsed = fitter.fit(
            _matrix(values, scaler, train),
            five.iloc[train][ticker].to_numpy(np.float32),
            _matrix(values, scaler, early),
            five.iloc[early][ticker].to_numpy(np.float32),
            2,
            50,
            None,
        )
        if abs(float(model.optimizer.learning_rate) - 0.001) > 1e-6:
            raise AssertionError(f"{ticker} 學習率不是 0.001")
        predicted = model.predict(_matrix(values, scaler, early), verbose=0).ravel()
        tau[ticker] = float(np.median(np.abs(predicted)))
        weights[ticker] = [item.astype(np.float32) for item in model.get_weights()]
        scalers[ticker] = _scaler_state(scaler)
        print(f"[V211] fitted {ticker} tau {tau[ticker]:.4f}", flush=True)
    assert_v211(WEIGHT_DIR / "weights.npz")
    arrays = {}
    for ticker in TICKERS:
        for i, item in enumerate(weights[ticker]):
            arrays[f"{ticker}_{i}"] = item
    np.savez(WEIGHT_DIR / "weights.npz", **arrays)
    document = {
        "model_version": MODEL_VERSION,
        "window": FROZEN_WINDOW,
        "window_source": FROZEN_WINDOW_SOURCE,
        "layers": 2,
        "units": 50,
        "dropout": 0.2,
        "seed": SEED,
        "time_step": TIME_STEP,
        "features": list(FEATURES),
        "learning_rate": 0.001,
        "early_stop_sessions": ES_LEN,
        "train_start": str(index[int(train[0])].date()),
        "train_end": str(index[int(train[-1])].date()),
        "early_stop_start": str(index[int(early[0])].date()),
        "early_stop_end": str(index[int(early[-1])].date()),
        "data_through": str(index[-1].date()),
        "train_dates": [str(index[int(pos)].date()) for pos in train],
        "tau": tau,
        "scalers": scalers,
        "weight_counts": {ticker: len(weights[ticker]) for ticker in TICKERS},
        "selected_by_test_roi": False,
        "official_555_replacement": False,
    }
    assert_v211(card).write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    document["weight_sha256"] = sha256(WEIGHT_DIR / "weights.npz")
    card.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document


def _load_weights(card):
    store = np.load(WEIGHT_DIR / "weights.npz")
    weights = {}
    for ticker in TICKERS:
        count = int(card["weight_counts"][ticker])
        weights[ticker] = [store[f"{ticker}_{i}"] for i in range(count)]
    return weights


def predict_rows(card, context, signal_date):
    verify_bundle(weight_dir=WEIGHT_DIR, card=card)
    if (list(FEATURES) != card["features"] or TIME_STEP != card["time_step"]
            or (card["layers"], card["units"], card["dropout"]) != (2, 50, 0.2)
            or card["model_version"] != MODEL_VERSION):
        raise ValueError("runtime feature order or network architecture differs from the frozen bundle")
    panel, frames, risk, _one, five, _excess = context
    index = frames[TICKERS[0]].index
    signal = pd.Timestamp(signal_date).normalize()
    if signal not in index:
        raise RuntimeError(f"價格沒有 {signal.date()}")
    position = int(index.get_loc(signal))
    train_dates = [pd.Timestamp(day) for day in card["train_dates"]]
    train = np.asarray([int(index.get_loc(day)) for day in train_dates], dtype=int)
    if int(train.max()) >= position:
        raise AssertionError("保存的訓練區間碰到了訊號日")
    values = {}
    for ticker in TICKERS:
        raw = frames[ticker][list(FEATURES)].to_numpy(dtype=float)
        if not np.isfinite(raw[position - TIME_STEP + 1:position + 1]).all():
            raise RuntimeError(f"{ticker} 在 {signal.date()} 的特徵不完整，停止產生訊號")
        values[ticker] = raw
    fitter = CorrectedFitter()
    weights = _load_weights(card)
    output = {}
    for ticker in TICKERS:
        scaler = _restore_scaler(card["scalers"][ticker])
        key, model = fitter._fitter._model(2, 50, len(FEATURES))
        model.set_weights(weights[ticker])
        output[ticker] = float(model.predict(_matrix(values[ticker], scaler, [position]), verbose=0).ravel()[0])
        if not np.isfinite(output[ticker]):
            raise RuntimeError(f"{ticker} 預測不是有限值")
    order = sorted(TICKERS, key=lambda ticker: output[ticker], reverse=True)
    ranks = {ticker: rank for rank, ticker in enumerate(order, start=1)}
    tau = np.asarray([card["tau"][ticker] for ticker in TICKERS], dtype=float)
    vector = np.asarray([output[ticker] for ticker in TICKERS], dtype=float)
    flag = bool(risk.loc[signal, "Risk_Off"])
    target = threshold_weights(vector, tau, flag)
    following = sessions(signal + pd.Timedelta(days=1), signal + pd.Timedelta(days=14))
    if not len(following):
        raise RuntimeError("找不到下一個交易日")
    rows = []
    for ticker, weight in zip(TICKERS, target):
        rows.append({
            "Ticker": ticker,
            "Predicted_Return_Pct": output[ticker],
            "Rank": ranks[ticker],
            "Target_Weight": float(weight),
            "Cash_Weight": float(1.0 - target.sum()),
            "Risk_Off": flag,
            "Close": float(panel.loc[signal, ticker]),
            "Daily_Sentiment": float(frames[ticker].loc[signal, "Daily_Sentiment"]),
            "Market_Vol_20d": float(frames[ticker].loc[signal, "Market_Vol_20d"]),
        })
    return pd.DataFrame(rows), str(pd.Timestamp(following[0]).date()), five


def _forecast_entry(signal_date):
    signal = pd.Timestamp(signal_date).normalize()
    following = sessions(signal + pd.Timedelta(days=1), signal + pd.Timedelta(days=40))
    if not len(following):
        raise RuntimeError("找不到下一個交易日")
    return pd.Timestamp(following[0]).normalize()


def _execution_anchor(signal_date):
    """The ledger's first forecast entry anchors its fixed five-session clock."""
    files = sorted(LIVE.glob("*.csv"))
    first = min(
        [pd.Timestamp(pd.read_csv(path, usecols=["Signal_Date"]).Signal_Date.iloc[0]).normalize()
         for path in files] + [pd.Timestamp(signal_date).normalize()]
    )
    return _forecast_entry(first)


def execution_eligibility(signal_date, first_execution_date=None):
    """Distinguish a forecast's next close from a scheduled portfolio trade.

    An ineligible forecast is never held over for the next rebalance. That
    rebalance requires the immediately preceding session's own saved signal.
    """
    entry = _forecast_entry(signal_date)
    anchor = (pd.Timestamp(first_execution_date).normalize()
              if first_execution_date is not None else _execution_anchor(signal_date))
    if entry < anchor:
        raise ValueError("forecast entry precedes the ledger execution anchor")
    calendar = sessions(anchor, entry + pd.Timedelta(days=40))
    if anchor not in calendar or entry not in calendar:
        raise ValueError("execution dates must be exchange sessions")
    position = int(calendar.get_loc(entry))
    next_position = position + (-position % 5)
    return {
        "forecast_entry_date": str(entry.date()),
        "scheduled_rebalance_date": str(pd.Timestamp(calendar[next_position]).date()),
        "trade_eligible": bool(position % 5 == 0),
        "execution_anchor": str(anchor.date()),
        "execution_policy": "exact_previous_session_signal_only; no carry-forward",
    }


def commit_signal(path, frame):
    path = assert_v211(path)
    if path.exists():
        raise FileExistsError(f"訊號已存在，拒絕覆蓋：{path.name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(assert_v211(temporary), index=False)
    temporary.replace(path)
    reread = pd.read_csv(path)
    if len(reread) != len(frame) or set(reread.Ticker) != set(frame.Ticker):
        raise AssertionError("訊號寫入後無法原樣讀回")
    return reread


def _signal_frame(rows, now, data_through, execution, card, source=None, decision_at=None):
    source = source or {}
    required = ("price_sha256", "sentiment_sha256", "fetched_at")
    missing = [name for name in required if not source.get(name)]
    if missing:
        raise ValueError("new signal source audit is incomplete: " + ", ".join(missing))
    eligibility = execution_eligibility(data_through)
    if str(execution) != eligibility["forecast_entry_date"]:
        raise ValueError("prediction entry differs from the next exchange session")
    bundle = verify_bundle(weight_dir=WEIGHT_DIR, card=card)
    frame = rows.copy()
    frame.insert(0, "Signal_Date", data_through)
    frame.insert(1, "Created_At", now.astimezone(TAIPEI).isoformat())
    frame.insert(2, "Data_Through", data_through)
    frame.insert(3, "Model_Version", MODEL_VERSION)
    frame.insert(4, "Model_Sha256", card.get("weight_sha256", sha256(WEIGHT_DIR / "weights.npz")))
    frame["Model_Bundle_Sha256"] = bundle["bundle_sha256"]
    frame["Decision_At"] = (decision_at or now).astimezone(TAIPEI).isoformat()
    frame["Planned_Execution_Date"] = execution
    frame["Forecast_Entry_Date"] = eligibility["forecast_entry_date"]
    frame["Scheduled_Rebalance_Date"] = eligibility["scheduled_rebalance_date"]
    frame["Trade_Eligible"] = eligibility["trade_eligible"]
    frame["Execution_Status"] = "eligible_unfilled" if eligibility["trade_eligible"] else "forecast_only"
    frame["Feature_AsOf"] = data_through
    frame["Price_AsOf"] = data_through
    frame["Sentiment_AsOf"] = data_through
    frame["Price_Source"] = source.get("price_source")
    frame["Sentiment_Source"] = source.get("sentiment_source")
    frame["Price_Sha256"] = source.get("price_sha256")
    frame["Sentiment_Sha256"] = source.get("sentiment_sha256")
    frame["Fetched_At"] = source.get("fetched_at")
    frame["Used_Previous_Day_Sentiment"] = False
    frame["Origin"] = "live_asof"
    frame["Execution_Price"] = np.nan
    frame["Actual_5d_Pct"] = np.nan
    return frame


def publish_today(now, context, card, source=None, publication_clock=None):
    panel = context[0]
    decision = publication_decision(now, panel.index)
    if not decision["publish"]:
        return decision
    signal = pd.Timestamp(decision["completed_session"])
    path = LIVE / f"{signal.strftime('%Y-%m-%d')}.csv"
    if path.exists():
        existing = pd.read_csv(path)
        decision["publish"] = False
        decision["reason"] = "already_written"
        decision["path"] = str(path.relative_to(ROOT))
        decision["created_at"] = str(existing.Created_At.iloc[0])
        decision["reread_rows"] = int(len(existing))
        return decision
    if source is None:
        from experiments.v2.v211_market import source_for

        source = source_for(signal)
    rows, execution, _five = predict_rows(card, context, signal)
    written_at = (publication_clock or (lambda: datetime.now(TZ)))()
    if written_at.tzinfo is None:
        raise ValueError("publication clock must contain timezone")
    if written_at.astimezone(TAIPEI).date() != now.astimezone(TAIPEI).date():
        raise RuntimeError("prediction finished on another date; refusing a backdated publication")
    if written_at < now:
        raise RuntimeError("publication clock precedes the decision time")
    frame = _signal_frame(rows, written_at, signal.strftime("%Y-%m-%d"), execution, card, source,
                          decision_at=now)
    reread = commit_signal(path, frame)
    decision["path"] = str(path.relative_to(ROOT))
    decision["created_at"] = str(reread.Created_At.iloc[0])
    decision["reread_rows"] = int(len(reread))
    decision["planned_execution"] = str(reread.Planned_Execution_Date.iloc[0])
    decision.update(execution_eligibility(signal))
    return decision


def label_maturity(signal_date, index):
    """The five-day label ends six sessions after the signal, at the close of that session."""
    signal = pd.Timestamp(signal_date).normalize()
    index = pd.DatetimeIndex(index)
    if signal not in index:
        return {"mature": False, "reason": "signal_not_in_prices", "label_end": None}
    position = int(index.get_loc(signal))
    if position + LABEL_GAP >= len(index):
        return {"mature": False, "reason": "waiting_for_future_sessions", "label_end": None}
    return {
        "mature": True,
        "reason": "five_sessions_after_fill_are_printed",
        "label_end": str(pd.Timestamp(index[position + LABEL_GAP]).date()),
    }


def projected_label_end(signal_date):
    window = sessions(pd.Timestamp(signal_date), pd.Timestamp(signal_date) + pd.Timedelta(days=40))
    if len(window) <= LABEL_GAP:
        return None
    return str(pd.Timestamp(window[LABEL_GAP]).date())


def score_matured(context):
    """Score a live signal only after the five-day label has printed in the price file."""
    validate_ledger_sources(ROOT, V211)
    panel, _frames, _risk, _one, five, _excess = context
    index = panel.index
    files = sorted(LIVE.glob("*.csv"))
    matured = []
    matured_today = []
    pending = 0
    pending_labels = []
    for path in files:
        frame = pd.read_csv(path)
        signal = pd.Timestamp(frame.Signal_Date.iloc[0]).normalize()
        state = label_maturity(signal, index)
        if not state["mature"]:
            pending += len(frame)
            pending_labels.append({"signal": str(signal.date()), "label_end": projected_label_end(signal)})
            continue
        if state["label_end"] == str(index[-1].date()):
            matured_today.append(str(signal.date()))
        position = int(index.get_loc(signal))
        actual = five.iloc[position]
        for _, row in frame.iterrows():
            matured.append({
                "Signal_Date": signal,
                "Ticker": row.Ticker,
                "Predicted": float(row.Predicted_Return_Pct),
                "Actual_5d_Pct": float(actual[row.Ticker]),
            })
    table = pd.DataFrame(matured, columns=["Signal_Date", "Ticker", "Predicted", "Actual_5d_Pct"])
    score_path = V211 / "matured_scores.csv"
    plan = append_plan(score_path, table, ("Signal_Date", "Ticker"))
    next_label_end = pending_labels[0]["label_end"] if pending_labels else None
    if table.empty:
        return {
            "matured_signal_days": 0,
            "pending_predictions": int(pending),
            "matured_today": matured_today,
            "pending_labels": pending_labels,
            "next_label_end": next_label_end,
            "metrics": None,
        }
    # Reject invalid predictions/labels before any realized output is written.
    metrics = classification_scores(table.Predicted, table.Actual_5d_Pct)
    days = table.groupby("Signal_Date")
    spearman = []
    for _day, part in days:
        if part.Predicted.nunique() > 1 and part.Actual_5d_Pct.nunique() > 1:
            spearman.append(float(part.Predicted.corr(part.Actual_5d_Pct, method="spearman")))
    metrics["Cross_Section_Spearman"] = float(np.nanmean(spearman)) if spearman else None
    metrics["Matured_Signal_Days"] = int(table.Signal_Date.nunique())
    metrics["Label_Overlap_Sessions"] = LABEL_GAP - 1
    added = apply_append(assert_v211(score_path), plan)
    record_ledger_sources(ROOT, V211, panel, "score_matured", {"matured_scores.csv": added})
    return {
        "matured_signal_days": int(table.Signal_Date.nunique()),
        "pending_predictions": int(pending),
        "matured_today": matured_today,
        "pending_labels": pending_labels,
        "next_label_end": next_label_end,
        "metrics": metrics,
    }


def audit_signal_frame(frame):
    """A published signal stays a record of the decision. Fills are stored elsewhere."""
    required = [
        "Signal_Date", "Created_At", "Data_Through", "Model_Version", "Model_Sha256",
        "Ticker", "Predicted_Return_Pct", "Rank", "Target_Weight", "Cash_Weight",
        "Risk_Off", "Close", "Planned_Execution_Date", "Execution_Price", "Actual_5d_Pct",
        "Used_Previous_Day_Sentiment",
    ]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise RuntimeError(f"訊號缺少欄位：{missing}")
    if len(frame) != len(TICKERS) or set(frame.Ticker) != set(TICKERS):
        raise RuntimeError("訊號必須是五檔各一列")
    created = str(frame.Created_At.iloc[0])
    if frame.Created_At.nunique() != 1 or not created.endswith("+08:00"):
        raise RuntimeError("發布時間必須是單一台北時間")
    card = json.loads((WEIGHT_DIR / "model_card.json").read_text(encoding="utf-8"))
    if str(frame.Model_Sha256.iloc[0]) != card["weight_sha256"]:
        raise RuntimeError("訊號裡的模型雜湊與凍結權重不同")
    if str(frame.Model_Version.iloc[0]) != MODEL_VERSION:
        raise RuntimeError("訊號裡的模型版本與凍結版本不同")
    signal_day = pd.Timestamp(frame.Signal_Date.iloc[0]).normalize()
    execution_day = pd.Timestamp(frame.Planned_Execution_Date.iloc[0]).normalize()
    if frame.Signal_Date.nunique() != 1 or frame.Planned_Execution_Date.nunique() != 1:
        raise RuntimeError("同一份訊號出現一個以上的日期")
    if execution_day <= signal_day:
        raise RuntimeError("成交日沒有晚於訊號日")
    if frame.Execution_Price.notna().any() or frame.Actual_5d_Pct.notna().any():
        raise RuntimeError("訊號檔不能事後填入成交價或五日報酬")
    weights = frame.Target_Weight.to_numpy(dtype=float)
    cash = float(frame.Cash_Weight.iloc[0])
    if frame.Cash_Weight.nunique() != 1 or (weights < 0).any() or weights.sum() > 1 + 1e-8:
        raise RuntimeError("股票權重與現金比例無法還原")
    if not np.isclose(weights.sum() + cash, 1.0, atol=1e-8):
        raise RuntimeError("股票權重與現金比例無法還原")
    if frame.Used_Previous_Day_Sentiment.astype(str).str.lower().isin(["true", "1"]).any():
        raise RuntimeError("訊號使用了前一日情緒")
    if "Model_Bundle_Sha256" in frame:
        required_audit = ("Price_Sha256", "Sentiment_Sha256", "Fetched_At", "Decision_At",
                          "Forecast_Entry_Date", "Scheduled_Rebalance_Date", "Trade_Eligible")
        if any(name not in frame or frame[name].isna().any() for name in required_audit):
            raise RuntimeError("新訊號缺少完整的來源或成交資格紀錄")
        eligibility = execution_eligibility(signal_day)
        if set(frame.Forecast_Entry_Date.astype(str)) != {eligibility["forecast_entry_date"]}:
            raise RuntimeError("新訊號的預測進場日與交易日曆不同")
        if set(frame.Scheduled_Rebalance_Date.astype(str)) != {eligibility["scheduled_rebalance_date"]}:
            raise RuntimeError("新訊號的再平衡日與交易日曆不同")
        flags = frame.Trade_Eligible.astype(str).str.lower().isin(["true", "1"])
        if not bool((flags == eligibility["trade_eligible"]).all()):
            raise RuntimeError("新訊號的成交資格與交易日曆不同")
        if (pd.to_datetime(frame.Created_At, utc=True) < pd.to_datetime(frame.Decision_At, utc=True)).any():
            raise RuntimeError("訊號發布時間早於決策時間")
    return {
        "created_at": created,
        "model_sha256": str(frame.Model_Sha256.iloc[0]),
        "signal_date": str(signal_day.date()),
        "planned_execution": str(execution_day.date()),
        "rows": int(len(frame)),
    }


def seal_signals():
    """Remember the original file bytes. A later run may add a new day, not rewrite an old one."""
    registry = V211 / "signal_registry.json"
    saved = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else {}
    found = {}
    for path in sorted(LIVE.glob("*.csv")):
        frame = pd.read_csv(path)
        audit = audit_signal_frame(frame)
        digest = sha256(path)
        found[path.name] = {**audit, "file_sha256": digest}
        previous = saved.get(path.name)
        if previous and previous.get("file_sha256") != digest:
            raise RuntimeError(f"已發布訊號被改寫：{path.name}")
        if previous and previous.get("created_at") != audit["created_at"]:
            raise RuntimeError(f"發布時間被改寫：{path.name}")
        if previous and previous.get("model_sha256") != audit["model_sha256"]:
            raise RuntimeError(f"模型雜湊被改寫：{path.name}")
    missing = [name for name in saved if name not in found]
    if missing:
        raise RuntimeError(f"已發布訊號檔消失：{missing}")
    if found != saved:
        assert_v211(registry).write_text(json.dumps(found, ensure_ascii=False, indent=2), encoding="utf-8")
    write_signal_audit()
    return found


def write_signal_audit():
    """Append provenance findings beside immutable legacy CSVs; never repair them."""
    path = V211 / "signal_audit.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    records = {}
    for signal_path in sorted(LIVE.glob("*.csv")):
        frame = pd.read_csv(signal_path)
        signal = frame.Signal_Date.iloc[0]
        missing = [name for name in ("Price_Sha256", "Sentiment_Sha256", "Fetched_At")
                   if name not in frame or frame[name].isna().any()
                   or frame[name].astype(str).str.strip().eq("").any()]
        records[signal_path.name] = {
            "file_sha256": sha256(signal_path),
            "recorded_created_at": str(frame.Created_At.iloc[0]),
            "execution": execution_eligibility(signal),
            "source_audit_status": "incomplete_legacy_record" if missing else "source_fields_present",
            "missing_source_fields": missing,
            "publication_evidence": ("developer_recorded_time_without_independent_pretrade_receipt"
                                     if str(signal) == "2026-11-05" else "repository_signal_record"),
            "legacy_csv_unchanged": True,
        }
    document = {"schema_version": 1, "signals": records,
                "note": "Retrospective audit only; present fields do not prove historical availability."}
    if previous != document:
        assert_v211(path).write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return document


def build_live_book(panel, signals):
    """Trade the published weights at the later session's close."""
    from experiments.calendar_rules import apply_rule
    from experiments.risk_experiment import BROKER_FEE_RATE, INITIAL_CAPITAL, SELL_TAX_RATE, rebalance

    if signals is None or len(signals) == 0:
        return {"status": "no_live_signals", "first_fill_completed": False, "fills": pd.DataFrame(), "daily": pd.DataFrame()}
    frame = signals.copy()
    frame["Signal_Date"] = pd.to_datetime(frame["Signal_Date"]).dt.normalize()
    frame["Planned_Execution_Date"] = pd.to_datetime(frame["Planned_Execution_Date"]).dt.normalize()
    if (frame.Planned_Execution_Date <= frame.Signal_Date).any():
        raise RuntimeError("成交日沒有晚於訊號日")
    expected_entries = frame.Signal_Date.map(_forecast_entry)
    if not bool((frame.Planned_Execution_Date == expected_entries).all()):
        raise RuntimeError("每筆預測只能對應下一交易日，不能挪用舊訊號至再平衡日")
    if "Created_At" in frame:
        created = pd.to_datetime(frame.Created_At, utc=True)
        deadlines = pd.DatetimeIndex(frame.Planned_Execution_Date).tz_localize(TAIPEI) + pd.Timedelta(hours=13, minutes=30)
        if (created.to_numpy() >= deadlines.tz_convert("UTC").to_numpy()).any():
            raise RuntimeError("訊號在預定收盤之後才發布，拒絕事後模擬成交")
    first = expected_entries.min()
    available = frame[frame.Planned_Execution_Date.isin(panel.index)]
    if available.empty:
        return {
            "status": "waiting_for_next_close",
            "first_fill_completed": False,
            "signals": int(frame.Signal_Date.nunique()),
            "fills": pd.DataFrame(),
            "daily": pd.DataFrame(),
        }
    by_exec = {}
    for day, part in available.groupby("Planned_Execution_Date", sort=True):
        if len(part) != len(TICKERS) or set(part.Ticker.astype(str)) != set(TICKERS):
            raise RuntimeError(f"{pd.Timestamp(day).date()} 的已發布訊號不完整")
        by_exec[pd.Timestamp(day).normalize()] = part.set_index("Ticker").loc[TICKERS]
    start = first
    if start not in panel.index:
        raise RuntimeError("缺少第一筆預測的進場價格，拒絕重設再平衡起點")
    index = panel.index[panel.index >= start]
    execution_calendar = sessions(start, panel.index[-1])
    books = {}
    for name in ("published", "equal"):
        shares = np.zeros(len(TICKERS))
        cash = float(INITIAL_CAPITAL)
        previous = None
        rows = []
        fills = []
        for day in index:
            day = pd.Timestamp(day).normalize()
            price = panel.loc[day, TICKERS].to_numpy(dtype=float)
            if not np.isfinite(price).all() or (price <= 0).any():
                raise RuntimeError(f"{day.date()} 收盤無效，停止入帳")
            held = shares.copy()
            change = np.zeros(len(TICKERS)) if previous is None else held * (price - previous)
            old_values = held * price
            pre_nav = float(old_values.sum() + cash)
            due = day in execution_calendar and int(execution_calendar.get_loc(day)) % 5 == 0
            info = by_exec.get(day)
            traded = False
            skipped = False
            ledger = {"Total_Cost": 0.0, "Turnover": 0.0, "Buy_Fee": 0.0, "Sell_Tax": 0.0}
            weights = old_values / pre_nav if pre_nav else np.zeros(len(TICKERS))
            if due:
                if info is None:
                    skipped = True
                else:
                    if name == "published":
                        weights = info.Target_Weight.to_numpy(dtype=float)
                    else:
                        flag = str(info.Risk_Off.iloc[0]).strip().lower() in ("true", "1")
                        weights = apply_rule("B", np.zeros(len(TICKERS)), np.zeros(len(TICKERS)), flag, np.ones(len(TICKERS)))
                    signal_close = info.Close.to_numpy(dtype=float)
                    target_values, cash, ledger = rebalance(old_values, cash, weights)
                    shares = target_values / price
                    traded = True
                    record = {
                        "Book": name,
                        "Signal_Date": str(pd.Timestamp(info.Signal_Date.iloc[0]).date()),
                        "Execution_Date": str(day.date()),
                        "Fill_Source": "execution_close",
                        "Cost": float(ledger["Total_Cost"]),
                        "Buy_Fee": float(ledger["Buy_Fee"]),
                        "Sell_Tax": float(ledger["Sell_Tax"]),
                        "Cash": float(cash),
                        "NAV_Open": float(target_values.sum() + cash),
                    }
                    for ticker, fill, signal_px, share, weight in zip(TICKERS, price, signal_close, shares, weights):
                        record[f"Fill_{ticker}"] = float(fill)
                        record[f"SignalClose_{ticker}"] = float(signal_px)
                        record[f"Shares_{ticker}"] = float(share)
                        record[f"Weight_{ticker}"] = float(weight)
                    fills.append(record)
            post = shares * price
            nav_open = float(post.sum() + cash)
            liquidation = float(post.sum() * (BROKER_FEE_RATE + SELL_TAX_RATE)) if day == index[-1] and post.sum() else 0.0
            rows.append({
                "Date": day,
                "Trade": traded,
                "Skipped": skipped,
                "NAV_Open": nav_open,
                "Rebalance_Cost": float(ledger["Total_Cost"]),
                "Turnover": float(ledger.get("Turnover", 0.0)),
                "Liquidation_If_Closed": liquidation,
                "Equity_Weight": float(post.sum() / nav_open) if nav_open else 0.0,
                "Cash_Weight": float(cash / nav_open) if nav_open else 1.0,
                "Stock_PnL": float(change.sum()),
            })
            previous = price
        daily = pd.DataFrame(rows)
        identity = (
            float(INITIAL_CAPITAL)
            + float(daily.Stock_PnL.sum())
            - float(daily.Rebalance_Cost.sum())
            - float(daily.NAV_Open.iloc[-1])
        )
        if abs(identity) > 1e-4:
            raise AssertionError(f"{name} 淨值恆等式不成立，差額 {identity}")
        books[name] = {"daily": daily, "fills": fills}
    published = books["published"]["daily"]
    return {
        "status": "updated",
        "first_fill_completed": bool(published.Trade.any()),
        "signals": int(frame.Signal_Date.nunique()),
        "fills": pd.DataFrame(books["published"]["fills"] + books["equal"]["fills"]),
        "daily": published,
        "equal_daily": books["equal"]["daily"],
        "rebalance_is_every": 5,
        "prediction_is": "daily",
        "fill_price": "next_session_close",
        "open_nav": float(published.NAV_Open.iloc[-1]),
        "cash_weight": float(published.Cash_Weight.iloc[-1]),
        "entry_cost": float(published.Rebalance_Cost.sum()),
        "sessions": int(len(published)),
    }


def _book_summary(result):
    summary = {
        "status": result["status"],
        "first_fill_completed": bool(result.get("first_fill_completed")),
        "signals": result.get("signals", 0),
        "rebalance_is_every": result.get("rebalance_is_every", 5),
        "prediction_is": "daily",
        "fill_price": result.get("fill_price", "next_session_close"),
    }
    if result["status"] == "updated":
        summary.update({
            "sessions": result["sessions"],
            "open_nav": result["open_nav"],
            "cash_weight": result["cash_weight"],
            "entry_cost": result["entry_cost"],
        })
    return summary


def update_book(context):
    validate_ledger_sources(ROOT, V211)
    files = sorted(LIVE.glob("*.csv"))
    if not files:
        return {"status": "no_live_signals", "first_fill_completed": False}
    panel = context[0]
    signals = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    result = build_live_book(panel, signals)
    if result["status"] != "updated":
        return _book_summary(result)
    # Validate every realized table before writing any of them. Recomputed
    # historical fills/NAV must agree; only newly available rows are appended.
    tables = (("live_fills.csv", result["fills"], ("Book", "Execution_Date"), ()),
              ("live_nav.csv", result["daily"], ("Date",), ("Liquidation_If_Closed",)),
              ("equal_nav.csv", result["equal_daily"], ("Date",), ("Liquidation_If_Closed",)))
    plans = [(name, append_plan(V211 / name, frame, keys, snapshot_columns=snapshots))
             for name, frame, keys, snapshots in tables]
    added = {name: apply_append(assert_v211(V211 / name), plan) for name, plan in plans}
    record_ledger_sources(ROOT, V211, panel, "update_book", added)
    published = result["daily"]
    equal = result["equal_daily"]
    compared = pd.DataFrame([
        {
            "Name": "rolling_selected_live",
            "NAV_Open": float(published.NAV_Open.iloc[-1]),
            "Entry_Cost": float(published.Rebalance_Cost.sum()),
            "Cash_Weight": float(published.Cash_Weight.iloc[-1]),
            "Rebalance_Days": int(published.Trade.sum()),
        },
        {
            "Name": "equal_weekly_live",
            "NAV_Open": float(equal.NAV_Open.iloc[-1]),
            "Entry_Cost": float(equal.Rebalance_Cost.sum()),
            "Cash_Weight": float(equal.Cash_Weight.iloc[-1]),
            "Rebalance_Days": int(equal.Trade.sum()),
        },
    ])
    compared["Gap_Vs_Equal_NAV"] = compared.NAV_Open - float(compared.loc[compared.Name == "equal_weekly_live", "NAV_Open"].iloc[0])
    compared.to_csv(assert_v211(V211 / "live_book.csv"), index=False)
    return _book_summary(result)


def rebalance_state(panel, now):
    """The first live fill is the anchor. The full price history is not."""
    files = sorted(LIVE.glob("*.csv"))
    today = pd.Timestamp(now.astimezone(TAIPEI).date())
    if not files:
        return {"rebalance_today": False, "reason": "no_live_execution_anchor", "today": str(today.date())}
    executions = []
    for path in files:
        frame = pd.read_csv(path, usecols=["Signal_Date"])
        executions.append(_forecast_entry(frame.Signal_Date.iloc[0]))
    first = min(executions)
    if first not in panel.index:
        return {
            "rebalance_today": False,
            "reason": "waiting_for_first_fill",
            "first_execution": str(first.date()),
            "today": str(today.date()),
        }
    if today not in panel.index:
        return {
            "rebalance_today": False,
            "reason": "today_not_in_prices",
            "first_execution": str(first.date()),
            "today": str(today.date()),
        }
    calendar = sessions(first, today)
    due = today in calendar and int(calendar.get_loc(today)) % 5 == 0
    previous = sessions(today - pd.Timedelta(days=40), today - pd.Timedelta(days=1))[-1]
    required_signal = str(pd.Timestamp(previous).date())
    return {
        "rebalance_today": bool(due),
        "reason": "every_5_sessions_from_first_fill",
        "first_execution": str(first.date()),
        "today": str(today.date()),
        "required_signal_date": required_signal if due else None,
        "required_signal_present": bool(due and (LIVE / f"{required_signal}.csv").exists()),
        "signal_carry_forward_allowed": False,
    }


def recommendation(scored):
    days = scored["matured_signal_days"]
    return {
        "minimum_matured_signal_days": 60,
        "matured_signal_days": days,
        "ready_for_first_review": days >= 60,
        "recommend_replace_official_555": False,
        "reason": "60 個重疊的五日標籤只是最低觀察數。程式不能更換正式 555。",
    }


def data_check(context):
    panel, frames, _risk, _one, _five, _excess = context
    official = pd.read_csv(ROOT / "data" / "sentiment_features_2026.csv")
    official["Date"] = pd.to_datetime(official["Date"])
    last = panel.index.max()
    feature_ok = True
    missing = []
    for ticker in TICKERS:
        row = frames[ticker].loc[last, list(FEATURES)]
        if not np.isfinite(row.to_numpy(dtype=float)).all():
            feature_ok = False
            missing.append(ticker)
    recent = official[official.Date >= "2026-10-01"]
    wide = recent.pivot(index="Date", columns="ticker", values="Daily_Sentiment")
    return {
        "prices_through": str(last.date()),
        "price_names_on_last_day": sorted(panel.columns),
        "official_sentiment_through": str(official.Date.max().date()),
        "official_sentiment_missing_since_october": int(wide.reindex(columns=TICKERS).isna().sum().sum()),
        "last_day_features_complete": feature_ok,
        "incomplete_tickers": missing,
        "frozen_window": FROZEN_WINDOW,
        "frozen_window_source": FROZEN_WINDOW_SOURCE,
        "weights_present_before_run": (WEIGHT_DIR / "model_card.json").exists(),
    }


def write_status(payload):
    assert_v211(V211 / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def resolve_context():
    from experiments.v2.v211_market import load_market_context, market_files_ready

    validate_ledger_sources(ROOT, V211)
    if market_files_ready():
        return load_market_context(), "v211_market"
    return load_context(), "frozen_555_snapshot"


def run(now=None, fetch=None, publication_clock=None):
    now = now or datetime.now(TZ)
    try:
        context, source_name = resolve_context()
    except Exception as exc:
        status = {
            "ok": False,
            "error": str(exc),
            "signal_published": False,
            "rebalance_today": False,
            "fetch": fetch,
            "official_555_replacement": False,
            "tracker_downloads_data": False,
        }
        write_status(status)
        return status
    check = data_check(context)
    check["context_source"] = source_name
    check["tracker_downloads_data"] = False
    if not check["last_day_features_complete"]:
        status = {
            "ok": False,
            "error": "最後交易日特徵不完整",
            "data": check,
            "signal_published": False,
            "rebalance_today": False,
            "fetch": fetch,
            "official_555_replacement": False,
        }
        write_status(status)
        return status
    weights_existed = (WEIGHT_DIR / "model_card.json").exists()
    card = fit_and_save(context)
    try:
        decision = publish_today(now, context, card, fetch, publication_clock=publication_clock)
        error = None
    except Exception as exc:
        decision = {"publish": False, "reason": "prediction_failed"}
        error = str(exc)
    scored = score_matured(context)
    book = update_book(context)
    reb = rebalance_state(context[0], now)
    status = {
        "ok": error is None,
        "error": error,
        "data_updated_through_latest_completed_session": publication_decision(now, context[0].index)["reason"] == "today_close_is_stored",
        "data": check,
        "model_version": MODEL_VERSION,
        "model_sha256": card.get("weight_sha256"),
        "weights_reused": weights_existed,
        "signal": decision,
        "signal_published": bool(decision.get("publish")),
        "rebalance_today": reb["rebalance_today"],
        "rebalance": reb,
        "labels_matured_today": scored.get("matured_today", []),
        "scorable_signal_days": scored["matured_signal_days"],
        "immature_predictions": scored["pending_predictions"],
        "scoring": scored,
        "book": book,
        "first_fill_completed": bool(book.get("first_fill_completed")),
        "next_label_date": scored.get("next_label_end"),
        "review": recommendation(scored),
        "fetch": fetch,
        "risk_cap": RISK,
        "official_555_replacement": False,
        "tracker_downloads_data": False,
    }
    write_status(status)
    return status


if __name__ == "__main__":
    run()
