"""Diagnose retained five-session forecasts without choosing a new model.

Only already published predictions and mature labels are scored. Zero forecasts
use the identical five-stock rows, dates and percentage-point units. This reader
does not train, download, fill labels, update a book or authenticate old releases.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOOK = "experiments/results/v2/v211"
TICKERS = frozenset(("2330.TW", "2317.TW", "2603.TW", "2002.TW", "2891.TW"))
WEIGHT = "53a1467fdba66e7f22a45e5f2fbc1cb78fc99825b15d516b3e7153d7bfc28144"
LABEL = "Close[i+6] / Close[i+1] − 1；五個交易日，單位為百分點"
LIMITATIONS = [
    "成熟股票列不是獨立時間樣本；相鄰五日標籤重疊，不能以股票列數宣稱模型有效。",
    "歷史訊號發布前可用性尚未完整證明；檔案hash與同日推論一致不等於前瞻證據。",
    "零預測是誤差參考，不是價格模型／無新聞配對；這個差額不能歸因新聞。",
    "這些已看過結果不選門檻、不換凍結權重；60成熟訊號日只作運作檢查，不自動核准模型或宣稱超越0050。",
]


class V2DiagnosticError(ValueError):
    """A saved forecast or its mature-label accounting cannot be reconciled."""


def _path(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
        raise V2DiagnosticError(f"缺少或不安全的V2診斷來源：{relative}")
    return path


def _json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, OSError) as exc:
        raise V2DiagnosticError(f"V2診斷JSON無法讀取：{path.name}") from exc
    if not isinstance(value, dict):
        raise V2DiagnosticError(f"V2診斷JSON格式錯誤：{path.name}")
    return value


def _dates(values, name):
    text = values.astype(str)
    if not text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise V2DiagnosticError(f"{name}必須是完整市場日期")
    try:
        parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="raise")
    except (ValueError, TypeError) as exc:
        raise V2DiagnosticError(f"{name}日期不合法") from exc
    return parsed


def _finite(frame, fields, name):
    if not set(fields).issubset(frame):
        raise V2DiagnosticError(f"{name}缺必要欄位")
    try:
        valid = np.isfinite(frame[list(fields)].to_numpy(float)).all()
    except (ValueError, TypeError) as exc:
        raise V2DiagnosticError(f"{name}含非數值") from exc
    if not valid:
        raise V2DiagnosticError(f"{name}含缺值或非有限值")


def _read_checked(root, relative, proof, sources):
    path = _path(root, BOOK + "/" + relative)
    raw = path.read_bytes()
    expected = proof.get("prefixes", {}).get(relative)
    digest = hashlib.sha256(raw).hexdigest()
    # No partially appended source is scored before its daily checkpoint exists.
    if not isinstance(expected, dict) or expected.get("bytes") != len(raw) or expected.get("sha256") != digest:
        raise V2DiagnosticError(f"V2保存來源完整性檢查失敗：{relative}")
    sources[BOOK + "/" + relative] = digest
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except (ValueError, OSError) as exc:
        raise V2DiagnosticError(f"V2診斷CSV無法讀取：{relative}") from exc


def _scores(prediction, actual):
    error = np.asarray(prediction, float) - np.asarray(actual, float)
    return {"MAE": float(np.abs(error).mean()), "RMSE": float(np.sqrt(np.square(error).mean())),
            "Direction_Accuracy": float(np.mean((np.asarray(prediction) > 0) == (np.asarray(actual) > 0)))}


def _paired_books(root, proof, sources):
    books, dates = [], None
    for filename, label in (("live_nav.csv", "凍結模型研究帳本"), ("equal_nav.csv", "相同政策等權帳本")):
        nav = _read_checked(root, filename, proof, sources)
        if nav.empty:
            return pd.DataFrame()
        _finite(nav, ("NAV_Open", "Rebalance_Cost", "Liquidation_If_Closed"), "估值帳本")
        if "Date" not in nav or nav.Date.duplicated().any():
            raise V2DiagnosticError("估值日期缺漏或重複")
        _dates(nav.Date, "估值日")
        nav = nav.sort_values("Date").reset_index(drop=True)
        if dates is not None and not nav.Date.equals(dates):
            raise V2DiagnosticError("兩本估值帳本日期未完整配對")
        dates = nav.Date
        if (nav.NAV_Open <= 0).any() or (nav[["Rebalance_Cost", "Liquidation_If_Closed"]] < 0).any().any():
            raise V2DiagnosticError("估值或成本範圍不合法")
        last = nav.iloc[-1]
        books.append({"Book": label, "First_Valuation_Date": nav.Date.iloc[0], "Last_Valuation_Date": last.Date,
                      "NAV_Open": float(last.NAV_Open), "If_Closed_Cost": float(last.Liquidation_If_Closed),
                      "NAV_If_Closed": float(last.NAV_Open - last.Liquidation_If_Closed),
                      "Rebalance_Cost_Sum": float(nav.Rebalance_Cost.sum())})
    return pd.DataFrame(books)


def load_v2_diagnostics(root=ROOT):
    """Reconcile mature five-day rows and compare them with zero forecasts.

    Exact checkpoint hashes guard the source snapshot. Calendar dates and label
    values are checked against saved closes, but raw prices are never returned.
    """
    root = Path(root)
    proof = _json(_path(root, BOOK + "/ledger_integrity.json"))
    registry = _json(_path(root, BOOK + "/signal_registry.json"))
    status = _json(_path(root, BOOK + "/status.json"))
    sources = {BOOK + "/" + name: hashlib.sha256(_path(root, BOOK + "/" + name).read_bytes()).hexdigest()
               for name in ("ledger_integrity.json", "signal_registry.json", "status.json")}
    rows = _read_checked(root, "matured_scores.csv", proof, sources)
    if not {"Signal_Date", "Ticker", "Predicted", "Actual_5d_Pct"}.issubset(rows):
        raise V2DiagnosticError("成熟預測表缺必要欄位")
    _dates(rows.Signal_Date, "訊號日")
    _finite(rows, ("Predicted", "Actual_5d_Pct"), "成熟預測")
    if rows.duplicated(["Signal_Date", "Ticker"]).any():
        raise V2DiagnosticError("成熟預測的日期／股票鍵重複")
    books = _paired_books(root, proof, sources)
    if rows.empty:
        return {"available": True, "summary": {"rows": 0, "signal_days": 0, "official_claim_blocked": True},
                "comparison": pd.DataFrame(), "by_day": pd.DataFrame(), "rows": rows,
                "books": books, "sources": sources, "label_contract": LABEL, "limitations": LIMITATIONS}
    prices = _read_checked(root, "market/prices.csv", proof, sources)
    if not {"Date", "ticker", "Close"}.issubset(prices) or prices.duplicated(["Date", "ticker"]).any():
        raise V2DiagnosticError("保存價格缺欄位或日期／股票重複")
    _dates(prices.Date, "價格日")
    _finite(prices, ("Close",), "保存價格")
    if (prices.Close <= 0).any():
        raise V2DiagnosticError("保存價格必須為正")
    calendar = sorted(prices.Date.unique())
    panel = prices.pivot(index="Date", columns="ticker", values="Close")
    per_day = []
    enriched = []
    for day, part in rows.groupby("Signal_Date", sort=True):
        if len(part) != 5 or set(part.Ticker) != TICKERS:
            raise V2DiagnosticError("每個成熟訊號日必須保留五檔唯一完整股票")
        try:
            position = calendar.index(day)
            entry_day, end_day = calendar[position + 1], calendar[position + 6]
        except (ValueError, IndexError) as exc:
            raise V2DiagnosticError("五日標籤尚未在完整保存交易日曆成熟") from exc
        if str(status.get("data", {}).get("prices_through", "")) < end_day:
            raise V2DiagnosticError("成熟標籤超過目前保存的資料截止日")
        path = _path(root, BOOK + "/daily/" + day + ".csv")
        meta = registry.get(day + ".csv", {})
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if meta.get("file_sha256") != digest or meta.get("model_sha256") != WEIGHT or meta.get("rows") != 5 or meta.get("signal_date") != day:
            raise V2DiagnosticError("成熟預測對應的原始訊號registry／hash不一致")
        sources[BOOK + "/daily/" + day + ".csv"] = digest
        signal = pd.read_csv(path, encoding="utf-8-sig")
        required = {"Signal_Date", "Ticker", "Predicted_Return_Pct", "Planned_Execution_Date", "Model_Sha256"}
        if (not required.issubset(signal) or len(signal) != 5 or signal.Ticker.duplicated().any()
                or set(signal.Ticker) != TICKERS or not signal.Signal_Date.eq(day).all()
                or not signal.Model_Sha256.eq(WEIGHT).all() or not signal.Planned_Execution_Date.eq(entry_day).all()
                or meta.get("planned_execution") != entry_day):
            raise V2DiagnosticError("原始五檔訊號或次日收盤進場日不一致")
        _finite(signal, ("Predicted_Return_Pct",), "原始預測")
        checked = part.set_index("Ticker").reindex(sorted(TICKERS)).copy()
        original = signal.set_index("Ticker").reindex(checked.index)
        if not np.allclose(checked.Predicted, original.Predicted_Return_Pct, rtol=0, atol=1e-12):
            raise V2DiagnosticError("成熟表的預測值與原始發布訊號不一致")
        needed = panel.reindex(index=calendar[position:position + 7], columns=checked.index)
        if needed.isna().any().any():
            raise V2DiagnosticError("五日標籤期間有股票缺價，不可靜默刪除交易日")
        actual = (needed.iloc[-1] / needed.iloc[1] - 1) * 100
        if not np.allclose(checked.Actual_5d_Pct, actual, rtol=0, atol=1e-8):
            raise V2DiagnosticError("成熟實際報酬不符合次日收盤至第六交易日收盤")
        model = _scores(checked.Predicted, checked.Actual_5d_Pct)
        zero = _scores(np.zeros(len(checked)), checked.Actual_5d_Pct)
        per_day.append({"Signal_Date": day, "Entry_Date": entry_day, "Label_End_Date": end_day,
                        "Rows": len(checked), "Model_MAE": model["MAE"], "Zero_MAE": zero["MAE"],
                        "MAE_Improvement_Pp": zero["MAE"] - model["MAE"], "Model_RMSE": model["RMSE"],
                        "Zero_RMSE": zero["RMSE"], "Model_Direction_Accuracy": model["Direction_Accuracy"],
                        "Zero_Direction_Accuracy": zero["Direction_Accuracy"]})
        checked["Entry_Date"], checked["Label_End_Date"] = entry_day, end_day
        checked["Zero_Baseline_Pct"] = 0.0
        checked["Model_Error_Pp"] = checked.Predicted - checked.Actual_5d_Pct
        checked["Zero_Error_Pp"] = -checked.Actual_5d_Pct
        enriched.append(checked.reset_index())
    rows = pd.concat(enriched, ignore_index=True)
    days = rows.Signal_Date.nunique()
    if status.get("scoring", {}).get("matured_signal_days") != days:
        raise V2DiagnosticError("成熟訊號日數與已保存狀態不一致")
    model = _scores(rows.Predicted, rows.Actual_5d_Pct)
    zero = _scores(np.zeros(len(rows)), rows.Actual_5d_Pct)
    comparison = pd.DataFrame([{"Forecast": "凍結模型", **model}, {"Forecast": "永遠預測0%", **zero}])
    summary = {"rows": len(rows), "signal_days": int(days), "first_signal_date": rows.Signal_Date.min(),
               "last_signal_date": rows.Signal_Date.max(), "model_mae": model["MAE"], "zero_mae": zero["MAE"],
               "mae_improvement_pp": zero["MAE"] - model["MAE"], "operations_review_threshold": 60,
               "official_claim_blocked": True,
               "historical_availability_certified": proof.get("historical_availability_certified") is True}
    return {"available": True, "summary": summary, "comparison": comparison, "by_day": pd.DataFrame(per_day),
            "rows": rows, "books": books, "sources": sources, "label_contract": LABEL, "limitations": LIMITATIONS}


def render_v2_diagnostics(root=ROOT):
    import streamlit as st

    st.subheader("V2下一步：先看成熟預測比零預測改善多少")
    try:
        result = load_v2_diagnostics(root)
    except (V2DiagnosticError, OSError, ValueError, KeyError) as exc:
        st.warning("成熟預測診斷暫不可用；不以缺值或DEMO補齊。")
        st.caption(str(exc))
        return
    summary = result["summary"]
    a, b, c = st.columns(3)
    a.metric("成熟訊號日", str(summary["signal_days"]))
    b.metric("成熟股票列", str(summary["rows"]))
    c.metric("首次運作檢查門檻", "60 個訊號日")
    st.caption(result["label_contract"] + "。預測誤差不是交易勝率或淨ROI。")
    if not summary["rows"]:
        st.info("尚無已成熟的完整五檔標籤，維持等待。")
        return
    st.warning(f"目前只有{summary['signal_days']}個訊號日；五日標籤重疊且歷史發布證據有限，尚不能據此換模型或宣稱超越0050。")
    table = result["comparison"].rename(columns={"Forecast": "預測", "MAE": "MAE（百分點）", "RMSE": "RMSE（百分點）",
                                                "Direction_Accuracy": "方向準確率（%）"}).copy()
    table["方向準確率（%）"] *= 100
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        name: st.column_config.NumberColumn(format="%.4f") for name in table.columns[1:]})
    st.caption("方向依現有二元定義：報酬>0為上漲，其餘為非上漲；零預測固定判非上漲。兩組使用完全相同的成熟列。")
    st.write(f"相對零預測的MAE改善：**{summary['mae_improvement_pp']:+.4f}個百分點**；正值表示誤差較小。這不是新聞效果或獲利增量。")
    with st.expander("逐訊號日與同列診斷", expanded=True):
        st.dataframe(result["by_day"], hide_index=True, width="stretch")
        st.dataframe(result["rows"], hide_index=True, width="stretch")
    if not result["books"].empty:
        with st.expander("同期間研究帳本的估值與成本"):
            st.dataframe(result["books"], hide_index=True, width="stretch")
            st.caption("NAV_Open是仍持倉的估值；NAV_If_Closed另扣已保存的假設期末平倉費。不是券商已平倉績效，成本差也不是獨立因果歸因。")
    with st.expander("研究限制與來源"):
        for limitation in result["limitations"]:
            st.write("• " + limitation)
        st.json(result["sources"])
    st.link_button("閱讀V2下一輪研究設計", "https://github.com/tKe1vin/tw-stock-lstm-research/blob/main/docs/v2_next_research.md")
