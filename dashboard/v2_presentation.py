"""A common V2 chart contract with separate open valuation and close assumptions.

No model runtime, provider client or write operation is used. Saved ledgers are
checked against their exact checkpoint; DEMO stays in its explicitly selected
fixture root and never borrows prices or diagnoses from the saved research.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.data import DataIntegrityError, ROOT
from dashboard.v2_diagnostics import V2DiagnosticError, WEIGHT, load_v2_diagnostics
from dashboard_v2.data_access.prospective import (
    display_sources, next_day_strategy, ridge_status, trading_journal,
    v211_root, v211_status_local,
)

ARMS = {"v211": "V2.11凍結模型", "equal_policy": "相同政策等權"}
TICKERS = ("2330.TW", "2317.TW", "2603.TW", "2002.TW", "2891.TW")
GITHUB = "https://github.com/tKe1vin/tw-stock-lstm-research/blob/main/"


class V2PresentationError(DataIntegrityError):
    """The selected source cannot support an aligned, trustworthy display."""


def _path(root, path):
    root, path = Path(root).resolve(), Path(path)
    path = path if path.is_absolute() else root / path
    if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
        raise V2PresentationError(f"V2來源缺少或不安全：{path.name}")
    return path


def _record(root, path, sources):
    path = _path(root, path)
    raw = path.read_bytes()
    name = path.relative_to(Path(root).resolve()).as_posix()
    digest = hashlib.sha256(raw).hexdigest()
    if name in sources and sources[name] != digest:
        raise V2PresentationError("來源在載入期間變動，等待完整同步後再讀取")
    sources[name] = digest
    return raw


def _json(root, path, sources):
    try:
        data = json.loads(_record(root, path, sources).decode("utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise V2PresentationError(f"V2 JSON無法讀取：{Path(path).name}") from exc
    if not isinstance(data, dict):
        raise V2PresentationError("V2來源必須是JSON物件")
    return data


def _csv(root, book, name, sources, proof, mode):
    path = _path(root, book / name)
    raw = _record(root, path, sources)
    if mode == "saved":
        entry = proof.get("prefixes", {}).get(name)
        if (not isinstance(entry, dict) or entry.get("bytes") != len(raw)
                or entry.get("sha256") != hashlib.sha256(raw).hexdigest()):
            raise V2PresentationError(f"V2保存帳本hash不符：{name}")
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, ValueError) as exc:
        raise V2PresentationError(f"V2 CSV無法讀取：{name}") from exc
    if mode == "demo" and ("Demo_Flag" not in frame or not frame.Demo_Flag.eq("SYNTHETIC_DEMO").all()):
        raise V2PresentationError("DEMO帳本缺合成資料標記")
    return frame


def _finite(frame, columns, name):
    if not set(columns).issubset(frame):
        raise V2PresentationError(f"{name}缺必要欄位")
    try:
        valid = np.isfinite(frame[list(columns)].to_numpy(float)).all()
    except (ValueError, TypeError) as exc:
        raise V2PresentationError(f"{name}含非數值") from exc
    if not valid:
        raise V2PresentationError(f"{name}含缺值或非有限值")


def _dates(frame, column, *, unique=False):
    if column not in frame or not frame[column].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise V2PresentationError(f"{column}日期格式不完整")
    try:
        dates = pd.to_datetime(frame[column], format="%Y-%m-%d", errors="raise")
    except (ValueError, TypeError) as exc:
        raise V2PresentationError(f"{column}日期不合法") from exc
    if unique and dates.duplicated().any():
        raise V2PresentationError(f"{column}日期重複")
    return dates


def _capital(root, sources):
    path = _path(root, "experiments/risk_experiment.py")
    tree = ast.parse(_record(root, path, sources).decode("utf-8-sig"))
    definitions = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "INITIAL_CAPITAL" for target in node.targets)]
    if len(definitions) != 1:
        raise V2PresentationError("找不到唯一的原協議初始資金")
    try:
        capital = float(ast.literal_eval(definitions[0].value))
    except (TypeError, ValueError) as exc:
        raise V2PresentationError("原協議初始資金不是固定數值") from exc
    if not np.isfinite(capital) or capital <= 0:
        raise V2PresentationError("原協議初始資金不合法")
    return capital


def _one(frame, column, default=None):
    if column not in frame:
        return default
    values = [value for value in frame[column].dropna().unique() if str(value).strip()]
    if len(values) > 1:
        raise V2PresentationError(f"同一訊號的{column}不一致")
    return values[0] if values else default


def _bool(value):
    if value is None or str(value).strip() == "":
        return None
    if str(value).lower() not in {"true", "false"}:
        raise V2PresentationError("交易資格欄位不是明確布林值")
    return str(value).lower() == "true"


def _signals(root, book, sources, payload, fills, diagnostics, mode):
    registry = _json(root, book / "signal_registry.json", sources)
    audit_path = book / "signal_audit.json"
    audit = _json(root, audit_path, sources).get("signals", {}) if audit_path.is_file() else {}
    pending = {row["signal"]: row.get("label_end") for row in payload.get("scoring", {}).get("pending_labels", [])}
    matured = {} if diagnostics is None else diagnostics["by_day"].set_index("Signal_Date").to_dict("index")
    if mode == "demo":
        matured = {row["Signal_Date"]: {} for row in pd.read_csv(book / "matured_scores.csv").to_dict("records")}
    rows, latest = [], None
    for name, meta in sorted(registry.items()):
        if not isinstance(meta, dict) or name != str(meta.get("signal_date")) + ".csv":
            raise V2PresentationError("訊號registry日期或檔名不一致")
        path = _path(root, book / "daily" / name)
        raw = _record(root, path, sources)
        if (hashlib.sha256(raw).hexdigest() != meta.get("file_sha256")
                or meta.get("model_sha256") != WEIGHT or meta.get("rows") != 5):
            raise V2PresentationError("已保存訊號hash不符")
        signal = pd.read_csv(path, encoding="utf-8-sig")
        day = str(meta["signal_date"])
        if (not {"Signal_Date", "Ticker", "Target_Weight", "Cash_Weight", "Model_Sha256"}.issubset(signal)
                or len(signal) != 5 or signal.Ticker.duplicated().any()
                or set(signal.Ticker) != set(TICKERS) or not signal.Signal_Date.eq(day).all()
                or not signal.Model_Sha256.eq(WEIGHT).all()):
            raise V2PresentationError("訊號必須包含同日完整五檔")
        _finite(signal, ("Target_Weight", "Cash_Weight"), "訊號目標權重")
        _dates(signal, "Signal_Date")
        recorded = audit.get(name, {})
        if recorded and recorded.get("file_sha256") != meta.get("file_sha256"):
            raise V2PresentationError("訊號日期說明與原始hash不一致")
        timing = recorded.get("execution", {})
        forecast = _one(signal, "Forecast_Entry_Date", _one(signal, "Planned_Execution_Date", meta.get("planned_execution")))
        schedule = _one(signal, "Scheduled_Rebalance_Date", timing.get("scheduled_rebalance_date"))
        eligible = _bool(_one(signal, "Trade_Eligible", timing.get("trade_eligible")))
        actual = fills[fills.Book.eq("published") & fills.Signal_Date.eq(day)] if {"Book", "Signal_Date"}.issubset(fills) else pd.DataFrame()
        execution = _one(actual, "Execution_Date")
        label_end = matured.get(day, {}).get("Label_End_Date") or pending.get(day)
        rows.append({"Signal_Date": day, "Published_At": _one(signal, "Created_At", meta.get("created_at")),
                     "Forecast_Entry_Date": forecast, "Scheduled_Rebalance_Date": schedule,
                     "Trade_Eligible": eligible, "Simulated_Execution_Date": execution,
                     "Label_End_Date": label_end,
                     "Label_State": "MATURED" if day in matured else "PENDING",
                     "Publication_Evidence": recorded.get("publication_evidence", "未提供獨立發布前證據"),
                     "Source": path.relative_to(Path(root).resolve()).as_posix()})
        latest = signal
    return pd.DataFrame(rows), latest


def _target_weights(latest, timeline):
    if latest is None:
        return pd.DataFrame()
    current = timeline.iloc[-1]
    cash = float(_one(latest, "Cash_Weight"))
    if (latest.Target_Weight < 0).any() or cash < 0 or abs(float(latest.Target_Weight.sum()) + cash - 1) > 1e-8:
        raise V2PresentationError("最新訊號的目標／現金權重無法對帳")
    rows = [{"Ticker": row.Ticker, "Weight_Pct": float(row.Target_Weight) * 100} for row in latest.itertuples()]
    rows.append({"Ticker": "CASH", "Weight_Pct": cash * 100})
    frame = pd.DataFrame(rows)
    for column in ("Signal_Date", "Forecast_Entry_Date", "Scheduled_Rebalance_Date", "Trade_Eligible"):
        frame[column] = current[column]
    return frame


def _holdings(root, book, nav, fills, sources, proof, missing, mode):
    if mode == "demo":
        missing.append("合成DEMO未交付完整股數與同源價格，不產生實際持倉權重。")
        return pd.DataFrame(), None
    try:
        execution = fills[fills.Book.eq("published")].sort_values("Execution_Date").iloc[-1]
        shares = np.asarray([float(execution[f"Shares_{ticker}"]) for ticker in TICKERS])
        if not np.isfinite(shares).all() or (shares < 0).any():
            raise ValueError("股數缺值或為負")
        prices = _csv(root, book, "market/prices.csv", sources, proof, mode)
        day = nav.Date.iloc[-1].strftime("%Y-%m-%d")
        chosen = prices[prices.Date.eq(day) & prices.ticker.isin(TICKERS)]
        if chosen.ticker.duplicated().any() or set(chosen.ticker) != set(TICKERS):
            raise ValueError("最後估值日價格缺五檔")
        closes = chosen.set_index("ticker").reindex(TICKERS).Close.to_numpy(float)
        if not np.isfinite(closes).all() or (closes <= 0).any():
            raise ValueError("最後估值日價格不合法")
        last = nav.iloc[-1]
        values = shares * closes
        cash = float(last.Cash_Weight) * float(last.NAV_Open)
        recomputed = float(values.sum() + cash)
        if not np.isclose(recomputed, float(last.NAV_Open), rtol=0, atol=1e-5):
            raise ValueError("持股估值加現金無法重算最後NAV")
        records = [{"Ticker": ticker, "Shares": float(share), "Weight_Pct": float(value / last.NAV_Open * 100)}
                   for ticker, share, value in zip(TICKERS, shares, values)]
        records.append({"Ticker": "CASH", "Shares": np.nan, "Weight_Pct": float(last.Cash_Weight) * 100})
        frame = pd.DataFrame(records)
        frame["As_Of_Date"], frame["Source_Execution_Date"] = day, execution.Execution_Date
        return frame, {"As_Of_Date": day, "Share_Value": float(values.sum()), "Cash_Value": cash,
                       "NAV_Recomputed": recomputed, "NAV_Open": float(last.NAV_Open),
                       "Difference": recomputed - float(last.NAV_Open)}
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        missing.append("實際持倉權重缺完整對帳來源：" + str(exc))
        return pd.DataFrame(), None


def load_v2_presentation(root=ROOT, mode="saved"):
    """Load one explicit source mode into V3-compatible cards and chart tables."""
    if mode not in {"saved", "demo"}:
        raise V2PresentationError("來源模式必須是saved或demo")
    root = Path(root).resolve()
    sources, missing = {}, []
    capital = _capital(root, sources)
    with display_sources(mode=mode, root=root):
        book = v211_root()
        proof = _json(root, book / "ledger_integrity.json", sources) if mode == "saved" else {}
        payload = _json(root, book / "status.json", sources)
        if payload.get("model_sha256") != WEIGHT:
            raise V2PresentationError("V2保存狀態的凍結模型版本不符")
        if mode == "demo" and payload.get("demo_flag") != "SYNTHETIC_DEMO":
            raise V2PresentationError("合成來源缺明確DEMO標記")
        try:
            diagnostics = load_v2_diagnostics(root) if mode == "saved" else None
        except V2DiagnosticError as exc:
            raise V2PresentationError(str(exc)) from exc
        if diagnostics:
            for name, digest in diagnostics["sources"].items():
                if name in sources and sources[name] != digest:
                    raise V2PresentationError("來源在載入期間變動，等待完整同步後再讀取")
                sources[name] = digest
        fills = _csv(root, book, "live_fills.csv", sources, proof, mode)
        navs, curves, exposures, metrics, costs = {}, [], [], [], []
        expected_dates = None
        for arm, filename, fill_name in (("v211", "live_nav.csv", "published"), ("equal_policy", "equal_nav.csv", "equal")):
            nav = _csv(root, book, filename, sources, proof, mode)
            if nav.empty:
                raise V2PresentationError("估值帳本為空，不能以零報酬代替")
            nav["Date"] = _dates(nav, "Date", unique=True)
            nav = nav.sort_values("Date").reset_index(drop=True)
            _finite(nav, ("NAV_Open", "Rebalance_Cost", "Liquidation_If_Closed", "Equity_Weight", "Cash_Weight", "Stock_PnL"), "估值帳本")
            if (nav.NAV_Open <= 0).any() or (nav[["Rebalance_Cost", "Liquidation_If_Closed", "Equity_Weight", "Cash_Weight"]] < -1e-10).any().any():
                raise V2PresentationError("帳本估值、曝險或成本範圍不合法")
            if not np.allclose(nav.Equity_Weight + nav.Cash_Weight, 1, rtol=0, atol=1e-8):
                raise V2PresentationError("股票與現金曝險未合計100%")
            if expected_dates is not None and not nav.Date.equals(expected_dates):
                raise V2PresentationError("模型與等權帳本缺少完整共同估值日期")
            expected_dates = nav.Date
            if mode == "saved" and not np.isclose(float(nav.NAV_Open.iloc[0] + nav.Rebalance_Cost.iloc[0] - nav.Stock_PnL.iloc[0]), capital, rtol=0, atol=1e-6):
                raise V2PresentationError("首日NAV／已發生成本與原協議初始資金不一致")
            selected = fills[fills.Book.eq(fill_name)] if "Book" in fills else pd.DataFrame()
            breakdown = {"Cost", "Buy_Fee", "Sell_Tax"}.issubset(selected)
            if breakdown:
                _finite(selected, ("Cost", "Buy_Fee", "Sell_Tax"), "模擬成交成本")
                if (selected[["Cost", "Buy_Fee", "Sell_Tax"]] < 0).any().any():
                    raise V2PresentationError("成交成本不可為負")
                actual_cost = float(selected.Cost.sum())
                if not np.isclose(actual_cost, float(nav.Rebalance_Cost.sum()), rtol=0, atol=1e-6):
                    raise V2PresentationError("成交成本加總與逐日再平衡費用不一致")
                if not np.allclose(selected.Cost, selected.Buy_Fee + selected.Sell_Tax, rtol=0, atol=1e-6):
                    raise V2PresentationError("成交費用／賣稅無法重算總成本")
            else:
                actual_cost = None
                missing.append(f"{ARMS[arm]}成交來源缺手續費／賣稅明細，不補零費用。")
                if mode == "saved":
                    raise V2PresentationError("正式保存成交檔缺完整成本欄位")
            wealth = nav.NAV_Open.to_numpy(float) / capital * 100
            peaks = np.maximum.accumulate(np.r_[100., wealth])[1:]
            drawdown = (wealth / peaks - 1) * 100
            last = nav.iloc[-1]
            closed = float(last.NAV_Open - last.Liquidation_If_Closed)
            if closed <= 0:
                raise V2PresentationError("假設期末平倉淨值不合法")
            nav["Arm"], nav["Display_Name"] = arm, ARMS[arm]
            # Only the selected endpoint is a hypothetical terminal close.
            # A daily close-cost column would imply repeatedly liquidating a
            # strategy whose later positions actually remained invested.
            nav["NAV_If_Closed"] = np.nan
            nav.loc[nav.index[-1], "NAV_If_Closed"] = closed
            nav["Cumulative_Execution_Cost"] = nav.Rebalance_Cost.cumsum()
            navs[arm] = nav
            curves.append(pd.DataFrame({"Date": nav.Date, "Arm": arm, "Display_Name": ARMS[arm], "Wealth_Index": wealth, "Drawdown_Pct": drawdown}))
            for field, label in (("Equity_Weight", "股票曝險"), ("Cash_Weight", "現金曝險")):
                exposures.append(pd.DataFrame({"Date": nav.Date, "Arm": arm, "Display_Name": ARMS[arm], "Series": ARMS[arm] + "／" + label, "Weight_Pct": nav[field] * 100}))
            metrics.append({"Arm": arm, "Display_Name": ARMS[arm], "Initial_Capital": capital,
                            "Start": nav.Date.iloc[0].strftime("%Y-%m-%d"), "End": last.Date.strftime("%Y-%m-%d"), "Sessions": len(nav),
                            "ROI_Pct": (float(last.NAV_Open) / capital - 1) * 100,
                            "ROI_Open_Pct": (float(last.NAV_Open) / capital - 1) * 100,
                            "ROI_If_Closed_Pct": (closed / capital - 1) * 100,
                            "MDD_Pct": float(drawdown.min()), "NAV_Open": float(last.NAV_Open), "NAV_If_Closed": closed,
                            "Total_Transaction_Cost": float(nav.Rebalance_Cost.sum()), "Hypothetical_Liquidation_Cost": float(last.Liquidation_If_Closed)})
            costs.append({"Arm": arm, "Display_Name": ARMS[arm], "Execution_Count": len(selected),
                          "Actual_Execution_Cost": actual_cost, "Buy_Fee": float(selected.Buy_Fee.sum()) if breakdown else None,
                          "Sell_Tax": float(selected.Sell_Tax.sum()) if breakdown else None, "NAV_Rebalance_Cost": float(nav.Rebalance_Cost.sum()),
                          "Hypothetical_Liquidation_Cost": float(last.Liquidation_If_Closed), "Cost_Breakdown_Available": breakdown})
        timeline, latest = _signals(root, book, sources, payload, fills, diagnostics, mode)
        targets = _target_weights(latest, timeline)
        holdings, reconciliation = _holdings(root, book, navs["v211"], fills, sources, proof, missing, mode)
        status, strategy, journal, ridge = v211_status_local(), next_day_strategy(), trading_journal(), ridge_status()
    metrics = pd.DataFrame(metrics)
    equal_roi = float(metrics.loc[metrics.Arm.eq("equal_policy"), "ROI_Open_Pct"].iloc[0])
    metrics["Gap_Vs_Benchmark_Pp"] = metrics.ROI_Open_Pct - equal_roi
    missing.extend(["未交付此V2帳本同期的0050可交易含息基準，不把相同政策等權改稱0050。",
                    "完整真實Ridge前瞻帳本與receipt／CHAIN未交付；合成資料不可補齊。"])
    if mode == "demo":
        missing.append("DEMO成熟日期缺完整同源市場日曆，未推算成真實成熟日期。")
        missing.append("DEMO進場日期是合成欄位，未認證次日收盤成交協議。")
    limitations = ["曲線使用NAV_Open，含已發生成本但仍有持倉；不是券商已平倉淨值。",
                   "ROI_If_Closed另扣來源最後一日假設平倉費，與已發生成交費分開。",
                   "權重沿用原帳本小數股；訊號目標與實際持倉分開，不把新預測當已成交。",
                   "少量重疊五日標籤與歷史發布時點不足以證明新聞增益或超越0050。"]
    for name, digest in sources.items():
        if hashlib.sha256(_path(root, name).read_bytes()).hexdigest() != digest:
            raise V2PresentationError("來源在載入期間變動，等待完整同步後再讀取")
    return {"available": True, "mode": mode, "is_demo": mode == "demo", "initial_capital": capital,
            "period_start": metrics.Start.iloc[0], "period_end": metrics.End.iloc[0],
            "model_arm": "v211", "benchmark_arm": "equal_policy", "benchmark_label": ARMS["equal_policy"],
            "metrics": metrics, "curves": pd.concat(curves, ignore_index=True),
            "nav": pd.concat(list(navs.values()), ignore_index=True), "exposure": pd.concat(exposures, ignore_index=True),
            "costs": pd.DataFrame(costs), "timeline": timeline, "holdings": holdings, "target_weights": targets,
            "holdings_reconciliation": reconciliation, "diagnostics": diagnostics, "status": status,
            "next_strategy": strategy, "trading_journal": pd.DataFrame(journal), "ridge": ridge,
            "missing": list(dict.fromkeys(missing)), "limitations": limitations, "sources": sources,
            "source_links": [{"path": path, "url": GITHUB + path} for path in sources],
            "curve_definition": "100 × NAV_Open / protocol initial capital; initial entry fee retained; drawdown includes initial wealth100"}
