"""Read-only views of the complete project's retained model evidence.

Only derived result tables, model status and research protocols are read here.
No model runtime, external provider, raw price panel or training is invoked.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.data import DataIntegrityError, ROOT, load_daily_status

GITHUB = "https://github.com/tKe1vin/tw-stock-lstm-research/blob/main/"
MODELS = {
    "444": ("444｜價格 LSTM", "單變量價格預測核心；固定持股為主研究基準。"),
    "555": ("555｜配置＋新聞＋風控", "與444共用預測，再檢驗動態權重、新聞與風控；不是重新訓練另一個主比較模型。"),
    "555_v2": ("555 V2｜凍結預測", "保留原凍結五日權重；每日輸出預測，不等同已發布交易訊號或已實現報酬。"),
    "v211": ("V2.11｜研究帳本", "依不可覆寫訊號追蹤次日收盤模擬成交、五日排程與成熟標籤。"),
}
PAIRED_NAMES = {
    "static_equal444": "444固定持股", "dynamic_no_news": "動態配置／無新聞",
    "dynamic_news_no_risk": "動態配置＋新聞", "dynamic_no_news_risk": "動態配置＋風控／無新聞",
    "dynamic_news_risk555": "555配置＋新聞＋風控", "equal_rebalanced_reference": "每日再平衡等權",
    "risk_matched_equal_reference": "相同政策曝險等權", "theoretical_twii_reference": "理論TWII指數參考",
}
TRACKER = "experiments/results/v2/v211"


def _file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise DataIntegrityError("結果來源不可超出專案目錄")
    return path


def _json(root, relative):
    path = _file(root, relative)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataIntegrityError(f"結果JSON無法讀取：{relative}") from exc
    if not isinstance(value, dict):
        raise DataIntegrityError(f"結果JSON格式錯誤：{relative}")
    return value


def _csv(root, relative):
    path = _file(root, relative)
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, ValueError, pd.errors.EmptyDataError) as exc:
        raise DataIntegrityError(f"結果CSV無法讀取：{relative}") from exc


def _date(value):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)):
        raise DataIntegrityError("請使用保存的ISO資料日")
    try:
        pd.Timestamp(value)
    except ValueError as exc:
        raise DataIntegrityError("資料日不合法") from exc
    return str(value)


def model_dates(model, root=ROOT):
    if model not in MODELS:
        raise DataIntegrityError("未知模型")
    parent = _file(root, f"outputs/{model}")
    return sorted((p.name for p in parent.iterdir() if p.is_dir()
                   and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)
                   and (p / "status.json").is_file()), reverse=True) if parent.is_dir() else []


def load_model_output(model, day, root=ROOT):
    if model not in MODELS:
        raise DataIntegrityError("未知模型")
    base = f"outputs/{model}/{_date(day)}"
    status = _json(root, base + "/status.json")
    if status is None:
        raise DataIntegrityError("所選資料日沒有保存模型狀態")
    tables = {}
    for name, relative in {
        "unified_signal": "unified_signal.csv", "recommendation": "native/next_day_recommendation.csv",
        "model_metrics": "native/model_metrics.csv", "native_strategy_metrics": "native/strategy_metrics.csv",
    }.items():
        value = _csv(root, base + "/" + relative)
        if value is not None:
            tables[name] = {"data": value, "path": base + "/" + relative}
    execution_audit = None
    if model == "555_v2":
        signal = tables.get("unified_signal", {}).get("data")
        saved = sorted(set(signal.execution_status.dropna().astype(str))) if signal is not None and "execution_status" in signal else []
        execution_audit = {
            "forecast_only": status.get("signal_published") is False,
            "saved_execution_statuses": saved,
            "legacy_status_discrepancy": any(value not in {"forecast_only", "forecast_expired"} for value in saved),
        }
    return {"model": model, "day": day, "status": status, "tables": tables, "base": base,
            "execution_audit": execution_audit}


def paired_dates(root=ROOT):
    parent = _file(root, "reports/daily")
    if not parent.is_dir():
        return []
    return sorted((p.name for p in parent.iterdir() if p.is_dir()
                   and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)
                   and (p / "paired/metrics.csv").is_file()), reverse=True)


def load_paired(day=None, root=ROOT, *, archived=False):
    base = "reports/research_revision/2026-11-06/paired" if archived else f"reports/daily/{_date(day)}/paired"
    protocol = _json(root, base + "/protocol.json")
    manifest = _json(root, base + "/reproduction_manifest.json")
    if not protocol or not manifest or protocol.get("protocol_version") != "paired-444-core-allocation-v1":
        raise DataIntegrityError("缺少共同444核心的配對協議或來源manifest")
    for name in ("metrics.csv", "nav.csv"):
        path = _file(root, base + "/" + name)
        wanted = manifest.get("outputs_sha256", {}).get(name)
        if not path.is_file() or not wanted or hashlib.sha256(path.read_bytes()).hexdigest() != wanted:
            raise DataIntegrityError(f"配對結果雜湊不符：{name}")
    metrics, nav = _csv(root, base + "/metrics.csv"), _csv(root, base + "/nav.csv")
    required = {"Arm", "ROI_Pct", "MDD_Pct", "First_Execution_Date", "Last_Valuation_Date", "Observations"}
    if not required.issubset(metrics) or not {"Arm", "Date", "NAV"}.issubset(nav):
        raise DataIntegrityError("配對績效或帳本缺必要欄位")
    if metrics.Arm.duplicated().any() or nav.duplicated(["Arm", "Date"]).any() or set(metrics.Arm) != set(nav.Arm):
        raise DataIntegrityError("配對結果組別或日期鍵不一致")
    if not {"static_equal444", "dynamic_news_risk555"}.issubset(metrics.Arm):
        raise DataIntegrityError("主比較缺少444／555")
    if metrics.First_Execution_Date.nunique() != 1 or metrics.Last_Valuation_Date.nunique() != 1:
        raise DataIntegrityError("配對績效使用不同期間")
    for frame, columns in ((metrics, ["ROI_Pct", "MDD_Pct", "Observations"]), (nav, ["NAV"])):
        try:
            finite = np.isfinite(frame[columns].to_numpy(float)).all()
        except (ValueError, TypeError) as exc:
            raise DataIntegrityError("配對績效含非數值") from exc
        if not finite:
            raise DataIntegrityError("配對績效缺值")
    capital = float(protocol.get("rules", {}).get("initial_capital", 0))
    if capital <= 0 or (nav.NAV <= 0).any():
        raise DataIntegrityError("配對帳本初始資金或NAV不合法")
    nav = nav.copy()
    nav.Date = pd.to_datetime(nav.Date, errors="raise")
    expected = None
    for row in metrics.itertuples():
        part = nav[nav.Arm.eq(row.Arm)].sort_values("Date")
        dates = part.Date.reset_index(drop=True)
        if expected is not None and not dates.equals(expected):
            raise DataIntegrityError("配對帳本日期未對齊")
        expected = dates
        if len(part) != int(row.Observations) or str(part.Date.iloc[0].date()) != str(row.First_Execution_Date)[:10] or str(part.Date.iloc[-1].date()) != str(row.Last_Valuation_Date)[:10]:
            raise DataIntegrityError("配對日期或觀察數與績效表不符")
        if abs((float(part.NAV.iloc[-1]) / capital - 1) * 100 - float(row.ROI_Pct)) > 1e-7:
            raise DataIntegrityError("配對帳本期末報酬無法重算")
    metrics = metrics.copy()
    metrics["組別"] = metrics.Arm.map(PAIRED_NAMES).fillna(metrics.Arm)
    nav["組別"] = nav.Arm.map(PAIRED_NAMES).fillna(nav.Arm)
    nav["淨值指數"] = nav.NAV / capital * 100
    return {"base": base, "metrics": metrics, "nav": nav, "protocol": protocol,
            "start": str(metrics.First_Execution_Date.iloc[0]), "end": str(metrics.Last_Valuation_Date.iloc[0])}


def load_tracker(root=ROOT):
    integrity = _json(root, TRACKER + "/ledger_integrity.json")
    tables = {}
    for name in ("live_nav", "equal_nav", "live_fills", "live_book", "matured_scores"):
        relative = TRACKER + "/" + name + ".csv"
        table = _csv(root, relative)
        if table is None:
            continue
        proof = (integrity or {}).get("prefixes", {}).get(name + ".csv")
        if name != "live_book":
            raw = _file(root, relative).read_bytes()
            if not proof or len(raw) < proof["bytes"] or hashlib.sha256(raw[:proof["bytes"]]).hexdigest() != proof["sha256"]:
                raise DataIntegrityError(f"V2.11保存帳本完整性檢查失敗：{name}")
        tables[name] = {"data": table, "path": relative}
    signal_files = sorted(_file(root, TRACKER + "/daily").glob("????-??-??.csv"))
    return {"tables": tables, "last_signal_date": signal_files[-1].stem if signal_files else None,
            "signal_count": len(signal_files), "historical_availability_certified": bool((integrity or {}).get("historical_availability_certified"))}


def _table(item, label, key):
    st.dataframe(item["data"], hide_index=True, width="stretch")
    st.caption("來源：" + item["path"])
    left, right = st.columns(2)
    left.link_button("查看來源檔", GITHUB + item["path"])
    right.download_button(label + " CSV", item["data"].to_csv(index=False).encode("utf-8-sig"),
                          file_name=Path(item["path"]).name, mime="text/csv", key=key)


def _paired_view(evidence, prefix):
    metrics = evidence["metrics"].set_index("Arm")
    a, b, c = st.columns(3)
    a.metric("444配對固定持股淨報酬", f"{metrics.loc['static_equal444', 'ROI_Pct']:.2f}%")
    b.metric("555配對新聞＋風控淨報酬", f"{metrics.loc['dynamic_news_risk555', 'ROI_Pct']:.2f}%")
    if {"dynamic_no_news_risk", "dynamic_news_risk555"}.issubset(metrics.index):
        increment = metrics.loc["dynamic_news_risk555", "ROI_Pct"] - metrics.loc["dynamic_no_news_risk", "ROI_Pct"]
        c.metric("相同風控下的新聞差額", f"{increment:+.2f} 個百分點")
    st.caption(f"同一協議及共同執行期間 {evidence['start']}～{evidence['end']}。新舊資料日的回測起點可能不同，不能串成真實持有績效。")
    st.info("探索性配對研究：共用444預測、次日收盤與相同成本；分數不是每日實際獲利。歷史新聞當時可用性未完整證明。TWII是理論價格指數參考，不是可交易0050或含息報酬。")
    options = list(metrics.index)
    defaults = [arm for arm in ("static_equal444", "dynamic_news_risk555", "dynamic_no_news_risk", "risk_matched_equal_reference") if arm in options]
    selected = st.multiselect("顯示比較組", options, default=defaults,
                              format_func=lambda x: PAIRED_NAMES.get(x, x), key=prefix + "arms")
    if selected:
        chart = alt.Chart(evidence["nav"][evidence["nav"].Arm.isin(selected)]).mark_line().encode(
            x=alt.X("Date:T", title="日期"), y=alt.Y("淨值指數:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("組別:N", title=None), tooltip=[alt.Tooltip("Date:T", format="%Y-%m-%d"), "組別:N", alt.Tooltip("淨值指數:Q", format=".2f")]
        ).properties(width="container", height=330)
        st.altair_chart(chart, use_container_width=True)
    columns = ["組別", "ROI_Pct", "Sharpe", "MDD_Pct", "Total_Transaction_Cost", "Average_Equity_Weight_Pct"]
    table = evidence["metrics"][[c for c in columns if c in evidence["metrics"]]].rename(columns={
        "ROI_Pct": "淨報酬（%）", "MDD_Pct": "最大回撤（%）", "Total_Transaction_Cost": "交易成本（元）", "Average_Equity_Weight_Pct": "平均股票曝險（%）"})
    _table({"data": table, "path": evidence["base"] + "/metrics.csv"}, "配對績效", prefix + "metrics_download")
    with st.expander("逐日帳本與共同協議"):
        _table({"data": evidence["nav"], "path": evidence["base"] + "/nav.csv"}, "配對帳本", prefix + "nav_download")
        st.json(evidence["protocol"])


def render_project(root=ROOT):
    st.subheader("444、555與延伸模型的研究結果")
    st.caption("只讀本次部署保存的資料；每個結果標明資料日與協議，預測、歷史回測與模擬帳本分開呈現。")
    for col, (model, (label, description)) in zip(st.columns(4), MODELS.items()):
        with col:
            with st.container(border=True):
                st.markdown("**" + label + "**")
                st.write(description)
    views = st.tabs(["444／555共同基準", "每日模型輸出", "V2.11帳本與成熟標籤", "修訂與延伸研究"])
    with views[0]:
        days = paired_dates(root)
        if not days:
            st.info("尚無保存的444／555配對研究；未以原生舊ROI代替共同基準。")
        else:
            day = st.selectbox("配對研究資料日", days, key="project_pair_day")
            try:
                _paired_view(load_paired(day, root), "project_pair_")
            except (DataIntegrityError, OSError, ValueError, KeyError) as exc:
                st.error("配對研究無法通過來源檢查；暫停顯示該比較。")
                st.code(str(exc), language=None)
    with views[1]:
        try:
            status = load_daily_status(root)
            if status.get("available"):
                st.write(f"最新整合狀態：資料日 **{status['as_of_date']}**，真正執行日 **{status['run_date']}**。")
                st.caption("完成與否以該保存資料日為準，不代表目前仍可成交；交易資格仍需查看訊號的排程與執行欄位。")
                rows = [{"模型": MODELS[item['model']][0], "保存狀態": item.get("status"), "資料日": item.get("observed_as_of_date"),
                         "真正生成時間": item.get("generated_at"), "用途／執行狀態": item.get("recommendation_execution_status"),
                         "僅維護": bool(item.get("maintenance_only"))} for item in status["models"]]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        except (DataIntegrityError, ValueError, KeyError) as exc:
            st.warning("整合狀態無法讀取：" + str(exc))
        model = st.selectbox("模型", list(MODELS), format_func=lambda x: MODELS[x][0], key="project_model")
        days = model_dates(model, root)
        if not days:
            st.info("此模型尚無保存的日期輸出。")
        else:
            day = st.selectbox("模型資料日", days, key="project_model_day_" + model)
            try:
                result = load_model_output(model, day, root)
                state = result["status"]
                st.write(f"**資料日 {day}** · 保存狀態 `{state.get('status', '未提供')}` · 實際執行日 `{state.get('run_date', '未提供')}`")
                st.caption(f"生成時間：{state.get('generated_at') or '未提供'}；用途：{state.get('recommendation_execution_status') or state.get('reason') or '未提供'}。")
                if model == "555_v2":
                    st.info("555 V2此處是凍結模型預測；不是已實現績效，亦非已發布研究交易訊號。")
                    audit = result["execution_audit"]
                    if audit["legacy_status_discrepancy"]:
                        st.warning("此日舊匯出列的執行狀態標記與僅預測用途不一致（"
                                   + "、".join(audit["saved_execution_statuses"])
                                   + "）；不視為可交易訊號或成交紀錄。下表與下載保留原檔數值。")
                if model == "v211":
                    st.info("V2.11狀態以此資料日保存檔為準；僅維護時不會補造當日訊號。帳本請看下一個分頁。")
                tables = result["tables"]
                primary = tables.get("unified_signal") or tables.get("recommendation")
                if primary is not None:
                    st.markdown("**已保存預測／研究目標**")
                    _table(primary, "模型輸出", "project_forecast_download_" + model)
                else:
                    st.info("所選日未保存預測表；缺檔不解讀為零預測或已執行。")
                if "model_metrics" in tables:
                    with st.expander("預測誤差與方向準確率"):
                        _table(tables["model_metrics"], "預測誤差", "project_error_download_" + model)
                if "native_strategy_metrics" in tables:
                    with st.expander("原生／歷史診斷績效（另有協議，不作主比較）"):
                        st.warning("原生策略的持有與成交協議可能不同。444對555的正式比較請回到共同基準分頁，不能混排此處ROI。")
                        _table(tables["native_strategy_metrics"], "原生診斷", "project_native_download_" + model)
                with st.expander("狀態原始紀錄"):
                    st.json(state)
                    st.link_button("查看來源狀態", GITHUB + result["base"] + "/status.json")
            except (DataIntegrityError, OSError, ValueError, KeyError) as exc:
                st.error("模型保存輸出無法讀取：" + str(exc))
    with views[2]:
        try:
            tracker = load_tracker(root)
            st.write(f"已保存 **{tracker['signal_count']}** 個訊號日；最後訊號日 **{tracker['last_signal_date'] or '未提供'}**。")
            st.warning("研究模擬帳本，無券商實際下單。初始歷史訊號缺獨立發布前證據；成熟五日標籤彼此重疊，少量觀察不能證明模型有效。")
            tables = tracker["tables"]
            if "live_nav" in tables:
                nav = tables["live_nav"]["data"]
                st.caption(f"保存帳本估值期間：{nav.Date.min()}～{nav.Date.max()}；NAV_Open是已持有部位估值，期末假設平倉費另列，不能直接當作已平倉淨ROI。")
                frames = []
                for name, label in (("live_nav", "V2.11研究帳本"), ("equal_nav", "相同政策等權帳本")):
                    if name in tables:
                        part = tables[name]["data"][["Date", "NAV_Open"]].copy()
                        part["帳本"] = label
                        frames.append(part)
                chart = alt.Chart(pd.concat(frames, ignore_index=True)).mark_line(point=True).encode(
                    x="Date:T", y=alt.Y("NAV_Open:Q", title="持倉估值（元）", scale=alt.Scale(zero=False)), color="帳本:N", tooltip=["Date:T", "帳本:N", "NAV_Open:Q"]
                ).properties(width="container", height=270)
                st.altair_chart(chart, use_container_width=True)
            for name, label in (("live_nav", "逐日估值"), ("equal_nav", "等權估值"), ("live_book", "帳本摘要"), ("live_fills", "模擬成交"), ("matured_scores", "成熟預測與實際五日報酬")):
                if name in tables:
                    with st.expander(label, expanded=name == "matured_scores"):
                        if name == "matured_scores":
                            frame = tables[name]["data"]
                            st.caption(f"{len(frame)}筆股票標籤、{frame.Signal_Date.nunique()}個訊號日；股票列數不是獨立時間樣本數。")
                        _table(tables[name], label, "project_tracker_" + name)
            if not tables:
                st.info("尚無保存的模擬帳本或成熟標籤。")
        except (DataIntegrityError, OSError, ValueError, KeyError) as exc:
            st.error("V2.11帳本無法通過來源檢查：" + str(exc))
    with views[3]:
        st.markdown("**2026-11-06修訂共同核心研究**")
        try:
            _paired_view(load_paired(root=root, archived=True), "project_archive_")
        except (DataIntegrityError, OSError, ValueError, KeyError) as exc:
            st.info("修訂研究來源未完整保存或未通過檢查：" + str(exc))
        st.divider()
        st.markdown("**固定多變量LSTM延伸（與V3、每日模型分開）**")
        base = "reports/research_extension/2026-11-06"
        try:
            summary, manifest = _json(root, base + "/summary.json"), _json(root, base + "/reproduction_manifest.json")
            table = _csv(root, base + "/strategy_metrics.csv")
            if summary is None or table is None or manifest is None:
                st.info("此checkout缺少延伸研究來源。")
            else:
                expected = manifest.get("files_sha256", {}).get("strategy_metrics.csv")
                if not expected or hashlib.sha256(_file(root, base + "/strategy_metrics.csv").read_bytes()).hexdigest() != expected:
                    raise DataIntegrityError("延伸研究績效來源hash不符")
                st.caption(f"保存帳本期間 {summary['portfolio_start']}～{summary['portfolio_end']}；驗證集採用候選：{summary['selected']}。")
                st.info("這是已看過歷史的探索研究，未替換正式模型。TWII比較組是理論指數成本代理，不是可交易ETF；不能與0050或其他期間直接排序。")
                _table({"data": table, "path": base + "/strategy_metrics.csv"}, "延伸績效", "project_extension_download")
                with st.expander("延伸研究摘要與限制"):
                    st.json(summary)
        except (DataIntegrityError, OSError, ValueError, KeyError) as exc:
            st.error("延伸研究來源檢查未通過：" + str(exc))
        st.link_button("延伸研究完整說明", GITHUB + base + "/summary.md")
