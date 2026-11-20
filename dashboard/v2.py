"""Native V2 views using the same visual contract as V3."""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.data import ROOT, DataIntegrityError
from dashboard.project import load_tracker
from dashboard.research_charts import comparison_bars, wealth_chart
from dashboard.research_layout import RESEARCH_TABS
from dashboard.v2_presentation import load_v2_presentation
from dashboard.v2_history import history_available, load_v2_history, render_v2_history
from dashboard_v2.data_access.load import NAMES, v2_comparison
from dashboard_v2.data_access.prospective import (
    day_review, display_sources, signal_dates, source_status, trading_journal,
)

GITHUB = "https://github.com/tKe1vin/tw-stock-lstm-research/blob/main/"


def _table(frame):
    st.dataframe(frame, hide_index=True, width="stretch")


def _bars(frame, category, value, title, **kwargs):
    st.altair_chart(comparison_bars(frame, category, value, title, benchmark_label="相同政策等權", **kwargs),
                    use_container_width=True)


def _overview(data):
    metrics = data["metrics"].set_index("Arm")
    model, equal = metrics.loc[data["model_arm"]], metrics.loc[data["benchmark_arm"]]
    a, b, c, d = st.columns(4)
    a.metric("V2.11成本後估值報酬", f"{model.ROI_Open_Pct:.2f}%")
    b.metric("等權基準估值報酬", f"{equal.ROI_Open_Pct:.2f}%")
    c.metric("模型對等權差額", f"{model.ROI_Open_Pct - equal.ROI_Open_Pct:+.2f} 個百分點")
    d.metric("模型最大回撤", f"{model.MDD_Pct:.2f}%")
    st.caption("保存的NAV_Open已扣發生的模擬成交費，仍有未平倉部位。假設現在全部平倉的費用與報酬另列，不冒充已成交結果。")
    with st.container(border=True):
        left, right = st.columns([3, 2])
        with left:
            st.subheader("同期間，成本後的資金路徑")
        with right:
            view = st.radio("圖表視角", ["成本後淨值", "相對基準", "回撤路徑"], horizontal=True,
                            key="v2_curve_view", label_visibility="collapsed")
        st.altair_chart(wealth_chart(data["curves"], view, [data["benchmark_arm"], data["model_arm"]],
                                    data["benchmark_arm"], benchmark_label="等權"), use_container_width=True)
        st.caption("起始資金＝100，保留第一次進場成本；相對財富＝模型淨值÷等權淨值−1。只比較完整配對的同日帳本。")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("兩組結果放在一起看")
        _table(data["metrics"][["Display_Name", "ROI_Open_Pct", "ROI_If_Closed_Pct", "MDD_Pct", "Total_Transaction_Cost"]].rename(columns={
            "Display_Name": "組別", "ROI_Open_Pct": "估值報酬（%）", "ROI_If_Closed_Pct": "若末日平倉報酬（%）",
            "MDD_Pct": "最大回撤（%）", "Total_Transaction_Cost": "已發生成交費（元）"}))
    with right:
        st.subheader("這一輪的結論")
        gap = model.ROI_Open_Pct - equal.ROI_Open_Pct
        st.write(f"本段模型較同政策等權 **{gap:+.3f} 個百分點**；僅有 {int(model.Sessions)} 個估值日，屬初步研究追蹤。")
        st.write("本帳本尚缺同期、同規則的0050含息基準，不能據此宣稱超越0050。")
        st.caption("V2與V3的期間、預測任務與成交假設不同；統一呈現方式不代表可以直接合併排名。")


def _predictions(data):
    st.subheader("預測有沒有勝過簡單基準？")
    diagnostics = data["diagnostics"]
    if diagnostics is None:
        st.info("SYNTHETIC_DEMO不借用正式成熟標籤；請切回正式保存資料查看可核對的預測診斷。")
        return
    summary = diagnostics["summary"]
    a, b, c, d = st.columns(4)
    a.metric("已成熟訊號日", str(summary["signal_days"]))
    b.metric("股票預測列", str(summary["rows"]))
    c.metric("模型MAE", f"{summary['model_mae']:.3f} 百分點")
    d.metric("對零預測MAE改善", f"{summary['mae_improvement_pp']:+.3f} 百分點")
    st.caption("五日標籤＝Close[i+6]÷Close[i+1]−1；尚未成熟的預測不當成0%。同日五檔及重疊五日標籤相關，列數不是獨立樣本數。")
    left, right = st.columns(2)
    with left:
        errors = diagnostics["comparison"].melt(id_vars="Forecast", value_vars=["MAE", "RMSE"], var_name="誤差指標", value_name="誤差")
        _bars(errors, "誤差指標", "誤差", "預測誤差", series="Forecast", unit="百分點")
    with right:
        rows = diagnostics["rows"].copy()
        bounds = [min(rows.Predicted.min(), rows.Actual_5d_Pct.min()), max(rows.Predicted.max(), rows.Actual_5d_Pct.max())]
        scatter = alt.Chart(rows).mark_circle(size=65, color="#2563eb", opacity=.75).encode(
            x=alt.X("Predicted:Q", title="預測五日報酬（%）", scale=alt.Scale(domain=bounds)),
            y=alt.Y("Actual_5d_Pct:Q", title="實際五日報酬（%）", scale=alt.Scale(domain=bounds)),
            tooltip=["Signal_Date:N", "Ticker:N", "Predicted:Q", "Actual_5d_Pct:Q"])
        diagonal = alt.Chart(pd.DataFrame({"x": bounds, "y": bounds})).mark_line(color="#7b8799", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
        st.altair_chart((scatter + diagonal).properties(height=250, width="container"), use_container_width=True)
    _bars(diagnostics["by_day"], "Signal_Date", "MAE_Improvement_Pp", "逐日MAE改善（正值較佳）", unit="百分點", height=220)
    _table(diagnostics["comparison"])
    with st.expander("逐日誤差與已成熟預測列"):
        _table(diagnostics["by_day"])
        _table(diagnostics["rows"])
    st.info("目前樣本不足且歷史發布時點尚未完整認證；方向準確率不是交易勝率，未計算短期年化Sharpe或宣稱新聞有顯著增益。")


def _costs(data):
    st.subheader("曝險、持倉與交易成本")
    costs = data["costs"]
    model = costs.loc[costs.Arm.eq(data["model_arm"])].iloc[0]
    a, b, c = st.columns(3)
    a.metric("已發生模擬成交次數", str(int(model.Execution_Count)))
    b.metric("已發生成交費", "未提供" if pd.isna(model.Actual_Execution_Cost) else f"{model.Actual_Execution_Cost:,.2f} 元")
    c.metric("末日假設平倉費 · 尚未發生", f"{model.Hypothetical_Liquidation_Cost:,.2f} 元")
    st.caption("每日Liquidation_If_Closed是當時保存的端點估計，不能加總成每日支出；假平倉只採最新端點，不畫成歷史已平倉曲線。")
    left, right = st.columns(2)
    with left:
        if costs.Actual_Execution_Cost.notna().all():
            _bars(costs, "Display_Name", "Actual_Execution_Cost", "已發生成交費", unit="元")
        else:
            st.info("DEMO未提供完整成交費明細，保留缺值，不以0元補齊。")
    with right:
        _bars(costs, "Display_Name", "Hypothetical_Liquidation_Cost", "末日假設平倉費", unit="元")
    exposure = data["exposure"].loc[data["exposure"].Arm.eq(data["model_arm"])].copy()
    chart = alt.Chart(exposure).mark_area(opacity=.85).encode(
        x=alt.X("Date:T", title=None), y=alt.Y("Weight_Pct:Q", title="資金比例（%）", stack="zero", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Series:N", title=None, scale=alt.Scale(range=["#d9e1ec", "#2563eb"]), legend=alt.Legend(orient="bottom")),
        tooltip=["Date:T", "Series:N", alt.Tooltip("Weight_Pct:Q", format=".2f")])
    st.altair_chart(chart.properties(width="container", height=240), use_container_width=True)
    holdings = data["holdings"]
    if not holdings.empty:
        st.subheader("實際持倉與最新訊號目標分開看")
        frames = []
        for frame, label in ((holdings, "末日實際持倉"), (data["target_weights"], "最新訊號目標 · 未等同成交")):
            part = frame[["Ticker", "Weight_Pct"]].copy()
            part["權重來源"] = label
            frames.append(part)
        _bars(pd.concat(frames), "Ticker", "Weight_Pct", "持倉權重", series="權重來源", unit="%")
        st.caption(f"持倉估值日 {holdings.As_Of_Date.iloc[0]}；最後成交來源 {holdings.Source_Execution_Date.iloc[0]}。新訊號目標不會自動改寫實際股數。")
        with st.expander("持倉對帳明細"):
            _table(holdings)
            _table(data["target_weights"])
            st.json(data["holdings_reconciliation"])
    _table(costs)
    st.caption("Buy_Fee為買進手續費；Sell_Fee為賣出手續費；Sell_Tax為賣稅。原凍結成交檔未另存Sell_Fee，展示依原協議以賣稅×手續費率÷賣稅率重建，再核對三者合計等於已發生總成本；原始帳本保持不變。")
    st.write("V2沿用原帳本的小數股、手續費及交易稅，未納入最低20元手續費與滑價；V3使用整數股、最低費用及滑價。兩版報酬不直接相減。")


def _next_strategy(data):
    plan = data["next_strategy"]
    if plan.get("state") != "LATEST_PUBLISHED_SIGNAL":
        st.info(plan.get("state", "DATA_UNAVAILABLE"))
        return
    for col, label, value in zip(st.columns(4), ["訊號日", "預測假設進場", "再平衡排程", "交易資格"],
                                 [plan.get("signal_date"), plan.get("forecast_entry"), plan.get("scheduled_rebalance"), plan.get("trade_eligible")]):
        col.metric(label, str(value) if value is not None else "未保存")
    st.caption(f"{plan.get('created_at')}；Execution_Status={plan.get('execution_status')}；最近模擬成交日 {plan.get('last_simulated_execution')}。")
    if str(plan.get("trade_eligible")).lower() != "true":
        st.info("此訊號未具備已保存的再平衡交易資格；預測進場日與目標權重不能當成隔日掛單。")
    st.caption("預測假設進場日是五日標籤用途；帳本只在排程與資格通過時記錄模擬成交。")
    _table(pd.DataFrame(plan.get("rows", [])))


def _daily(data):
    status = data["status"]
    section = st.selectbox("每日運作內容", ["V2.11", "交易日誌", "當日檢討", "隔日策略", "本機前瞻狀態"], key="v2_section")
    if section == "V2.11":
        a, b, c = st.columns(3)
        a.metric("最後保存訊號日", status.get("latest_signal_date") or "未提供")
        b.metric("行情資料至", status.get("prices_through") or "未提供")
        c.metric("成熟訊號日", str(status.get("matured_signal_days", 0)))
        st.caption(f"來源 {status.get('source')}；{status.get('confirmation')}。權重核對 {status.get('weight_ok')}，registry核對 {status.get('registry_ok')}。")
        timeline = data["timeline"]
        events = timeline.melt(id_vars="Signal_Date", value_vars=["Forecast_Entry_Date", "Scheduled_Rebalance_Date", "Simulated_Execution_Date", "Label_End_Date"],
                               var_name="事件", value_name="日期").dropna(subset=["日期"])
        if not events.empty:
            events["日期"] = pd.to_datetime(events["日期"])
            events["事件"] = events["事件"].replace({"Forecast_Entry_Date": "預測假設進場", "Scheduled_Rebalance_Date": "排程再平衡", "Simulated_Execution_Date": "已保存模擬成交", "Label_End_Date": "標籤成熟"})
            chart = alt.Chart(events).mark_point(filled=True, size=70).encode(
                x=alt.X("日期:T", title="事件日期"), y=alt.Y("Signal_Date:N", title="訊號日"),
                color=alt.Color("事件:N", legend=alt.Legend(orient="bottom")), tooltip=["Signal_Date:N", "事件:N", "日期:T"])
            st.altair_chart(chart.properties(width="container", height=270), use_container_width=True)
        _table(timeline)
        st.caption("排程與標籤成熟點是日期說明，不代表當天已成交或歷史發布證據已認證。")
    elif section == "交易日誌":
        left, right = st.columns(2)
        day = left.selectbox("日誌訊號日", ["全部", *signal_dates()], key="v2_journal_day")
        ticker = right.selectbox("日誌股票", ["全部", *[key for key in NAMES if key != "^TWII"]], key="v2_ticker",
                                 format_func=lambda value: value if value == "全部" else f"{value} {NAMES[value]}")
        _table(pd.DataFrame(trading_journal(day, ticker)))
    elif section == "當日檢討":
        review = day_review()
        a, b, c = st.columns(3)
        a.metric("標籤狀態", review["label_state"])
        b.metric("成交狀態", review["trade_state"])
        c.metric("流程", "OK" if review["flow_ok"] else "ERROR")
        for label, key in (("成熟評分列", "matured_rows"), ("已發生模擬成交", "fills")):
            st.subheader(label)
            _table(pd.DataFrame(review[key]))
        st.caption(f"價格到 {review['prices_through']}；下一標籤日 {review['next_label_end']}。")
    elif section == "隔日策略":
        _next_strategy(data)
    else:
        st.write("正式模式讀GitHub保存帳本；來源名稱與hash一致不等於歷史發布時點已認證。")
        st.json(source_status(), expanded=False)


def _archive(data, section):
    if section == "Ridge 前瞻":
        ridge = data["ridge"]
        st.info(f"Ridge {ridge['status']}｜一日Close-only任務，不能與V2.11五日預測合併。")
        a, b, c = st.columns(3)
        a.metric("Ridge MAE", "未提供" if ridge["mae"] is None else f"{ridge['mae']:.4f}")
        b.metric("Ridge RMSE", "未提供" if ridge.get("rmse") is None else f"{ridge['rmse']:.4f}")
        c.metric("Ridge可評分交易日", str(ridge.get("scorable_days", 0)))
        st.caption("交付包舊快照不代表當前正式來源已附完整帳本。缺原始帳本及receipt／CHAIN時，不以DEMO或舊摘要認證成效。")
        st.json(ridge, expanded=False)
        return
    if section == "原 555 與 V2":
        _table(pd.DataFrame([
            {"版本": "原555", "任務": "六特徵、一日LSTM", "比較限制": "舊測試窗與成交協議不同"},
            {"版本": "V2歷史", "任務": "rolling五日、次日收盤、五日再平衡", "比較限制": "只和同帳equal_weekly比較"},
            {"版本": "V2.11", "任務": "凍結模型每日登錄、成熟後評分", "比較限制": "短期研究追蹤，時點證據尚未完整"},
            {"版本": "V3", "任務": "0050基準與新聞增量探索", "比較限制": "獨立協議，不進V2排名"}]))
        return
    if section == "限制":
        for note in data["limitations"] + data["missing"]:
            st.write("• " + note)
        return
    history = v2_comparison()
    st.warning("以下是歷史實驗彙總，期間及協議與上方V2.11每日帳本不同，不合併成一條曲線。各折權重未完整保存，仍為EXPLORATORY／UNVERIFIED。")
    ranked = pd.DataFrame(history["ranked"])
    if not ranked.empty:
        _bars(ranked, "Name", "ROI_Pct", "歷史同協議可比組報酬", unit="%")
        _table(ranked)
    if section == "歷史實驗":
        _table(pd.DataFrame(history["quality"]))
        _table(pd.DataFrame(history["blocked"]))
    st.caption("舊rolling_selected只有彙總，未交付完整每日NAV；保留缺口，不從彙總重建假曲線。")


def _sources(data):
    st.subheader("已補上什麼，還缺什麼？")
    _table(pd.DataFrame([
        {"項目": "每日NAV／相對等權／回撤圖", "狀態": "已補", "說明": "保存帳本衍生，保留首次成本"},
        {"項目": "曝險／實際持倉／訊號目標", "狀態": "已補" if not data["holdings"].empty else "部分可用", "說明": "股數×保存價格核對NAV；目標不作成交"},
        {"項目": "預測診斷／零預測基準", "狀態": "已補" if data["diagnostics"] is not None else "DEMO不混入正式診斷", "說明": "成熟後才評分，顯示樣本日數"},
        {"項目": "成本／時間線／來源雜湊", "狀態": "已補", "說明": "已發生與假設分列，發布／進場／再平衡／成交分開"},
        {"項目": "同期同規則0050含息帳本", "狀態": "待建立", "說明": "缺可交易基準，不能宣稱勝過0050"},
        {"項目": "Ridge完整原始前瞻鏈", "狀態": "缺原始資料", "說明": "receipt、CHAIN與完整每日帳本未提供"},
        {"項目": "舊實驗各折權重及完整NAV", "狀態": "缺原始資料", "說明": "舊結果保留UNVERIFIED，不補假曲線"},
        {"項目": "歷史新聞與發布時點證明", "狀態": "仍未完整", "說明": "新取得的標題／hash不能證明過去已可用"}]))
    st.caption("新聞標題、情緒與可用時點請看側欄「新聞與情緒」；舊快取缺標題的列維持缺值，後取得的標題附來源與取得時間。")
    section = st.selectbox("歷史與其他來源", ["V2 總覽", "原 555 與 V2", "歷史實驗", "Ridge 前瞻", "限制"], key="v2_archive_section")
    _archive(data, section)
    with st.expander("下載與研究來源"):
        a, b, c = st.columns(3)
        a.download_button("下載V2指標CSV", data["metrics"].to_csv(index=False).encode("utf-8-sig"), "v211_metrics.csv", "text/csv", width="stretch")
        b.download_button("下載V2淨值指數CSV", data["curves"].to_csv(index=False).encode("utf-8-sig"), "v211_wealth_indexes.csv", "text/csv", width="stretch")
        c.link_button("本輪補強與剩餘缺口", GITHUB + "docs/v2_v3_visualization_update.md", width="stretch")
        st.caption("只提供衍生摘要與淨值指數；不另散布完整供應商行情。檔案hash證明完整性，不等於歷史可用性。")
        st.json(data["sources"], expanded=False)


def render_v2(root=ROOT):
    root = Path(root)
    st.title("V2研究與每日檢討")
    st.caption("與V3採相同的摘要卡、互動圖與六個分頁；讀取保存研究及每日資料，每五分鐘重新讀取。沒有券商下單功能。")
    mode = st.radio("前瞻／日誌資料", ["saved", "demo"], horizontal=True, key="v2_data_mode",
                    format_func=lambda value: {"saved": "正式保存資料", "demo": "合成DEMO展示"}[value])
    available = history_available(root) if mode == "saved" else False
    if mode == "demo":
        st.session_state["v2_research_scope"] = "tracking"
        st.warning("SYNTHETIC_DEMO｜合成資料，只供介面展示，不是實際模型績效，也不計入正式結果。")
        st.caption("DEMO只展示每日追蹤介面，不借用正式歷史回測的預測、曲線或績效。")
    scope = st.radio("研究範圍", ["historical", "tracking"], horizontal=True,
                     index=0 if available else 1, key="v2_research_scope", disabled=mode == "demo",
                     format_func=lambda value: {"historical": "同標準歷史回測（150日）", "tracking": "每日凍結追蹤"}[value])
    if mode == "saved" and not available:
        st.warning("同標準歷史報告尚未同步；每日凍結追蹤仍可查看。選擇歷史範圍可查看來源檢查訊息。")
    if mode == "saved" and scope == "historical":
        try:
            history = load_v2_history(root)
        except (DataIntegrityError, OSError, ValueError, KeyError, RuntimeError) as exc:
            st.error("同標準歷史回測來源缺少或未通過完整性檢查，暫停顯示歷史績效。可手動切換「每日凍結追蹤」查看V2.11。")
            st.caption(str(exc))
            return

        def tracking_section(section=None):
            try:
                load_tracker(root)
                tracking = load_v2_presentation(root=root, mode="saved")
                with display_sources(mode="saved", root=root):
                    if section is None:
                        _daily(tracking)
                    else:
                        _archive(tracking, section)
            except (DataIntegrityError, OSError, ValueError, KeyError, RuntimeError) as exc:
                st.warning("V2.11每日追蹤來源未完整同步，歷史回測仍保留獨立呈現。")
                st.caption(str(exc))

        render_v2_history(history, daily_renderer=tracking_section, archive_renderer=tracking_section)
        return
    if mode == "saved":
        st.info("正式保存資料｜V2.11凍結模型的研究追蹤，歷史發布時點尚未完整認證，維持PENDING。缺少的Ridge來源不以DEMO補齊。")
    try:
        if mode == "saved":
            load_tracker(root)
        data = load_v2_presentation(root=root, mode=mode)
        st.caption(f"估值期間 {data['period_start']} — {data['period_end']}；基準：{data['benchmark_label']}；初始資金 {data['initial_capital']:,.0f} 元。")
        with display_sources(mode=mode, root=root):
            tabs = st.tabs(RESEARCH_TABS)
            with tabs[0]: _overview(data)
            with tabs[1]: _predictions(data)
            with tabs[2]: _costs(data)
            with tabs[3]: _daily(data)
            with tabs[4]:
                if mode == "demo":
                    st.info("優化診斷只使用正式保存的成熟標籤；請切回「正式保存資料」。")
                else:
                    from dashboard.v2_diagnostics import render_v2_diagnostics
                    render_v2_diagnostics(root)
                st.write("下一步先固定新聞有／無、動態權重有／無及風控有／無的配對實驗，再建立同期0050含息帳本；完整保存各折權重、資料時點及成本，不以已看過的測試報酬選模。")
            with tabs[5]: _sources(data)
    except (DataIntegrityError, OSError, ValueError, KeyError, RuntimeError) as exc:
        st.error("V2來源尚未完整同步或未通過檢查，暫停顯示此頁結果。網站會定時重新讀取。")
        st.caption(str(exc))
