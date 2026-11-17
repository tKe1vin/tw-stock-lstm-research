"""Read-only project portal for retained models, journals and news evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.data import DataIntegrityError, STUDIES, load_daily_status, load_study

GITHUB = "https://github.com/tKe1vin/tw-stock-lstm-research/blob/main/"
COLORS = {"0050 基準": "#7b8799", "新聞 Ridge": "#2563eb",
          "價格 Ridge": "#c48b29", "0050＋五股等權": "#8063c5"}
REASONS = {
    "insufficient_positive_folds": "正超額折數不足",
    "nonpositive_validation_excess": "驗證累積超額未為正",
    "news_does_not_beat_same_family_control": "新聞未優於同家族價格控制",
    "tracking_error_limit": "追蹤誤差超過限制",
    "nonpositive_slippage_stress_excess": "滑價壓力情境未為正",
}
CSS = """
<style>
html, body, [class*="css"], .stApp {font-family: 'Noto Sans CJK TC', 'Microsoft JhengHei', sans-serif;}
.block-container {max-width: 1320px; padding-top: 2.6rem; padding-bottom: 3rem;}
h1 {font-size: 2.2rem !important; letter-spacing: -.035em; font-weight: 750 !important;}
h2 {font-size: 1.45rem !important; letter-spacing: -.02em;}
h3 {font-size: 1.07rem !important;}
.eyebrow {font-size: .72rem; letter-spacing: .16em; font-weight: 700; color: #5d7393; margin-bottom: .45rem;}
.subline {color: #65758c; font-size: .96rem; margin-top: -.55rem; margin-bottom: 1.25rem; line-height: 1.8;}
[data-testid="stSidebar"] {border-right: 1px solid #e2e8f0;}
[data-testid="stMetric"] {background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.05rem 1.15rem; min-height: 115px;}
[data-testid="stMetricValue"] {font-size: 1.8rem; font-weight: 650;}
button[data-baseweb="tab"] {font-size: .9rem; padding-left: .6rem; padding-right: .6rem;}
.section-note {color:#65758c; font-size:.87rem; line-height:1.8;}
.app-footer {font-size:.77rem; color:#7b8799; border-top:1px solid #e2e8f0; padding-top:1rem; margin-top:2rem;}
</style>
"""


def pp(value):
    return f"{value:+.2f} 個百分點"


def curve_chart(curves, metric, arms):
    """Chart complete common dates; never combine different study periods."""
    frame = curves.copy()
    benchmark = frame.loc[frame.Arm.eq("B0_benchmark"), ["Date", "Wealth_Index"]].rename(
        columns={"Wealth_Index": "Benchmark_Wealth"})
    frame = frame.merge(benchmark, on="Date", validate="many_to_one")
    frame["Relative_Wealth_Pct"] = (frame.Wealth_Index / frame.Benchmark_Wealth - 1) * 100
    frame = frame[frame.Arm.isin(arms)].copy()
    field, title = {"成本後淨值": ("Wealth_Index", "淨值指數（起始資金＝100）"),
                    "相對0050": ("Relative_Wealth_Pct", "相對0050財富增長（%）"),
                    "回撤路徑": ("Drawdown_Pct", "回撤（%）")}[metric]
    names = [name for name in COLORS if name in set(frame.Display_Name)]
    palette = alt.Scale(domain=names, range=[COLORS[name] for name in names])
    hover = alt.selection_point(fields=["Date"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(frame).encode(
        x=alt.X("Date:T", title=None, axis=alt.Axis(format="%m/%d", grid=False)),
        y=alt.Y(f"{field}:Q", title=title, scale=alt.Scale(zero=False)),
        color=alt.Color("Display_Name:N", title=None, scale=palette,
                        legend=alt.Legend(orient="bottom", labelLimit=200)),
        tooltip=[alt.Tooltip("Date:T", title="日期", format="%Y-%m-%d"),
                 alt.Tooltip("Display_Name:N", title="組別"), alt.Tooltip(f"{field}:Q", title=title, format=".2f")])
    line = base.mark_line(strokeWidth=2.4)
    points = base.mark_circle(size=55).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0))).add_params(hover)
    reference = alt.Chart(pd.DataFrame({"value": [100 if metric == "成本後淨值" else 0]})).mark_rule(
        color="#bdc8d7", strokeDash=[4, 5]).encode(y="value:Q")
    return (reference + line + points).properties(width="container", height=345).configure_view(strokeWidth=0).configure_axis(
        labelColor="#6b7a90", titleColor="#6b7a90", gridColor="#e9edf4", titleFontWeight="normal")


def overview(study):
    metrics = study["metrics"].set_index("Arm")
    news, price, benchmark = (metrics.loc[study[key]] for key in ("news_arm", "price_arm", "benchmark_arm"))
    a, b, c, d = st.columns(4)
    a.metric("新聞候選淨報酬 · 未採用", f"{news.ROI_Pct:.2f}%")
    b.metric("0050基準淨報酬", f"{benchmark.ROI_Pct:.2f}%")
    c.metric("新聞對0050差額", pp(news.ROI_Pct - benchmark.ROI_Pct))
    d.metric("新聞對同架構價格控制", pp(news.ROI_Pct - price.ROI_Pct))
    st.caption("摘要使用完整留出期與相同成本假設；新聞增量只對照同家族價格模型，不能由單段留出結果決定採用。")
    with st.container(border=True):
        left, right = st.columns([3, 2])
        with left:
            st.subheader("同期間，成本後的資金路徑")
        with right:
            metric = st.radio("圖表視角", ["成本後淨值", "相對0050", "回撤路徑"], horizontal=True,
                              label_visibility="collapsed", key="curve_view")
        include_equal = st.checkbox("加入0050＋五股等權控制組", value=False, key="show_equal")
        arms = [study["benchmark_arm"], study["price_arm"], study["news_arm"]]
        if include_equal:
            arms.append("B1_equal")
        # Pinned 1.51 needs the legacy Vega sizing flag to resize the SVG itself.
        st.altair_chart(curve_chart(study["curves"], metric, arms), use_container_width=True)
        st.caption("淨值由已保存帳本計算，保留進場及期末成本。相對財富使用「策略淨值 ÷ 0050淨值 − 1」，與摘要的報酬百分點差額分開。")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("四組結果放在一起看")
        table = study["metrics"][["Display_Name", "ROI_Pct", "Gap_Vs_Benchmark_Pp", "MDD_Pct", "Total_Transaction_Cost"]].rename(
            columns={"Display_Name": "組別", "ROI_Pct": "淨報酬（%）", "Gap_Vs_Benchmark_Pp": "對0050（百分點）",
                     "MDD_Pct": "最大回撤（%）", "Total_Transaction_Cost": "交易成本（元）"})
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            key: st.column_config.NumberColumn(format="%.2f") for key in table.columns[1:]})
    with right:
        st.subheader("這一輪的結論")
        st.write("目前沒有可採用的新聞候選。三折驗證採用結果仍是 **0050基準（B0）**。")
        st.write("等權控制組可幫助辨認股票池與配置的影響；它的勝出不能直接算成新聞的效果。")
        st.caption("新聞模型的留出期結果為失敗診斷，沒有因留出表現改選，也沒有替換每日模型。")


def validation(study):
    st.subheader("先通過驗證，才考慮採用")
    st.write("三折皆使用時間切分與成熟標籤隔離。採用決策在留出期評分前寫入；已看過的歷史仍屬探索。")
    rows = []
    for candidate, part in study["validation"].groupby("Candidate", sort=False):
        rows.append({"候選": candidate, "正超額折數": f"{int(part.Relative_Wealth_Excess_Pct.gt(0).sum())}/{len(part)}",
                     "驗證累積相對財富（%）": (np.prod(1 + part.Relative_Wealth_Excess_Pct / 100) - 1) * 100,
                     "壓力情境相對財富（%）": (np.prod(1 + part.Stress_Relative_Wealth_Excess_Pct / 100) - 1) * 100,
                     "最高年化追蹤誤差（%）": part.Tracking_Error_Annualized.max() * 100})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        name: st.column_config.NumberColumn(format="%.2f") for name in rows[0] if "（%）" in name})
    for assessment in study["selection"].get("assessments", []):
        with st.expander(f"{assessment['candidate']}｜未通過的門檻", expanded=True):
            for reason in assessment.get("rejection_reasons", []):
                st.write("• " + REASONS.get(reason, reason))
    st.caption("至少兩折正超額、驗證累積及滑價壓力情境為正、勝過同家族無新聞控制，且追蹤誤差不超過登記上限；沒有通過者就維持0050。")
    with st.expander("逐折數字與選模原始紀錄"):
        st.dataframe(study["validation"], hide_index=True, width="stretch")
        st.json(study["selection"])


def costs(study):
    st.subheader("落差有多少來自成本？")
    metrics = study["metrics"].set_index("Arm")
    n, b = metrics.loc[study["news_arm"]], metrics.loc[study["benchmark_arm"]]
    gap = float(n.ROI_Pct - b.ROI_Pct)
    cost_gap = float((n.Total_Transaction_Cost - b.Total_Transaction_Cost) / n.Initial_Capital * 100)
    c1, c2, c3 = st.columns(3)
    c1.metric("對0050淨報酬差", pp(gap))
    c2.metric("額外實現成本的影響", pp(-cost_gap))
    c3.metric("其餘持倉／現金路徑差", pp(gap + cost_gap))
    st.caption("這是同一組已實現帳本的會計差異分解。移除成本會改變開倉門檻與交易路徑，不能把它直接當成減費後的回測或因果歸因。")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("成本與回撤")
        names = ["Display_Name", "Total_Transaction_Cost", "MDD_Pct", "Tracking_Error_Annualized"]
        table = study["metrics"][names].rename(columns={"Display_Name": "組別", "Total_Transaction_Cost": "成本（元）",
            "MDD_Pct": "最大回撤（%）", "Tracking_Error_Annualized": "年化追蹤誤差（%）"})
        table["年化追蹤誤差（%）"] *= 100
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            name: st.column_config.NumberColumn(format="%.2f") for name in table.columns[1:]})
    with right:
        st.subheader("固定交易假設")
        execution = study["config"]["execution"]
        st.write(f"每 **{study['hold_sessions']} 個交易日**排程，次日收盤理想化成交；整數股、不借款。")
        st.write("手續費每腿0.1425%、最低20元；股票賣稅0.3%、ETF賣稅0.1%。")
        st.write(f"每邊滑價 **{execution['slippage_bps_each_side']:g} bps**；壓力情境 **{execution['stress_slippage_bps_each_side']:g} bps**。")
        st.write("股利先列應收，付款後才可使用；股票與0050切換的各交易腿均計費。")
    st.info("主要限制：歷史新聞當時可用性尚未完整證明、原五股缺研究前選定證據、收盤成交仍為模型假設。")


def daily():
    status = load_daily_status()
    st.subheader("每日研究的最新保存狀態")
    if not status.get("available"):
        st.info("此checkout尚無每日狀態；V3封存結果仍可獨立展示。")
        return
    a, b, c = st.columns(3)
    a.metric("資料截止交易日", status.get("as_of_date") or "未提供")
    b.metric("真正執行日", status.get("run_date") or "未提供")
    c.metric("444／555研究", "完成" if status.get("research_status") == "complete" else "尚未完成")
    st.caption("此頁讀取checkout保存的日誌。V3研究與每日444／555分開；歷史結果不代表今天的即時績效。")
    rows = []
    for item in status.get("models", []):
        label = "僅維護，未補寫訊號" if item.get("maintenance_only") else "完成" if item.get("current") else "等待／未完成"
        rows.append({"系統": item["model"], "狀態": label, "資料日": item.get("observed_as_of_date"),
                     "實際生成時間": item.get("generated_at") or "—"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if status.get("status") != "complete":
        st.info("四系統狀態仍為部分完成：延遲V2.11只維護既有紀錄，沒有補造過去前瞻訊號。444／555的研究已分開呈現。")
    news = status.get("news", {})
    st.write(f"新聞來源掃描至 **{news.get('calendar_scan_through') or '未提供'}**；資料日完整掃描：**{'是' if news.get('signal_session_scan_complete') else '否'}**。")
    st.caption(f"實際完成時間：{news.get('completed_at') or '未提供'}。舊新聞不因現在取得而被認證為當時已可用。")
    cols = st.columns(3)
    for col, key, label in zip(cols, ("444", "555", "attempts"), ("444累積日誌", "555累積日誌", "實際執行紀錄")):
        path = status.get("links", {}).get(key)
        if path:
            col.link_button(label, GITHUB + path, width="stretch")


def next_steps():
    st.subheader("有機會超越0050；目前還沒有足夠證據")
    notes = json.loads((Path(__file__).parent / "research_notes.json").read_text(encoding="utf-8"))
    st.write(notes["verdict"])
    for experiment in notes["experiments"]:
        with st.container(border=True):
            st.markdown(f"### {experiment['label']}")
            st.caption("下一輪設計，尚未實作或取得結果")
            st.write(experiment["finding"])
            st.markdown("**要檢驗的假設**　" + experiment["hypothesis"])
            st.markdown("**具體實驗**　" + experiment["next_experiment"])
            st.markdown("**未通過時**　" + experiment["fallback"])
    st.link_button("閱讀完整研究討論", GITHUB + "docs/v3_next_research.md")


def render_v3(version):
    try:
        study = load_study(version)
    except (DataIntegrityError, OSError, ValueError, KeyError) as exc:
        st.error("封存資料檢查未通過，暫停顯示績效。")
        st.code(str(exc), language=None)
        return
    st.markdown('<div class="eyebrow">TAIWAN EQUITY · RESEARCH PREVIEW</div>', unsafe_allow_html=True)
    st.title("新聞訊號，值得加碼嗎？")
    st.markdown(f'<div class="subline">{study["label"]}　／　留出期 {study["period_start"]} — {study["period_end"]}<br>以0050為基準，分開檢驗模型、新聞與交易成本。</div>', unsafe_allow_html=True)
    st.warning("探索性結果｜新聞候選尚未通過採用門檻，維持0050基準；每日模型未被替換。", icon="🧪")
    tabs = st.tabs(["結果總覽", "配對驗證", "成本與風險", "每日運作", "下一輪研究"])
    with tabs[0]: overview(study)
    with tabs[1]: validation(study)
    with tabs[2]: costs(study)
    with tabs[3]: daily()
    with tabs[4]: next_steps()
    with st.expander("下載與研究來源"):
        a, b, c = st.columns(3)
        a.download_button("下載留出期指標 CSV", study["metrics"].to_csv(index=False).encode("utf-8-sig"),
                          file_name=f"{version}_holdout_metrics.csv", mime="text/csv", width="stretch")
        b.download_button("下載三折驗證 CSV", study["validation"].to_csv(index=False).encode("utf-8-sig"),
                          file_name=f"{version}_validation_metrics.csv", mime="text/csv", width="stretch")
        c.link_button("查看原始研究報告", GITHUB + study["report_dir"] + "/summary.md", width="stretch")
        st.caption("圖表為保存帳本衍生的淨值指數。完整行情與新聞原文未隨展示散布；雜湊用於檔案完整性，不等於歷史新聞可用時點的證明。")
        st.json(study["provenance"], expanded=False)


def _navigate(page):
    st.session_state["project_page"] = page


def project_overview():
    st.markdown('<div class="eyebrow">TAIWAN EQUITY · PROJECT RESEARCH PORTAL</div>', unsafe_allow_html=True)
    st.title("台股研究，從模型到每一天")
    st.markdown('<div class="subline">444價格預測 × 555動態配置 × 新聞情緒 × 0050延伸研究<br>模型結果、研究日誌與新聞來源，在同一個入口追溯。</div>', unsafe_allow_html=True)
    from dashboard.journals import available_dates
    a, b = st.columns(2)
    a.metric("研究系統", "4 套")
    b.metric("保存逐日報告的日期", f"{len(available_dates(root=ROOT))} 天")
    try:
        daily()
    except (DataIntegrityError, OSError, ValueError, KeyError) as exc:
        st.warning("最新狀態無法確認；其他保存資料仍可由側欄查看。")
        st.caption(str(exc))
    st.divider()
    st.subheader("從這裡開始")
    items = [
        ("模型與結果", "比較444／555共同核心、各模型每日預測、V2.11帳本，以及原修訂與延伸研究。"),
        ("研究日誌", "依資料日閱讀444／555報告、搜尋累積日誌，或按真正執行日追蹤成功與失敗。"),
        ("新聞與情緒", "查看保存的文章、情緒分數、股票關聯、原文連結、來源掃描與取得時間。"),
        ("V3研究", "檢視V3.2／V3.3與0050的配對驗證、留出績效、成本及下一輪設計。"),
        ("文件與重現", "閱讀企畫書、研究協議、130題審查與重現說明，下載文件及原始Notebook。"),
    ]
    for start in range(0, len(items), 2):
        for col, (name, text) in zip(st.columns(2), items[start:start + 2]):
            with col:
                with st.container(border=True):
                    st.markdown("### " + name)
                    st.write(text)
                    st.button("開啟" + name, key="open_" + name, on_click=_navigate, args=(name,))
    st.caption("本頁顯示目前部署checkout保存的研究，資料更新取決於GitHub每日流程及Streamlit同步；重新整理不會在網站內執行模型訓練或抓取新聞。")


DOCUMENTS = {
    "企畫與主題": {"修訂企畫書": "docs/proposal_revised.md", "專案說明": "README.md"},
    "研究協議": {"444／555配對協議": "docs/research_protocol_v2.md", "P0十項改善": "docs/p0_remediation.md",
                 "V2.11帳本協議": "research/v211/research_protocol.md", "V3研究設計": "research/v3/design.md",
                 "V3.3研究設計": "research/v3_3/design.md", "V3資料契約": "research/v3/data_contract.md"},
    "審查與重現": {"130題逐題狀態": "docs/audit_130_status.md", "關鍵修正與歷史限制": "docs/audit_130_critical.md",
                 "驗證紀錄": "VALIDATION.md", "交付與重現說明": "docs/delivery_v2_1.md",
                 "每日恢復說明": "docs/daily_recovery_2026-11-17.md", "Streamlit使用說明": "docs/streamlit_demo.md"},
    "下一輪研究": {"V3下一輪討論": "docs/v3_next_research.md"},
}


def documents():
    from dashboard.journals import display_markdown
    st.header("專題文件與重現入口")
    st.caption("文件直接來自本次部署的GitHub版本；歷史限制與原協議保留。")
    group = st.selectbox("文件分類", list(DOCUMENTS), key="docs_group")
    label = st.selectbox("閱讀文件", list(DOCUMENTS[group]), key="docs_document_" + group)
    relative = DOCUMENTS[group][label]
    path = ROOT / relative
    if path.is_file():
        text = path.read_text(encoding="utf-8-sig")
        a, b = st.columns(2)
        a.link_button("在GitHub查看原檔", GITHUB + relative)
        b.download_button("下載這份文件", text, file_name=path.name, mime="text/markdown", key="docs_download")
        with st.container(border=True):
            st.markdown(display_markdown(text, relative), unsafe_allow_html=False)
    else:
        st.info("此checkout沒有保存這份文件。")
    st.divider()
    st.subheader("企畫書與模型Notebook")
    files = {
        "修訂企畫書 Word": "docs/proposal_revised.docx", "444原始Notebook": "models/model_444/444.ipynb",
        "555原六特徵Notebook": "models/model_555/555.ipynb", "555 V2凍結研究Notebook": "models/model_555_v2/555_V2.ipynb",
        "444研究Notebook": "notebooks/444_baseline.ipynb", "555多變量延伸Notebook": "notebooks/555_multivariate.ipynb",
    }
    for label, relative in files.items():
        a, b = st.columns([3, 1])
        a.link_button(label, GITHUB + relative)
        path = ROOT / relative
        if path.is_file():
            b.download_button("下載", path.read_bytes(), file_name=path.name,
                              mime="application/octet-stream", key="docs_file_" + relative)
    st.caption("下載Notebook不會在網站執行訓練。主研究555共用444預測；原六特徵Notebook保留為獨立歷史／延伸研究。完整來源程式、原交付壓縮檔與版本紀錄可由GitHub專案查看。")


def main():
    st.set_page_config(page_title="台股專題｜模型・日誌・新聞", page_icon="📊", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### TW RESEARCH")
        st.caption("台股LSTM・新聞情緒・動態配置")
        page = st.radio("專案導覽", ["專案總覽", "模型與結果", "研究日誌", "新聞與情緒", "V3研究", "文件與重現"], key="project_page")
        if page == "V3研究":
            version = st.selectbox("研究版本", list(STUDIES), key="study_version",
                                  format_func=lambda x: {"v3.3": "V3.3｜20日・Ridge / HGB", "v3.2": "V3.2｜5日・Ridge / LSTM"}[x])
            st.caption("版本期間不同，請於各自期間比較，不將跨版本ROI排序視為改善。")
        st.divider()
        st.markdown("**展示模式**　保存結果・唯讀")
        st.caption("閱讀保存報告，不需模型套件或行情金鑰。更新程式與資料由GitHub同步。")
        if st.button("重新讀取保存報告", width="stretch"):
            st.rerun()
        st.page_link("https://github.com/tKe1vin/tw-stock-lstm-research", label="GitHub完整專案", icon="↗️")
    if page == "專案總覽":
        project_overview()
    elif page == "模型與結果":
        from dashboard.project import render_project
        render_project()
    elif page == "研究日誌":
        from dashboard.journals import render_journals
        render_journals()
    elif page == "新聞與情緒":
        from dashboard.news import render_news
        render_news()
    elif page == "V3研究":
        render_v3(version)
    else:
        documents()
    st.markdown('<div class="app-footer">TW STOCK RESEARCH　·　來源、日期與協議可追溯　·　研究模擬，無券商下單功能</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
