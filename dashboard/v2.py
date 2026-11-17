"""V2 views inside the existing portal, with explicit per-render data sources."""
from pathlib import Path

import streamlit as st

from dashboard.data import ROOT, DataIntegrityError
from dashboard.project import load_tracker
from dashboard_v2.data_access.load import NAMES
from dashboard_v2.data_access.prospective import display_sources, signal_dates, v211_status_local
from dashboard_v2.views.part2 import render as render_part2

SECTIONS = ["V2 總覽", "原 555 與 V2", "歷史實驗", "本機前瞻狀態", "V2.11", "Ridge 前瞻",
            "交易日誌", "當日檢討", "隔日策略", "優化診斷", "限制"]
CSS = """
<style>
.cards {display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:12px 0 20px;}
.card {background:white; border:1px solid #e2e8f0; padding:14px 16px; border-radius:12px; color:#17253c;}
.card p {margin:0; color:#65758c; font-size:13px;}
.card strong {display:block; margin:8px 0; font-size:21px; overflow-wrap:anywhere;}
.card small {color:#65758c; overflow-wrap:anywhere;}
.table-wrap {overflow:auto; border:1px solid #e2e8f0; border-radius:10px; background:white; color:#17253c;}
.table-wrap table {width:100%; border-collapse:collapse; font-size:13px;}
.table-wrap th,.table-wrap td {padding:8px 10px; border-bottom:1px solid #e2e8f0; text-align:left; white-space:nowrap;}
.gap {padding:18px; border:1px dashed #c4a76d; color:#775b22; border-radius:10px;}
[data-testid="stMarkdownContainer"] svg[role="img"] {background:#101418; border-radius:10px; padding:8px; max-width:100%;}
</style>
"""


def render_v2(root=ROOT):
    root = Path(root)
    st.title("V2研究與每日檢討")
    st.caption("V2.11五日預測、研究帳本、每日檢討與隔日策略；沿用本網站的每日資料同步與五分鐘重新讀取。沒有券商下單功能。")
    mode = st.radio("前瞻／日誌資料", ["saved", "demo"], horizontal=True, key="v2_data_mode",
                    format_func=lambda value: {"saved": "正式保存資料", "demo": "合成DEMO展示"}[value])
    if mode == "saved":
        st.info("正式保存資料｜讀取GitHub每日保存的V2.11訊號與模擬帳本。歷史發布時點尚未完整認證，維持研究追蹤；缺少的Ridge來源不以DEMO補齊。")
    else:
        st.warning("SYNTHETIC_DEMO｜前瞻追蹤與日誌使用合成資料，不是實際模型績效，也不計入正式結果。")
    st.caption("此模式只切換前瞻追蹤／日誌來源；歷史實驗固定讀原研究檔，與每日帳本分開。Ridge是一日任務，不能和V2五日預測合併績效。")
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        if mode == "saved":
            load_tracker(root)  # Guard retained ledger prefixes before using the legacy views.
        with display_sources(mode=mode, root=root):
            status = v211_status_local()
            st.caption(f"最後保存訊號日：{status.get('latest_signal_date') or '未提供'}；行情資料至：{status.get('prices_through') or '未提供'}。來源模式與檔案日期不等於前瞻績效已驗證。")
            section = st.selectbox("V2研究頁面", SECTIONS, index=4, key="v2_section",
                                   format_func=lambda value: "保存資料與前瞻狀態" if value == "本機前瞻狀態" else value)
            a, b = st.columns(2)
            day = a.selectbox("日誌訊號日", ["全部", *signal_dates()], key="v2_journal_day")
            ticker = b.selectbox("日誌股票", ["全部", *[key for key in NAMES if key != "^TWII"]],
                                 format_func=lambda value: value if value == "全部" else f"{value} {NAMES[value]}", key="v2_ticker")
            if section == "優化診斷":
                if mode == "demo":
                    st.info("優化診斷只使用正式保存的成熟標籤；請切回「正式保存資料」。")
                else:
                    from dashboard.v2_diagnostics import render_v2_diagnostics
                    render_v2_diagnostics(root)
            else:
                render_part2(section, day, ticker)
    except (DataIntegrityError, OSError, ValueError, KeyError, RuntimeError) as exc:
        st.error("V2來源尚未完整同步或未通過檢查，暫停顯示此頁結果。網站會定時重新讀取。")
        st.caption(str(exc))
