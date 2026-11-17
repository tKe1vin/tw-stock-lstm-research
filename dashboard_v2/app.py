"""Two-part research dashboard. Refresh only rereads saved files."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from dashboard_v2.data_access.prospective import signal_dates
from dashboard_v2.views.part1 import render as render_part1
from dashboard_v2.views.part2 import render as render_part2

CSS = """
<style>
.stApp {background:#101418; color:#e8eef4;}
[data-testid="stWidgetLabel"] p,
[data-testid="stRadio"] label p,
[data-testid="stCheckbox"] label p {color:#e8eef4 !important;}
.card {background:#1b242c; border:1px solid #31404c; padding:12px 14px; border-radius:8px;}
.cards {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:8px 0 16px;}
.card p {margin:0; color:#9fb0c0; font-size:13px;}
.card strong {display:block; margin:6px 0; font-size:22px; word-break:break-all;}
.card small {color:#b7c4d0;}
.table-wrap {overflow:auto; border:1px solid #31404c; border-radius:8px;}
table {width:100%; border-collapse:collapse; font-size:13px;}
th,td {padding:6px 8px; border-bottom:1px solid #2a3844; text-align:left; white-space:nowrap;}
.gap {padding:18px; border:1px dashed #8a6a32; color:#f0d7a2; border-radius:8px;}
.demo-banner {padding:10px 12px; margin:0 0 14px; border:1px solid #8a6a32; color:#f0d7a2; border-radius:8px; background:#1a1710;}
</style>
"""


def main():
    st.set_page_config(page_title="台股 LSTM｜555 V2 研究展示", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("555 V2／V2.11／Ridge 研究展示（非 V3）")
    st.markdown(
        '<div class="demo-banner"><b>SYNTHETIC_DEMO｜FIXTURE_NOT_OFFICIAL_PERFORMANCE</b>：'
        "預設讀取合成 fixtures；不可當作正式績效、論文結果或唯一帳本。"
        " 真實前瞻僅能透過本機 env（TW_V211_LOCAL／TW_RIDGE_ROOT）唯讀載入。"
        " 缺 live 時維持 PENDING／DATA_UNAVAILABLE／SOURCE_CONFLICT 展示。重新整理不訓練、不下單、不改正式帳本。</div>",
        unsafe_allow_html=True,
    )
    st.caption("只讀已保存研究檔與本機／fixtures 前瞻快照。與組長 V3 Dashboard（dashboard/）分離。")
    area = st.radio("研究入口", ["444／555／TWII 正式比較", "555 升級研究與每日交易"], horizontal=True)
    auto = st.checkbox("每 60 秒自動刷新", value=False)
    if area.startswith("444"):
        protocol = st.radio("協議", ["原始歷史比較", "新版配對研究"], horizontal=True)
        ticker = st.selectbox("新聞股票", ["全部", "2330.TW", "2317.TW", "2603.TW", "2002.TW", "2891.TW"])
        klass = st.selectbox("情緒分類", ["全部", "正向", "負向", "門檻內", "ZERO_UNVERIFIED", "UNVERIFIED"])
        keyword = st.text_input("新聞編號或時間關鍵字")
        args = (protocol, ticker, klass, keyword)
        runner = render_part1
    else:
        section = st.radio(
            "升級研究子頁",
            [
                "V2 總覽",
                "原 555 與 V2",
                "歷史實驗",
                "本機前瞻狀態",
                "V2.11",
                "Ridge 前瞻",
                "交易日誌",
                "當日檢討",
                "隔日策略",
                "限制",
            ],
            horizontal=True,
        )
        days = ["全部", *signal_dates()]
        day = st.selectbox("日誌日期", days)
        ticker = st.selectbox("日誌股票", ["全部", "2330.TW", "2317.TW", "2603.TW", "2002.TW", "2891.TW"])
        args = (section, day, ticker)
        runner = render_part2
    if auto:
        _auto(runner, args, uuid4().hex)
    else:
        runner(*args)


@st.fragment(run_every=timedelta(seconds=60))
def _auto(runner, args, render_id):
    # Reload selectors as well as the body when another saved date arrives.
    if st.session_state.get("_v2_auto_render") == render_id:
        st.rerun(scope="app")
    st.session_state["_v2_auto_render"] = render_id
    runner(*args)


if __name__ == "__main__":
    main()
