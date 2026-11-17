"""Periodically reread the deployed snapshot without touching research files."""
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st


@st.fragment(run_every=300)
def automatic_refresh(render_id):
    """A timer repeats this fragment; a full page render always gets a new ID.

    The first call only draws the status. A repeat for that same page requests
    one full rerun, so every view rereads its files and there is no rerun loop.
    Repository syncing is handled by the hosting platform, not by this timer.
    """
    if st.session_state.get("_automatic_refresh_render") == render_id:
        st.rerun(scope="app")
    st.session_state["_automatic_refresh_render"] = render_id
    st.caption("開啟頁面時，每5分鐘自動讀取保存資料。")
    st.caption("本頁讀取：" + datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S") + "（台北）")
