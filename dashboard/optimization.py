"""Display a separately archived allocation experiment without changing V3 selection."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.data import ROOT, DataIntegrityError

REPORT = "reports/research_v3_4/2026-11-17"
FILES = ("summary.md", "comparison_metrics.csv", "registration.json")


def load_cost_experiment(root=ROOT):
    base = Path(root) / REPORT
    if not (base / "manifest.json").is_file():
        return {"available": False}
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != "v3_4_cost_replay_v1" or manifest.get("no_retraining") is not True
            or manifest.get("holdout_used_for_selection") is not False
            or manifest.get("raw_provider_data_included") is not False
            or manifest.get("historical_alpha_proven") is not False
            or manifest.get("legacy_replay_verified") is not True):
        raise DataIntegrityError("V3.4研究角色或重播證據不符")
    hashes = manifest.get("files_sha256", {})
    for name in FILES:
        path = base / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(name):
            raise DataIntegrityError("V3.4優化實驗來源hash不符：" + name)
    return {"available": True, "summary": (base / "summary.md").read_text(encoding="utf-8"),
            "metrics": pd.read_csv(base / "comparison_metrics.csv"),
            "registration": json.loads((base / "registration.json").read_text()), "manifest": manifest}


def render_cost_experiment(root=ROOT):
    st.subheader("V3.4｜實際換倉成本的配對實驗")
    st.caption("此實驗獨立於側欄版本切換，使用V3.3原期間與已保存的價格／新聞預測，只調整配置器。舊研究、模型及每日流程保留；已看過歷史仍屬探索。")
    try:
        result = load_cost_experiment(root)
    except (OSError, ValueError, DataIntegrityError) as exc:
        st.error("優化實驗來源檢查未通過，暫停顯示此實驗。")
        st.caption(str(exc))
        return
    if not result["available"]:
        st.info("第一個優化實驗已限定範圍：同一批價格／新聞預測，比較原配置與實際增量換倉成本配置。結果尚未封存，完成後在這裡展示。")
        return
    frame = result["metrics"]
    st.markdown(result["summary"], unsafe_allow_html=False)
    table = frame[frame.Split.eq("holdout")][["Allocator", "Arm", "ROI_Pct", "Net_Excess_ROI_Pp", "MDD_Pct", "Total_Transaction_Cost"]].rename(
        columns={"Allocator": "配置", "Arm": "組別", "ROI_Pct": "淨報酬（%）", "Net_Excess_ROI_Pp": "對0050（百分點）",
                 "MDD_Pct": "最大回撤（%）", "Total_Transaction_Cost": "交易成本（元）"})
    table["配置"] = table["配置"].replace({"benchmark": "0050", "legacy": "原配置", "incremental": "增量成本配置"})
    table["組別"] = table["組別"].replace({"B0_benchmark": "0050基準", "ridge_price": "價格模型", "ridge_news": "新聞模型"})
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        name: st.column_config.NumberColumn(format="%.4f") for name in table.columns[2:]})
    st.download_button("下載V3.4配對實驗指標", frame.to_csv(index=False).encode("utf-8-sig"),
                       file_name="v3_4_comparison_metrics.csv", mime="text/csv", key="v34_metrics_download")
    with st.expander("實驗固定設定與來源"):
        st.dataframe(frame, hide_index=True, width="stretch")
        st.json(result["registration"])
        st.json(result["manifest"])
