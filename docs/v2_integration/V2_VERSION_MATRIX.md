# V2 Version Matrix

| 項目 | 最新 main (`3e64240`) | group-clone 研究分支 | development Ridge | 正式 research 倉 | 本整合分支策略 |
| --- | --- | --- | --- | --- | --- |
| 444／555 主協議 | 已改良 paired 研究 | 舊展示邏輯 | 引用 DEV reports | dirty 歷史報告 | **用 main**；Dashboard 只讀展示 |
| V3／V3.3 | **已存在**（0050 新聞研究） | 無／過時 | 無 | 無 | **保留 main，不覆寫** |
| V3 Streamlit | `dashboard/` | 無 | 8501 另套 | 無 | **保留** |
| 555 V2 雙主題 UI | 無 | **獨有** `dashboard/` | 無 | 無 | **移植為 `dashboard_v2/`** |
| V2.11 權重 | SHA `53a1467f…c28144` | 可能過期複本 | 展示鏡像 | 同 SHA | **以 main／正式倉為準** |
| 跨午夜／delayed_run | **已在** `run_all_daily.py` | 本機修補文件 | — | — | **已存在於 main**；文件交叉引用 |
| Ridge 前瞻 | 無完整 phase5 | — | **獨有** ledger | 無 | **SYNTHETIC_DEMO＋狀態文**；帳本私下／env |
| SOURCE_CONFLICT | 無完整包 | 舊真實分析 | — | 本地訊號 | **合成衝突證據**（Schema 保留） |
| v26～v210 腳本 | 部分（v28/v29/v211） | 完整歷史 | 完整歷史 | 完整歷史 | **不整包移植**；見 HISTORICAL_SCRIPTS |
| GitHub Actions | 最新 | 勿覆蓋 | — | — | **不移植舊 Actions** |

## 分類標籤（整合決策）

| 功能 | 判定 |
| --- | --- |
| V3 Dashboard／V3.3 研究 | 已存在於 main／已改良 → 不覆蓋 |
| delayed_run／maintenance_only | 已存在於 main |
| 雙主題 444/555 + V2.11/Ridge UI | 研究分支獨有 → 以 `dashboard_v2` 交付 |
| Ridge 原始 ledger／forward-evidence | 需保留但不能整包合併進 Git → Manifest＋私下路徑 |
| 舊 Desktop 絕對路徑、舊 clone dashboard 覆寫 main dashboard | 過時／不應交付 |
| prospective 路徑 | 需修改才相容 → 已改相對路徑＋env 覆寫 |
