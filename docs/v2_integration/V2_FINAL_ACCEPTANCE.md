# V2 最終驗收（10 項）

**Branch：** `research/v2-integration-20261117`  
**Base：** `origin/main` @ `3e64240`  
**狀態：** 驗收完成；**等待批准後才可 Push**（尚未 Commit／Push／Merge）

| # | 項目 | 結果 |
| --- | --- | --- |
| 1 | Dashboard 是否使用安全合成 fixtures | **是** — `SYNTHETIC_DEMO`／`DEMO_META.json`；橫幅標示非正式績效 |
| 2 | 真實 Ridge／前瞻結構副本是否排除在待提交內容外 | **是** — fixtures／conflict evidence 已合成；完整 forward-evidence 未納入 |
| 3 | V2 模型是否可在 Integration Worktree 成功推論 | **是** — `.venv-tf-verify` + `verify_v211_inference.py` |
| 4 | 是否與正式凍結模型取得一致預測 | **是** — `2026-11-05` max_abs_diff=`0.0`；權重 SHA `53a1467f…c28144` |
| 5 | V2.6～V2.10 必要獨有功能 | 見 `V2_HISTORICAL_SCRIPTS.md`：無須再移植；方法學已在 main 鏈 |
| 6 | 哪些功能已存在 main | V2.11 執行鏈、`v28`／`v29`、凍結權重、model bundle、協議文件 |
| 7 | V2.11 成熟評分與帳本是否可驗證 | **是** — 倉內 `matured_scores`／`live_nav`／`ledger_*` 唯讀可查；未改寫 |
| 8 | 所有 V2 測試通過數量 | **`dashboard_v2/tests` → 30 passed** |
| 9 | 尚未解決的相容性／重現性問題 | 見下方 |
| 10 | 是否可正式建立完整 V2 Pull Request | **內容就緒，可建 PR**；須你批准後才 Commit／Push |

## 剩餘限制（不得宣稱已解）

1. Ridge 完整 live ledger／外部收據鏈仍僅本機；Git 僅 DEMO。  
2. 正式衝突原始輸入若遺失，仍無法 ROI 選邊重現（協議正確行為）。  
3. `load_context()` 凍結 555 行情只到 2026-11-04；前瞻日需 `v211_market` 快照（驗證腳本已走此路徑）。  
4. Dashboard 預設 DEMO；讀真實績效必須設 `TW_V211_LOCAL`／`TW_RIDGE_ROOT`。  
5. 未移植 v26～v210 歷史腳本本體（刻意；非執行阻擋）。

## 快速重現指令

```text
# Dashboard 測試（輕量 venv）
.\.venv-integration\Scripts\python.exe -m pytest dashboard_v2/tests -q

# 凍結推論（TF venv；不改權重）
.\.venv-tf-verify\Scripts\python.exe docs\v2_integration\verify_v211_inference.py --signal-date 2026-11-05

# Dashboard
.\.venv-integration\Scripts\python.exe -m streamlit run streamlit_v2_app.py
```
