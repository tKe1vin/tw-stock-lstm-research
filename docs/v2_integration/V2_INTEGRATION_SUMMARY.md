# V2 Integration Summary

**Branch:** `research/v2-integration-20261117`  
**Base:** `origin/main` @ `3e64240e04aa0af1f5f606f4254aeaaa0cf3f26e`  
**Worktree:** `tw-stock-lstm-workspace/tw-stock-lstm-v2-integration`  
**狀態：** 最終驗收完成；**尚未 Commit／Push／Merge**。

## 目標

在不覆寫組長 V3 看板的前提下，交付可重現的 555 V2 研究：凍結模型推論、正式帳本核對、安全 DEMO Dashboard、與 Ridge／V2.11 分離狀態。

## 已完成

| 區塊 | 說明 |
| --- | --- |
| `dashboard_v2/` | 雙主題展示；DEMO≠正式績效 |
| SYNTHETIC_DEMO fixtures | Schema／PENDING／SOURCE_CONFLICT／DATA_UNAVAILABLE 保留 |
| TF 驗證 | `.venv-tf-verify` + `verify_v211_inference.py` → 與正式日訊號一致 |
| 文件 | `V2_*.md`、`V2_FINAL_ACCEPTANCE.md`、`V2_HISTORICAL_SCRIPTS.md` |

## 明確未做

- 未改 `dashboard/`（V3）、正式權重、訊號、帳本、排程  
- 未上傳完整 Ridge forward-evidence  
- 未整包移植 v26～v210 歷史腳本  
- 未 Commit／Push

## 驗收一句話

Dashboard 用安全合成 fixtures；真實研究可經 env／倉內正式 V2.11 唯讀重現；**30** 項 Dashboard 測試通過；凍結推論 **PASS**。詳見 `V2_FINAL_ACCEPTANCE.md`。
