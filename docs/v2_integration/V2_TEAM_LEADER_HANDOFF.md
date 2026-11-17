# 組長交接（V2 only）

## 你會拿到什麼

1. Integration branch 工作樹：`dashboard_v2` + 文件 + 合成 fixtures  
2. main 既有：V2.11 凍結權重、推論程式、正式帳本  
3. 可重現推論：`.venv-tf-verify` + `docs/v2_integration/verify_v211_inference.py`  
4. Dashboard 測試：**30 passed**

## 不會／不應拿到

- 完整 Ridge forward-evidence  
- 真實 26 份前瞻結構副本（已改 DEMO）  
- 對 V3／正式權重／排程的修改  
- 自動 Push（需批准）

## 建議審查順序

1. 讀 `V2_FINAL_ACCEPTANCE.md`  
2. 跑 Dashboard 測試與推論驗證  
3. 確認 fixtures 全為 `SYNTHETIC_DEMO`  
4. 以 `TW_V211_LOCAL` 指向正式結果（可選）核對 live  
5. 批准後再 Commit／Push／開 PR
