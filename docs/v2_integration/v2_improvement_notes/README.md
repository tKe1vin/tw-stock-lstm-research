# V2 隔離稽核

這個目錄只讀 `reports/v210_comparison/comparison.json`，檢查已保存帳本能不能與等權比較。

不會訓練模型，也不會寫入 V2.11、444、555 或 Ridge。

測試：

`python -m pytest docs/v2_integration/v2_improvement_notes/tests -q`

此附錄僅檢查已保存的歷史表，不重算缺失帳本、不證明前瞻alpha。原本的research/v2_improvement路徑已改成目前交付位置；檔案讀取以repository根目錄為準。
