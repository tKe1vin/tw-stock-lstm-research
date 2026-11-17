# SYNTHETIC_DEMO fixtures

**標記：`SYNTHETIC_DEMO`／`FIXTURE_NOT_OFFICIAL_PERFORMANCE`**

本目錄為合成展示／測試資料，保留 Schema 與異常情境（PENDING、SOURCE_CONFLICT、DATA_UNAVAILABLE）。

| 規則 | 說明 |
| --- | --- |
| 正式績效 | **不可**計入；API 設 `official_claim_blocked` |
| 真實資料 | 僅 `TW_V211_LOCAL`／`TW_RIDGE_ROOT` 唯讀 |
| 重建 | `python dashboard_v2/fixtures/_build_synthetic_demo.py` |
| 權重 | 不在此目錄；核對 `experiments/results/v2/v211/weights/` |
