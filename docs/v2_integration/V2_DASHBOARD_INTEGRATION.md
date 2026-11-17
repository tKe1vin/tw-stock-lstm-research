# V2 Dashboard 整合說明

- 入口：`streamlit_v2_app.py` → `dashboard_v2/app.py`  
- 與 V3：`dashboard/` **分離**  
- 預設資料：`fixtures/display_snapshots`（`SYNTHETIC_DEMO`）  
- Live 覆寫：`TW_V211_LOCAL`／`TW_RIDGE_ROOT`／`TW_GITHUB_REVIEW_ROOT`（優先於 fixtures）  
- 狀態標記：`data_kind`、`demo_flag`、`not_official_performance`、`official_claim_blocked`  
- 異常情境：PENDING、SOURCE_CONFLICT、DATA_UNAVAILABLE（NAV_Close）保留  
- 重建 DEMO：`python dashboard_v2/fixtures/_build_synthetic_demo.py`
