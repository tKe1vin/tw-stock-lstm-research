# V2 交付清單（僅 555 V2｜不含 V3）

**Branch：** `research/v2-integration-20261117`  
**Base：** `origin/main` @ `3e64240e04aa0af1f5f606f4254aeaaa0cf3f26e`  
**狀態：** 驗收完成；**未 Commit／Push／Merge**（等待批准）

---

## 1. 已整合進本分支（工作樹）

| 項目 | 位置 |
| --- | --- |
| 雙主題 Dashboard | `dashboard_v2/`、`streamlit_v2_app.py` |
| **SYNTHETIC_DEMO** fixtures | `dashboard_v2/fixtures/`（含 `_build_synthetic_demo.py`） |
| SOURCE_CONFLICT 合成證據 | `docs/v2_integration/evidence/source_conflict_1110/` |
| 狀態／限制摘錄 | `docs/v2_integration/evidence/*.md` |
| TF 推論驗證腳本 | `docs/v2_integration/verify_v211_inference.py` |
| 最終 10 項驗收 | `docs/v2_integration/V2_FINAL_ACCEPTANCE.md` |
| 歷史腳本分類 | `docs/v2_integration/V2_HISTORICAL_SCRIPTS.md` |
| 忽略本機 venv | `.gitignore`（`.venv-integration/`、`.venv-tf-verify/`） |

路徑：相對路徑＋可選 `TW_V211_LOCAL`／`TW_RIDGE_ROOT`／`TW_GITHUB_REVIEW_ROOT`（live 優先於 DEMO）。  
UI 橫幅：`SYNTHETIC_DEMO｜FIXTURE_NOT_OFFICIAL_PERFORMANCE`。

---

## 2. 已存在 main、無須重複提交

| 功能 | main 位置 |
| --- | --- |
| V2.11 執行／排程／市場 | `experiments/v2/v211*.py`、`v28*`、`v29.py`、`books.py`、`guard.py` |
| 凍結權重＋card | `experiments/results/v2/v211/weights/`（SHA `53a1467f…c28144`） |
| model bundle | `models/model_555_v2/` + `scripts/model_bundle.py` |
| 研究協議 | `docs/research_protocol_v2.md` |

---

## 3. 歷史腳本（v26～v210）

見 `V2_HISTORICAL_SCRIPTS.md`。本次**未整包複製**；無執行必要缺口。

---

## 4. Dashboard／測試

- Part1／Part2 完整；PENDING／DATA_UNAVAILABLE／SOURCE_CONFLICT 保留  
- DEMO 與 live 嚴格分離（`data_kind`／`official_claim_blocked`）  
- **`dashboard_v2/tests` → 30 passed**  
- 啟動：`python -m streamlit run streamlit_v2_app.py`（`.venv-integration`）

---

## 5. 推論重現

| 檢查 | 結果 |
| --- | --- |
| 獨立 TF 環境 | `.venv-tf-verify`（`tensorflow_cpu==2.20.0`） |
| `verify_bundle` | PASS |
| 固定輸入 `2026-11-05` | 與正式 daily **完全一致**（diff 0） |
| 權重／排程／帳本 | **未修改** |

---

## 6. Fixtures 公開安全

| 檢查 | 結果 |
| --- | --- |
| 合成標記 | `SYNTHETIC_DEMO` |
| 真實 Ridge／衝突快照 | **已替換／不提交** |
| 私人絕對路徑 | 交付文件已淨化 |
| 安全稽核 | `acceptance_evidence/FIXTURE_SECURITY_AUDIT.json`（0 issues） |

---

## 7. 禁止事項（維持）

- 不 Commit／Push／Merge（待你批准）  
- 不改 V3、正式權重、已發布訊號、正式帳本、Windows 排程  
- 不上傳完整 Ridge forward-evidence  
