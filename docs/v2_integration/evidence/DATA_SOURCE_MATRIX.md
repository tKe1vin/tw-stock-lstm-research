# 資料來源矩陣（V2 交付）

| 來源 | Git 內 | 本機 env | 可否計入正式績效 |
| --- | --- | --- | --- |
| Dashboard fixtures | `SYNTHETIC_DEMO` | — | **否** |
| V2.11 凍結權重／model_card | `experiments/results/v2/v211/weights/` | — | 權重核對用 |
| V2.11 正式 ledger | main 既有 `experiments/results/v2/v211/` | `TW_V211_LOCAL` 可覆寫 | 是（正式帳本） |
| Ridge ledger | 無完整包 | `TW_RIDGE_ROOT` | 是（獨立協議；未滿 252 日仍 PENDING） |
| GitHub review 配對研究 | DEMO metadata | `TW_GITHUB_REVIEW_ROOT` | 研究 as_of ≠ V2.11 發布 |
| forward-evidence | **不提交** | 私下唯讀 | 封印證據 |

衝突展示：`docs/v2_integration/evidence/source_conflict_1110/` 現為 **SYNTHETIC_DEMO**。
