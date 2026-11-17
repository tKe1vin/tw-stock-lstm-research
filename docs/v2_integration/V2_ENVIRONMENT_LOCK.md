# V2 Environment Lock

## Integration Worktree

| 項目 | 值 |
| --- | --- |
| Python | 3.12.10 |
| Dashboard venv | `.venv-integration/`（gitignore；streamlit／pandas／pytest） |
| 推論驗證 venv | `.venv-tf-verify/`（gitignore；`tensorflow_cpu==2.20.0`、scikit-learn、pandas） |
| 正式研究依賴基準 | `requirements.txt`／`requirements-ci.txt` |
| Dashboard V2 | `dashboard_v2/requirements.txt` |

## 凍結模型

| 項目 | SHA-256 |
| --- | --- |
| `weights.npz` | `53a1467fdba66e7f22a45e5f2fbc1cb78fc99825b15d516b3e7153d7bfc28144` |
| Bundle | `scripts/model_bundle.verify_bundle` PASS（本機 TF 驗證時） |

## 可選本機覆寫

| 環境變數 | 用途 |
| --- | --- |
| `TW_V211_LOCAL` | 覆蓋本機 V2.11 根（**優先於 DEMO fixtures**） |
| `TW_RIDGE_ROOT` | 真實 Ridge ledger |
| `TW_GITHUB_REVIEW_ROOT` | GitHub review worktree |

未設定時 Dashboard 使用 `SYNTHETIC_DEMO` fixtures，並標記 `official_claim_blocked`。
