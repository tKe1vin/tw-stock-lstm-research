# GitHub 跨午夜 Skip 修正提案（Phase 8 交付）

**狀態：APPROVAL_REQUIRED。未修改 main／Workflow／本機排程。**

## 根因

`run_all_daily.py` 在 `completed_session != run_date` 時直接 `skipped` + `exit_code=0`。  
Actions 常於台北午夜後才進入 → 名義日已換、可處理 session 仍是上一交易日 → 正式流程與標籤維護皆被跳過。

## 隔離交付物

| 路徑 | 說明 |
| --- | --- |
| `integration_audit_20261117/github_midnight_fix/session_policy.py` | 決策核心 |
| `test_session_policy.py` | 7 項回歸（passed） |
| `MINIMAL_DIFF_PROPOSAL.md` | 對正式檔的最小差異構想 |
| `APPROVAL_REQUIRED.md` | 批准前禁令 |

## 行為摘要

- 午夜後：允許 **標籤維護**；**禁止**回填發布  
- 錯過決策窗：`MISSED`／`partial`  
- 已發布 session：不重發（冪等）  
- `SKIPPED_NOT_SUCCESS` ≠ `FULL_SUCCESS`  
- 不改凍結權重  

詳見 Phase 7 回歸表與 Phase 8 `MINIMAL_DIFF_PROPOSAL.md`。
