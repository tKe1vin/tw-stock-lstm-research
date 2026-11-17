# GitHub 跨午夜回歸測試

**執行時間：** 2026-11-17T13:19+08:00  
**路徑：** `integration_audit_20261117/github_midnight_fix/test_session_policy.py`  
**結果：** 7 passed / 0 failed

## 覆蓋情境

| 測試 | 情境 |
| --- | --- |
| `test_before_close_uses_previous_session` | 收盤前用上一已完成 session |
| `test_after_close_same_day_uses_today` | 同日收盤後可用當日 |
| `test_successful_evening_run_can_publish` | 晚間完整執行可發布 |
| `test_after_midnight_does_not_backfill_publish` | 跨午夜 → partial，不可回填發布 |
| `test_weekend_morning_uses_friday_session_without_publish` | 週末維護標籤、不發布 |
| `test_already_published_session_is_not_republished` | 同 session 不可二次發布 |
| `test_skip_summary_is_not_full_success` | SKIPPED_NOT_SUCCESS ≠ FULL_SUCCESS |

## 明確不測／不改

- 不改凍結 `weights.npz`（`53a1467f...`）
- 不改正式 Actions YAML
- 不 Push
