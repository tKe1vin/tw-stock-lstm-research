# Phase 7 測試結果

**執行時間：** 2026-11-17T13:19+08:00  
**解譯器：** `tw-stock-lstm-group-clone\.venv\Scripts\python.exe`  
**指令：**

```text
python -m pytest dashboard/tests/test_dashboard_sources.py integration_audit_20261117/github_midnight_fix/test_session_policy.py -v
```

## 本環境合計（勿與其他倉相加）

| 項目 | 數值 |
| --- | ---: |
| Collected | 18 |
| Passed | 18 |
| Failed | 0 |
| Skipped | 0 |
| Collection Errors | 0 |

## 分檔

| 檔案 | Collected | Passed |
| --- | ---: | ---: |
| `dashboard/tests/test_dashboard_sources.py` | 11 | 11 |
| `integration_audit_20261117/github_midnight_fix/test_session_policy.py` | 7 | 7 |

## Dashboard 覆蓋對照

| 需求 | 測試 |
| --- | --- |
| 8502 讀本機 V2.11 快照 | `test_v211_reads_local_prospective_snapshot` |
| Ridge ≠ V2.11 期間 | `test_ridge_not_merged_with_v211_and_scores_from_files` |
| 成熟／未成熟分離 | `test_matured_and_pending_labels_are_separated` |
| SOURCE_CONFLICT 可見 | `test_source_conflict_is_visible` |
| 原 444／555 不混 | `test_historical_tokens_stay_with_their_csv` |
| 配對協議分離 | `test_paired_protocol_is_separate` |
| V2 排行不混帳 | `test_v2_ranking_keeps_incomparable_books_out` |
| 交易日誌／隔日真實帳本 | `test_journal_and_next_day_use_real_local_ledger` |
| 半寫入拒收 | `test_partial_file_is_rejected` |
| 前端不碰權重 | `test_dashboard_read_does_not_touch_weights` |
| 新聞不造假 | `test_news_does_not_invent_titles_or_neutral_zeros` |

## Smoke（非 pytest）

`integration_audit_20261117/_phase7_smoke.py` @ 13:19:57+08:00

- Ridge scorable_days=5, MAE≈2.1807, PENDING
- V2.11 matured=2, latest=2026-11-16, open_nav=100741.19751310395
- 11/09 未宣告成熟（收盤前）
- SOURCE_CONFLICT=True
- next_day signal_date=2026-11-16
