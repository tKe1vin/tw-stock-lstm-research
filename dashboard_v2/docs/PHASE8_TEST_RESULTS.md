# Phase 8 測試結果

**執行時間：** 2026-11-17T13:28+08:00  
**環境：** `tw-stock-lstm-group-clone\.venv`（勿與其他倉相加）

```text
python -m pytest dashboard/tests/test_dashboard_sources.py dashboard/tests/test_phase8_forward.py integration_audit_20261117/github_midnight_fix/test_session_policy.py -v
```

| 項目 | 數值 |
| --- | ---: |
| Collected | 27 |
| Passed | 27 |
| Failed | 0 |
| Skipped | 0 |
| Collection Errors | 0 |

## 分檔

| 檔案 | Collected | Passed |
| --- | ---: | ---: |
| `test_dashboard_sources.py`（既有） | 11 | 11 |
| `test_phase8_forward.py`（新增） | 9 | 9 |
| `test_session_policy.py`（隔離） | 7 | 7 |

## 新增 Phase 8 覆蓋

1. 11/17 成熟閘門（收盤前 PENDING）  
2. 衝突訊號 SHA 不變  
3. 成熟列不重複  
4. Ridge／V2.11 分離且 RMSE 存在  
5. 未成熟不進績效宣稱  
6. SOURCE_CONFLICT 仍可見  
7. NAV 與帳本一致  
8. 協議不混合  
9. 前端讀取不寫正式檔／權重
