# V2 測試結果

## Dashboard（`.venv-integration`）

```text
.\.venv-integration\Scripts\python.exe -m pytest dashboard_v2/tests -q
→ 30 passed
```

涵蓋：歷史／配對協議、V2.11 DEMO 快照、成熟／PENDING 分離、SOURCE_CONFLICT、Ridge 未合併、日誌、DEMO／live 分離、權重未改寫。

## 凍結推論（`.venv-tf-verify`）

```text
.\.venv-tf-verify\Scripts\python.exe docs\v2_integration\verify_v211_inference.py --signal-date 2026-11-05
→ ok=true, weight_ok, bundle_ok, matched, max_abs_diff=0.0
```

## Fixtures 安全稽核

`acceptance_evidence/FIXTURE_SECURITY_AUDIT.json` → issues = 0（SYNTHETIC_DEMO）。
