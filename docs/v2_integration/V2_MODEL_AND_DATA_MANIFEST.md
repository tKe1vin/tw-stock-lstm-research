# V2 模型與資料 Manifest

| Artifact | 路徑 | 備註 |
| --- | --- | --- |
| 權重 | `experiments/results/v2/v211/weights/weights.npz` | SHA `53a1467f…c28144` |
| Model card（含 scalers／tau） | `…/weights/model_card.json` | 凍結契約 |
| Bundle manifest | `models/model_555_v2/bundle_manifest.json` | `verify_bundle` |
| 正式 daily／NAV／matured | `experiments/results/v2/v211/` | 唯讀；未改 |
| Market 快照 | `…/v211/market/` | 前瞻推論輸入 |
| DEMO fixtures | `dashboard_v2/fixtures/display_snapshots/` | 非正式 |
| 推論驗證 | `docs/v2_integration/verify_v211_inference.py` | 唯讀重播 |
