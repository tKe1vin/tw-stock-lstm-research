# Risk-Off 消融盤點

檢查檔：`reports/v210_comparison/comparison.json`。

`weights_saved` 是 false。`books` 只有各策略的彙總 ROI、成本、平均現金與換手，沒有 rolling_selected 的逐日預測、逐日目標權重或逐日 NAV。

因此相同預測、取消 Risk-Off 的控制組是 BLOCKED。不能用平均現金約 11.8% 反推每日權重，也不能把 V2.11 的 2026-11 帳本當成 2022–2026 的歷史消融。

V2.11 的 `live_fills.csv` 只有 2026-11-06 一筆模擬進場，風險上限 0.50，屬於前瞻帳，不是歷史消融樣本。
