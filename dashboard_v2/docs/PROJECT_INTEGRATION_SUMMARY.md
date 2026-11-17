# 整合說明

使用的組長版本是 `b62034e4fa850933a632c8db6321b309ebad85f9`，分支 `research/v2-controlled-improvement-20261110`。沒有 commit 或 push。

新增的是 `dashboard/` 與先前的 `research/v2_improvement/`。正式倉、development、排程與凍結權重沒有納入修改。

第一部分能切換歷史快照與 2026-11-06 配對帳。第二部分能看 V2 可比結果、V2.11 未成熟狀態、模擬日誌、當日檢討與隔日策略。

V2 消融仍是 BLOCKED，見 `research/v2_improvement/reports/RISK_OFF_ABLATION_INVENTORY.md`。新聞品質不足以重建文章。交易日誌只來自 V2.11 已保存的模擬帳。

測試是 13 passed。TensorFlow 全套未跑。建議組長先決定是否要提供 rolling_selected 的逐日權重，再談新候選模型。
