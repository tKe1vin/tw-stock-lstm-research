# 交易日誌規格

訊號來自 `experiments/results/v2/v211/daily/*.csv`。`forecast_only` 或 `Trade_Eligible=False` 顯示 SIGNAL_ONLY。

模擬成交只讀 `live_fills.csv`，標成 SIMULATED_EXECUTION 與 PAPER_TRADING。手續費在 Buy_Fee，交易稅在 Sell_Tax。

當日檢討只使用 status.json 已成熟欄位。metrics 為 null 時不計算五日誤差。隔日策略使用 2026-11-09 訊號，並同時顯示 forecast_entry_date 2026-11-10 與 scheduled_rebalance_date 2026-11-13。
