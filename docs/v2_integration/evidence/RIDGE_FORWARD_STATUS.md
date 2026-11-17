# Ridge 前瞻狀態（交付摘錄）

**標記：** 真實 Ridge ledger／forward-evidence **不進本 PR**。  
Dashboard 預設僅顯示 `SYNTHETIC_DEMO` fixtures（`official_claim_blocked=true`）。

| 項目 | 交付狀態 |
| --- | --- |
| 協議 | 273-day prospective close-only ridge；claim 門檻 252 scorable days |
| Git 內展示 | `dashboard_v2/fixtures/display_snapshots/ridge/`（合成） |
| 真實 ledger | 僅本機 `TW_RIDGE_ROOT` 唯讀載入 |
| 與 V2.11 | **不得合併** NAV／績效 |

真實首日 prediction SHA（live only）：見本機 lock／ledger；DEMO 使用 `DEMO_META.json` 內合成雜湊。
