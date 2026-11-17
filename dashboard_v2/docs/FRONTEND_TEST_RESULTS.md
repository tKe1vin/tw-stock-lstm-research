# 前端測試

`dashboard/tests/test_dashboard_sources.py` 與既有 `research/v2_improvement/tests/test_controls.py` 一起執行：13 passed。

pyarrow 26.0.1 匯入時被 Windows 應用程式控制封鎖。已從克隆虛擬環境移除，看板改走 SVG 與 HTML。沒有要求關閉安全機制。

既有完整測試套件需要 TensorFlow，這次沒有安裝，也沒有執行。
