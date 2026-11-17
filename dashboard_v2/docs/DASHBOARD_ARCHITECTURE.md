# 看板架構

入口是 `dashboard/app.py`，畫面函式放在 `dashboard/views/`，避免被 Streamlit 當成額外分頁。埠 8502。上方只有兩個研究入口，篩選放在自動刷新片段之外。

資料只從克隆內已保存的 CSV 與 JSON 讀取。`stable_text` 會連讀兩次，內容不一致就拒絕。頁面不呼叫訓練、下單或新聞抓取。

圖表使用 SVG。這台電腦的應用程式控制會封鎖 pyarrow，因此已從新虛擬環境移除 pyarrow，也不使用 `st.dataframe`。
