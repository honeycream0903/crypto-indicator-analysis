# 📊 Crypto Indicator Analysis

本專案分析常見技術指標（如 SMA、RSI、MACD 等）在加密貨幣價格預測中的有效性，使用 XGBoost 統計其特徵重要性。  
資料處理與模型訓練皆於 Google Colab 環境完成，無需手動下載資料或設定環境，所有安裝與資料載入皆在 Notebook 中自動完成。
你可以點擊下方按鈕，在 Google Colab 直接開啟並執行專案：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/honeycream0903/crypto-indicator-analysis/blob/main/crypto_indicator_analysis.ipynb)



# 📦 環境需求
- Python 3
- pandas
- pandas-ta
- xgboost
- gdown

可於 Colab 中直接安裝必要套件：
```bash
pip install pandas_ta xgboost gdown

```

# 📥 資料下載與解壓縮

```bash
# 下載每日價格的壓縮資料並解壓縮
gdown --id 1_x08rXnQuvRzdCWVoD2Qftt6RmihaeSJ
unzip Cryptos_1day.zip
```

📌 所有資料皆為 `.csv` 格式，包含如 `BTC.csv`, `ETH.csv`, `ADA.csv` 等幣種的歷史價格資訊。
