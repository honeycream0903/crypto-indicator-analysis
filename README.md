# 📊 Crypto Indicator Analysis

本專案分析常見技術指標（如 SMA、RSI、MACD 等）在加密貨幣價格預測中的有效性，使用 XGBoost 統計其特徵重要性。  
資料處理與模型訓練皆於 Google Colab 環境完成，無需手動下載資料或設定環境，所有安裝與資料載入皆在 Notebook 中自動完成。
你可以點擊下方按鈕，在 Google Colab 直接開啟並執行專案：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)(https://colab.research.google.com/drive/1c4DSuPSMAFO7b7sSfLmrRu8m82O2ATew#scrollTo=x_3BQwHBmtAr)]



# 📦 環境需求
- Python 3
- pandas
- pandas-ta
- xgboost
- gdown

可於 Colab 中直接安裝必要套件：
```bash
!pip install pandas_ta
!pip install xgboost
```

# 📥 資料下載與解壓縮

```bash
# 下載每分/每日價格的壓縮資料並解壓縮
"""
!gdown --id 1mDQP9sIAdG4813ehJ9P0Ortad8gJEU_h
!unzip '/content/Crypto_1min.zip'
"""
!gdown --id 1_x08rXnQuvRzdCWVoD2Qftt6RmihaeSJ
!unzip '/content/Cryptos_1day.zip'
```

📌 所有資料皆為 `.csv` 格式，包含如 `BTC.csv`, `ETH.csv`, `ADA.csv` 等幣種的歷史價格資訊。


#結果呈現每一幣種前十名有效預測之數學技術指標與分數

圖示以 BTC 為例
![outputＢＴＣ](https://github.com/user-attachments/assets/2ac7e467-212c-45a5-bf3d-b51bdd3a39ec)

# ADA: iterations = 100
<pre>{'SMA_7': {'count': 86, 'score': 35.34580198605201},
 'SMA_30': {'count': 52, 'score': 24.962968176637915},
 'ROC_1': {'count': 45, 'score': 17.47844725908166},
 'ATRr_7': {'count': 37, 'score': 16.299778634085357},
 'ICS_26': {'count': 36, 'score': 15.40062045739566},
 'RSI_14': {'count': 35, 'score': 12.426192166752466},
 'ATRr_30': {'count': 23, 'score': 14.973758668545713},
 'MACDh_12_26_9': {'count': 22, 'score': 13.038482309807783},
 'MACDs_12_26_9': {'count': 19, 'score': 20.325090869629502},
 'ATRr_14': {'count': 18, 'score': 14.13159274824288}}</pre>
# ALGO: iterations = 83
<pre>{'SMA_7': {'count': 75, 'score': 34.253869699817066}, 
 'ROC_1': {'count': 52, 'score': 15.35281079646284}, 
 'SMA_30': {'count': 43, 'score': 22.45089428900566}, 
 'ATRr_7': {'count': 43, 'score': 14.789956491102219},
 'ICS_26': {'count': 37, 'score': 16.546014713348825},
 'RSI_14': {'count': 32, 'score': 11.594159473396292},
 'MACDh_12_26_9': {'count': 25, 'score': 14.118534736492855},
 'ATRr_30': {'count': 13, 'score': 10.589672579677638}, 
 'ATRr_14': {'count': 12, 'score': 13.603508142943042},
 'MACD_12_26_9': {'count': 11, 'score': 9.99637245395541}}</pre>
# ATOM: iterations = 88
<pre>{'SMA_7': {'count': 84, 'score': 36.14112698113785}, 
 'ROC_1': {'count': 55, 'score': 13.961849930551756}, 
 'SMA_30': {'count': 54, 'score': 19.462283388433985},
 'ATRr_7': {'count': 43, 'score': 14.335961629200419},
 'ICS_26': {'count': 34, 'score': 13.412820077543628},
 'RSI_14': {'count': 30, 'score': 13.160777470898216}, 
 'MACDh_12_26_9': {'count': 23, 'score': 10.757741914691943},
 'MACD_12_26_9': {'count': 17, 'score': 12.405006077618847}, 
 'MACDs_12_26_9': {'count': 14, 'score': 13.09519268150255}, 
 'ISA_9': {'count': 12, 'score': 11.553163421897837}}</pre>
# AVAX: iterations = 37
<pre>{'SMA_7': {'count': 34, 'score': 39.62325354583327}, 
 'ATRr_7': {'count': 21, 'score': 16.326748890654674}, 
 'ROC_1': {'count': 20, 'score': 15.609046382980159}, 
 'SMA_30': {'count': 18, 'score': 18.17512968906947}, 
 'MACDh_12_26_9': {'count': 14, 'score': 10.863054756535146}, 
 'RSI_14': {'count': 12, 'score': 12.23441095478276}, 
 'ICS_26': {'count': 11, 'score': 8.324679265962335}, 
 'MACD_12_26_9': {'count': 10, 'score': 9.466292142118155},
 'ISA_9': {'count': 8, 'score': 8.998771295397978},
 'ATRr_30': {'count': 6, 'score': 15.21045538931652}}</pre>
# BNB: iterations = 100
<pre>{'SMA_7': {'count': 99, 'score': 33.08301858308677}, 
 'ATRr_7': {'count': 59, 'score': 13.716041191038201},
 'SMA_30': {'count': 53, 'score': 17.33094833074927},
 'ROC_1': {'count': 53, 'score': 15.199737151588218}, 
 'MACDh_12_26_9': {'count': 36, 'score': 11.509514592601356}, 
 'RSI_14': {'count': 36, 'score': 12.350780436655402},
 'ICS_26': {'count': 32, 'score': 13.682957847493283},
 'MACD_12_26_9': {'count': 22, 'score': 11.927223390572903},
 'ISA_9': {'count': 17, 'score': 12.664006025358427},
 'ATRr_14': {'count': 16, 'score': 16.522381519541366}}</pre>
# BTC: iterations = 100
<pre>{'SMA_7': {'count': 98, 'score': 37.14451320102745},
 'SMA_30': {'count': 59, 'score': 17.961984551702262}, 
 'ATRr_7': {'count': 59, 'score': 14.20809878825749},
 'ROC_1': {'count': 50, 'score': 13.003722317712993},
 'ICS_26': {'count': 42, 'score': 12.075257835968412},
 'RSI_14': {'count': 35, 'score': 10.869409839607194}, 
 'MACDh_12_26_9': {'count': 34, 'score': 12.074784179802808},
 'MACD_12_26_9': {'count': 23, 'score': 12.66013944953382}, 
 'ATRr_14': {'count': 16, 'score': 11.896041155834176}, 
 'ATRr_30': {'count': 15, 'score': 13.057841314169044}}</pre>
# DOT: iterations = 41
<pre>{'SMA_7': {'count': 41, 'score': 33.32510216104093}, 
 'ROC_1': {'count': 22, 'score': 13.567408343487786},
 'ATRr_7': {'count': 21, 'score': 17.962045003146372}, 
 'SMA_30': {'count': 20, 'score': 15.436903454519438},
 'ICS_26': {'count': 16, 'score': 8.910982231321125}, 
 'RSI_14': {'count': 15, 'score': 12.985550220115675},
 'MACDh_12_26_9': {'count': 13, 'score': 14.056118823614046},
 'MACD_12_26_9': {'count': 10, 'score': 13.488354262929949}, 
 'ISA_9': {'count': 10, 'score': 16.730168744945782}, 
 'MACDs_12_26_9': {'count': 8, 'score': 17.732337861056745}}</pre>
# EGLD: iterations = 39
<pre>{'SMA_7': {'count': 39, 'score': 37.782242433903356}, 
 'ATRr_7': {'count': 22, 'score': 17.05367907140483},
 'ROC_1': {'count': 19, 'score': 13.53337380328696}, 
 'SMA_30': {'count': 18, 'score': 16.682191506160507}, 
 'RSI_14': {'count': 18, 'score': 12.12766830701307}, 
 'ICS_26': {'count': 16, 'score': 11.919305261395682},
 'ATRr_30': {'count': 10, 'score': 12.93295073864851},
 'MACD_12_26_9': {'count': 9, 'score': 13.673301379793067},
 'MACDh_12_26_9': {'count': 8, 'score': 11.091152547322318},
 'ATRr_14': {'count': 7, 'score': 7.886122952304183}}</pre>
# ETH: iterations = 100
<pre>{'SMA_7': {'count': 99, 'score': 33.588873204613556}, 
 'ROC_1': {'count': 57, 'score': 13.615792700415492}, 
 'ATRr_7': {'count': 55, 'score': 16.92832469569538}, 
 'SMA_30': {'count': 50, 'score': 18.97059140906744}, 
 'RSI_14': {'count': 42, 'score': 10.82514770305735},
 'MACDh_12_26_9': {'count': 34, 'score': 12.514469143723153}, 
 'ICS_26': {'count': 34, 'score': 13.035300784671596}, 
 'MACD_12_26_9': {'count': 25, 'score': 13.997925610088048},
 'ATRr_14': {'count': 18, 'score': 11.439592841944311}, 
 'ATRr_30': {'count': 18, 'score': 13.154221886540236}}</pre>
# FLOW: iterations = 6
<pre> {'SMA_7': {'count': 6, 'score': 31.605953991807027}, 
 'RSI_14': {'count': 5, 'score': 12.867942825390347},
 'MACD_12_26_9': {'count': 3, 'score': 11.057779051379205}, 
 'ROC_1': {'count': 3, 'score': 8.10027688122351}, 
 'ICS_26': {'count': 3, 'score': 11.553481707206473}, 
 'ATRr_7': {'count': 2, 'score': 15.16918738784927}, 
 'MACDh_12_26_9': {'count': 2, 'score': 29.688832054560955},
 'SMA_30': {'count': 1, 'score': 32.974137931034484}, 
 'ATRr_14': {'count': 1, 'score': 12.340425531914894}, 
 'MACDs_12_26_9': {'count': 1, 'score': 25.70281124497992}}</pre>
# FTM: iterations = 84
<pre>{'SMA_7': {'count': 57, 'score': 42.617691578928984}, 
 'SMA_30': {'count': 30, 'score': 34.403081344122754}, 
 'ROC_1': {'count': 26, 'score': 17.933447012202606}, 
 'ATRr_7': {'count': 24, 'score': 21.536930372427538}, 
 'ICS_26': {'count': 23, 'score': 14.10500395180317}, 
 'RSI_14': {'count': 22, 'score': 13.76901643343529}, 
 'MACDh_12_26_9': {'count': 14, 'score': 18.687726668391992},
 'MACDs_12_26_9': {'count': 11, 'score': 25.164748581253786}, 
 'ISA_9': {'count': 11, 'score': 18.892179674623318}, 
 'ATRr_30': {'count': 10, 'score': 16.153530095723337}}</pre>
# KLAY: iterations = 10
<pre>{'SMA_7': {'count': 9, 'score': 35.71631917479462}, 
 'ATRr_7': {'count': 6, 'score': 22.689450748286006}, 
 'ICS_26': {'count': 6, 'score': 10.653749241249994}, 
 'SMA_30': {'count': 5, 'score': 21.922050103522643}, 
 'ROC_1': {'count': 5, 'score': 11.474859448369608},
 'MACDh_12_26_9': {'count': 4, 'score': 13.864526271257938},
 'RSI_14': {'count': 4, 'score': 14.935421308390072}, 
 'ATRr_14': {'count': 2, 'score': 8.536693191865606},
 'RMA_30': {'count': 2, 'score': 8.305002428363283}, 
 'HMA_9': {'count': 2, 'score': 20.37474593495935}}</pre>
# LUNA: iterations = 40
<pre>{'SMA_7': {'count': 37, 'score': 32.99606223109505}, 
 'ATRr_7': {'count': 30, 'score': 14.82376822868193},
 'SMA_30': {'count': 23, 'score': 19.83541852257771},
 'ROC_1': {'count': 18, 'score': 18.398315301586976},
 'RSI_14': {'count': 17, 'score': 11.38771419248924}, 
 'ICS_26': {'count': 14, 'score': 13.631257077340944},
 'MACDh_12_26_9': {'count': 13, 'score': 10.937334149339607},
 'MACDs_12_26_9': {'count': 8, 'score': 15.365382605211876}, 
 'MACD_12_26_9': {'count': 7, 'score': 9.62648588602577}, 
 'ATRr_14': {'count': 6, 'score': 11.815473210672366}}</pre>
# MATIC: iterations = 89
<pre>{'SMA_7': {'count': 64, 'score': 44.81879201513878}, 
 'SMA_30': {'count': 44, 'score': 39.33639692882581}, 
 'ROC_1': {'count': 32, 'score': 15.290539837816334},
 'ATRr_7': {'count': 28, 'score': 24.230041920675472},
 'ICS_26': {'count': 28, 'score': 13.941845459164337},
 'RSI_14': {'count': 21, 'score': 11.853851920858155},
 'ATRr_14': {'count': 11, 'score': 19.602549333721075}, 
 'EMA_20': {'count': 10, 'score': 19.799389444600045}, 
 'MACDs_12_26_9': {'count': 9, 'score': 24.749801425032942}, 
 'EMA_50': {'count': 9, 'score': 14.625040842882953}}</pre>
# NEAR: iterations = 35
<pre>{'SMA_7': {'count': 35, 'score': 34.79521776304074}, 
 'ROC_1': {'count': 21, 'score': 16.32819595567596}, 
 'ICS_26': {'count': 20, 'score': 13.946534890989998},
 'ATRr_7': {'count': 19, 'score': 16.156953219804414}, 
 'RSI_14': {'count': 13, 'score': 8.108333416824099}, 
 'SMA_30': {'count': 12, 'score': 18.214147109654014},
 'MACDh_12_26_9': {'count': 11, 'score': 13.058959034178654}, 
 'MACD_12_26_9': {'count': 8, 'score': 8.176931279803123}, 
 'ATRr_14': {'count': 7, 'score': 8.433367132188444}, 
 'ISA_9': {'count': 7, 'score': 15.422711581956706}}</pre>
# ONE: iterations = 85
<pre>{'SMA_7': {'count': 39, 'score': 43.90976236537295},
 'SMA_30': {'count': 26, 'score': 34.634000899876035}, 
 'ROC_1': {'count': 18, 'score': 19.539649123283976}, 
 'RSI_14': {'count': 17, 'score': 19.433274752236173}, 
 'ATRr_7': {'count': 14, 'score': 17.828361584590414}, 
 'ICS_26': {'count': 14, 'score': 13.910218374420312}, 
 'MACDh_12_26_9': {'count': 8, 'score': 11.255102842071972}, 
 'ATRr_14': {'count': 7, 'score': 18.752097922944717},
 'ATRr_30': {'count': 7, 'score': 16.592873206966747}, 
 'MACD_12_26_9': {'count': 6, 'score': 16.4131314709297}}</pre>
# SOL: iterations = 41
<pre>{'SMA_7': {'count': 40, 'score': 37.966428616552314},
 'ATRr_7': {'count': 28, 'score': 14.678418883123515}, 
 'SMA_30': {'count': 21, 'score': 18.44063444191997}, 
 'ROC_1': {'count': 20, 'score': 14.837468262066679},
 'RSI_14': {'count': 19, 'score': 9.35162310640872}, 
 'ICS_26': {'count': 15, 'score': 12.074304176012197}, 
 'MACDh_12_26_9': {'count': 14, 'score': 14.312982066333364},
 'MACD_12_26_9': {'count': 10, 'score': 11.602393520056813}, 
 'ATRr_14': {'count': 7, 'score': 20.601348186957058}, 
 'HMA_9': {'count': 7, 'score': 10.96274512256768}}</pre>
# TRX: iterations = 100
<pre>{'SMA_7': {'count': 58, 'score': 52.756523106296825},
 'SMA_30': {'count': 34, 'score': 51.25909797523626},
 'ICS_26': {'count': 29, 'score': 16.821115466663855},
 'ATRr_7': {'count': 25, 'score': 23.31357367542781},
 'ROC_1': {'count': 22, 'score': 15.227397516433676}, 
 'RSI_14': {'count': 21, 'score': 19.652328460482778},
 'MACDh_12_26_9': {'count': 17, 'score': 21.77442244838017},
 'MACD_12_26_9': {'count': 13, 'score': 21.597873644071964},
 'ATRr_30': {'count': 9, 'score': 24.891649605088645}, 
 'ATRr_14': {'count': 8, 'score': 23.028762903762903}}</pre>
# XTZ: iterations = 73
<pre> {'SMA_7': {'count': 71, 'score': 32.27933768846761},
 'ROC_1': {'count': 47, 'score': 16.802672179043835}, 
 'ATRr_7': {'count': 41, 'score': 17.2900627279014},
 'SMA_30': {'count': 38, 'score': 19.098899471609794},
 'ICS_26': {'count': 31, 'score': 10.07434779989969}, 
 'RSI_14': {'count': 28, 'score': 12.524840102831693}, 
 'MACDh_12_26_9': {'count': 18, 'score': 10.588560121899683},
 'MACD_12_26_9': {'count': 13, 'score': 15.507807117731655}, 
 'ATRr_14': {'count': 10, 'score': 12.138377204600593}, 
 'ISA_9': {'count': 10, 'score': 10.135567342207043}}</pre>
# ZIL: iterations = 95
<pre> {'SMA_7': {'count': 54, 'score': 40.71848638975902}, 
 'SMA_30': {'count': 43, 'score': 38.803182571242466},
 'ROC_1': {'count': 31, 'score': 17.802160344972243}, 
 'ATRr_7': {'count': 26, 'score': 19.421490022277652}, 
 'ICS_26': {'count': 20, 'score': 24.287065069291824}, 
 'RSI_14': {'count': 12, 'score': 18.867337037003235},
 'ATRr_14': {'count': 11, 'score': 21.35765275994755},
 'MACDh_12_26_9': {'count': 11, 'score': 24.036160793922136}, 
 'MACD_12_26_9': {'count': 10, 'score': 15.4294087554382},
 'ISA_9': {'count': 9, 'score': 17.35072828263949}}</pre>
