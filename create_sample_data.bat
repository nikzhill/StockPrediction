@echo off
REM Create folders
mkdir C:\Projects\StockPrediction\data\raw
mkdir C:\Projects\StockPrediction\data\processed
mkdir C:\Projects\StockPrediction\data\sentiment

REM Create sample news CSV
echo date,text> C:\Projects\StockPrediction\data\raw\AAPL_news.csv
echo 2025-09-01,Apple stock rises after new iPhone launch>> C:\Projects\StockPrediction\data\raw\AAPL_news.csv
echo 2025-09-02,Concerns grow about slowing iPhone sales in China>> C:\Projects\StockPrediction\data\raw\AAPL_news.csv
echo 2025-09-03,Apple announces record-breaking quarterly profits>> C:\Projects\StockPrediction\data\raw\AAPL_news.csv

REM Create sample price CSV
echo date,open,high,low,close,volume> C:\Projects\StockPrediction\data\processed\AAPL_prices.csv
echo 2025-09-01,150,152,149,151,1000000>> C:\Projects\StockPrediction\data\processed\AAPL_prices.csv
echo 2025-09-02,151,153,150,152,1100000>> C:\Projects\StockPrediction\data\processed\AAPL_prices.csv
echo 2025-09-03,152,154,151,153,1050000>> C:\Projects\StockPrediction\data\processed\AAPL_prices.csv

echo Sample data created successfully!
pause
