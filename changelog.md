feat(data): implement yfinance ingestion and validation pipeline

- Added download.py to fetch daily OHLCV from yfinance and cache as Parquet
- Added validate.py to check schema, NaNs, duplicates, gaps, and column integrity
- Normalized columns (date, open, high, low, close, adj_close, volume, ticker)
- Verified data integrity for AAPL, MSFT, SPY from 2015 onwards


Added changelog