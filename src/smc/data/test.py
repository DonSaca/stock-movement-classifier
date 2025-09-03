import yfinance as yf

df = yf.download("AAPL", start="2010-01-01", end=None, interval="1d", auto_adjust=False, actions=False, progress=False, threads=False)

print("Type:", type(df))
print("Columns:", df.columns)
print(df.head(5))
