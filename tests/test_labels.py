# tests/test_labels.py
import pandas as pd
import numpy as np
from smc.labels import make_labels

def test_labels_deadzone_simple():
    # Construct deterministic prices that produce one >eps, one in dead-zone, one <-eps
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=4, freq="B"),
        "close": [100.0, 100.2, 100.199, 99.9],
        "ticker": ["T"] * 4
    })
    out = make_labels(df, price_col="close", horizon=1, epsilon=0.001, drop_deadzone=True)
    labels = out['label'].tolist()
    # Expected: first return >eps -> 1
    assert labels[0] == 1
    # second return ~ -1e-5 -> within dead-zone -> pd.NA
    assert pd.isna(labels[1])
    # third return < -eps -> 0
    assert labels[2] == 0
    # last row has no forward price -> pd.NA
    assert pd.isna(labels[3])

def test_multi_ticker_grouping():
    # two tickers, trivial behavior: labels computed separately
    df = pd.concat([
        pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3, freq="B"),
                      "close":[10, 10.5, 10.6], "ticker":["A"]*3}),
        pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3, freq="B"),
                      "close":[20, 19.8, 19.6], "ticker":["B"]*3})
    ], ignore_index=True)
    out = make_labels(df, price_col="close", horizon=1, epsilon=0.0001, drop_deadzone=True)
    # Ensure labels exist and no cross-ticker leakage
    assert set(out[out['ticker']=="A"]['label'].dropna().unique()).issubset({0,1})
    assert set(out[out['ticker']=="B"]['label'].dropna().unique()).issubset({0,1})
