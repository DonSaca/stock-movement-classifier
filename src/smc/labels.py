# src/smc/labels.py
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd

def compute_next_log_return(price: pd.Series, horizon: int = 1) -> pd.Series:
    """
    price: pandas Series sorted by date ascending (one ticker).
    horizon: days ahead for the return (int).
    Returns a Series aligned with the same index, NaN where forward price missing.
    """
    return np.log(price.shift(-horizon) / price)

def label_from_returns(returns: pd.Series,
                       epsilon: float = 0.001,
                       drop_deadzone: bool = True) -> pd.Series:
    """
    returns: next-horizon log returns
    epsilon: dead-zone threshold (e.g., 0.001 = 0.1%)
    drop_deadzone: if True set labels within [-eps, +eps] to pd.NA (so they can be dropped)
                   if False fill them (default 0) — not recommended initially.
    Returns a pandas Series with dtype 'Int64' (nullable int).
    """
    labels = pd.Series(pd.NA, index=returns.index, dtype="Int64")
    labels.loc[returns > epsilon] = 1
    labels.loc[returns < -epsilon] = 0
    if not drop_deadzone:
        labels = labels.fillna(0).astype("Int64")
    return labels

def make_labels(df: pd.DataFrame,
                price_col: str = "adj_close",
                horizon: int = 1,
                epsilon: float = 0.001,
                drop_deadzone: bool = True,
                group_col: Optional[str] = "ticker",
                sort_col: str = "date") -> pd.DataFrame:
    """
    Add two columns to a copy of df:
      - next_{horizon}d_logret (float)
      - label (Int64 nullable)  -- values 1, 0 or pd.NA
    df must contain price_col and if group_col is provided, grouping is applied per ticker.
    """
    df = df.copy()
    if group_col:
        def _proc(g):
            g = g.sort_values(sort_col)
            g[f"next_{horizon}d_logret"] = compute_next_log_return(g[price_col], horizon)
            g["label"] = label_from_returns(g[f"next_{horizon}d_logret"], epsilon, drop_deadzone)
            return g
        out = df.groupby(group_col, group_keys=False).apply(_proc)
    else:
        out = df.sort_values(sort_col)
        out[f"next_{horizon}d_logret"] = compute_next_log_return(out[price_col], horizon)
        out["label"] = label_from_returns(out[f"next_{horizon}d_logret"], epsilon, drop_deadzone)

    # keep original index order
    out = out.sort_index()
    return out

def save_labels(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # drop pandas NA-int type problems by letting parquet store nullable ints
    df.to_parquet(out_path, index=False)
