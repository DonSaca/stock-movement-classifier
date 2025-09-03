# src/smc/data/download.py
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yfinance as yf
from tqdm import tqdm


REQUIRED_COLS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


def _today_utc_date_str() -> str:
    # yfinance end is exclusive; add +1 day so we include today's bar when available
    return (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()


def _normalize_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize yfinance output into columns:
      date, open, high, low, close, adj_close, volume, ticker

    Handles:
      - Series -> single-row DataFrame
      - MultiIndex columns -> flattened
      - several column name variants (case-insensitive)
      - safe numeric conversion using .values to ensure 1-D input to pd.to_numeric
    """
    import pandas as pd

    REQUIRED_COLS = ["date", "open", "high", "low", "close", "adj_close", "volume"]

    # empty / none handling
    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.DataFrame(columns=REQUIRED_COLS + ["ticker"])

    # If a Series (happens sometimes), convert to single-row DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame().T

    # Flatten MultiIndex columns if present (e.g., (TICKER, 'Open') -> 'Open')
    if getattr(df.columns, "nlevels", 1) > 1:
        def _flatten(col):
            if isinstance(col, tuple):
                # prefer the last level if non-empty, else first
                return col[-1] if col[-1] not in (None, "") else col[0]
            return col
        df = df.copy()
        df.columns = [_flatten(c) for c in df.columns]

    # Work on a copy and reset index to get 'Date' as a column if it was the index
    df = df.copy()
    df.reset_index(inplace=True)

    # Build a flexible case-insensitive rename map for common column names
    col_map = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ("date", "index", "datetime"):
            col_map[c] = "date"
        elif lc in ("open", "o"):
            col_map[c] = "open"
        elif lc == "high":
            col_map[c] = "high"
        elif lc == "low":
            col_map[c] = "low"
        elif lc in ("close", "c"):
            col_map[c] = "close"
        elif lc in ("adj close", "adj_close", "adjusted close", "adjclose"):
            col_map[c] = "adj_close"
        elif lc in ("volume", "vol"):
            col_map[c] = "volume"

    df.rename(columns=col_map, inplace=True)

    # Ensure expected columns exist (fill with NA if missing)
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        if col not in df.columns:
            df[col] = pd.NA

    # Date conversion (robust)
    try:
        df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)
    except Exception:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Numeric conversions: use .values so to_numeric receives a 1-D array
    for col in ("open", "high", "low", "close", "adj_close"):
        df[col] = pd.to_numeric(df[col].values, errors="coerce")

    # Volume: coerce then try Int64, fallback to float if Int64 fails
    vol_vals = pd.to_numeric(df["volume"].values, errors="coerce")
    try:
        df["volume"] = pd.Series(vol_vals).astype("Int64")
    except Exception:
        df["volume"] = pd.Series(vol_vals)

    df["ticker"] = ticker.upper()
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    # Return columns in stable order
    return df[["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"]]


def _fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    # actions=False speeds up; auto_adjust=False so we keep both close and adj_close
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    return _normalize_df(df, ticker)


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_COLS + ["ticker"])
    return pd.read_parquet(path)


def _merge_incremental(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        merged = incoming
    else:
        merged = pd.concat([existing, incoming], axis=0, ignore_index=True)
    merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return merged


def download_and_cache(
    tickers: Iterable[str],
    start: str,
    end: str | None,
    out_dir: Path,
    force_full: bool,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    if end is None:
        end = _today_utc_date_str()

    for t in tqdm([t.strip().upper() for t in tickers if t.strip()], desc="Downloading"):
        out_path = out_dir / f"{t}.parquet"
        existing = _load_existing(out_path)

        # incremental start: day after last available date
        inc_start = start
        if not force_full and not existing.empty:
            last_date = existing["date"].max()
            if pd.notna(last_date):
                inc_start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).date().isoformat()

        incoming = _fetch(t, inc_start, end)

        # If force_full: re-fetch from start and replace
        if force_full:
            merged = incoming
        else:
            merged = _merge_incremental(existing, incoming)

        # Final sort/dedup before save
        merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

        # Save parquet (snappy)
        merged.to_parquet(out_path, index=False)
        saved_paths.append(out_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download daily OHLCV via yfinance and cache per-ticker Parquet."
    )
    parser.add_argument("--tickers", nargs="+", required=True, help="Tickers, e.g. AAPL MSFT SPY")
    parser.add_argument("--start", type=str, default="2010-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (exclusive). Defaults to tomorrow UTC.")
    parser.add_argument("--out-dir", type=str, default=os.getenv("SMC_DATA_DIR", "data/raw"),
                        help="Output directory for Parquet files.")
    parser.add_argument("--force-full", action="store_true", help="Re-download from start and overwrite existing.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    paths = download_and_cache(args.tickers, args.start, args.end, out_dir, args.force_full)

    print("\nSaved files:")
    for p in paths:
        print(f"  - {p.resolve()}")


if __name__ == "__main__":
    main()
#sept 3