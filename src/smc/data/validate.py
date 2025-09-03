# src/smc/data/validate.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd


REQUIRED_COLS = ["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"]


def _scan_files(base: Path, tickers: List[str] | None) -> List[Path]:
    if tickers:
        return [base / f"{t.strip().upper()}.parquet" for t in tickers if t.strip()]
    return sorted(base.glob("*.parquet"))


def _basic_fixes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return df


def _check_df(df: pd.DataFrame) -> dict:
    report = {}

    report["row_count"] = int(df.shape[0])
    report["col_ok"] = all(c in df.columns for c in REQUIRED_COLS)

    # Missing values count
    na_counts = df[["open", "high", "low", "close", "adj_close", "volume"]].isna().sum()
    report["na_any"] = bool(na_counts.sum() > 0)
    report["na_breakdown"] = {k: int(v) for k, v in na_counts.items()}

    # Flag if any entire column is missing/empty
    empty_cols = []
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        if c not in df.columns or df[c].isna().all() or (df[c] == 0).all():
            empty_cols.append(c)
    report["empty_columns"] = empty_cols

    # Duplicates and order
    report["duplicate_dates"] = int(df.duplicated(subset=["date"]).sum())
    report["sorted"] = bool(df["date"].is_monotonic_increasing)

    # Sanity checks for values
    neg_prices = (df[["open", "high", "low", "close", "adj_close"]] < 0).sum().sum()
    report["negative_prices"] = int(neg_prices)

    zero_vol_days = int((df["volume"].fillna(0) == 0).sum())
    report["zero_volume_days"] = zero_vol_days

    # Simple gap metric (rough, not trading-calendar aware)
    if df.shape[0] >= 2:
        gaps = df["date"].diff().dt.days.fillna(0)
        report["gaps_gt3_days"] = int((gaps > 3).sum())
    else:
        report["gaps_gt3_days"] = 0

    return report


def validate_files(files: List[Path], fix: bool, write_back: bool) -> Tuple[pd.DataFrame, List[Path]]:
    reports = []
    fixed_paths: List[Path] = []

    for f in files:
        if not f.exists():
            reports.append({"ticker_file": str(f), "exists": False})
            continue

        df = pd.read_parquet(f)
        # Soft fixes: sort + drop duplicate dates
        df_fixed = _basic_fixes(df) if fix else df

        rep = _check_df(df_fixed)
        rep.update({"ticker_file": str(f), "exists": True})

        reports.append(rep)

        if fix and write_back:
            df_fixed.to_parquet(f, index=False)
            fixed_paths.append(f)

    report_df = pd.DataFrame(reports)
    return report_df, fixed_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate cached Parquet OHLCV files.")
    p.add_argument("--data-dir", type=str, default=os.getenv("SMC_DATA_DIR", "data/raw"), help="Directory with *.parquet")
    p.add_argument("--tickers", nargs="*", help="Subset of tickers to validate, e.g. AAPL MSFT")
    p.add_argument("--fix", action="store_true", help="Apply non-destructive fixes (sort + de-dup).")
    p.add_argument("--write-back", action="store_true", help="Persist fixes to the same files.")
    p.add_argument("--report-out", type=str, default=None, help="Optional path to write CSV report.")
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(args.data_dir)
    files = _scan_files(base, args.tickers)

    report_df, fixed = validate_files(files, fix=args.fix, write_back=args.write_back)

    if args.report_out:
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(out_path, index=False)
        print(f"Validation report written to: {out_path.resolve()}")

    print("\nSummary:")
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(report_df)

    if fixed:
        print("\nFiles updated (sorted/de-duped):")
        for p in fixed:
            print(f"  - {p.resolve()}")


if __name__ == "__main__":
    main()
