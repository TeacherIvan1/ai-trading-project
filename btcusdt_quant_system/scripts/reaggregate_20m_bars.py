"""
scripts/reaggregate_20m_bars.py
===============================
Re-aggregates the cached 30-day BTCUSDT aggTrades using dollar_threshold = 20_000_000.0 ($20M Dollar Bars)
to increase sample density for machine learning features and labels.
"""

import io
import os
import sys
import time
import zipfile
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from core.data_bars import DollarBarAndMicrostructureEngine


def reaggregate_20m(
    dollar_threshold: float = 20_000_000.0,
    output_path: str = None,
):
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "data", "btcusdt_dollar_bars_20m_30d.parquet")
    cache_dir = os.path.join(PROJECT_ROOT, "data", "cache_zips")

    zip_files = sorted([os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".zip")])
    if not zip_files:
        raise RuntimeError("No cached zip files found in data/cache_zips")

    print(f"[*] Re-aggregating {len(zip_files)} daily zip files at ${dollar_threshold/1e6:.1f}M threshold...")
    t0 = time.time()

    engine = DollarBarAndMicrostructureEngine(dollar_threshold=dollar_threshold)
    total_trades = 0

    for idx, zf_path in enumerate(zip_files, 1):
        with zipfile.ZipFile(zf_path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(".csv"):
                    continue
                with zf.open(name) as f:
                    df = pd.read_csv(f, usecols=[1, 2, 5, 6], header=0, names=["price", "qty", "ts", "ibm"])
                    p = df["price"].to_numpy(dtype=np.float64)
                    q = df["qty"].to_numpy(dtype=np.float64)
                    ts = df["ts"].to_numpy(dtype=np.int64)
                    ibm = (df["ibm"].astype(str).str.lower() == "true").to_numpy(dtype=bool)

                    n = len(p)
                    total_trades += n
                    for i in range(n):
                        engine.process_agg_trade(int(ts[i]), float(p[i]), float(q[i]), bool(ibm[i]))
        print(f"  [{idx:02d}/30] Processed {os.path.basename(zf_path)} -> Cumulative Bars: {len(engine.completed_bars):,}")

    t_agg = time.time() - t0
    print(f"[*] Aggregated {total_trades:,} trades in {t_agg:.2f}s. Total $20M Bars: {len(engine.completed_bars):,}")

    bars_df = pd.DataFrame(engine.completed_bars)
    bars_df["timestamp"] = pd.to_datetime(bars_df["close_ts"], unit="ms")
    bars_df = bars_df.sort_values("timestamp").reset_index(drop=True)

    print("[*] Computing VPIN and Kyle's Lambda...")
    bars_df = DollarBarAndMicrostructureEngine.compute_vpin_and_kyle_lambda(bars_df, window=50)

    bars_df.to_parquet(output_path, index=False)
    csv_fallback = output_path.replace(".parquet", ".csv")
    bars_df.to_csv(csv_fallback, index=False)
    print(f"[+] Saved $20M Dollar Bars to:\n    {output_path}\n    {csv_fallback}")
    return bars_df


if __name__ == "__main__":
    reaggregate_20m()
