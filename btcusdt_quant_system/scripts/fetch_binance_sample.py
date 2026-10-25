"""
scripts/fetch_binance_sample.py
===============================
Lightweight, high-throughput historical aggTrades downloader for Binance USD(S)-M Futures.
Streams 30 days of daily compressed zip CSVs from data.binance.vision directly in-memory,
aggregates them into Information-Driven Dollar Bars, and computes VPIN & Kyle's Lambda.
"""

import concurrent.futures
import io
import os
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from core.data_bars import DollarBarAndMicrostructureEngine


def download_daily_zip(symbol: str, date_str: str, cache_dir: str) -> tuple[str, bytes]:
    """Downloads or reads cached daily aggTrades zip from data.binance.vision."""
    cached_file = os.path.join(cache_dir, f"{symbol}-aggTrades-{date_str}.zip")
    if os.path.exists(cached_file) and os.path.getsize(cached_file) > 1024:
        with open(cached_file, "rb") as f:
            return date_str, f.read()

    url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 200:
                    data = resp.read()
                    try:
                        with open(cached_file, "wb") as f_out:
                            f_out.write(data)
                    except Exception:
                        pass
                    return date_str, data
        except Exception as e:
            if attempt == 2:
                print(f"Warning: Failed to download {date_str}: {e}")
                return date_str, b""
            time.sleep(1.0)
    return date_str, b""


def fetch_and_convert_30d(
    symbol: str = "BTCUSDT",
    start_date: str = "2024-06-01",
    days: int = 30,
    dollar_threshold: float = 50_000_000.0,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Downloads 30 days of aggTrades, streams them through DollarBarAndMicrostructureEngine,
    and returns a DataFrame with completed bars, VPIN, and Kyle's Lambda.
    """
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "data", f"{symbol.lower()}_dollar_bars_30d.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cache_dir = os.path.join(PROJECT_ROOT, "data", "cache_zips")
    os.makedirs(cache_dir, exist_ok=True)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    date_list = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]

    print(f"[*] Downloading {days} days of {symbol} aggTrades from data.binance.vision ({date_list[0]} to {date_list[-1]})...")
    t0 = time.time()

    raw_zips = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(download_daily_zip, symbol, d, cache_dir): d for d in date_list}
        for future in concurrent.futures.as_completed(future_map):
            d_str, data = future.result()
            if data:
                raw_zips[d_str] = data
                print(f"  [+] Fetched {d_str} ({len(data)/1024/1024:.2f} MB)")

    print(f"[*] Data fetched in {time.time() - t0:.2f}s. Total days available: {len(raw_zips)}/{days}")
    print("[*] Streaming trades into DollarBarAndMicrostructureEngine ($50M threshold)...")

    engine = DollarBarAndMicrostructureEngine(dollar_threshold=dollar_threshold)
    total_trades = 0

    for d_str in sorted(raw_zips.keys()):
        data = raw_zips[d_str]
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
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

    print(f"[*] Processed {total_trades:,} raw aggTrades in {time.time() - t0:.2f}s.")
    print(f"[*] Total Dollar Bars generated: {len(engine.completed_bars):,}")

    bars_df = pd.DataFrame(engine.completed_bars)
    if bars_df.empty:
        raise RuntimeError("No Dollar Bars were generated. Verify input data.")

    bars_df["timestamp"] = pd.to_datetime(bars_df["close_ts"], unit="ms")
    bars_df = bars_df.sort_values("timestamp").reset_index(drop=True)

    print("[*] Computing VPIN and Kyle's Lambda...")
    bars_df = DollarBarAndMicrostructureEngine.compute_vpin_and_kyle_lambda(bars_df, window=50)

    try:
        bars_df.to_parquet(output_path, index=False)
        print(f"[+] Saved Dollar Bars to: {output_path}")
    except Exception as e:
        print(f"Warning saving parquet: {e}")

    csv_fallback = output_path.replace(".parquet", ".csv")
    bars_df.to_csv(csv_fallback, index=False)
    print(f"[+] Saved Dollar Bars to: {csv_fallback}")
    return bars_df


if __name__ == "__main__":
    fetch_and_convert_30d()
