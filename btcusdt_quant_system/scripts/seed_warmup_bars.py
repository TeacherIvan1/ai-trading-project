"""
scripts/seed_warmup_bars.py
===========================
Fetches 60 1m klines for all 10 universe symbols and writes them to
logs/live_dollar_bars.csv so candlestick charts render immediately.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LIVE_BARS_CSV = LOGS_DIR / "live_dollar_bars.csv"

DEFAULT_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"
]

def fetch_klines(symbol: str, limit: int = 60):
    urls = [
        f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1m&limit={limit}",
        f"https://testnet.binancefuture.com/fapi/v1/klines?symbol={symbol}&interval=1m&limit={limit}"
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, list) and len(data) >= 10:
                    rows = []
                    for k in data:
                        vol = float(k[5])
                        taker_buy_vol = float(k[9])
                        net_flow = 2.0 * taker_buy_vol - vol
                        rows.append({
                            "open_ts": int(k[0]),
                            "close_ts": int(k[6]),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": vol,
                            "dollar_volume": float(k[7]),
                            "net_order_flow": net_flow,
                            "ticks": int(k[8]),
                            "symbol": symbol
                        })
                    return rows
        except Exception:
            continue
    return []

def main():
    existing_rows = []
    if LIVE_BARS_CSV.exists():
        try:
            df_exist = pd.read_csv(LIVE_BARS_CSV)
            existing_rows = df_exist.to_dict(orient="records")
        except Exception:
            existing_rows = []

    existing_syms = set()
    for r in existing_rows:
        sym = r.get("symbol")
        if sym:
            existing_syms.add(sym)

    all_rows = []
    for s in DEFAULT_UNIVERSE:
        bars = fetch_klines(s, limit=60)
        print(f"Fetched {len(bars)} warmup bars for {s}")
        all_rows.extend(bars)

    # Combine existing and new rows, preserving chronological order
    all_rows.extend(existing_rows)

    df_out = pd.DataFrame(all_rows)
    # Deduplicate by symbol and close_ts
    df_out = df_out.drop_duplicates(subset=["symbol", "close_ts"]).sort_values(by=["symbol", "close_ts"])
    df_out.to_csv(LIVE_BARS_CSV, index=False)
    print(f"Successfully wrote {len(df_out)} bars across {df_out['symbol'].nunique()} symbols to {LIVE_BARS_CSV}")

if __name__ == "__main__":
    main()

