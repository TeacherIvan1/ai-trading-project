"""
scripts/cancel_all_open_orders.py
==================================
Immediate cleanup script to cancel all resting open orders on Binance Futures Testnet.
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from execution.binance_testnet_client import BinanceFuturesTestnetClient

DEFAULT_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"
]

async def cancel_all():
    client = BinanceFuturesTestnetClient()
    open_orders = await client.get_open_orders()
    print(f"[PRE-CLEANUP] Found {len(open_orders)} open orders resting on Binance Futures Testnet.")

    cancelled_count = 0
    symbols_with_orders = set(o.get("symbol") for o in open_orders if o.get("symbol"))

    # Also sweep default universe symbols to ensure thorough cancellation
    all_symbols = symbols_with_orders.union(set(DEFAULT_UNIVERSE))

    for sym in all_symbols:
        try:
            res = await client.cancel_all_open_orders(sym)
            if isinstance(res, dict) and res.get("code") == 200:
                print(f"  Cancelled orders for {sym}: {res.get('msg')}")
                cancelled_count += 1
            else:
                print(f"  Sweep {sym}: {res}")
        except Exception as e:
            print(f"  Error cancelling {sym}: {e}")

    await asyncio.sleep(1.0)
    remaining = await client.get_open_orders()
    print(f"\n[POST-CLEANUP] Remaining open orders: {len(remaining)}")
    await client.close()

if __name__ == "__main__":
    asyncio.run(cancel_all())

