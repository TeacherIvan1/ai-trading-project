import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from execution.binance_testnet_client import BinanceFuturesTestnetClient

async def main():
    c = BinanceFuturesTestnetClient()
    await c.sync_server_time()
    
    # 1. Query before
    bal_before = await c.get_usdt_balance()
    pos_before = await c.get_position_risk()
    active_before = [p for p in pos_before if abs(float(p.get('positionAmt', 0))) > 0]
    print("=== BEFORE FLATTEN ===")
    print("Wallet Balance:", bal_before)
    total_unreal = 0.0
    for p in active_before:
        sym = p['symbol']
        amt = float(p['positionAmt'])
        entry = float(p['entryPrice'])
        unreal = float(p['unRealizedProfit'])
        total_unreal += unreal
        notional = abs(amt) * entry
        print(f"  {sym:<8} | Amt={amt} | Entry={entry} | Notional=${notional:.2f} | UnrealPnL=${unreal:.2f}")
    print(f"Total Unrealized PnL to lock in: ${total_unreal:.2f}")
        
    print("\n=== FLATTENING ALL POSITIONS ===")
    res = await c.flatten_all_positions()
    for r in res:
        print(" ", r)
        
    await asyncio.sleep(2.0)
    
    # 2. Query after
    bal_after = await c.get_usdt_balance()
    pos_after = await c.get_position_risk()
    active_after = [p for p in pos_after if abs(float(p.get('positionAmt', 0))) > 0]
    print("\n=== AFTER FLATTEN ===")
    print("Wallet Balance:", bal_after)
    print("Active Positions Remaining:", len(active_after))
    for p in active_after:
        print(" ", p['symbol'], p['positionAmt'])
        
    await c.close()

if __name__ == "__main__":
    asyncio.run(main())
