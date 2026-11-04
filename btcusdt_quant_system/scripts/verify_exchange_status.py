import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from execution.binance_testnet_client import BinanceFuturesTestnetClient

async def main():
    c = BinanceFuturesTestnetClient()
    bal = await c.get_usdt_balance()
    print(f"WALLET TOTAL BALANCE: ${bal.get('total_balance'):.2f} USDT")
    print(f"WALLET AVAILABLE BALANCE: ${bal.get('available_balance'):.2f} USDT")

    pos = await c.get_position_risk()
    active_p = [p for p in pos if abs(float(p.get("positionAmt", 0))) > 0]
    print(f"\nACTIVE POSITIONS: {len(active_p)}")
    for p in active_p:
        amt = float(p.get("positionAmt", 0))
        ep = float(p.get("entryPrice", 0))
        notional = abs(amt) * ep
        print(f"  {p.get('symbol')}: amt={amt} entryPrice={ep} notional=${notional:.2f} lev={p.get('leverage')}x isolated={p.get('isolated')}")

    orders = await c.get_open_orders()
    print(f"\nACTIVE EXCHANGE ORDERS: {len(orders)}")
    for o in orders:
        print(f"  {o.get('symbol')}: id={o.get('orderId')} type={o.get('type')} side={o.get('side')} price={o.get('price')} stopPrice={o.get('stopPrice')} closePos={o.get('closePosition')}")

    await c.close()

if __name__ == "__main__":
    asyncio.run(main())
