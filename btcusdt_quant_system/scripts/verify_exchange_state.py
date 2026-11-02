import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from execution.binance_testnet_client import BinanceFuturesTestnetClient, DEFAULT_UNIVERSE

async def main():
    client = BinanceFuturesTestnetClient()
    print("=== BINANCE FUTURES TESTNET LIVE AUDIT ===")
    
    # 1. Balance
    bal = await client.get_usdt_balance()
    print(f"Total Equity: ${bal.get('total_balance'):,.2f} USDT")
    print(f"Available Balance: ${bal.get('available_balance'):,.2f} USDT")
    
    # 2. Positions
    positions = await client.get_position_risk()
    active_pos = [p for p in positions if abs(float(p.get("positionAmt", 0.0))) > 0]
    print(f"\nActive Open Positions: {len(active_pos)}")
    for p in active_pos:
        print(f"  {p.get('symbol')}: amt={p.get('positionAmt')}, entry={p.get('entryPrice')}, leverage={p.get('leverage')}x, isolated={p.get('isolated')}")
    
    # 3. Leverage and Margin Type per universe symbol
    print(f"\nUniverse Clamping Verification (10 Symbols):")
    for s in DEFAULT_UNIVERSE:
        pos_info = next((p for p in positions if p.get("symbol") == s), None)
        if pos_info:
            lev = pos_info.get("leverage")
            m_type = "ISOLATED" if pos_info.get("isolated") else "CROSS"
            amt = float(pos_info.get("positionAmt", 0.0))
            print(f"  {s:<10} | Leverage: {lev}x | Margin: {m_type:<8} | Pos Amt: {amt}")
        else:
            print(f"  {s:<10} | No record in positionRisk")
            
    # 4. Open Orders
    total_orders = 0
    print(f"\nOpen Orders on Exchange:")
    for s in DEFAULT_UNIVERSE:
        orders = await client.get_open_orders(s)
        total_orders += len(orders)
        if orders:
            print(f"  {s}: {len(orders)} open orders")
            for o in orders:
                print(f"    OrderID: {o.get('orderId')} | Type: {o.get('type')} | Side: {o.get('side')} | StopPrice: {o.get('stopPrice')} | Price: {o.get('price')} | closePosition: {o.get('closePosition')}")
    print(f"Total Open Orders across all 10 symbols: {total_orders}")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
