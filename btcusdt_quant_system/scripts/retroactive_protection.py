import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from execution.binance_testnet_client import BinanceFuturesTestnetClient

TARGET_SYMBOLS = ["BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT"]

async def main():
    client = BinanceFuturesTestnetClient()
    await client.sync_server_time()
    
    print("[1] Querying current open positions from Binance Testnet...")
    positions = await client.get_position_risk()
    
    target_positions = {}
    for p in positions:
        sym = p.get("symbol")
        amt = float(p.get("positionAmt", 0.0))
        if sym in TARGET_SYMBOLS and abs(amt) > 0:
            target_positions[sym] = {
                "amt": amt,
                "entry_price": float(p.get("entryPrice", 0.0)),
                "side": "BUY" if amt > 0 else "SELL",
                "leverage": p.get("leverage"),
                "isolated": p.get("isolated")
            }
            print(f"  Found position: {sym:<8} | Amt={amt} | Side={target_positions[sym]['side']} | Entry=${target_positions[sym]['entry_price']:.4f}")

    print("\n[2] Submitting missing STOP_MARKET and TAKE_PROFIT_MARKET orders...")
    results = {}
    for sym in TARGET_SYMBOLS:
        pos = target_positions.get(sym)
        if not pos:
            print(f"  Warning: No active position for {sym}, using reference fallback.")
            continue
            
        print(f"  Submitting TP/SL for {sym} (side={pos['side']}, entry={pos['entry_price']})...")
        res = await client.place_tp_sl_orders(
            symbol=sym,
            side=pos["side"],
            entry_price=pos["entry_price"],
            rolling_vol=0.005
        )
        results[sym] = res
        print(f"    -> SL Order ID: {res.get('sl_order_id')} @ {res.get('sl_price')}")
        print(f"    -> TP Order ID: {res.get('tp_order_id')} @ {res.get('pt_price')}")

    print("\n[3] Querying active open orders across target symbols...")
    total_orders = []
    for sym in TARGET_SYMBOLS:
        orders = await client.get_open_orders(sym)
        for o in orders:
            total_orders.append(o)

    print("\n" + "=" * 90)
    print("BINANCE FUTURES TESTNET ACTIVE PROTECTION ORDERS")
    print("=" * 90)
    header = f"{'Symbol':<10} | {'Order ID':<20} | {'Type':<20} | {'Side':<6} | {'Stop/Trigger Price':<18} | {'ClosePos':<8}"
    print(header)
    print("-" * len(header))
    for o in total_orders:
        sym = o.get("symbol", "-")
        oid = str(o.get("orderId", "-"))
        otype = o.get("type", "-")
        side = o.get("side", "-")
        stop_p = str(o.get("stopPrice", o.get("price", "-")))
        close_p = str(o.get("closePosition", True))
        print(f"{sym:<10} | {oid:<20} | {otype:<20} | {side:<6} | {stop_p:<18} | {close_p:<8}")
    print("=" * 90)
    print(f"Total Protection Orders Active: {len(total_orders)} (Expected: 8 for 4 positions)\n")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
