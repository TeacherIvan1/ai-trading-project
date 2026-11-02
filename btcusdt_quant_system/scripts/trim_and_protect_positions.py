import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from execution.binance_testnet_client import BinanceFuturesTestnetClient, format_order_qty

PROTECTED_SYMBOLS = ["BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT"]

async def main():
    client = BinanceFuturesTestnetClient()
    await client.sync_server_time()

    print("[STEP 1] Auditing all open positions on Binance Futures Testnet...")
    positions = await client.get_position_risk()
    active_positions = {p["symbol"]: p for p in positions if abs(float(p.get("positionAmt", 0.0))) > 0}

    print(f"  Total open positions found: {len(active_positions)} ({list(active_positions.keys())})")

    # A. Close excess positions (e.g. ETHUSDT, BTCUSDT)
    for sym, pos in list(active_positions.items()):
        if sym not in PROTECTED_SYMBOLS:
            amt = float(pos.get("positionAmt", 0.0))
            side = "SELL" if amt > 0 else "BUY"
            raw_qty = abs(amt)
            qty_str = format_order_qty(sym, raw_qty)
            print(f"  -> Trimming excess position on {sym}: {side} {qty_str} (Amt was {amt})...")
            try:
                close_res = await client.place_order(
                    symbol=sym,
                    side=side,
                    qty=qty_str,
                    order_type="MARKET",
                    reduce_only=True
                )
                print(f"     Trimmed {sym}: status={close_res.get('status')} | orderId={close_res.get('orderId')}")
            except Exception as e:
                print(f"     Trim error on {sym}: {e}")

            # Cancel lingering limit / stop orders for this trimmed symbol
            try:
                await client.cancel_all_open_orders(sym)
            except Exception:
                pass

    # Cancel lingering un-triggered limit orders on protected symbols (e.g. non-closePosition orders)
    for sym in PROTECTED_SYMBOLS:
        try:
            orders = await client.get_open_orders(sym)
            for o in orders:
                # Cancel entry limit orders (not closePosition)
                if not o.get("closePosition") and o.get("type") == "LIMIT":
                    oid = o.get("orderId")
                    print(f"  Cancelling lingering entry quote {oid} on {sym}...")
                    await client.cancel_order(sym, oid)
        except Exception as e:
            print(f"  Order cleanup error on {sym}: {e}")

    # Re-audit positions to confirm only the 4 protected symbols remain
    print("\n[STEP 2] Verifying remaining 4 protected positions...")
    positions2 = await client.get_position_risk()
    active_now = {p["symbol"]: p for p in positions2 if abs(float(p.get("positionAmt", 0.0))) > 0}
    print(f"  Remaining open positions: {len(active_now)} ({list(active_now.keys())})")
    for sym, p in active_now.items():
        amt = float(p.get("positionAmt", 0.0))
        entry_p = float(p.get("entryPrice", 0.0))
        lev = p.get("leverage")
        m_type = "ISOLATED" if p.get("isolated") else "CROSS"
        print(f"  {sym:<10} | Amt: {amt:>10} | Entry: ${entry_p:<9.4f} | Leverage: {lev}x | Margin: {m_type}")

    # B. Ensure all 4 positions have active STOP_MARKET and TAKE_PROFIT_MARKET orders
    print("\n[STEP 3] Ensuring STOP_MARKET and TAKE_PROFIT_MARKET active for each of the 4 positions...")
    for sym in PROTECTED_SYMBOLS:
        p = active_now.get(sym)
        if not p:
            print(f"  Warning: {sym} not found in active positions!")
            continue

        amt = float(p.get("positionAmt", 0.0))
        entry_p = float(p.get("entryPrice", 0.0))
        pos_side = "BUY" if amt > 0 else "SELL"

        # Check existing protection orders
        existing_orders = await client.get_open_orders(sym)
        has_sl = any(o.get("type") == "STOP_MARKET" for o in existing_orders)
        has_tp = any(o.get("type") == "TAKE_PROFIT_MARKET" for o in existing_orders)

        if not (has_sl and has_tp):
            print(f"  Submitting TP/SL for {sym} (side={pos_side}, entry={entry_p})...")
            tp_sl_res = await client.place_tp_sl_orders(
                symbol=sym,
                side=pos_side,
                entry_price=entry_p,
                rolling_vol=0.005
            )
            print(f"    -> SL Order ID: {tp_sl_res.get('sl_order_id')} @ {tp_sl_res.get('sl_price')}")
            print(f"    -> TP Order ID: {tp_sl_res.get('tp_order_id')} @ {tp_sl_res.get('pt_price')}")
        else:
            print(f"  {sym} already has active TP/SL protection.")

    # C. Query and display all 8 protection orders
    print("\n" + "=" * 95)
    print("BINANCE FUTURES TESTNET ACTIVE PROTECTION ORDERS TABLE")
    print("=" * 95)
    header = f"{'Symbol':<10} | {'Order ID':<20} | {'Type':<20} | {'Side':<6} | {'Stop/Trigger Price':<18} | {'ClosePos':<8}"
    print(header)
    print("-" * len(header))

    total_prot_orders = []
    for sym in PROTECTED_SYMBOLS:
        orders = await client.get_open_orders(sym)
        for o in orders:
            if o.get("type") in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
                total_prot_orders.append(o)
                oid = str(o.get("orderId", "-"))
                otype = o.get("type", "-")
                side = o.get("side", "-")
                stop_p = str(o.get("stopPrice", o.get("price", "-")))
                close_p = str(o.get("closePosition", True))
                print(f"{sym:<10} | {oid:<20} | {otype:<20} | {side:<6} | {stop_p:<18} | {close_p:<8}")

    print("=" * 95)
    print(f"Total Protection Orders: {len(total_prot_orders)} / 8 Expected (2 per position)\n")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())

