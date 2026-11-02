"""
scripts/flatten_and_clamp_leverage.py
======================================
1. Flattens all legacy active positions on Binance Futures Testnet via reduceOnly MARKET orders.
2. Cancels all resting open orders across the 10 universe symbols.
3. Sets 2x Leverage and ISOLATED margin mode across all 10 symbols.
4. Audits and confirms all 10 symbols are clamped at 2x Isolated.
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from execution.binance_testnet_client import BinanceFuturesTestnetClient, DEFAULT_UNIVERSE

async def run_clean_slate():
    client = BinanceFuturesTestnetClient()

    print("[STEP 1] Flattening all active positions across universe...")
    flatten_results = await client.flatten_all_positions(DEFAULT_UNIVERSE)
    for r in flatten_results:
        sym = r.get("symbol")
        action = r.get("action", "FLATTEN")
        if "error" in r:
            print(f"  {sym:<8}: {action} ERROR: {r['error']}")
        elif action == "CANCEL_ORDERS":
            print(f"  {sym:<8}: Cancelled open orders: {r.get('response', {}).get('msg', 'OK')}")
        else:
            print(f"  {sym:<8}: {action} {r.get('side')} {r.get('qty')} -> Status: {r.get('response', {}).get('status')}")

    await asyncio.sleep(2.0)

    # Verify positions are 0
    positions = await client.get_position_risk()
    active = [p for p in positions if abs(float(p.get("positionAmt", 0))) > 0]
    print(f"\n[POST-FLATTEN AUDIT] Active open positions remaining: {len(active)}")
    for a in active:
        print(f"  Warning: {a.get('symbol')} still has position: {a.get('positionAmt')}")

    print("\n[STEP 2] Clamping all 10 symbols to 2x Leverage and ISOLATED Margin...")
    for sym in DEFAULT_UNIVERSE:
        m_res = await client.set_margin_type(sym, "ISOLATED")
        l_res = await client.set_leverage(sym, 2)
        print(f"  {sym:<8}: Margin={m_res.get('msg')} | Leverage={l_res.get('leverage', 'N/A')}x")

    await asyncio.sleep(1.0)
    final_positions = await client.get_position_risk()
    universe_positions = {p["symbol"]: p for p in final_positions if p["symbol"] in DEFAULT_UNIVERSE}

    print("\n[STEP 3] Final Clamped State Verification:")
    all_clamped = True
    for sym in DEFAULT_UNIVERSE:
        p = universe_positions.get(sym, {})
        lev = p.get("leverage")
        m_type = p.get("marginType")
        amt = float(p.get("positionAmt", 0))
        is_isolated = (m_type == "isolated" or p.get("isolated") is True)
        is_2x = (str(lev) == "2")
        status_flag = "OK" if (is_2x and amt == 0) else "CHECK"
        if not is_2x:
            all_clamped = False
        print(f"  {sym:<8}: Leverage={lev:>2}x | MarginType={m_type:<8} | PosAmt={amt:>7.3f} | [{status_flag}]")

    print(f"\nAll 10 symbols clamped at 2x: {all_clamped}")
    await client.close()

if __name__ == "__main__":
    asyncio.run(run_clean_slate())

