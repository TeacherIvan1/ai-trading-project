"""
scripts/configure_leverage_and_margin.py
=========================================
Configures 2x Leverage and ISOLATED Margin across all 10 symbols in the universe.
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

async def configure_symbols():
    client = BinanceFuturesTestnetClient()
    print("[CONFIG] Enforcing 2x Leverage and ISOLATED margin across universe...")
    for sym in DEFAULT_UNIVERSE:
        m_res = await client.set_margin_type(sym, "ISOLATED")
        l_res = await client.set_leverage(sym, 2)
        print(f"  {sym:<8}: MarginType={m_res.get('msg')} | Leverage={l_res.get('leverage', 'N/A')}x")

    await asyncio.sleep(1.0)
    positions = await client.get_position_risk()
    active = [p for p in positions if float(p.get("positionAmt", 0)) != 0]
    print(f"\n[ACTIVE POSITIONS AUDIT] ({len(active)} active positions):")
    total_notional = 0.0
    for p in active:
        sym = p.get("symbol")
        amt = float(p.get("positionAmt", 0))
        entry = float(p.get("entryPrice", 0))
        notional = abs(amt) * entry
        total_notional += notional
        lev = p.get("leverage")
        m_type = p.get("marginType")
        pnl = float(p.get("unRealizedProfit", 0))
        print(f"  {sym:<8} Amt={amt:>10.3f} Entry={entry:>10.2f} Notional=${notional:>10.2f} Lev={lev:>2}x Margin={m_type} UnrealPnL=${pnl:>7.2f}")
    print(f"Total Portfolio Active Notional: ${total_notional:.2f} USDT")
    await client.close()

if __name__ == "__main__":
    asyncio.run(configure_symbols())

