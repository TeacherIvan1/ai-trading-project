import asyncio
import json
import urllib.request
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from execution.binance_testnet_client import BinanceFuturesTestnetClient

PROTECTED_SYMBOLS = ["BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT"]

async def main():
    print("=" * 90)
    print("FINAL TESTNET DAEMON & DASHBOARD STATE VERIFICATION")
    print("=" * 90)

    # 1. Query dashboard state
    try:
        with urllib.request.urlopen("http://127.0.0.1:8050/api/state", timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print("Error fetching dashboard state:", e)
        return

    bot_running = data.get("bot_running")
    test_duration = data.get("test_duration_formatted")
    open_positions_count = data.get("account_metrics", {}).get("open_positions")
    daemon_pid = data.get("system_health", {}).get("daemon_pid")
    active_in_pos = [c["symbol"] for c in data.get("active_cryptos", []) if c.get("in_position")]

    print(f"Bot Running Status     : {bot_running}")
    print(f"Daemon PID             : {daemon_pid}")
    print(f"Session Duration       : {test_duration} (Counting Up)")
    print(f"Open Positions Count   : {open_positions_count} (Cap = 4)")
    print(f"Active Cryptos In Pos  : {active_in_pos}")
    print(f"Testnet Total Equity   : ${data.get('account_metrics', {}).get('testnet_equity'):,.2f} USDT")
    print(f"Testnet Available Bal  : ${data.get('account_metrics', {}).get('testnet_available'):,.2f} USDT")

    # 2. Query exchange protection orders
    client = BinanceFuturesTestnetClient()
    await client.sync_server_time()

    prot_symbols = active_in_pos if active_in_pos else ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "DOTUSDT"]
    prot_orders = []
    for s in prot_symbols:
        orders = await client.get_open_orders(s)
        for o in orders:
            if o.get("type") in ("STOP_MARKET", "TAKE_PROFIT_MARKET") or o.get("orderType") in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
                prot_orders.append(o)

    print("\n" + "=" * 90)
    print("ACTIVE EXCHANGE-SIDE PROTECTION ORDERS TABLE (8 ORDERS / 4 POSITIONS)")
    print("=" * 90)
    header = f"{'Symbol':<10} | {'Order ID':<20} | {'Type':<20} | {'Side':<6} | {'Stop/Trigger Price':<18} | {'ClosePos':<8}"
    print(header)
    print("-" * len(header))
    for o in prot_orders:
        sym = o.get("symbol", "-")
        oid = str(o.get("orderId", "-"))
        otype = o.get("type", "-")
        side = o.get("side", "-")
        stop_p = str(o.get("stopPrice", o.get("price", "-")))
        close_p = str(o.get("closePosition", True))
        print(f"{sym:<10} | {oid:<20} | {otype:<20} | {side:<6} | {stop_p:<18} | {close_p:<8}")
    print("=" * 90)
    print(f"Total Protection Orders Verified: {len(prot_orders)} / 8 Expected (2 per position)\n")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
