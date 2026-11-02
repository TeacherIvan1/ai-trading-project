"""
scripts/verify_testnet.py
=========================
Verification script for Binance Futures Testnet connectivity & authentication:
1. Public REST Connectivity (/fapi/v1/ping and /fapi/v1/depth?symbol=BTCUSDT&limit=5)
2. Live WebSocket Tick Streaming (5-second window on wss://stream.binancefuture.com/ws/btcusdt@aggTrade)
3. Authenticated Account Balance Query (GET /fapi/v2/balance) via BinanceFuturesTestnetClient
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import websockets
from dotenv import load_dotenv
from execution.binance_testnet_client import BinanceFuturesTestnetClient

# Load environment
for env_candidate in [PROJECT_ROOT / ".env", Path(".env")]:
    if env_candidate.exists():
        load_dotenv(dotenv_path=env_candidate)
        break


async def verify_public_rest(client: BinanceFuturesTestnetClient):
    t0 = time.perf_counter()
    ping_ok = await client.ping()
    ping_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    depth_data = await client.get_depth("BTCUSDT", limit=5)
    depth_ms = (time.perf_counter() - t1) * 1000

    bids = depth_data.get("bids", [])
    asks = depth_data.get("asks", [])
    depth_ok = len(bids) > 0 and len(asks) > 0
    top_bid = float(bids[0][0]) if depth_ok else 0.0
    top_ask = float(asks[0][0]) if depth_ok else 0.0

    return {
        "ping_ok": ping_ok,
        "ping_ms": ping_ms,
        "depth_ok": depth_ok,
        "depth_ms": depth_ms,
        "top_bid": top_bid,
        "top_ask": top_ask,
    }


async def verify_websocket_ticks(duration_sec: float = 5.0):
    url = "wss://stream.binancefuture.com/ws/btcusdt@aggTrade"
    t_start = time.time()
    tick_count = 0
    first_price = None
    last_price = None

    try:
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            while time.time() - t_start < duration_sec:
                try:
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=1.5)
                    data = json.loads(raw_msg)
                    tick_count += 1
                    price = float(data.get("p", 0.0))
                    if first_price is None:
                        first_price = price
                    last_price = price
                except asyncio.TimeoutError:
                    continue
        return {
            "ws_ok": tick_count > 0,
            "tick_count": tick_count,
            "first_price": first_price,
            "last_price": last_price,
            "duration_sec": time.time() - t_start,
        }
    except Exception as e:
        return {
            "ws_ok": False,
            "tick_count": 0,
            "error": str(e),
            "duration_sec": time.time() - t_start,
        }


async def verify_auth_balance(client: BinanceFuturesTestnetClient):
    api_key = os.getenv("BINANCE_FUTURES_TESTNET_API_KEY", "").strip()
    api_secret = os.getenv("BINANCE_FUTURES_TESTNET_API_SECRET", "").strip()

    if not api_key or not api_secret:
        return {
            "auth_configured": False,
            "status_message": "PUBLIC_TESTNET_OK | WAITING_FOR_API_KEYS_IN_ENV",
            "available_usdt": None,
            "total_usdt": None,
        }

    try:
        bal = await client.get_usdt_balance()
        if "error" in bal:
            return {
                "auth_configured": True,
                "auth_ok": False,
                "error": bal["error"],
                "available_usdt": 0.0,
                "total_usdt": 0.0,
            }
        return {
            "auth_configured": True,
            "auth_ok": True,
            "available_usdt": bal.get("available_balance", 0.0),
            "total_usdt": bal.get("total_balance", 0.0),
        }
    except Exception as e:
        return {
            "auth_configured": True,
            "auth_ok": False,
            "error": str(e),
            "available_usdt": 0.0,
            "total_usdt": 0.0,
        }


async def main():
    async with BinanceFuturesTestnetClient() as client:
        # Run verifications
        rest_res = await verify_public_rest(client)
        ws_res = await verify_websocket_ticks(5.0)
        auth_res = await verify_auth_balance(client)

        header = "=" * 80
        print("\n" + header)
        print("BINANCE FUTURES TESTNET VERIFICATION AUDIT")
        print(header)
        print(f"{'Verification Check':<38} | {'Status':<15} | {'Details':<22}")
        print("-" * 80)

        # 1. REST Ping
        ping_status = "PASS" if rest_res["ping_ok"] else "FAIL"
        print(f"{'REST Ping (/fapi/v1/ping)':<38} | {ping_status:<15} | {rest_res['ping_ms']:.1f} ms RTT")

        # 2. REST Depth
        depth_status = "PASS" if rest_res["depth_ok"] else "FAIL"
        depth_details = f"Bid: {rest_res['top_bid']:.1f} Ask: {rest_res['top_ask']:.1f}"
        print(f"{'L2 Depth (/fapi/v1/depth limit=5)':<38} | {depth_status:<15} | {depth_details:<22}")

        # 3. WebSocket Feed
        ws_status = "PASS" if ws_res["ws_ok"] else "FAIL"
        ws_details = f"{ws_res['tick_count']} ticks in {ws_res['duration_sec']:.1f}s"
        print(f"{'Live WebSocket (@aggTrade 5s test)':<38} | {ws_status:<15} | {ws_details:<22}")

        # 4. Auth Account Balance
        if not auth_res["auth_configured"]:
            auth_status = "PENDING_KEYS"
            auth_details = auth_res["status_message"]
        elif auth_res.get("auth_ok"):
            auth_status = "PASS"
            auth_details = f"Avail: ${auth_res['available_usdt']:,.2f} USDT"
        else:
            auth_status = "FAIL"
            auth_details = f"Err: {str(auth_res.get('error'))[:20]}"
        print(f"{'Private Account Balance (/fapi/v2/balance)':<38} | {auth_status:<15} | {auth_details:<22}")

        print(header)

        if not auth_res["auth_configured"]:
            print(f"STATUS: {auth_res['status_message']}\n")
        elif auth_res.get("auth_ok") and rest_res["ping_ok"] and ws_res["ws_ok"]:
            print(f"STATUS: TESTNET_VERIFIED_AND_AUTHENTICATED | Total Equity: ${auth_res['total_usdt']:,.2f} USDT\n")
        else:
            print("STATUS: TESTNET_VERIFICATION_FAILED\n")


if __name__ == "__main__":
    asyncio.run(main())
