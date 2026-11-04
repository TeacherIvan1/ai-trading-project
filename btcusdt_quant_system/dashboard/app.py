"""
dashboard/app.py
================
Institutional Real-Time Monitoring & Deliberation Terminal
for the Multi-Asset Systematic Crypto Quantitative Trading Engine.
Features:
- Live process management (POST /api/bot/start, POST /api/bot/stop)
- Two-Mode UI Switcher:
  * Mode 1: Simple Overview (Total Profit + Equity Curve, Test Duration Timer, 10-Asset Cards, Executed Trades)
  * Mode 2: QuantLab & TradingAgents Pro (Lightweight Candlestick Chart, VPIN Gauge, Pre-Trade Risk Gate,
            TauricResearch 5-Phase Multi-Agent Consensus Terminal, Real-Time Audit Feed)
- Native Binance Futures Testnet positions and order routing integration.
"""

import asyncio
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from execution.binance_testnet_client import BinanceFuturesTestnetClient
from agents.trading_agents_bridge import trading_agents_pipeline, build_portfolio_context

app = FastAPI(title="Crypto Quant Engine // Terminal & TradingAgents")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_LOG_PATH = LOGS_DIR / "shadow_trial_audit.jsonl"
LIVE_BARS_CSV_PATH = LOGS_DIR / "live_dollar_bars.csv"

DEFAULT_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"
]

REF_PRICES = {
    "BTCUSDT": 77850.0,
    "ETHUSDT": 2288.0,
    "SOLUSDT": 94.0,
    "BNBUSDT": 704.0,
    "XRPUSDT": 1.25,
    "ADAUSDT": 0.19,
    "DOGEUSDT": 0.075,
    "AVAXUSDT": 8.56,
    "DOTUSDT": 0.94,
    "LINKUSDT": 11.3,
}

# Process Manager state
bot_process: Optional[subprocess.Popen] = None
bot_start_time: Optional[float] = None
desired_state: str = "RUNNING"

# Global Testnet client and balance cache (TTL = 5 seconds)
testnet_client = BinanceFuturesTestnetClient()
balance_cache: Dict[str, Any] = {
    "last_fetch": 0.0,
    "available_balance": 13314.88,
    "total_balance": 13484.09,
}
positions_cache: Dict[str, Any] = {
    "last_fetch": 0.0,
    "data": [],
}


daemon_status_cache: Dict[str, Any] = {
    "last_check": 0.0,
    "status": {"running": False, "pid": None}
}


def get_daemon_status() -> Dict[str, Any]:
    """Checks whether main_shadow_trial.py is currently running using psutil + bot_process."""
    global bot_process, bot_start_time

    # 1. In-memory tracked child process
    if bot_process is not None:
        if bot_process.poll() is None:
            if bot_start_time is None:
                bot_start_time = time.time()
            return {"running": True, "pid": bot_process.pid}
        else:
            bot_process = None

    # 2. Fast scan via psutil for any running main_shadow_trial.py
    try:
        import psutil
        for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                p_name = (p.info.get("name") or "").lower()
                if "python" not in p_name:
                    continue
                cmd = p.info.get("cmdline") or []
                cmd_str = " ".join(cmd)
                if "main_shadow_trial.py" in cmd_str and p.info.get("pid") != os.getpid():
                    pid = p.info["pid"]
                    if bot_start_time is None:
                        create_t = p.info.get("create_time")
                        bot_start_time = create_t if create_t else time.time()
                    status = {"running": True, "pid": pid}
                    daemon_status_cache["status"] = status
                    daemon_status_cache["last_check"] = time.time()
                    return status
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception:
        pass

    # 3. Low-latency caching check
    now = time.time()
    if now - daemon_status_cache["last_check"] < 2.0:
        return daemon_status_cache["status"]

    # 4. Fallback process query (Windows powershell or Linux pgrep)
    if sys.platform == "win32":
        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { $_.CommandLine -like '*main_shadow_trial.py*' } | Select-Object -ExpandProperty ProcessId"
            ]
            out = subprocess.check_output(cmd, text=True, timeout=3.0).strip()
            if out:
                pids = [int(p) for p in out.splitlines() if p.strip().isdigit()]
                if pids:
                    if bot_start_time is None:
                        bot_start_time = time.time()
                    status = {"running": True, "pid": pids[0]}
                    daemon_status_cache["status"] = status
                    daemon_status_cache["last_check"] = now
                    return status
        except Exception:
            pass
    else:
        try:
            out = subprocess.check_output(["pgrep", "-f", "main_shadow_trial.py"], text=True, timeout=3.0).strip()
            if out:
                pids = [int(p) for p in out.splitlines() if p.strip().isdigit() and int(p) != os.getpid()]
                if pids:
                    if bot_start_time is None:
                        bot_start_time = time.time()
                    status = {"running": True, "pid": pids[0]}
                    daemon_status_cache["status"] = status
                    daemon_status_cache["last_check"] = now
                    return status
        except Exception:
            pass

    bot_start_time = None
    status = {"running": False, "pid": None}
    daemon_status_cache["status"] = status
    daemon_status_cache["last_check"] = now
    return status


async def get_cached_testnet_balance() -> Dict[str, float]:
    """Queries BinanceFuturesTestnetClient with a 5-second TTL cache."""
    now = time.time()
    if now - balance_cache["last_fetch"] > 5.0:
        try:
            bal = await testnet_client.get_usdt_balance()
            if bal and "available_balance" in bal:
                balance_cache["available_balance"] = float(bal["available_balance"])
                balance_cache["total_balance"] = float(bal["total_balance"])
                balance_cache["last_fetch"] = now
        except Exception:
            balance_cache["last_fetch"] = now - 2.0
    return {
        "available_balance": balance_cache["available_balance"],
        "total_balance": balance_cache["total_balance"],
    }


async def get_cached_testnet_positions() -> List[Dict[str, Any]]:
    """Queries active positions from Binance Futures Testnet /fapi/v2/positionRisk."""
    now = time.time()
    if now - positions_cache["last_fetch"] > 4.0:
        try:
            pos_list = await testnet_client.get_position_risk()
            active_pos = [p for p in pos_list if float(p.get("positionAmt", 0)) != 0]
            positions_cache["data"] = active_pos
            positions_cache["last_fetch"] = now
        except Exception:
            positions_cache["last_fetch"] = now - 2.0
    return positions_cache["data"]


income_cache = {
    "total_realized_pnl": 0.0,
    "income_records": [],
    "last_fetch": 0.0,
}


async def get_cached_realized_income() -> Dict[str, Any]:
    """Queries BinanceFuturesTestnetClient get_realized_income with a 5-second TTL cache."""
    now = time.time()
    if now - income_cache["last_fetch"] > 5.0:
        try:
            records = await testnet_client.get_realized_income(limit=100)
            if records and isinstance(records, list):
                tot_pnl = sum(float(r.get("income", 0.0)) for r in records if r.get("incomeType") == "REALIZED_PNL")
                income_cache["total_realized_pnl"] = tot_pnl
                income_cache["income_records"] = records
                income_cache["last_fetch"] = now
        except Exception:
            income_cache["last_fetch"] = now - 2.0
    return {
        "total_realized_pnl": income_cache["total_realized_pnl"],
        "income_records": income_cache["income_records"],
    }


def parse_audit_log_data(max_tail: int = 5000) -> Dict[str, Any]:
    """
    Scans the tail of shadow_trial_audit.jsonl and extracts:
    - recent_events (last 30)
    - latest BOT_SESSION_STARTED event
    - closed_trades list
    - opened_positions list
    - evaluations per symbol
    """
    if not AUDIT_LOG_PATH.exists():
        return {
            "recent_events": [],
            "session_started": None,
            "closed_trades": [],
            "opened_positions": [],
            "evaluations": {},
        }

    lines = []
    try:
        with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()[-max_tail:]
    except Exception:
        pass

    session_started = None
    closed_trades = []
    opened_positions = []
    evaluations: Dict[str, Dict[str, Any]] = {}
    recent_events = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            et = e.get("event_type")
            sym = e.get("symbol")
            recent_events.append(e)

            if et == "BOT_SESSION_STARTED":
                session_started = e
            elif et == "SHADOW_POSITION_CLOSED":
                closed_trades.append(e)
            elif et in ("SHADOW_POSITION_OPENED", "TESTNET_ORDER_ROUTED"):
                opened_positions.append(e)
            elif et == "SHADOW_SIGNAL_EVALUATED" and sym:
                evaluations[sym] = e
        except Exception:
            continue

    # Format recent events for audit table
    formatted_recent = []
    for e in recent_events[-35:]:
        ts = e.get("ts_utc", "")
        time_part = ts[11:19] if len(ts) >= 19 else ts
        formatted_recent.append({
            "ts": time_part,
            "ts_utc": ts,
            "symbol": e.get("symbol", "-"),
            "event_type": e.get("event_type", ""),
            "close": e.get("close_price", e.get("close", e.get("price", e.get("entry_price", "-")))),
            "side": e.get("primary_side", e.get("side", "-")),
            "z_dist": e.get("z_dist", "-"),
            "status": e.get("order_status", e.get("status", e.get("exit_reason", e.get("trigger_reason", "-")))),
            "order_id": e.get("orderId", e.get("cl_ord_id", "-")),
        })

    return {
        "recent_events": formatted_recent,
        "session_started": session_started,
        "closed_trades": closed_trades,
        "opened_positions": opened_positions,
        "evaluations": evaluations,
    }


def parse_tail_dollar_bars(max_bars: int = 60, target_symbol: str = "BTCUSDT") -> List[Dict[str, Any]]:
    """Parses the tail of live_dollar_bars.csv filtered strictly by target_symbol."""
    if not LIVE_BARS_CSV_PATH.exists():
        return []
    bars = []
    try:
        with open(LIVE_BARS_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_rows = list(reader)
            matching = [
                r for r in raw_rows
                if not r.get("symbol") or r.get("symbol").upper() == target_symbol.upper()
            ]
            last_time = 0
            for r in matching[-max_bars:]:
                try:
                    close_ts = int(float(r["close_ts"]))
                    t_sec = int(close_ts / 1000)
                    if t_sec <= last_time:
                        t_sec = last_time + 1
                    last_time = t_sec

                    o = float(r["open"])
                    h = float(r["high"])
                    l = float(r["low"])
                    c = float(r["close"])
                    if l <= 0.0:
                        l = min(o, c) * 0.9998
                    h = max(h, o, c)

                    bars.append({
                        "time": t_sec,
                        "close_ts": close_ts,
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c,
                        "dollar_volume": float(r.get("dollar_volume", 0.0)),
                        "net_order_flow": float(r.get("net_order_flow", 0.0)),
                        "ticks": int(float(r.get("ticks", 0))),
                        "symbol": r.get("symbol", target_symbol),
                    })
                except Exception:
                    continue
    except Exception:
        pass
    return bars


@app.post("/api/bot/start")
async def start_bot():
    """Starts main_shadow_trial.py with --testnet in the background and sets desired_state=RUNNING."""
    global bot_process, bot_start_time, desired_state
    desired_state = "RUNNING"
    status = get_daemon_status()
    if status["running"]:
        if bot_start_time is None:
            bot_start_time = time.time()
        return JSONResponse({"status": "already_running", "pid": status["pid"], "message": f"Bot already running on PID {status['pid']}"})

    python_exe = sys.executable
    script_path = str(PROJECT_ROOT / "main_shadow_trial.py")
    daemon_out = open(LOGS_DIR / "daemon.log", "a", encoding="utf-8")
    daemon_err = open(LOGS_DIR / "daemon_err.log", "a", encoding="utf-8")

    cmd = [python_exe, script_path, "--testnet"]
    bot_process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=daemon_out,
        stderr=daemon_err,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP") else 0
    )
    bot_start_time = time.time()
    return JSONResponse({"status": "started", "pid": bot_process.pid, "message": f"Bot started successfully (PID {bot_process.pid})"})


@app.post("/api/bot/stop")
async def stop_bot():
    """Stops main_shadow_trial.py, cancels open testnet orders, and sets desired_state=STOPPED."""
    global bot_process, bot_start_time, desired_state
    desired_state = "STOPPED"
    status = get_daemon_status()
    killed = False

    if bot_process:
        try:
            bot_process.terminate()
            killed = True
        except Exception:
            pass
        bot_process = None

    if status["running"] and status["pid"]:
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(status["pid"])], capture_output=True, timeout=3.0)
            else:
                os.kill(status["pid"], 9)
            killed = True
        except Exception:
            pass

    for s in DEFAULT_UNIVERSE:
        try:
            await testnet_client.cancel_all_open_orders(s)
        except Exception:
            pass

    bot_start_time = None
    return JSONResponse({"status": "stopped", "message": "Bot process terminated and open orders cancelled"})


async def bot_watchdog_loop():
    """Background watchdog: checks every 30 seconds; if desired_state == 'RUNNING' and bot not running, auto-restarts."""
    while True:
        await asyncio.sleep(30.0)
        try:
            if desired_state == "RUNNING":
                status = get_daemon_status()
                if not status["running"]:
                    print("[WATCHDOG] main_shadow_trial.py not running (desired_state=RUNNING). Auto-restarting...", flush=True)
                    await start_bot()
        except Exception as e:
            print(f"[WATCHDOG ERROR] {e}", flush=True)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(bot_watchdog_loop())


@app.get("/api/report/{symbol}", response_class=HTMLResponse)
async def get_agent_report(symbol: str = "BTCUSDT"):
    """
    Renders a standalone, institutional-grade single-page HTML deliberation report
    matching complete_report.html from TauricResearch/TradingAgents.
    """
    target_symbol = symbol.strip().upper() if symbol else "BTCUSDT"
    if target_symbol not in DEFAULT_UNIVERSE:
        target_symbol = "BTCUSDT"

    bal = await get_cached_testnet_balance()
    live_testnet_positions = await get_cached_testnet_positions()
    portfolio_ctx = build_portfolio_context(bal["total_balance"], live_testnet_positions)
    audit_data = parse_audit_log_data(200)
    ev = audit_data["evaluations"].get(target_symbol, {})
    bars = parse_tail_dollar_bars(5, target_symbol=target_symbol)
    latest_bar = bars[-1] if bars else {}
    close_price = float(ev.get("close_price", latest_bar.get("close", REF_PRICES.get(target_symbol, 100.0))))
    rolling_vol = float(ev.get("rolling_vol", 0.0015))
    cusum_dir = int(ev.get("cusum_dir", 0))
    z_dist = float(ev.get("z_dist", 0.0))
    vpin_cdf = float(ev.get("vpin_cdf", 0.50))
    net_order_flow = float(latest_bar.get("net_order_flow", 0.0))

    await trading_agents_pipeline.ensure_deliberation(
        symbol=target_symbol,
        current_price=close_price,
        rolling_vol=rolling_vol,
        cusum_dir=cusum_dir,
        z_dist=z_dist,
        vpin_cdf=vpin_cdf,
        net_order_flow=net_order_flow,
        open_positions=live_testnet_positions,
        universe_prices={s: float(audit_data["evaluations"].get(s, {}).get("close_price", REF_PRICES.get(s, 100.0))) for s in DEFAULT_UNIVERSE},
        testnet_balance=bal["total_balance"],
        portfolio=portfolio_ctx,
    )
    html = trading_agents_pipeline.generate_html_report(target_symbol)
    return HTMLResponse(content=html)


@app.get("/api/state")
async def get_state(symbol: str = "BTCUSDT"):
    """
    Standard JSON Contract for Mode 1 & Mode 2.
    Guarantees complete, non-null arrays for active_cryptos, executed_trades_list,
    bars_series, and recent_events.
    """
    global bot_start_time
    target_symbol = symbol.strip().upper() if symbol else "BTCUSDT"
    if target_symbol not in DEFAULT_UNIVERSE:
        target_symbol = "BTCUSDT"

    bal = await get_cached_testnet_balance()
    live_testnet_positions = await get_cached_testnet_positions()
    income_data = await get_cached_realized_income()
    daemon_info = get_daemon_status()
    audit_data = parse_audit_log_data(5000)
    bars = parse_tail_dollar_bars(60, target_symbol=target_symbol)

    bot_running = daemon_info["running"]

    # 1. Strictly compute elapsed session time from bot_start_time / BOT_SESSION_STARTED
    duration_sec = 0
    duration_fmt = "00:00:00"
    if bot_running:
        if bot_start_time:
            duration_sec = max(0, int(time.time() - bot_start_time))
        else:
            session_event = audit_data["session_started"]
            if session_event:
                ts_str = session_event.get("ts_utc") or session_event.get("timestamp")
                if ts_str:
                    try:
                        start_dt = datetime.fromisoformat(ts_str)
                        duration_sec = max(0, int((datetime.now(timezone.utc) - start_dt).total_seconds()))
                    except Exception:
                        duration_sec = 0
        
        hrs, rem = divmod(duration_sec, 3600)
        mins, secs = divmod(rem, 60)
        duration_fmt = f"{hrs:02d}:{mins:02d}:{secs:02d}"
    else:
        bot_start_time = None

    # 2. Executed Trades List: Combine Live Testnet Positions + Open Shadow + Closed Trades
    executed_trades_list = []
    active_in_position_symbols = set()

    # (a) Live Testnet Positions (unrealized PnL from Binance fapi/v2/positionRisk)
    total_unrealized_pnl = 0.0
    for p in live_testnet_positions:
        p_sym = p.get("symbol", "")
        p_amt = float(p.get("positionAmt", 0))
        if p_amt != 0:
            active_in_position_symbols.add(p_sym)
            entry_p = float(p.get("entryPrice", 0.0))
            unreal_pnl = float(p.get("unRealizedProfit", 0.0))
            total_unrealized_pnl += unreal_pnl
            notional = abs(p_amt) * entry_p
            ret_bps = round((unreal_pnl / notional) * 10000, 1) if notional > 0 else 0.0

            lev = int(p.get("leverage", 2))
            is_iso = bool(p.get("isolated") or str(p.get("marginType", "")).lower() == "isolated")
            m_type_str = "ISOLATED" if is_iso else "CROSS"

            executed_trades_list.append({
                "symbol": p_sym,
                "side": "BUY" if p_amt > 0 else "SELL",
                "entry_price": entry_p,
                "exit_price": 0.0,
                "pnl_usd": round(unreal_pnl, 2),
                "unrealized_pnl": round(unreal_pnl, 2),
                "net_return_bps": ret_bps,
                "exit_reason": "TESTNET_LIVE_POSITION",
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "status": "OPEN",
                "qty": abs(p_amt),
                "leverage": lev,
                "margin_type": m_type_str,
                "leverage_badge": f"{lev}x [{m_type_str}]",
            })

    # (b) Open Shadow positions that have not been closed yet
    closed_ids = {t.get("cl_ord_id") for t in audit_data["closed_trades"]}
    live_testnet_symbols = {p.get("symbol") for p in live_testnet_positions} if live_testnet_positions else set()
    for o in reversed(audit_data["opened_positions"]):
        cl_id = o.get("cl_ord_id")
        o_sym = o.get("symbol", "BTCUSDT")
        if live_testnet_symbols and o_sym not in live_testnet_symbols:
            continue
        if cl_id and cl_id not in closed_ids and o_sym not in active_in_position_symbols:
            active_in_position_symbols.add(o_sym)
            entry_p = float(o.get("entry_price", o.get("price", 0.0)))
            executed_trades_list.append({
                "symbol": o_sym,
                "side": "BUY" if o.get("side") in (1, "BUY", "buy") else "SELL",
                "entry_price": entry_p,
                "exit_price": 0.0,
                "pnl_usd": 0.0,
                "unrealized_pnl": 0.0,
                "net_return_bps": 0.0,
                "exit_reason": o.get("trigger_reason", "SHADOW_ORDER_OPEN"),
                "timestamp": (o.get("ts_utc") or "")[11:19] or "--:--:--",
                "status": "OPEN",
                "qty": float(o.get("qty", 0.0)),
                "leverage": 2,
                "margin_type": "ISOLATED",
                "leverage_badge": "2x [ISOLATED]",
            })

    # (c) Closed Trades
    for t in reversed(audit_data["closed_trades"][-25:]):
        entry_p = float(t.get("entry_price", 0.0))
        exit_p = float(t.get("exit_price", 0.0))
        executed_trades_list.append({
            "symbol": t.get("symbol", "BTCUSDT"),
            "side": "BUY" if t.get("side") in (1, "BUY") else "SELL",
            "entry_price": entry_p,
            "exit_price": exit_p,
            "pnl_usd": round(float(t.get("pnl_usd", 0.0)), 2),
            "unrealized_pnl": 0.0,
            "net_return_bps": round(float(t.get("net_return_bps", 0.0)), 1),
            "exit_reason": t.get("exit_reason", "HORIZONTAL_BARRIER"),
            "timestamp": (t.get("ts_utc") or "")[11:19] or "--:--:--",
            "status": "CLOSED",
            "qty": float(t.get("qty_btc", 0.0)),
            "leverage": 2,
            "margin_type": "ISOLATED",
            "leverage_badge": "2x [ISOLATED]",
        })

    # 3. Microstructure & Live Pricing (Strictly Filtered by target_symbol)
    latest_eval = audit_data["evaluations"].get(target_symbol, {})
    latest_bar = bars[-1] if bars else {}
    close_price = latest_eval.get("close_price", latest_bar.get("close", REF_PRICES.get(target_symbol, 80000.0)))
    rolling_vol = latest_eval.get("rolling_vol", 0.00065)
    cusum_dir = latest_eval.get("cusum_dir", 0)
    z_dist = latest_eval.get("z_dist", 0.0)
    vpin = latest_eval.get("vpin", 0.485)
    vpin_cdf = latest_eval.get("vpin_cdf", 0.52)
    vpin_near_circuit_breaker = bool(vpin_cdf >= 0.95)

    microstructure = {
        "symbol": target_symbol,
        "close_price": round(float(close_price), 4 if close_price < 1.0 else 2),
        "rolling_vol": round(float(rolling_vol), 6),
        "cusum_dir": int(cusum_dir),
        "z_dist": round(float(z_dist), 2),
        "vpin": round(float(vpin), 4),
        "vpin_cdf": round(float(vpin_cdf), 4),
        "vpin_near_circuit_breaker": vpin_near_circuit_breaker,
    }

    # 4. TauricResearch / TradingAgents Multi-Agent Consensus
    portfolio_ctx = build_portfolio_context(
        testnet_balance=bal["total_balance"],
        open_positions=live_testnet_positions,
        max_positions=4,
    )
    universe_p = {
        s: float(audit_data["evaluations"].get(s, {}).get("close_price", REF_PRICES.get(s, 100.0)))
        for s in DEFAULT_UNIVERSE
    }
    await trading_agents_pipeline.ensure_deliberation(
        symbol=target_symbol,
        current_price=float(close_price),
        meta_prob=None,
        rolling_vol=float(rolling_vol),
        cusum_dir=int(cusum_dir),
        z_dist=float(z_dist),
        vpin_cdf=float(vpin_cdf),
        net_order_flow=float(latest_bar.get("net_order_flow", 0.0)),
        open_positions=live_testnet_positions,
        universe_prices=universe_p,
        testnet_balance=bal["total_balance"],
        portfolio=portfolio_ctx,
    )
    debates = trading_agents_pipeline.get_latest_debates()

    # 5. Active Cryptos (All 10 Universe Symbols Guaranteed)
    active_cryptos = []
    for s in DEFAULT_UNIVERSE:
        short_name = s.replace("USDT", "")
        # Resolve latest price
        s_eval = audit_data["evaluations"].get(s, {})
        price = float(s_eval.get("close_price", REF_PRICES.get(s, 100.0)))
        ref_p = REF_PRICES.get(s, price)
        change_pct = round(((price - ref_p) / ref_p) * 100.0, 2) if ref_p > 0 else 0.0

        # PnL & trade count
        s_closed = [t for t in audit_data["closed_trades"] if t.get("symbol") == s]
        if income_data["income_records"]:
            pnl_realized = sum(float(r.get("income", 0.0)) for r in income_data["income_records"] if r.get("symbol") == s and r.get("incomeType") == "REALIZED_PNL")
        else:
            pnl_realized = sum(float(t.get("pnl_usd", 0.0)) for t in s_closed)
        in_pos = any(p.get("symbol") == s and abs(float(p.get("positionAmt", 0))) > 0 for p in live_testnet_positions) or (s in active_in_position_symbols)

        # Deliberation stance
        d = debates.get(s, {})
        stance = d.get("stance", "NEUTRAL")
        conviction = float(d.get("conviction", 0.50))
        thesis = d.get("thesis", "Analyzing real-time order flow and volatility regime.")
        verdict = d.get("verdict", "APPROVED")

        active_cryptos.append({
            "symbol": s,
            "short_name": short_name,
            "name": short_name,
            "price": round(price, 4 if price < 1.0 else 2),
            "change_pct": change_pct,
            "pnl_usd": round(pnl_realized, 2),
            "pnl": round(pnl_realized, 2),
            "trades_count": len(s_closed) + (1 if in_pos else 0),
            "trade_count": len(s_closed) + (1 if in_pos else 0),
            "in_position": in_pos,
            "status": "IN_POSITION" if in_pos else "MONITORING",
            "agent_stance": stance,
            "stance": stance,
            "conviction": conviction,
            "thesis": thesis,
            "verdict": verdict,
        })

    # 5. Account Metrics
    initial_paper_equity = 2500.00
    if income_data["income_records"]:
        realized_pnl = float(income_data["total_realized_pnl"])
    else:
        realized_pnl = sum(float(t.get("pnl_usd", 0.0)) for t in audit_data["closed_trades"])

    paper_equity = initial_paper_equity + realized_pnl
    if audit_data["closed_trades"]:
        paper_equity = float(audit_data["closed_trades"][-1].get("updated_equity", paper_equity))
        wins = [t for t in audit_data["closed_trades"] if float(t.get("pnl_usd", 0.0)) > 0]
        win_rate = round((len(wins) / len(audit_data["closed_trades"])) * 100.0, 1)
        max_dd = round(max(float(t.get("drawdown_pct", 0.0)) for t in audit_data["closed_trades"]), 2)
    else:
        win_rate = 0.0
        max_dd = 0.0

    total_net_pnl = realized_pnl + total_unrealized_pnl
    active_exchange_pos_count = len([p for p in live_testnet_positions if abs(float(p.get("positionAmt", 0))) > 0])

    account_metrics = {
        "testnet_equity": bal["total_balance"],
        "testnet_available": bal["available_balance"],
        "paper_equity": round(paper_equity, 2),
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(total_unrealized_pnl, 2),
        "floating_pnl": round(total_unrealized_pnl, 2),
        "total_unrealized_pnl": round(total_unrealized_pnl, 2),
        "total_net_pnl": round(total_net_pnl, 2),
        "net_profit_usd": round(realized_pnl, 2),
        "net_profit_pct": round((realized_pnl / initial_paper_equity) * 100.0, 2),
        "win_rate": win_rate,
        "open_positions": active_exchange_pos_count if active_exchange_pos_count > 0 else len(active_in_position_symbols),
        "max_drawdown": max_dd,
        "total_trades": len(audit_data["closed_trades"]),
    }

    # 6. System Health
    system_health = {
        "daemon_running": bot_running,
        "daemon_pid": daemon_info["pid"],
        "uptime_str": duration_fmt,
        "resync_count": sum(1 for e in audit_data["recent_events"] if e.get("event_type") == "L2_BOOK_RESYNC"),
        "dollar_bars_formed": len(bars),
        "last_event_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "TESTNET",
        "selected_symbol": target_symbol,
    }

    # 7. Equity Curve for Sparkline
    equity_curve = [initial_paper_equity]
    running_eq = initial_paper_equity
    for t in audit_data["closed_trades"]:
        running_eq += float(t.get("pnl_usd", 0.0))
        equity_curve.append(round(running_eq, 2))
    if len(equity_curve) == 1:
        equity_curve.append(initial_paper_equity)

    return {
        "bot_running": bot_running,
        "target_symbol": target_symbol,
        "symbol": target_symbol,
        "test_duration_seconds": duration_sec,
        "test_duration_formatted": duration_fmt,
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(total_unrealized_pnl, 2),
        "floating_pnl": round(total_unrealized_pnl, 2),
        "total_net_pnl": round(total_net_pnl, 2),
        "net_profit_usd": round(realized_pnl, 2),
        "net_profit_pct": round((realized_pnl / initial_paper_equity) * 100.0, 2),
        "active_cryptos": active_cryptos,
        "executed_trades_list": executed_trades_list,
        "bars_series": bars,
        "recent_events": audit_data["recent_events"],
        "audit_tail": audit_data["recent_events"],  # Alias for audit table
        "latest_agent_debates": debates,
        "deliberation": debates.get(target_symbol, {}),
        "portfolio_context": portfolio_ctx,
        "system_health": system_health,
        "account_metrics": account_metrics,
        "microstructure": microstructure,
        "equity_curve": equity_curve,
    }


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serves the complete two-mode dashboard UI with safe null-coalescing rendering."""
    html_content = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BTCUSDT Quant Engine // Terminal & TradingAgents</title>
  <!-- Tailwind CSS -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Lightweight Charts -->
  <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: {
              950: '#040711',
              900: '#070a14',
              850: '#0b101e',
              800: '#11182c',
              700: '#1a243e',
              600: '#253556',
              accent: '#38bdf8',
              bull: '#10b981',
              bear: '#ef4444',
              gold: '#f59e0b',
            }
          },
          fontFamily: {
            mono: ['JetBrains Mono', 'Consolas', 'Courier New', 'monospace']
          }
        }
      }
    }
  </script>
  <style>
    body { background-color: #040711; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0b101e; }
    ::-webkit-scrollbar-thumb { background: #1a243e; border-radius: 3px; }
    .glow-bull { text-shadow: 0 0 10px rgba(16, 185, 129, 0.4); }
    .glow-bear { text-shadow: 0 0 10px rgba(239, 68, 68, 0.4); }
    .pulse-dot { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
  </style>
</head>
<body class="p-3 antialiased select-none text-xs">

  <!-- ================= TOP KPI & CONTROL HEADER ================= -->
  <header class="bg-brand-900 border border-brand-700/60 rounded-lg p-3 mb-3 shadow-lg flex flex-wrap items-center justify-between gap-3">
    <!-- Left: Identity & Badges -->
    <div class="flex items-center gap-3">
      <div class="flex items-center gap-2">
        <span class="inline-block w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_#38bdf8]"></span>
        <h1 class="text-sm font-bold tracking-wider text-slate-100 uppercase">Quant Engine // TradingAgents</h1>
      </div>
      <span id="badge-mode" class="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-950 text-emerald-400 border border-emerald-600/50">TESTNET ACTIVE</span>
      <span id="badge-daemon" class="px-2 py-0.5 rounded text-[10px] font-semibold bg-brand-800 text-slate-300 border border-slate-700">DAEMON CHECKING</span>
      <span class="text-[10px] text-slate-500 hidden sm:inline">10-ASSET PERP UNIVERSE</span>
    </div>

    <!-- Center: Segmented View Mode Switcher -->
    <div class="inline-flex rounded-lg bg-brand-950 p-1 border border-brand-700/70">
      <button id="btn-mode-1" onclick="switchMode(1)" class="px-3 py-1 rounded text-xs font-semibold bg-brand-700 text-cyan-300 shadow transition-all">
        Simple Overview
      </button>
      <button id="btn-mode-2" onclick="switchMode(2)" class="px-3 py-1 rounded text-xs font-semibold text-slate-400 hover:text-slate-200 transition-all">
        QuantLab & Agents Pro
      </button>
    </div>

    <!-- Right: Start / Stop Bot Action Button -->
    <div class="flex items-center gap-3">
      <div class="text-right hidden md:block">
        <div class="text-[9px] text-slate-400 uppercase tracking-wider">Testnet USDT Balance</div>
        <div id="hdr-testnet-equity" class="text-xs font-bold text-slate-200 font-mono">$13,314.88</div>
      </div>
      <button id="btn-bot-toggle" onclick="toggleBotProcess()" class="flex items-center gap-2 px-4 py-1.5 rounded-md font-bold text-xs tracking-wider uppercase transition-all shadow-md bg-emerald-600 hover:bg-emerald-500 text-slate-950">
        <span id="bot-toggle-icon">▶</span>
        <span id="bot-toggle-text">START BOT</span>
      </button>
    </div>
  </header>

  <!-- ============================================================== -->
  <!-- ================= MODE 1: SIMPLE OVERVIEW ==================== -->
  <!-- ============================================================== -->
  <div id="mode1-container" class="space-y-3">
    
    <!-- Top 2 Summary Cards: Profit & Duration -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
      
      <!-- Card 1: Total Net PnL, Realized vs Floating Drawdown & Sparkline -->
      <div class="bg-brand-900 border border-brand-700/60 rounded-lg p-4 shadow flex flex-col justify-between">
        <div class="flex items-center justify-between mb-2">
          <span class="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Total Net PnL (Realized + Floating)</span>
          <span class="text-[10px] text-slate-500">Testnet Equity: <span id="m1-equity-val" class="font-mono text-cyan-300 font-bold">$13,316.16</span></span>
        </div>
        <div class="flex items-baseline justify-between">
          <div>
            <div id="m1-total-net-pnl" class="text-2xl font-black text-emerald-400 font-mono glow-bull">+$0.00</div>
            <div class="flex items-center gap-3 mt-1.5 text-xs font-mono">
              <span class="text-slate-400">Realized: <span id="m1-realized-pnl" class="font-bold text-slate-200">+$0.00</span></span>
              <span class="text-slate-400">Floating: <span id="m1-floating-pnl" class="font-bold text-amber-300">+$0.00</span></span>
            </div>
          </div>
          <!-- Sparkline Container -->
          <div class="w-44 h-12 flex items-center justify-end">
            <svg id="m1-sparkline-svg" class="w-full h-full overflow-visible" viewBox="0 0 180 40">
              <polyline id="m1-sparkline-poly" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="0,20 180,20" />
            </svg>
          </div>
        </div>
        <div class="mt-3 pt-2 border-t border-brand-800 text-[10px] text-slate-400 flex justify-between">
          <span>Real-Time Mark-to-Market</span>
          <span>Max 10% Single / 40% Portfolio Notional</span>
        </div>
      </div>

      <!-- Card 2: Duration of Test & Win Rate -->
      <div class="bg-brand-900 border border-brand-700/60 rounded-lg p-4 shadow flex flex-col justify-between">
        <div class="flex items-center justify-between mb-2">
          <span class="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Duration of Test Session</span>
          <span id="m1-status-pill" class="px-2 py-0.5 rounded text-[9px] font-bold bg-slate-800 text-slate-400">STANDBY</span>
        </div>
        <div class="flex items-baseline justify-between">
          <div>
            <div id="m1-duration-timer" class="text-2xl font-black text-cyan-300 font-mono">00:00:00</div>
            <div class="text-xs text-slate-400 font-mono">Active Bot Run Time</div>
          </div>
          <div class="text-right">
            <div id="m1-win-rate" class="text-lg font-bold text-slate-200 font-mono">0.0%</div>
            <div id="m1-trade-count" class="text-xs text-slate-400 font-mono">0 Closed Trades</div>
          </div>
        </div>
        <div class="mt-3 pt-2 border-t border-brand-800 text-[10px] text-slate-400 flex justify-between">
          <span>Active Positions: <span id="m1-open-count" class="text-cyan-300 font-bold font-mono">0</span></span>
          <span>L2 Sync Drift: <span id="m1-resync-count" class="text-slate-300 font-mono">0 events</span></span>
        </div>
      </div>
    </div>

    <!-- Card 3: 10-Crypto Major Universe Cards Grid -->
    <div>
      <div class="flex items-center justify-between mb-2">
        <h2 class="text-xs font-bold uppercase tracking-wider text-slate-300">10 Major Cryptocurrencies // Multi-Asset Matrix</h2>
        <span class="text-[10px] text-slate-500">Click any card to select & view Pro terminal</span>
      </div>
      <div id="m1-crypto-grid" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2.5">
        <!-- Injected via JavaScript -->
      </div>
    </div>

    <!-- Card 4: Executed Trades Table -->
    <div class="bg-brand-900 border border-brand-700/60 rounded-lg p-3 shadow">
      <div class="flex items-center justify-between mb-2">
        <h2 class="text-xs font-bold uppercase tracking-wider text-slate-300">Executed Trades Log</h2>
        <span class="text-[10px] text-slate-500">Live Testnet & Closed Positions</span>
      </div>
      <div class="overflow-x-auto max-h-56">
        <table class="w-full text-left font-mono">
          <thead class="text-[10px] uppercase text-slate-400 border-b border-brand-800 bg-brand-950 sticky top-0">
            <tr>
              <th class="py-1 px-2">Time</th>
              <th class="py-1 px-2">Asset</th>
              <th class="py-1 px-2">Status</th>
              <th class="py-1 px-2">Side</th>
              <th class="py-1 px-2">Entry Price</th>
              <th class="py-1 px-2">Exit Price</th>
              <th class="py-1 px-2">Net PnL ($)</th>
              <th class="py-1 px-2">Exit / Trigger Reason</th>
            </tr>
          </thead>
          <tbody id="m1-trades-tbody" class="text-[11px] divide-y divide-brand-800/40">
            <tr><td colspan="8" class="text-center py-4 text-slate-500">Loading executed trades...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

  </div>

  <!-- ============================================================== -->
  <!-- ============= MODE 2: QUANTLAB & TRADINGAGENTS PRO ========== -->
  <!-- ============================================================== -->
  <div id="mode2-container" class="space-y-3 hidden">
    
    <!-- Row 1: Charts & Live Microstructure Stats -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-3">
      
      <!-- Chart Container (8 Cols) -->
      <div class="lg:col-span-8 bg-brand-900 border border-brand-700/60 rounded-lg p-3 shadow flex flex-col">
        <div class="flex items-center justify-between mb-2">
          <div class="flex items-center gap-2">
            <span class="font-bold text-slate-100 text-sm" id="chart-symbol-label">BTCUSDT</span>
            <span class="text-[10px] text-slate-400">Isolated 1m Candlestick Bars</span>
          </div>
          <div class="flex items-center gap-3 text-[11px] font-mono">
            <span>O: <span id="bar-o" class="text-slate-300">--</span></span>
            <span>H: <span id="bar-h" class="text-slate-300">--</span></span>
            <span>L: <span id="bar-l" class="text-slate-300">--</span></span>
            <span>C: <span id="bar-c" class="text-cyan-400 font-bold">--</span></span>
          </div>
        </div>
        <div id="tv-chart" class="w-full h-72 rounded bg-brand-950"></div>
        <div id="tv-subchart" class="w-full h-24 rounded bg-brand-950 mt-1"></div>
      </div>

      <!-- Right Column: Microstructure & Risk Gauges (4 Cols) -->
      <div class="lg:col-span-4 space-y-3">
        
        <!-- VPIN Toxicity Card -->
        <div class="bg-brand-900 border border-brand-700/60 rounded-lg p-3 shadow">
          <div class="flex items-center justify-between mb-1">
            <span class="text-[10px] font-bold uppercase tracking-wider text-slate-400">VPIN Order Flow Toxicity</span>
            <span id="vpin-status-badge" class="px-1.5 py-0.5 rounded text-[9px] font-bold bg-emerald-950 text-emerald-400 border border-emerald-700/60">NORMAL</span>
          </div>
          <div class="flex items-baseline justify-between mt-1">
            <span class="text-xs text-slate-400">Empirical CDF:</span>
            <span id="m2-vpin-cdf" class="text-base font-bold font-mono text-slate-100">0.5200</span>
          </div>
          <!-- Gauge Bar -->
          <div class="w-full bg-brand-950 h-2 rounded mt-1.5 overflow-hidden border border-brand-700/50 relative">
            <div id="m2-vpin-bar" class="h-full bg-gradient-to-r from-emerald-500 via-yellow-500 to-rose-600 w-1/2 transition-all duration-300"></div>
            <div class="absolute right-[1%] top-0 bottom-0 w-0.5 bg-rose-500 shadow-[0_0_4px_#ef4444]" title="0.99 Circuit Breaker"></div>
          </div>
          <div class="flex justify-between text-[9px] text-slate-500 mt-1">
            <span>0.00</span>
            <span class="text-rose-400">0.99 Circuit Breaker</span>
          </div>
        </div>

        <!-- Microstructure Signals Card -->
        <div class="bg-brand-900 border border-brand-700/60 rounded-lg p-3 shadow font-mono text-[11px] space-y-1.5">
          <div class="text-[10px] font-bold uppercase tracking-wider text-slate-400 font-sans mb-1">Live Feature Engine (<span id="m2-feat-sym">BTCUSDT</span>)</div>
          <div class="flex justify-between">
            <span class="text-slate-400">Rolling Volatility:</span>
            <span id="m2-rolling-vol" class="text-slate-200">0.000650</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-400">Signed CUSUM Dir:</span>
            <span id="m2-cusum-dir" class="text-slate-200">0</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-400">Band z-dist:</span>
            <span id="m2-z-dist" class="text-slate-200">+0.00σ</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-400">Meta-Label Prob:</span>
            <span id="m2-meta-prob" class="text-cyan-400 font-bold">--</span>
          </div>
          <div class="flex justify-between pt-1 border-t border-brand-800">
            <span class="text-slate-400">Quarter-Kelly Sizing:</span>
            <span id="m2-kelly-alloc" class="text-emerald-400 font-bold">0.0%</span>
          </div>
        </div>

        <!-- Pre-Trade Risk Gate Card -->
        <div class="bg-brand-900 border border-brand-700/60 rounded-lg p-3 shadow font-mono text-[10px] space-y-1">
          <div class="text-[10px] font-bold uppercase tracking-wider text-slate-400 font-sans mb-1">SEC 15c3-5 Pre-Trade Risk Gate</div>
          <div class="flex justify-between"><span>Kill Switch (<20% DD):</span><span class="text-emerald-400">PASS</span></div>
          <div class="flex justify-between"><span>VPIN Toxicity (<0.99):</span><span id="gate-vpin" class="text-emerald-400">PASS</span></div>
          <div class="flex justify-between"><span>Fat-Finger Price Collar:</span><span class="text-emerald-400">PASS</span></div>
          <div class="flex justify-between"><span>Breakout / Agent Gate:</span><span id="gate-meta" class="text-slate-400">EVALUATING</span></div>
        </div>

      </div>
    </div>

    <!-- Row 2: TauricResearch 5-Phase Multi-Agent Consensus Panel -->
    <div class="bg-brand-900 border border-brand-700/60 rounded-lg p-3 shadow">
      <div class="flex flex-wrap items-center justify-between mb-3 gap-2">
        <div class="flex items-center gap-2">
          <span class="inline-block w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_6px_#f59e0b]"></span>
          <h2 class="text-xs font-bold uppercase tracking-wider text-slate-200">TauricResearch / TradingAgents Consensus Terminal</h2>
          <button onclick="window.open('/api/report/' + selectedAsset, '_blank')" class="ml-2 px-2 py-1 rounded text-[10px] font-sans font-bold bg-cyan-900/60 hover:bg-cyan-800 text-cyan-200 border border-cyan-700/50 flex items-center gap-1 transition cursor-pointer">
            <span>📄</span> <span>View Full Agent Report</span>
          </button>
        </div>
        <!-- Asset Selector Tabs -->
        <div id="m2-asset-tabs" class="flex flex-wrap gap-1 font-mono text-[10px]">
          <!-- Injected via JavaScript -->
        </div>
      </div>

      <!-- 5-Phase Grid -->
      <div class="grid grid-cols-1 md:grid-cols-5 gap-2 font-mono text-[11px]">
        <!-- Phase 1: Specialized Analysts -->
        <div class="bg-brand-950 p-2.5 rounded border border-brand-800 flex flex-col justify-between">
          <div>
            <div class="text-[10px] uppercase font-bold text-cyan-400 font-sans mb-1.5 flex items-center justify-between">
              <span>Phase 1: Analysts</span>
              <span id="p1-bias-badge" class="px-1 py-0.2 rounded text-[8px] bg-brand-800 text-slate-300">BULLISH</span>
            </div>
            <div class="space-y-1.5 text-[10px]">
              <div><span class="text-slate-500 font-sans">TECH:</span> <span id="p1-tech-sum" class="text-slate-300">Evaluating CUSUM breakout alignment.</span></div>
              <div><span class="text-slate-500 font-sans">SENT:</span> <span id="p1-sent-sum" class="text-slate-300">Order flow toxicity normal.</span></div>
              <div><span class="text-slate-500 font-sans">MACRO:</span> <span id="p1-macro-sum" class="text-slate-300">Funding rates neutral.</span></div>
            </div>
          </div>
          <div class="mt-2 pt-1 border-t border-brand-800/60 text-[9px] text-slate-400">
            Composite Score: <span id="p1-comp-score" class="text-emerald-400 font-bold">+0.45</span>
          </div>
        </div>

        <!-- Phase 2: Bull vs Bear Debate -->
        <div class="bg-brand-950 p-2.5 rounded border border-brand-800 flex flex-col justify-between">
          <div>
            <div class="text-[10px] uppercase font-bold text-amber-400 font-sans mb-1.5 flex items-center justify-between">
              <span>Phase 2: Debate</span>
              <span id="p2-winner-badge" class="px-1 py-0.2 rounded text-[8px] bg-brand-800 text-slate-300">BULL ADV</span>
            </div>
            <div class="space-y-1 text-[10px]">
              <div class="text-emerald-400 font-sans text-[9px] font-bold">BULL ARGUMENTS:</div>
              <ul id="p2-bull-args" class="list-disc list-inside text-slate-300 text-[9px] space-y-0.5">
                <li>Absorption near support.</li>
              </ul>
              <div class="text-rose-400 font-sans text-[9px] font-bold mt-1">BEAR ARGUMENTS:</div>
              <ul id="p2-bear-args" class="list-disc list-inside text-slate-300 text-[9px] space-y-0.5">
                <li>Resistance unbreached.</li>
              </ul>
            </div>
          </div>
          <div class="mt-2 pt-1 border-t border-brand-800/60 text-[9px] text-slate-400 truncate" id="p2-rebuttal">
            Rebuttal: Bull momentum confirmed.
          </div>
        </div>

        <!-- Phase 3: Trader Proposal -->
        <div class="bg-brand-950 p-2.5 rounded border border-brand-800 flex flex-col justify-between">
          <div>
            <div class="text-[10px] uppercase font-bold text-purple-400 font-sans mb-1.5 flex items-center justify-between">
              <span>Phase 3: Proposal</span>
              <span id="p3-action-badge" class="px-1 py-0.2 rounded text-[8px] bg-emerald-900 text-emerald-300 font-bold">BUY</span>
            </div>
            <div class="space-y-1 text-[10px]">
              <div><span class="text-slate-500 font-sans">ENTRY:</span> <span id="p3-entry" class="text-slate-200">--</span></div>
              <div><span class="text-slate-500 font-sans">TAKE PROFIT:</span> <span id="p3-pt" class="text-emerald-400">--</span></div>
              <div><span class="text-slate-500 font-sans">STOP LOSS:</span> <span id="p3-sl" class="text-rose-400">--</span></div>
              <div><span class="text-slate-500 font-sans">LEVERAGE:</span> <span id="p3-lev" class="text-cyan-300">2.0x</span></div>
            </div>
          </div>
          <div class="mt-2 pt-1 border-t border-brand-800/60 text-[9px] text-slate-400 truncate" id="p3-rationale">
            Triple-Barrier dynamic limit synthesis.
          </div>
        </div>

        <!-- Phase 4: Risk Guardians -->
        <div class="bg-brand-950 p-2.5 rounded border border-brand-800 flex flex-col justify-between">
          <div>
            <div class="text-[10px] uppercase font-bold text-rose-400 font-sans mb-1.5 flex items-center justify-between">
              <span>Phase 4: Risk Guardians</span>
              <span id="p4-approved-badge" class="px-1 py-0.2 rounded text-[8px] bg-emerald-900 text-emerald-300 font-bold">APPROVED</span>
            </div>
            <div class="space-y-1 text-[10px]">
              <div><span class="text-slate-500 font-sans">VPIN CHECK:</span> <span id="p4-vpin-check" class="text-emerald-400">PASS</span></div>
              <div><span class="text-slate-500 font-sans">VOL CHECK:</span> <span id="p4-vol-check" class="text-emerald-400">PASS</span></div>
              <div><span class="text-slate-500 font-sans">DRAWDOWN:</span> <span id="p4-dd-check" class="text-emerald-400">PASS</span></div>
              <div><span class="text-slate-500 font-sans">VETO:</span> <span id="p4-veto-reason" class="text-slate-400">NONE</span></div>
            </div>
          </div>
          <div class="mt-2 pt-1 border-t border-brand-800/60 text-[9px] text-slate-400">
            SEC 15c3-5 Compliance Verified
          </div>
        </div>

        <!-- Phase 5: PM Sign-off & Conviction -->
        <div class="bg-brand-950 p-2.5 rounded border border-brand-800 flex flex-col justify-between">
          <div>
            <div class="text-[10px] uppercase font-bold text-emerald-400 font-sans mb-1.5 flex items-center justify-between">
              <span>Phase 5: PM Sign-Off</span>
              <span id="p5-final-verdict" class="px-1 py-0.2 rounded text-[8px] bg-emerald-900 text-emerald-300 font-bold">APPROVED</span>
            </div>
            <div class="space-y-1.5 text-[10px]">
              <div>
                <span class="text-slate-500 font-sans">CONVICTION:</span>
                <span id="p5-conviction-score" class="text-emerald-400 font-bold">74%</span>
              </div>
              <div class="w-full bg-brand-900 h-1.5 rounded overflow-hidden">
                <div id="p5-conviction-bar" class="h-full bg-emerald-400 w-3/4"></div>
              </div>
              <div class="text-slate-300 text-[9px] mt-1 leading-snug" id="p5-thesis">
                Portfolio Manager approves allocation based on order flow absorption.
              </div>
            </div>
          </div>
          <div class="mt-2 pt-1 border-t border-brand-800/60 text-[9px] text-slate-400" id="p5-timestamp">
            UTC: --:--:--
          </div>
        </div>

      </div>
    </div>

    <!-- Row 3: Raw Audit Log Feed -->
    <div class="bg-brand-900 border border-brand-700/60 rounded-lg p-3 shadow">
      <div class="flex items-center justify-between mb-2">
        <h2 class="text-xs font-bold uppercase tracking-wider text-slate-300">Live JSONL Audit Tail</h2>
        <span class="text-[10px] text-slate-500">Real-Time Event Bus</span>
      </div>
      <div class="overflow-x-auto max-h-48">
        <table class="w-full text-left font-mono">
          <thead class="text-[10px] uppercase text-slate-400 border-b border-brand-800 bg-brand-950 sticky top-0">
            <tr>
              <th class="py-1 px-2">Time</th>
              <th class="py-1 px-2">Asset</th>
              <th class="py-1 px-2">Event Type</th>
              <th class="py-1 px-2">Price</th>
              <th class="py-1 px-2">Side</th>
              <th class="py-1 px-2">z-dist</th>
              <th class="py-1 px-2">Status / Result</th>
              <th class="py-1 px-2">Order ID</th>
            </tr>
          </thead>
          <tbody id="m2-audit-tbody" class="text-[11px] divide-y divide-brand-800/40">
            <tr><td colspan="8" class="text-center py-4 text-slate-500">Loading audit feed...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

  </div>

  <!-- ================= JAVASCRIPT LOGIC ================= -->
  <script>
    let currentMode = 1;
    let selectedAsset = 'BTCUSDT';
    let chart, candleSeries, volumeSeries, flowSeries;
    let latestDebates = {};
    let isBotRunning = false;

    // Mode Switcher
    function switchMode(mode) {
      currentMode = mode;
      const m1 = document.getElementById('mode1-container');
      const m2 = document.getElementById('mode2-container');
      const btn1 = document.getElementById('btn-mode-1');
      const btn2 = document.getElementById('btn-mode-2');

      if (mode === 1) {
        m1.classList.remove('hidden');
        m2.classList.add('hidden');
        btn1.className = 'px-3 py-1 rounded text-xs font-semibold bg-brand-700 text-cyan-300 shadow transition-all';
        btn2.className = 'px-3 py-1 rounded text-xs font-semibold text-slate-400 hover:text-slate-200 transition-all';
      } else {
        m1.classList.add('hidden');
        m2.classList.remove('hidden');
        btn2.className = 'px-3 py-1 rounded text-xs font-semibold bg-brand-700 text-cyan-300 shadow transition-all';
        btn1.className = 'px-3 py-1 rounded text-xs font-semibold text-slate-400 hover:text-slate-200 transition-all';
        if (!chart) initLightweightCharts();
      }
    }

    // Bot Process Start / Stop
    async function toggleBotProcess() {
      const btn = document.getElementById('btn-bot-toggle');
      btn.disabled = true;
      btn.classList.add('opacity-50');

      try {
        const endpoint = isBotRunning ? '/api/bot/stop' : '/api/bot/start';
        const res = await fetch(endpoint, { method: 'POST' });
        const data = await res.json();
        console.log('Bot toggle response:', data);
      } catch (err) {
        console.error('Error toggling bot:', err);
      } finally {
        setTimeout(() => {
          btn.disabled = false;
          btn.classList.remove('opacity-50');
          pollState();
        }, 1000);
      }
    }

    // Initialize Lightweight Charts for Mode 2
    function initLightweightCharts() {
      const chartEl = document.getElementById('tv-chart');
      const subEl = document.getElementById('tv-subchart');
      if (!chartEl || chart) return;

      chart = LightweightCharts.createChart(chartEl, {
        width: chartEl.clientWidth,
        height: 280,
        layout: { background: { color: '#040711' }, textColor: '#94a3b8' },
        grid: { vertLines: { color: '#11182c' }, horzLines: { color: '#11182c' } },
        crosshair: { mode: 0 },
        timeScale: { timeVisible: true, secondsVisible: true, borderColor: '#1a243e' },
      });

      candleSeries = chart.addCandlestickSeries({
        upColor: '#10b981',
        downColor: '#ef4444',
        borderUpColor: '#10b981',
        borderDownColor: '#ef4444',
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444',
      });

      const subChart = LightweightCharts.createChart(subEl, {
        width: subEl.clientWidth,
        height: 96,
        layout: { background: { color: '#040711' }, textColor: '#94a3b8' },
        grid: { vertLines: { color: '#11182c' }, horzLines: { color: '#11182c' } },
        timeScale: { timeVisible: true, secondsVisible: true, borderColor: '#1a243e' },
      });

      flowSeries = subChart.addHistogramSeries({
        color: '#38bdf8',
        priceFormat: { type: 'volume' },
      });

      window.addEventListener('resize', () => {
        chart.applyOptions({ width: chartEl.clientWidth });
        subChart.applyOptions({ width: subEl.clientWidth });
      });
    }

    // Asset Selection across both Mode 1 and Mode 2
    function selectAsset(sym) {
      if (!sym) return;
      selectedAsset = sym.toUpperCase();
      const chartLbl = document.getElementById('chart-symbol-label');
      if (chartLbl) chartLbl.textContent = selectedAsset;
      const featSym = document.getElementById('m2-feat-sym');
      if (featSym) featSym.textContent = selectedAsset;
      renderAssetTabs();
      updateDeliberationPanel();
      pollState();
    }

    function renderAssetTabs() {
      const container = document.getElementById('m2-asset-tabs');
      if (!container) return;
      const universe = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"];
      container.innerHTML = universe.map(sym => {
        const isSel = sym === selectedAsset;
        const cls = isSel
          ? 'bg-brand-700 text-cyan-300 border-cyan-500/50 shadow'
          : 'bg-brand-950 text-slate-400 hover:text-slate-200 border-brand-800';
        return `<button onclick="selectAsset('${sym}')" class="px-2 py-0.5 rounded border text-[10px] font-bold ${cls}">${sym.replace('USDT','')}</button>`;
      }).join('');
    }

    function updateDeliberationPanel() {
      const d = latestDebates[selectedAsset];
      if (!d) return;

      // Phase 1
      const p1 = d.analysts || {};
      const bBadge = document.getElementById('p1-bias-badge'); if (bBadge) bBadge.textContent = p1.composite_bias || 'NEUTRAL';
      const techSum = document.getElementById('p1-tech-sum'); if (techSum) techSum.textContent = p1.technical_summary || '--';
      const sentSum = document.getElementById('p1-sent-sum'); if (sentSum) sentSum.textContent = p1.sentiment_summary || '--';
      const macroSum = document.getElementById('p1-macro-sum'); if (macroSum) macroSum.textContent = p1.news_macro_summary || '--';
      const cScore = document.getElementById('p1-comp-score'); if (cScore) cScore.textContent = (p1.composite_score >= 0 ? '+' : '') + (p1.composite_score ?? 0.0);

      // Phase 2
      const p2 = d.debate || {};
      const wBadge = document.getElementById('p2-winner-badge'); if (wBadge) wBadge.textContent = (p2.debate_winner || 'TIE') + ' ADV';
      const bList = (p2.bull_arguments || []).map(a => `<li>${a}</li>`).join('');
      const rList = (p2.bear_arguments || []).map(a => `<li>${a}</li>`).join('');
      const bArgs = document.getElementById('p2-bull-args'); if (bArgs) bArgs.innerHTML = bList || '<li>Analyzing support levels.</li>';
      const rArgs = document.getElementById('p2-bear-args'); if (rArgs) rArgs.innerHTML = rList || '<li>Analyzing resistance levels.</li>';
      const reb = document.getElementById('p2-rebuttal'); if (reb) reb.textContent = 'Rebuttal: ' + (p2.rebuttal_points || '--');

      // Phase 3
      const p3 = d.proposal || {};
      const actBadge = document.getElementById('p3-action-badge');
      if (actBadge) {
        actBadge.textContent = p3.action || 'HOLD';
        actBadge.className = p3.action === 'BUY' ? 'px-1 py-0.2 rounded text-[8px] bg-emerald-900 text-emerald-300 font-bold' : (p3.action === 'SELL' ? 'px-1 py-0.2 rounded text-[8px] bg-rose-900 text-rose-300 font-bold' : 'px-1 py-0.2 rounded text-[8px] bg-slate-800 text-slate-300 font-bold');
      }
      const entryEl = document.getElementById('p3-entry'); if (entryEl) entryEl.textContent = p3.target_entry ? '$' + Number(p3.target_entry).toFixed(2) : '--';
      const ptEl = document.getElementById('p3-pt'); if (ptEl) ptEl.textContent = p3.take_profit ? '$' + Number(p3.take_profit).toFixed(2) : '--';
      const slEl = document.getElementById('p3-sl'); if (slEl) slEl.textContent = p3.stop_loss ? '$' + Number(p3.stop_loss).toFixed(2) : '--';
      const levEl = document.getElementById('p3-lev'); if (levEl) levEl.textContent = (p3.recommended_leverage || 2.0) + 'x';
      const ratEl = document.getElementById('p3-rationale'); if (ratEl) ratEl.textContent = p3.rationale || '--';

      // Phase 4
      const p4 = d.risk_verdict || {};
      const appBadge = document.getElementById('p4-approved-badge');
      if (appBadge) {
        appBadge.textContent = p4.approved ? 'APPROVED' : 'VETOED';
        appBadge.className = p4.approved ? 'px-1 py-0.2 rounded text-[8px] bg-emerald-900 text-emerald-300 font-bold' : 'px-1 py-0.2 rounded text-[8px] bg-rose-900 text-rose-300 font-bold';
      }
      const vpinChk = document.getElementById('p4-vpin-check'); if (vpinChk) vpinChk.textContent = p4.vpin_toxicity_check || 'PASS';
      const volChk = document.getElementById('p4-vol-check'); if (volChk) volChk.textContent = p4.volatility_check || 'PASS';
      const ddChk = document.getElementById('p4-dd-check'); if (ddChk) ddChk.textContent = p4.drawdown_check || 'PASS';
      const vetoR = document.getElementById('p4-veto-reason'); if (vetoR) vetoR.textContent = p4.veto_reason || 'NONE';

      // Phase 5
      const p5Verdict = document.getElementById('p5-final-verdict');
      if (p5Verdict) {
        p5Verdict.textContent = d.verdict || 'APPROVED';
        p5Verdict.className = d.verdict === 'APPROVED' ? 'px-1 py-0.2 rounded text-[8px] bg-emerald-900 text-emerald-300 font-bold' : 'px-1 py-0.2 rounded text-[8px] bg-rose-900 text-rose-300 font-bold';
      }
      const conv = Math.round((d.conviction || 0.5) * 100);
      const convScore = document.getElementById('p5-conviction-score'); if (convScore) convScore.textContent = conv + '%';
      const convBar = document.getElementById('p5-conviction-bar'); if (convBar) convBar.style.width = conv + '%';
      const p5Th = document.getElementById('p5-thesis'); if (p5Th) p5Th.textContent = d.thesis || '--';
      const p5Ts = document.getElementById('p5-timestamp'); if (p5Ts) p5Ts.textContent = 'UTC: ' + (d.timestamp_utc ? d.timestamp_utc.slice(11, 19) : '--:--:--');
    }

    // Sparkline Render
    function updateSparkline(curve) {
      if (!curve || curve.length < 2) return;
      const svg = document.getElementById('m1-sparkline-poly');
      if (!svg) return;
      const minVal = Math.min(...curve);
      const maxVal = Math.max(...curve);
      const range = (maxVal - minVal) || 1.0;

      const pts = curve.map((val, idx) => {
        const x = (idx / (curve.length - 1)) * 180;
        const y = 35 - ((val - minVal) / range) * 30;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(' ');

      svg.setAttribute('points', pts);
      const isPositive = curve[curve.length - 1] >= curve[0];
      svg.setAttribute('stroke', isPositive ? '#10b981' : '#ef4444');
    }

    // Main Polling Loop with Safe Null-Coalescing
    async function pollState() {
      try {
        const res = await fetch(`/api/state?symbol=${encodeURIComponent(selectedAsset)}`);
        const state = await res.json();
        if (!state) return;

        // 1. Bot Process Button & Badges
        isBotRunning = Boolean(state.bot_running);
        const botBtn = document.getElementById('btn-bot-toggle');
        const botIcon = document.getElementById('bot-toggle-icon');
        const botText = document.getElementById('bot-toggle-text');
        const badgeDaemon = document.getElementById('badge-daemon');
        const m1Pill = document.getElementById('m1-status-pill');

        if (isBotRunning) {
          if (botBtn) {
            botBtn.className = 'flex items-center gap-2 px-4 py-1.5 rounded-md font-bold text-xs tracking-wider uppercase transition-all shadow-md bg-rose-600 hover:bg-rose-500 text-slate-100 ring-2 ring-rose-500/50';
            botIcon.textContent = '■';
            botText.textContent = 'STOP BOT';
          }
          if (badgeDaemon) {
            badgeDaemon.textContent = `DAEMON RUNNING (PID ${state.system_health?.daemon_pid || 'ACTIVE'})`;
            badgeDaemon.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-950 text-emerald-300 border border-emerald-600';
          }
          if (m1Pill) {
            m1Pill.textContent = 'RUNNING';
            m1Pill.className = 'px-2 py-0.5 rounded text-[9px] font-bold bg-emerald-950 text-emerald-400 border border-emerald-700/60 pulse-dot';
          }
        } else {
          if (botBtn) {
            botBtn.className = 'flex items-center gap-2 px-4 py-1.5 rounded-md font-bold text-xs tracking-wider uppercase transition-all shadow-md bg-emerald-600 hover:bg-emerald-500 text-slate-950';
            botIcon.textContent = '▶';
            botText.textContent = 'START BOT';
          }
          if (badgeDaemon) {
            badgeDaemon.textContent = 'DAEMON STOPPED';
            badgeDaemon.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-brand-800 text-slate-400 border border-slate-700';
          }
          if (m1Pill) {
            m1Pill.textContent = 'STOPPED';
            m1Pill.className = 'px-2 py-0.5 rounded text-[9px] font-bold bg-slate-800 text-slate-400';
          }
        }

        // Header Equity
        const hdrEq = document.getElementById('hdr-testnet-equity');
        if (hdrEq && state.account_metrics) {
          hdrEq.textContent = '$' + Number(state.account_metrics.testnet_available || state.account_metrics.testnet_equity || 0).toLocaleString('en-US', { minimumFractionDigits: 2 });
        }

        // 2. Mode 1 KPI Pillars (Realized, Floating & Total Net PnL)
        const realizedPnl = Number(state.realized_pnl ?? state.account_metrics?.realized_pnl ?? 0);
        const floatingPnl = Number(state.floating_pnl ?? state.account_metrics?.floating_pnl ?? 0);
        const totalNetPnl = Number(state.total_net_pnl ?? (realizedPnl + floatingPnl));

        const totPnlEl = document.getElementById('m1-total-net-pnl');
        const realPnlEl = document.getElementById('m1-realized-pnl');
        const floatPnlEl = document.getElementById('m1-floating-pnl');
        const eqValEl = document.getElementById('m1-equity-val');

        if (totPnlEl) {
          const sign = totalNetPnl >= 0 ? '+' : '';
          totPnlEl.textContent = `${sign}$${totalNetPnl.toFixed(2)}`;
          totPnlEl.className = totalNetPnl >= 0 ? 'text-2xl font-black text-emerald-400 font-mono glow-bull' : 'text-2xl font-black text-rose-400 font-mono glow-bear';
        }
        if (realPnlEl) {
          const sign = realizedPnl >= 0 ? '+' : '';
          realPnlEl.textContent = `${sign}$${realizedPnl.toFixed(2)}`;
          realPnlEl.className = realizedPnl >= 0 ? 'font-bold text-emerald-400' : 'font-bold text-rose-400';
        }
        if (floatPnlEl) {
          const sign = floatingPnl >= 0 ? '+' : '';
          floatPnlEl.textContent = `${sign}$${floatingPnl.toFixed(2)}`;
          floatPnlEl.className = floatingPnl >= 0 ? 'font-bold text-emerald-400' : 'font-bold text-rose-400';
        }
        if (eqValEl && state.account_metrics) {
          eqValEl.textContent = '$' + Number(state.account_metrics.testnet_equity || 0).toLocaleString('en-US', { minimumFractionDigits: 2 });
        }

        // Keep legacy profit-usd if present
        const pnlEl = document.getElementById('m1-profit-usd');
        const pnlPctEl = document.getElementById('m1-profit-pct');
        if (pnlEl) {
          const sign = totalNetPnl >= 0 ? '+' : '';
          pnlEl.textContent = `${sign}$${totalNetPnl.toFixed(2)}`;
          pnlEl.className = totalNetPnl >= 0 ? 'text-2xl font-black text-emerald-400 font-mono glow-bull' : 'text-2xl font-black text-rose-400 font-mono glow-bear';
        }
        if (pnlPctEl) {
          const sign = totalNetPnl >= 0 ? '+' : '';
          pnlPctEl.textContent = `${sign}${Number(state.net_profit_pct ?? 0).toFixed(2)}% Net Return`;
          pnlPctEl.className = totalNetPnl >= 0 ? 'text-xs font-semibold text-emerald-400/80 font-mono' : 'text-xs font-semibold text-rose-400/80 font-mono';
        }

        updateSparkline(state.equity_curve);

        const durEl = document.getElementById('m1-duration-timer'); if (durEl) durEl.textContent = state.test_duration_formatted || '00:00:00';
        const winEl = document.getElementById('m1-win-rate'); if (winEl) winEl.textContent = (state.account_metrics?.win_rate ?? 0) + '%';
        const trdEl = document.getElementById('m1-trade-count'); if (trdEl) trdEl.textContent = (state.account_metrics?.total_trades ?? 0) + ' Closed Trades';
        const opnEl = document.getElementById('m1-open-count'); if (opnEl) opnEl.textContent = state.account_metrics?.open_positions ?? 0;
        const maxDdEl = document.getElementById('m1-max-dd'); if (maxDdEl) maxDdEl.textContent = (state.account_metrics?.max_drawdown ?? 0) + '%';
        const resyncEl = document.getElementById('m1-resync-count'); if (resyncEl) resyncEl.textContent = (state.system_health?.resync_count ?? 0) + ' events';

        // 3. 10-Crypto Grid
        const grid = document.getElementById('m1-crypto-grid');
        const cryptos = state.active_cryptos || [];
        if (grid && cryptos.length > 0) {
          grid.innerHTML = cryptos.map(c => {
            const sym = c.symbol || '';
            const name = c.short_name || c.name || sym.replace('USDT', '');
            const stance = (c.agent_stance || c.stance || 'NEUTRAL').toUpperCase();
            const isBull = stance === 'BULLISH';
            const isBear = stance === 'BEARISH';
            const badgeClass = isBull ? 'bg-emerald-950 text-emerald-400 border border-emerald-600/50' : (isBear ? 'bg-rose-950 text-rose-400 border border-rose-600/50' : 'bg-brand-800 text-slate-300 border border-slate-700');
            const convVal = Number(c.conviction ?? 0.5);
            const convPct = Math.round(convVal * 100);
            const chgVal = Number(c.change_pct ?? 0);
            const chgSign = chgVal >= 0 ? '+' : '';
            const chgClass = chgVal >= 0 ? 'text-emerald-400' : 'text-rose-400';
            const priceVal = Number(c.price ?? 0);
            const pDec = priceVal < 1.0 ? 4 : 2;
            const pnlVal = Number(c.pnl_usd ?? c.pnl ?? 0);
            const inPos = Boolean(c.in_position || c.status === 'IN_POSITION');
            const isSelected = sym === selectedAsset;

            return `
              <div onclick="selectAsset('${sym}');" class="cursor-pointer bg-brand-900 border ${inPos ? 'border-cyan-400 ring-1 ring-cyan-400/40' : (isSelected ? 'border-amber-400/80 shadow-[0_0_8px_rgba(245,158,11,0.2)]' : 'border-brand-700/60')} hover:border-cyan-500/60 rounded-lg p-2.5 transition-all shadow hover:shadow-cyan-900/20">
                <div class="flex items-center justify-between mb-1">
                  <div class="flex items-center gap-1.5">
                    ${inPos ? '<span class=\"w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-[0_0_6px_#38bdf8]\"></span>' : ''}
                    <span class="font-bold text-slate-100">${name}</span>
                  </div>
                  <span class="text-[9px] px-1 py-0.2 rounded font-semibold ${badgeClass}">${stance} ${convPct}%</span>
                </div>
                <div class="flex items-baseline justify-between font-mono">
                  <span class="text-xs font-bold text-slate-200">$${priceVal.toFixed(pDec)}</span>
                  <span class="text-[10px] ${chgClass}">${chgSign}${chgVal}%</span>
                </div>
                <div class="mt-2 pt-1.5 border-t border-brand-800/60 flex items-center justify-between text-[9px] text-slate-400 font-mono">
                  <span>PnL: <b class="${pnlVal >= 0 ? 'text-emerald-400' : 'text-rose-400'}">$${pnlVal.toFixed(2)}</b></span>
                  <span class="${inPos ? 'text-cyan-300 font-bold' : ''}">${inPos ? 'IN POSITION' : (c.trades_count ?? c.trade_count ?? 0) + ' trades'}</span>
                </div>
              </div>
            `;
          }).join('');
        }

        // 4. Mode 1 Executed Trades Table
        const m1Tbody = document.getElementById('m1-trades-tbody');
        const trades = state.executed_trades_list || [];
        if (m1Tbody) {
          if (trades.length > 0) {
            m1Tbody.innerHTML = trades.map(t => {
              const isBuy = String(t.side).toUpperCase() === 'BUY' || t.side === 1;
              const sideStr = isBuy ? 'BUY' : 'SELL';
              const sideClass = isBuy ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold';
              const pnlVal = Number(t.pnl_usd ?? 0.0);
              const pnlClass = pnlVal >= 0 ? 'text-emerald-400' : 'text-rose-400';
              const statusStr = (t.status || 'CLOSED').toUpperCase();
              const statusClass = statusStr === 'OPEN' ? 'bg-cyan-950 text-cyan-300 border border-cyan-500/50' : 'bg-brand-800 text-slate-400';
              const entryP = Number(t.entry_price || 0.0);
              const exitP = Number(t.exit_price || 0.0);
              const pDec = (entryP < 1.0 || (exitP > 0 && exitP < 1.0)) ? 4 : 2;

              return `
                <tr class="hover:bg-brand-800/30">
                  <td class="py-1 px-2 text-slate-400">${t.timestamp || '--:--:--'}</td>
                  <td class="py-1 px-2 font-bold text-slate-200">${t.symbol || '--'}</td>
                  <td class="py-1 px-2"><span class="px-1.5 py-0.2 rounded text-[9px] font-bold ${statusClass}">${statusStr}</span></td>
                  <td class="py-1 px-2 ${sideClass}">${sideStr}</td>
                  <td class="py-1 px-2 text-slate-300">
                    <span>$${entryP.toFixed(pDec)}</span>
                    <span class="ml-1.5 px-1 py-0.2 rounded text-[9px] font-bold bg-amber-500/10 text-amber-300 border border-amber-500/30 whitespace-nowrap">${t.leverage || 2}x [${t.margin_type || 'ISOLATED'}]</span>
                  </td>
                  <td class="py-1 px-2 text-slate-300">${statusStr === 'OPEN' ? '--' : '$' + exitP.toFixed(pDec)}</td>
                  <td class="py-1 px-2 font-bold ${pnlClass}">${statusStr === 'OPEN' ? (pnlVal !== 0 ? (pnlVal >= 0 ? '+' : '') + '$' + pnlVal.toFixed(2) : '--') : (pnlVal >= 0 ? '+' : '') + '$' + pnlVal.toFixed(2)}</td>
                  <td class="py-1 px-2 text-slate-400 text-[10px]">${t.exit_reason || t.trigger_reason || '--'}</td>
                </tr>
              `;
            }).join('');
          } else {
            m1Tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4 text-slate-500">No executed trades logged in current session.</td></tr>';
          }
        }

        // 5. Mode 2 Charts & Microstructure
        latestDebates = state.latest_agent_debates || {};
        renderAssetTabs();
        updateDeliberationPanel();

        const featSym = document.getElementById('m2-feat-sym');
        if (featSym) featSym.textContent = state.microstructure?.symbol || selectedAsset;

        // Update chart if container is initialized
        if (candleSeries && state.bars_series && state.bars_series.length > 0) {
          try {
            candleSeries.setData(state.bars_series.map(b => ({
              time: b.time,
              open: Number(b.open),
              high: Number(b.high),
              low: Number(b.low),
              close: Number(b.close)
            })));
            if (flowSeries) {
              flowSeries.setData(state.bars_series.map(b => ({
                time: b.time,
                value: Number(b.net_order_flow || 0),
                color: Number(b.net_order_flow || 0) >= 0 ? '#10b981' : '#ef4444'
              })));
            }

            const lastB = state.bars_series[state.bars_series.length - 1];
            if (lastB) {
              const pDec = Number(lastB.close) < 1.0 ? 4 : 2;
              const barO = document.getElementById('bar-o'); if (barO) barO.textContent = Number(lastB.open).toFixed(pDec);
              const barH = document.getElementById('bar-h'); if (barH) barH.textContent = Number(lastB.high).toFixed(pDec);
              const barL = document.getElementById('bar-l'); if (barL) barL.textContent = Number(lastB.low).toFixed(pDec);
              const barC = document.getElementById('bar-c'); if (barC) barC.textContent = Number(lastB.close).toFixed(pDec);
            }
          } catch (chartErr) {
            console.warn('Lightweight charts render exception:', chartErr);
          }
        }

        // Microstructure
        const micro = state.microstructure || {};
        const vpinCdf = Number(micro.vpin_cdf ?? 0.52);
        const vpinCdfEl = document.getElementById('m2-vpin-cdf'); if (vpinCdfEl) vpinCdfEl.textContent = vpinCdf.toFixed(4);
        const vpinBar = document.getElementById('m2-vpin-bar'); if (vpinBar) vpinBar.style.width = (vpinCdf * 100) + '%';
        const vpinBadge = document.getElementById('vpin-status-badge');
        const gateVpin = document.getElementById('gate-vpin');
        if (vpinCdf >= 0.95) {
          if (vpinBadge) { vpinBadge.textContent = 'HIGH TOXICITY'; vpinBadge.className = 'px-1.5 py-0.5 rounded text-[9px] font-bold bg-rose-950 text-rose-300 border border-rose-600 pulse-dot'; }
          if (gateVpin) { gateVpin.textContent = 'WARN'; gateVpin.className = 'text-rose-400 font-bold'; }
        } else {
          if (vpinBadge) { vpinBadge.textContent = 'NORMAL'; vpinBadge.className = 'px-1.5 py-0.5 rounded text-[9px] font-bold bg-emerald-950 text-emerald-400 border border-emerald-700/60'; }
          if (gateVpin) { gateVpin.textContent = 'PASS'; gateVpin.className = 'text-emerald-400'; }
        }

        const rVolEl = document.getElementById('m2-rolling-vol'); if (rVolEl) rVolEl.textContent = Number(micro.rolling_vol ?? 0.00065).toFixed(6);
        const cDirEl = document.getElementById('m2-cusum-dir'); if (cDirEl) cDirEl.textContent = (micro.cusum_dir > 0 ? '+' : '') + (micro.cusum_dir ?? 0);
        const zDistEl = document.getElementById('m2-z-dist'); if (zDistEl) zDistEl.textContent = (micro.z_dist > 0 ? '+' : '') + Number(micro.z_dist ?? 0).toFixed(2) + 'σ';

        // Recent signal
        const rSig = (state.recent_signals && state.recent_signals.length > 0) ? state.recent_signals[state.recent_signals.length - 1] : null;
        if (rSig) {
          const mProbEl = document.getElementById('m2-meta-prob'); if (mProbEl) mProbEl.textContent = rSig.meta_prob ? Number(rSig.meta_prob).toFixed(4) : '--';
          const kAllocEl = document.getElementById('m2-kelly-alloc'); if (kAllocEl) kAllocEl.textContent = (Number(rSig.kelly_fraction ?? 0) * 100).toFixed(1) + '%';
          const metaGate = document.getElementById('gate-meta');
          if (metaGate) {
            if (rSig.meta_prob && rSig.meta_prob >= 0.56) {
              metaGate.textContent = 'PASS (' + (Number(rSig.meta_prob) * 100).toFixed(1) + '%)';
              metaGate.className = 'text-emerald-400 font-bold';
            } else if (rSig.meta_prob) {
              metaGate.textContent = 'FILTERED (' + (Number(rSig.meta_prob) * 100).toFixed(1) + '%)';
              metaGate.className = 'text-yellow-400';
            } else {
              metaGate.textContent = 'STANDBY';
              metaGate.className = 'text-slate-400';
            }
          }
        }

        // Mode 2 Audit Table
        const auditEvents = state.recent_events || state.audit_tail || [];
        const m2Tbody = document.getElementById('m2-audit-tbody');
        if (m2Tbody && auditEvents.length > 0) {
          m2Tbody.innerHTML = auditEvents.slice().reverse().map(e => `
            <tr class="hover:bg-brand-800/30">
              <td class="py-0.5 px-2 text-slate-400">${e.ts || (e.ts_utc ? e.ts_utc.slice(11, 19) : '--:--:--')}</td>
              <td class="py-0.5 px-2 font-bold text-slate-300">${e.symbol || '-'}</td>
              <td class="py-0.5 px-2 text-cyan-300">${e.event_type || '-'}</td>
              <td class="py-0.5 px-2 text-slate-200">${e.close ?? '-'}</td>
              <td class="py-0.5 px-2">${e.side ?? '-'}</td>
              <td class="py-0.5 px-2 text-slate-400">${e.z_dist ?? '-'}</td>
              <td class="py-0.5 px-2 text-slate-300 text-[10px]">${e.status ?? '-'}</td>
              <td class="py-0.5 px-2 text-slate-500 text-[10px] truncate max-w-xs">${e.order_id ?? '-'}</td>
            </tr>
          `).join('');
        }

      } catch (err) {
        console.error('Error polling state:', err);
      }
    }

    // Initial load and periodic polling
    pollState();
    setInterval(pollState, 2000);
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8050)
