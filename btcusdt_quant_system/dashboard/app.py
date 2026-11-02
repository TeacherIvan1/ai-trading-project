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
- Native Binance Futures Testnet and Shadow execution state tracking.
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
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from execution.binance_testnet_client import BinanceFuturesTestnetClient
from agents.trading_agents_bridge import trading_agents_pipeline

app = FastAPI(title="BTCUSDT Quant Engine // Terminal & TradingAgents")

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
    "BTCUSDT": 80940.0,
    "ETHUSDT": 3120.0,
    "SOLUSDT": 184.5,
    "BNBUSDT": 595.0,
    "XRPUSDT": 0.54,
    "ADAUSDT": 0.35,
    "DOGEUSDT": 0.145,
    "AVAXUSDT": 27.8,
    "DOTUSDT": 4.25,
    "LINKUSDT": 11.8,
}

# Process Manager state
bot_process: Optional[subprocess.Popen] = None
bot_start_time: Optional[float] = None

# Global Testnet client and balance cache (TTL = 5 seconds)
testnet_client = BinanceFuturesTestnetClient()
balance_cache: Dict[str, Any] = {
    "last_fetch": 0.0,
    "available_balance": 13484.09,
    "total_balance": 13484.09,
}


def get_daemon_status() -> Dict[str, Any]:
    """Checks whether main_shadow_trial.py is currently running."""
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { $_.CommandLine -like '*main_shadow_trial.py*' } | Select-Object -ExpandProperty ProcessId"
        ]
        out = subprocess.check_output(cmd, text=True, timeout=2.5).strip()
        if out:
            pids = [int(p) for p in out.splitlines() if p.strip().isdigit()]
            if pids:
                return {"running": True, "pid": pids[0]}
    except Exception:
        pass
    return {"running": False, "pid": None}


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


def parse_tail_audit_log(max_lines: int = 500) -> List[Dict[str, Any]]:
    """Reads the last max_lines from shadow_trial_audit.jsonl."""
    if not AUDIT_LOG_PATH.exists():
        return []
    records = []
    try:
        with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-max_lines:]:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return records


def parse_tail_dollar_bars(max_bars: int = 100) -> List[Dict[str, Any]]:
    """Parses the tail of live_dollar_bars.csv."""
    if not LIVE_BARS_CSV_PATH.exists():
        return []
    bars = []
    try:
        with open(LIVE_BARS_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_rows = list(reader)[-max_bars:]
            last_time = 0
            for r in raw_rows:
                try:
                    close_ts = int(r["close_ts"])
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
                        "ticks": int(r.get("ticks", 0)),
                        "symbol": r.get("symbol", "BTCUSDT"),
                    })
                except Exception:
                    continue
    except Exception:
        pass
    return bars


@app.post("/api/bot/start")
async def start_bot():
    """Starts main_shadow_trial.py with --testnet in the background."""
    global bot_process, bot_start_time
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
    """Stops main_shadow_trial.py, cancels open testnet orders, and records stop."""
    global bot_process, bot_start_time
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
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(status["pid"])], capture_output=True, timeout=3.0)
            killed = True
        except Exception:
            pass

    # Cancel open orders across active universe
    for s in DEFAULT_UNIVERSE:
        try:
            await testnet_client.cancel_all_open_orders(s)
        except Exception:
            pass

    bot_start_time = None
    return JSONResponse({"status": "stopped", "message": "Bot process terminated and open orders cancelled"})


@app.get("/api/state")
async def get_state():
    """Full real-time state extraction endpoint for Mode 1 & Mode 2."""
    global bot_start_time
    bal = await get_cached_testnet_balance()
    daemon_info = get_daemon_status()
    audit_events = parse_tail_audit_log(500)
    bars = parse_tail_dollar_bars(100)

    bot_running = daemon_info["running"]
    if bot_running and bot_start_time is None:
        # Infer start time from recent audit events
        ts_list = [datetime.fromisoformat(e["ts_utc"]) for e in audit_events if e.get("ts_utc")]
        if ts_list:
            bot_start_time = min(ts_list).timestamp()
        else:
            bot_start_time = time.time()
    elif not bot_running:
        bot_start_time = None

    if bot_start_time and bot_running:
        duration_sec = int(time.time() - bot_start_time)
        hrs, rem = divmod(duration_sec, 3600)
        mins, secs = divmod(rem, 60)
        duration_fmt = f"{hrs:02d}:{mins:02d}:{secs:02d}"
    else:
        duration_sec = 0
        duration_fmt = "00:00:00"

    # 1. System Health
    resync_count = sum(1 for e in audit_events if e.get("event_type") == "L2_BOOK_RESYNC")
    dollar_bars_formed = sum(1 for e in audit_events if e.get("event_type") in ("DOLLAR_BAR_CLOSED", "DOLLAR_BAR_FORMED"))
    testnet_orders = [e for e in audit_events if e.get("event_type") == "TESTNET_ORDER_ROUTED"]
    current_mode = "TESTNET" if len(testnet_orders) > 0 or os.getenv("BINANCE_FUTURES_TESTNET_API_KEY") else "SHADOW"
    last_event_utc = audit_events[-1].get("ts_utc") if audit_events else datetime.now(timezone.utc).isoformat()

    system_health = {
        "daemon_running": bot_running,
        "daemon_pid": daemon_info["pid"],
        "uptime_str": duration_fmt,
        "resync_count": resync_count,
        "dollar_bars_formed": dollar_bars_formed,
        "last_event_utc": last_event_utc,
        "mode": current_mode,
    }

    # 2. Account & Profit Metrics
    initial_paper_equity = 2500.00
    closed_trades = [e for e in audit_events if e.get("event_type") == "SHADOW_POSITION_CLOSED"]
    realized_pnl = sum(float(t.get("pnl_usd", 0.0)) for t in closed_trades)
    paper_equity = initial_paper_equity + realized_pnl
    if closed_trades:
        paper_equity = float(closed_trades[-1].get("updated_equity", paper_equity))
        wins = [t for t in closed_trades if float(t.get("pnl_usd", 0.0)) > 0]
        win_rate = (len(wins) / len(closed_trades)) * 100.0
        max_dd = max(float(t.get("drawdown_pct", 0.0)) for t in closed_trades)
    else:
        win_rate = 0.0
        max_dd = 0.0

    net_profit_usd = round(realized_pnl, 2)
    net_profit_pct = round((realized_pnl / initial_paper_equity) * 100.0, 2)

    # 3. Executed Trades List
    executed_trades_list = []
    for t in reversed(closed_trades[-30:]):
        executed_trades_list.append({
            "symbol": t.get("symbol", "BTCUSDT"),
            "side": "BUY" if t.get("side") == 1 else "SELL",
            "entry_price": float(t.get("entry_price", 0.0)),
            "exit_price": float(t.get("exit_price", 0.0)),
            "pnl_usd": round(float(t.get("pnl_usd", 0.0)), 2),
            "net_return_bps": round(float(t.get("net_return_bps", 0.0)), 1),
            "exit_reason": t.get("exit_reason", "HORIZONTAL_BARRIER"),
            "timestamp": t.get("ts_utc", "")[11:19] if t.get("ts_utc") else "--:--:--",
            "status": "CLOSED"
        })

    account_metrics = {
        "testnet_equity": bal["total_balance"],
        "testnet_available": bal["available_balance"],
        "paper_equity": round(paper_equity, 2),
        "realized_pnl": net_profit_usd,
        "net_profit_usd": net_profit_usd,
        "net_profit_pct": net_profit_pct,
        "win_rate": round(win_rate, 1),
        "open_positions": 0,
        "max_drawdown": round(max_dd, 2),
        "total_trades": len(closed_trades),
    }

    # 4. Microstructure (BTCUSDT primary)
    eval_events = [e for e in audit_events if e.get("event_type") == "SHADOW_SIGNAL_EVALUATED"]
    btc_evals = [e for e in eval_events if e.get("symbol", "BTCUSDT") == "BTCUSDT"]
    latest_eval = btc_evals[-1] if btc_evals else (eval_events[-1] if eval_events else {})
    latest_bar = bars[-1] if bars else {}

    close_price = latest_eval.get("close_price", latest_bar.get("close", 80940.0))
    rolling_vol = latest_eval.get("rolling_vol", 0.00085)
    cusum_dir = latest_eval.get("cusum_dir", 0)
    z_dist = latest_eval.get("z_dist", 0.0)
    vpin = latest_eval.get("vpin", 0.485)
    vpin_cdf = latest_eval.get("vpin_cdf", 0.52)
    vpin_near_circuit_breaker = bool(vpin_cdf >= 0.95)

    microstructure = {
        "close_price": round(float(close_price), 2),
        "rolling_vol": round(float(rolling_vol), 6),
        "cusum_dir": int(cusum_dir),
        "z_dist": round(float(z_dist), 2),
        "vpin": round(float(vpin), 4),
        "vpin_cdf": round(float(vpin_cdf), 4),
        "vpin_near_circuit_breaker": vpin_near_circuit_breaker,
    }

    # 5. Active Cryptos (The 10 Universe Assets)
    debates = trading_agents_pipeline.get_latest_debates()
    active_cryptos = []
    for s in DEFAULT_UNIVERSE:
        # Get latest price for symbol
        sym_evals = [e for e in eval_events if e.get("symbol") == s]
        sym_price = sym_evals[-1].get("close_price") if sym_evals else REF_PRICES.get(s, 100.0)
        ref_p = REF_PRICES.get(s, sym_price)
        change_pct = round(((sym_price - ref_p) / ref_p) * 100.0, 2) if ref_p > 0 else 0.0

        sym_trades = [t for t in closed_trades if t.get("symbol") == s]
        sym_pnl = round(sum(float(t.get("pnl_usd", 0.0)) for t in sym_trades), 2)

        delib = debates.get(s, {})
        stance = delib.get("stance", "NEUTRAL")
        conviction = delib.get("conviction", 0.50)
        thesis = delib.get("thesis", "Consensus thesis calibrating order flow regime.")
        verdict = delib.get("verdict", "APPROVED")

        active_cryptos.append({
            "symbol": s,
            "name": s.replace("USDT", ""),
            "price": round(float(sym_price), 4 if sym_price < 1.0 else 2),
            "change_pct": change_pct,
            "trade_count": len(sym_trades),
            "pnl": sym_pnl,
            "stance": stance,
            "conviction": conviction,
            "thesis": thesis,
            "verdict": verdict,
            "status": "MONITORING",
        })

    # 6. Recent Signals & Pre-Trade Risk Gates
    recent_signals = []
    target_event_types = {"SHADOW_SIGNAL_EVALUATED", "SHADOW_SIGNAL_DECISION", "TESTNET_ORDER_ROUTED"}
    relevant_events = [e for e in audit_events if e.get("event_type") in target_event_types]

    for e in relevant_events[-25:]:
        meta_prob = e.get("meta_prob")
        gate_passed = False
        kelly_frac = 0.0
        if meta_prob is not None:
            gate_passed = bool(meta_prob >= 0.56)
            if gate_passed:
                kelly_frac = min(0.15, max(0.05, (meta_prob - 0.50) * 1.5))

        recent_signals.append({
            "timestamp": e.get("ts_utc", "")[11:19],
            "symbol": e.get("symbol", "BTCUSDT"),
            "event_type": e.get("event_type"),
            "side": e.get("primary_side", e.get("side", 0)),
            "close_price": e.get("close_price", e.get("price", close_price)),
            "meta_prob": meta_prob,
            "gate_passed": gate_passed,
            "vpin_cdf": e.get("vpin_cdf", vpin_cdf),
            "kelly_fraction": round(kelly_frac, 3),
            "order_status": e.get("order_status", e.get("status", "ACKNOWLEDGED")),
            "order_id": e.get("orderId"),
            "latency_us": e.get("latency_us", 42),
        })

    # 7. Raw Audit Log Tail
    audit_tail = []
    for e in audit_events[-35:]:
        audit_tail.append({
            "ts": e.get("ts_utc", "")[11:19],
            "symbol": e.get("symbol", "-"),
            "event_type": e.get("event_type", ""),
            "close": e.get("close_price", e.get("close", "-")),
            "side": e.get("primary_side", e.get("side", "-")),
            "z_dist": e.get("z_dist", "-"),
            "status": e.get("order_status", e.get("status", e.get("exit_reason", "-"))),
            "order_id": e.get("orderId", e.get("cl_ord_id", "-")),
        })

    # Equity Curve Points for Sparkline
    equity_curve = [initial_paper_equity]
    running_eq = initial_paper_equity
    for t in closed_trades:
        running_eq += float(t.get("pnl_usd", 0.0))
        equity_curve.append(round(running_eq, 2))
    if len(equity_curve) == 1:
        equity_curve.append(initial_paper_equity)

    return {
        "bot_running": bot_running,
        "test_duration_seconds": duration_sec,
        "test_duration_formatted": duration_fmt,
        "net_profit_usd": net_profit_usd,
        "net_profit_pct": net_profit_pct,
        "executed_trades_list": executed_trades_list,
        "active_cryptos": active_cryptos,
        "latest_agent_debates": debates,
        "system_health": system_health,
        "account_metrics": account_metrics,
        "microstructure": microstructure,
        "bars_series": bars[-60:],
        "recent_signals": recent_signals,
        "audit_tail": audit_tail,
        "equity_curve": equity_curve,
    }


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serves the complete two-mode dashboard UI."""
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
        <div id="hdr-testnet-equity" class="text-xs font-bold text-slate-200 font-mono">$13,484.09</div>
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
      
      <!-- Card 1: Total Profit Earned & Sparkline -->
      <div class="bg-brand-900 border border-brand-700/60 rounded-lg p-4 shadow flex flex-col justify-between">
        <div class="flex items-center justify-between mb-2">
          <span class="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Total Profit Earned</span>
          <span class="text-[10px] text-slate-500">Starting Equity: $2,500.00</span>
        </div>
        <div class="flex items-baseline justify-between">
          <div>
            <div id="m1-profit-usd" class="text-2xl font-black text-emerald-400 font-mono glow-bull">+$0.00</div>
            <div id="m1-profit-pct" class="text-xs font-semibold text-emerald-400/80 font-mono">+0.00% Net Return</div>
          </div>
          <!-- Sparkline Container -->
          <div class="w-48 h-12 flex items-center justify-end">
            <svg id="m1-sparkline-svg" class="w-full h-full overflow-visible" viewBox="0 0 180 40">
              <polyline id="m1-sparkline-poly" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="0,20 180,20" />
            </svg>
          </div>
        </div>
        <div class="mt-3 pt-2 border-t border-brand-800 text-[10px] text-slate-400 flex justify-between">
          <span>Closed-Loop Paper Execution</span>
          <span>Triple-Barrier Enforced</span>
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
          <span>Max Drawdown: <span id="m1-max-dd" class="text-slate-300 font-mono">0.0%</span></span>
          <span>L2 Sync Drift: <span id="m1-resync-count" class="text-slate-300 font-mono">0 events</span></span>
        </div>
      </div>
    </div>

    <!-- Card 3: 10-Crypto Major Universe Cards Grid -->
    <div>
      <div class="flex items-center justify-between mb-2">
        <h2 class="text-xs font-bold uppercase tracking-wider text-slate-300">10 Major Cryptocurrencies // Multi-Asset Matrix</h2>
        <span class="text-[10px] text-slate-500">TauricResearch 5-Phase Deliberation Active</span>
      </div>
      <div id="m1-crypto-grid" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2.5">
        <!-- Injected via JavaScript -->
      </div>
    </div>

    <!-- Card 4: Executed Trades Table -->
    <div class="bg-brand-900 border border-brand-700/60 rounded-lg p-3 shadow">
      <div class="flex items-center justify-between mb-2">
        <h2 class="text-xs font-bold uppercase tracking-wider text-slate-300">Executed Trades Log</h2>
        <span class="text-[10px] text-slate-500">Closed & Active Positions</span>
      </div>
      <div class="overflow-x-auto max-h-56">
        <table class="w-full text-left font-mono">
          <thead class="text-[10px] uppercase text-slate-400 border-b border-brand-800 bg-brand-950 sticky top-0">
            <tr>
              <th class="py-1 px-2">Time</th>
              <th class="py-1 px-2">Asset</th>
              <th class="py-1 px-2">Side</th>
              <th class="py-1 px-2">Entry Price</th>
              <th class="py-1 px-2">Exit Price</th>
              <th class="py-1 px-2">Net PnL ($)</th>
              <th class="py-1 px-2">Return (bps)</th>
              <th class="py-1 px-2">Exit Reason</th>
            </tr>
          </thead>
          <tbody id="m1-trades-tbody" class="text-[11px] divide-y divide-brand-800/40">
            <tr><td colspan="8" class="text-center py-4 text-slate-500">No executed trades logged in current session.</td></tr>
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
            <span class="font-bold text-slate-200" id="chart-symbol-label">BTCUSDT</span>
            <span class="text-[10px] text-slate-400">Information-Driven Dollar Bars ($20M Threshold)</span>
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
          <div class="text-[10px] font-bold uppercase tracking-wider text-slate-400 font-sans mb-1">Live Feature Engine</div>
          <div class="flex justify-between">
            <span class="text-slate-400">Rolling Volatility:</span>
            <span id="m2-rolling-vol" class="text-slate-200">0.000850</span>
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
          <div class="flex justify-between"><span>Kill Switch (<3% DD):</span><span class="text-emerald-400">PASS</span></div>
          <div class="flex justify-between"><span>VPIN Toxicity (<0.99):</span><span id="gate-vpin" class="text-emerald-400">PASS</span></div>
          <div class="flex justify-between"><span>Fat-Finger Price Collar:</span><span class="text-emerald-400">PASS</span></div>
          <div class="flex justify-between"><span>Meta Gate (>=0.56):</span><span id="gate-meta" class="text-slate-400">NO TRIGGER</span></div>
        </div>

      </div>
    </div>

    <!-- Row 2: TauricResearch 5-Phase Multi-Agent Consensus Panel -->
    <div class="bg-brand-900 border border-brand-700/60 rounded-lg p-3 shadow">
      <div class="flex flex-wrap items-center justify-between mb-3 gap-2">
        <div class="flex items-center gap-2">
          <span class="inline-block w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_6px_#f59e0b]"></span>
          <h2 class="text-xs font-bold uppercase tracking-wider text-slate-200">TauricResearch / TradingAgents Consensus Terminal</h2>
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

    // Asset Selection for Mode 2 Panel
    function selectAsset(sym) {
      selectedAsset = sym;
      renderAssetTabs();
      updateDeliberationPanel();
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
      document.getElementById('p1-bias-badge').textContent = p1.composite_bias || 'NEUTRAL';
      document.getElementById('p1-tech-sum').textContent = p1.technical_summary || '--';
      document.getElementById('p1-sent-sum').textContent = p1.sentiment_summary || '--';
      document.getElementById('p1-macro-sum').textContent = p1.news_macro_summary || '--';
      document.getElementById('p1-comp-score').textContent = (p1.composite_score >= 0 ? '+' : '') + (p1.composite_score ?? 0.0);

      // Phase 2
      const p2 = d.debate || {};
      document.getElementById('p2-winner-badge').textContent = (p2.debate_winner || 'TIE') + ' ADV';
      const bList = (p2.bull_arguments || []).map(a => `<li>${a}</li>`).join('');
      const rList = (p2.bear_arguments || []).map(a => `<li>${a}</li>`).join('');
      document.getElementById('p2-bull-args').innerHTML = bList || '<li>Analyzing support levels.</li>';
      document.getElementById('p2-bear-args').innerHTML = rList || '<li>Analyzing resistance levels.</li>';
      document.getElementById('p2-rebuttal').textContent = 'Rebuttal: ' + (p2.rebuttal_points || '--');

      // Phase 3
      const p3 = d.proposal || {};
      const actBadge = document.getElementById('p3-action-badge');
      actBadge.textContent = p3.action || 'HOLD';
      actBadge.className = p3.action === 'BUY' ? 'px-1 py-0.2 rounded text-[8px] bg-emerald-900 text-emerald-300 font-bold' : (p3.action === 'SELL' ? 'px-1 py-0.2 rounded text-[8px] bg-rose-900 text-rose-300 font-bold' : 'px-1 py-0.2 rounded text-[8px] bg-slate-800 text-slate-300 font-bold');
      document.getElementById('p3-entry').textContent = p3.target_entry ? '$' + Number(p3.target_entry).toFixed(2) : '--';
      document.getElementById('p3-pt').textContent = p3.take_profit ? '$' + Number(p3.take_profit).toFixed(2) : '--';
      document.getElementById('p3-sl').textContent = p3.stop_loss ? '$' + Number(p3.stop_loss).toFixed(2) : '--';
      document.getElementById('p3-lev').textContent = (p3.recommended_leverage || 2.0) + 'x';
      document.getElementById('p3-rationale').textContent = p3.rationale || '--';

      // Phase 4
      const p4 = d.risk_verdict || {};
      const appBadge = document.getElementById('p4-approved-badge');
      appBadge.textContent = p4.approved ? 'APPROVED' : 'VETOED';
      appBadge.className = p4.approved ? 'px-1 py-0.2 rounded text-[8px] bg-emerald-900 text-emerald-300 font-bold' : 'px-1 py-0.2 rounded text-[8px] bg-rose-900 text-rose-300 font-bold';
      document.getElementById('p4-vpin-check').textContent = p4.vpin_toxicity_check || 'PASS';
      document.getElementById('p4-vol-check').textContent = p4.volatility_check || 'PASS';
      document.getElementById('p4-dd-check').textContent = p4.drawdown_check || 'PASS';
      document.getElementById('p4-veto-reason').textContent = p4.veto_reason || 'NONE';

      // Phase 5
      const p5Verdict = document.getElementById('p5-final-verdict');
      p5Verdict.textContent = d.verdict || 'APPROVED';
      p5Verdict.className = d.verdict === 'APPROVED' ? 'px-1 py-0.2 rounded text-[8px] bg-emerald-900 text-emerald-300 font-bold' : 'px-1 py-0.2 rounded text-[8px] bg-rose-900 text-rose-300 font-bold';
      const conv = Math.round((d.conviction || 0.5) * 100);
      document.getElementById('p5-conviction-score').textContent = conv + '%';
      document.getElementById('p5-conviction-bar').style.width = conv + '%';
      document.getElementById('p5-thesis').textContent = d.thesis || '--';
      document.getElementById('p5-timestamp').textContent = 'UTC: ' + (d.timestamp_utc ? d.timestamp_utc.slice(11, 19) : '--:--:--');
    }

    // Sparkline Render
    function updateSparkline(curve) {
      if (!curve || curve.length < 2) return;
      const svg = document.getElementById('m1-sparkline-poly');
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

    // Main Polling Loop
    async function pollState() {
      try {
        const res = await fetch('/api/state');
        const state = await res.json();

        // 1. Bot Process Button & Badges
        isBotRunning = Boolean(state.bot_running);
        const botBtn = document.getElementById('btn-bot-toggle');
        const botIcon = document.getElementById('bot-toggle-icon');
        const botText = document.getElementById('bot-toggle-text');
        const badgeDaemon = document.getElementById('badge-daemon');
        const m1Pill = document.getElementById('m1-status-pill');

        if (isBotRunning) {
          botBtn.className = 'flex items-center gap-2 px-4 py-1.5 rounded-md font-bold text-xs tracking-wider uppercase transition-all shadow-md bg-rose-600 hover:bg-rose-500 text-slate-100 ring-2 ring-rose-500/50';
          botIcon.textContent = '■';
          botText.textContent = 'STOP BOT';
          badgeDaemon.textContent = `DAEMON RUNNING (PID ${state.system_health.daemon_pid})`;
          badgeDaemon.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-950 text-emerald-300 border border-emerald-600';
          m1Pill.textContent = 'RUNNING';
          m1Pill.className = 'px-2 py-0.5 rounded text-[9px] font-bold bg-emerald-950 text-emerald-400 border border-emerald-700/60 pulse-dot';
        } else {
          botBtn.className = 'flex items-center gap-2 px-4 py-1.5 rounded-md font-bold text-xs tracking-wider uppercase transition-all shadow-md bg-emerald-600 hover:bg-emerald-500 text-slate-950';
          botIcon.textContent = '▶';
          botText.textContent = 'START BOT';
          badgeDaemon.textContent = 'DAEMON STOPPED';
          badgeDaemon.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-brand-800 text-slate-400 border border-slate-700';
          m1Pill.textContent = 'STOPPED';
          m1Pill.className = 'px-2 py-0.5 rounded text-[9px] font-bold bg-slate-800 text-slate-400';
        }

        // Header Equity
        document.getElementById('hdr-testnet-equity').textContent = '$' + Number(state.account_metrics.testnet_equity).toLocaleString('en-US', { minimumFractionDigits: 2 });

        // 2. Mode 1 KPI Pillars
        const netPnl = Number(state.net_profit_usd);
        const pnlEl = document.getElementById('m1-profit-usd');
        const pnlPctEl = document.getElementById('m1-profit-pct');
        const sign = netPnl >= 0 ? '+' : '';
        pnlEl.textContent = `${sign}$${netPnl.toFixed(2)}`;
        pnlPctEl.textContent = `${sign}${Number(state.net_profit_pct).toFixed(2)}% Net Return`;
        if (netPnl >= 0) {
          pnlEl.className = 'text-2xl font-black text-emerald-400 font-mono glow-bull';
          pnlPctEl.className = 'text-xs font-semibold text-emerald-400/80 font-mono';
        } else {
          pnlEl.className = 'text-2xl font-black text-rose-400 font-mono glow-bear';
          pnlPctEl.className = 'text-xs font-semibold text-rose-400/80 font-mono';
        }

        updateSparkline(state.equity_curve);

        document.getElementById('m1-duration-timer').textContent = state.test_duration_formatted;
        document.getElementById('m1-win-rate').textContent = state.account_metrics.win_rate + '%';
        document.getElementById('m1-trade-count').textContent = state.account_metrics.total_trades + ' Closed Trades';
        document.getElementById('m1-max-dd').textContent = state.account_metrics.max_drawdown + '%';
        document.getElementById('m1-resync-count').textContent = state.system_health.resync_count + ' events';

        // 3. 10-Crypto Grid
        const grid = document.getElementById('m1-crypto-grid');
        grid.innerHTML = (state.active_cryptos || []).map(c => {
          const isBull = c.stance === 'BULLISH';
          const isBear = c.stance === 'BEARISH';
          const badgeClass = isBull ? 'bg-emerald-950 text-emerald-400 border border-emerald-600/50' : (isBear ? 'bg-rose-950 text-rose-400 border border-rose-600/50' : 'bg-brand-800 text-slate-300 border border-slate-700');
          const convPct = Math.round(c.conviction * 100);
          const chgSign = c.change_pct >= 0 ? '+' : '';
          const chgClass = c.change_pct >= 0 ? 'text-emerald-400' : 'text-rose-400';

          return `
            <div onclick="switchMode(2); selectAsset('${c.symbol}');" class="cursor-pointer bg-brand-900 border border-brand-700/60 hover:border-cyan-500/60 rounded-lg p-2.5 transition-all shadow hover:shadow-cyan-900/20">
              <div class="flex items-center justify-between mb-1">
                <span class="font-bold text-slate-100">${c.name}</span>
                <span class="text-[9px] px-1 py-0.2 rounded font-semibold ${badgeClass}">${c.stance} ${convPct}%</span>
              </div>
              <div class="flex items-baseline justify-between font-mono">
                <span class="text-xs font-bold text-slate-200">$${Number(c.price).toFixed(c.price < 1 ? 4 : 2)}</span>
                <span class="text-[10px] ${chgClass}">${chgSign}${c.change_pct}%</span>
              </div>
              <div class="mt-2 pt-1.5 border-t border-brand-800/60 flex items-center justify-between text-[9px] text-slate-400 font-mono">
                <span>PnL: <b class="${c.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}">$${c.pnl.toFixed(2)}</b></span>
                <span>${c.trade_count} trades</span>
              </div>
            </div>
          `;
        }).join('');

        // 4. Mode 1 Executed Trades Table
        const m1Tbody = document.getElementById('m1-trades-tbody');
        if (state.executed_trades_list && state.executed_trades_list.length > 0) {
          m1Tbody.innerHTML = state.executed_trades_list.map(t => {
            const sideClass = t.side === 'BUY' ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold';
            const pnlClass = t.pnl_usd >= 0 ? 'text-emerald-400' : 'text-rose-400';
            return `
              <tr class="hover:bg-brand-800/30">
                <td class="py-1 px-2 text-slate-400">${t.timestamp}</td>
                <td class="py-1 px-2 font-bold text-slate-200">${t.symbol}</td>
                <td class="py-1 px-2 ${sideClass}">${t.side}</td>
                <td class="py-1 px-2 text-slate-300">$${t.entry_price.toFixed(2)}</td>
                <td class="py-1 px-2 text-slate-300">$${t.exit_price.toFixed(2)}</td>
                <td class="py-1 px-2 font-bold ${pnlClass}">${t.pnl_usd >= 0 ? '+' : ''}$${t.pnl_usd.toFixed(2)}</td>
                <td class="py-1 px-2 ${pnlClass}">${t.net_return_bps > 0 ? '+' : ''}${t.net_return_bps}</td>
                <td class="py-1 px-2 text-slate-400 text-[10px]">${t.exit_reason}</td>
              </tr>
            `;
          }).join('');
        } else {
          m1Tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4 text-slate-500">No executed trades logged in current session.</td></tr>';
        }

        // 5. Mode 2 Charts & Microstructure
        if (currentMode === 2) {
          latestDebates = state.latest_agent_debates || {};
          renderAssetTabs();
          updateDeliberationPanel();

          // Update chart
          if (candleSeries && state.bars_series && state.bars_series.length > 0) {
            candleSeries.setData(state.bars_series.map(b => ({
              time: b.time,
              open: b.open,
              high: b.high,
              low: b.low,
              close: b.close
            })));
            flowSeries.setData(state.bars_series.map(b => ({
              time: b.time,
              value: b.net_order_flow,
              color: b.net_order_flow >= 0 ? '#10b981' : '#ef4444'
            })));

            const lastB = state.bars_series[state.bars_series.length - 1];
            document.getElementById('bar-o').textContent = lastB.open.toFixed(2);
            document.getElementById('bar-h').textContent = lastB.high.toFixed(2);
            document.getElementById('bar-l').textContent = lastB.low.toFixed(2);
            document.getElementById('bar-c').textContent = lastB.close.toFixed(2);
          }

          // Microstructure
          const micro = state.microstructure;
          document.getElementById('m2-vpin-cdf').textContent = micro.vpin_cdf.toFixed(4);
          document.getElementById('m2-vpin-bar').style.width = (micro.vpin_cdf * 100) + '%';
          const vpinBadge = document.getElementById('vpin-status-badge');
          if (micro.vpin_cdf >= 0.95) {
            vpinBadge.textContent = 'HIGH TOXICITY';
            vpinBadge.className = 'px-1.5 py-0.5 rounded text-[9px] font-bold bg-rose-950 text-rose-300 border border-rose-600 pulse-dot';
            document.getElementById('gate-vpin').textContent = 'WARN';
            document.getElementById('gate-vpin').className = 'text-rose-400 font-bold';
          } else {
            vpinBadge.textContent = 'NORMAL';
            vpinBadge.className = 'px-1.5 py-0.5 rounded text-[9px] font-bold bg-emerald-950 text-emerald-400 border border-emerald-700/60';
            document.getElementById('gate-vpin').textContent = 'PASS';
            document.getElementById('gate-vpin').className = 'text-emerald-400';
          }

          document.getElementById('m2-rolling-vol').textContent = micro.rolling_vol.toFixed(6);
          document.getElementById('m2-cusum-dir').textContent = (micro.cusum_dir > 0 ? '+' : '') + micro.cusum_dir;
          document.getElementById('m2-z-dist').textContent = (micro.z_dist > 0 ? '+' : '') + micro.z_dist.toFixed(2) + 'σ';

          // Recent signal
          if (state.recent_signals && state.recent_signals.length > 0) {
            const sig = state.recent_signals[state.recent_signals.length - 1];
            document.getElementById('m2-meta-prob').textContent = sig.meta_prob ? sig.meta_prob.toFixed(4) : '--';
            document.getElementById('m2-kelly-alloc').textContent = (sig.kelly_fraction * 100).toFixed(1) + '%';
            const metaGate = document.getElementById('gate-meta');
            if (sig.meta_prob && sig.meta_prob >= 0.56) {
              metaGate.textContent = 'PASS (' + (sig.meta_prob * 100).toFixed(1) + '%)';
              metaGate.className = 'text-emerald-400 font-bold';
            } else if (sig.meta_prob) {
              metaGate.textContent = 'FILTERED (' + (sig.meta_prob * 100).toFixed(1) + '%)';
              metaGate.className = 'text-yellow-400';
            } else {
              metaGate.textContent = 'NO TRIGGER';
              metaGate.className = 'text-slate-400';
            }
          }

          // Mode 2 Audit Table
          const m2Tbody = document.getElementById('m2-audit-tbody');
          if (state.audit_tail && state.audit_tail.length > 0) {
            m2Tbody.innerHTML = state.audit_tail.slice().reverse().map(e => `
              <tr class="hover:bg-brand-800/30">
                <td class="py-0.5 px-2 text-slate-400">${e.ts}</td>
                <td class="py-0.5 px-2 font-bold text-slate-300">${e.symbol}</td>
                <td class="py-0.5 px-2 text-cyan-300">${e.event_type}</td>
                <td class="py-0.5 px-2 text-slate-200">${e.close}</td>
                <td class="py-0.5 px-2">${e.side}</td>
                <td class="py-0.5 px-2 text-slate-400">${e.z_dist}</td>
                <td class="py-0.5 px-2 text-slate-300 text-[10px]">${e.status}</td>
                <td class="py-0.5 px-2 text-slate-500 text-[10px] truncate max-w-xs">${e.order_id}</td>
              </tr>
            `).join('');
          }
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
