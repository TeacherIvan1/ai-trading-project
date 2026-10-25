"""
scripts/report_shadow_trial.py
==============================
Lightweight CLI status reporter for the live zero-risk shadow execution trial.
Parses logs/shadow_trial_audit.jsonl and outputs real-time metrics:
- Live Uptime
- L2 Resync Count
- Dollar Bars Formed
- Shadow Signals Evaluated (SHADOW_SIGNAL_EVALUATED)
- Closed Paper Trades
- Win Rate & Net PnL
- Current Paper Equity
- Diagnostic trace of 3 most recent bar evaluations
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    elif minutes > 0:
        return f"{minutes}m {sec:02d}s"
    else:
        return f"{sec}s"


def parse_audit_log(log_path: Path):
    if not log_path.exists():
        return None

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def generate_status_report(log_path: Path = None):
    project_root = Path(__file__).parent.parent.resolve()
    candidate_paths = [
        project_root / "logs" / "shadow_trial_audit.jsonl",
        Path("logs/shadow_trial_audit.jsonl").resolve(),
    ]

    selected_path = None
    if log_path is not None and log_path.exists():
        selected_path = log_path
    else:
        for p in candidate_paths:
            if p.exists() and p.stat().st_size > 0:
                selected_path = p
                break
        if selected_path is None:
            for p in candidate_paths:
                if p.exists():
                    selected_path = p
                    break

    if selected_path is None or not selected_path.exists():
        print(f"[REPORT ERROR] Audit log file not found at: {candidate_paths[0]}")
        return

    records = parse_audit_log(selected_path)
    if not records:
        print(f"[REPORT] Audit log file ({selected_path}) is empty. No live shadow trial events recorded yet.")
        return

    # Parse timestamps
    ts_list = []
    for r in records:
        ts_str = r.get("ts_utc")
        if ts_str:
            try:
                ts_list.append(datetime.fromisoformat(ts_str))
            except Exception:
                pass

    if ts_list:
        start_ts = min(ts_list)
        end_ts = max(ts_list)
        uptime_sec = (end_ts - start_ts).total_seconds()
        start_str = start_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
        end_str = end_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        uptime_sec = 0.0
        start_str = "N/A"
        end_str = "N/A"

    # Event metrics
    resync_events = [r for r in records if r.get("event_type") == "L2_BOOK_RESYNC"]
    dollar_bars = [r for r in records if r.get("event_type") in ("DOLLAR_BAR_CLOSED", "DOLLAR_BAR_FORMED")]
    signal_evals = [r for r in records if r.get("event_type") == "SHADOW_SIGNAL_EVALUATED"]
    closed_trades = [r for r in records if r.get("event_type") == "SHADOW_POSITION_CLOSED"]

    # Filtered / Opened decisions
    opened_positions = [
        r for r in signal_evals
        if r.get("order_status") == "OPENED_SHADOW_POSITION" or
           (isinstance(r.get("decision"), dict) and r["decision"].get("status") == "OPENED_SHADOW_POSITION")
    ]

    # Performance & Equity
    initial_equity = 2500.00
    if closed_trades:
        current_equity = closed_trades[-1].get("updated_equity", initial_equity)
        total_pnl = sum(t.get("pnl_usd", 0.0) for t in closed_trades)
        winning_trades = [t for t in closed_trades if t.get("pnl_usd", 0.0) > 0.0]
        losing_trades = [t for t in closed_trades if t.get("pnl_usd", 0.0) <= 0.0]
        win_rate = (len(winning_trades) / len(closed_trades)) * 100.0
        tot_win = sum(t.get("pnl_usd", 0.0) for t in winning_trades)
        tot_loss = abs(sum(t.get("pnl_usd", 0.0) for t in losing_trades))
        profit_factor = (tot_win / (tot_loss + 1e-9)) if tot_loss > 0 else 99.0
        max_dd = max(t.get("drawdown_pct", 0.0) for t in closed_trades)
    else:
        current_equity = initial_equity
        total_pnl = 0.00
        win_rate = 0.0
        profit_factor = 0.0
        max_dd = 0.0

    ret_pct = ((current_equity - initial_equity) / initial_equity) * 100.0

    # Output Terminal Summary
    header = "=" * 80
    print("\n" + header)
    print("LIVE SHADOW TRIAL STATUS AUDIT REPORT")
    print(header)
    print(f"{'Audit Metric':<40} | {'Value':<35}")
    print("-" * 80)
    print(f"{'Log File Path':<40} | {str(selected_path):<35}")
    print(f"{'Session Start':<40} | {start_str:<35}")
    print(f"{'Last Event Timestamp':<40} | {end_str:<35}")
    print(f"{'Live Session Uptime':<40} | {format_duration(uptime_sec):<35}")
    print(f"{'L2 Order Book Resync Count':<40} | {len(resync_events):<35}")
    print(f"{'Dollar Bars Formed':<40} | {len(dollar_bars):<35}")
    print(f"{'Shadow Signals Evaluated':<40} | {len(signal_evals):<35}")
    print(f"{'Shadow Positions Opened':<40} | {len(opened_positions):<35}")
    print(f"{'Closed Paper Trades':<40} | {len(closed_trades):<35}")
    print(f"{'Paper Trading Win Rate':<40} | {win_rate:>6.1f}%{'':<28}")
    print(f"{'Realized Net PnL (USD)':<40} | ${total_pnl:>+8.2f} ({ret_pct:>+5.2f}%){'':<17}")
    print(f"{'Current Paper Equity':<40} | ${current_equity:>8.2f}{'':<26}")
    print(f"{'Max Peak-to-Trough Drawdown':<40} | {max_dd:>6.2f}%{'':<28}")
    print(header + "\n")

    # Step 3: Print diagnostic values of the 3 most recent bar evaluations
    recent_evals = [r for r in records if r.get("event_type") == "SHADOW_SIGNAL_EVALUATED"]
    if recent_evals:
        print("RECENT 3 BAR SIGNAL EVALUATIONS (DIAGNOSTIC TRACE):")
        e_header = f"{'Timestamp (UTC)':<20} | {'Close':<10} | {'CUSUM Dir':<10} | {'z_dist':<8} | {'Side':<5} | {'Meta Prob':<10} | {'Status':<25}"
        print("-" * len(e_header))
        print(e_header)
        print("-" * len(e_header))
        for ev in recent_evals[-3:]:
            ts = ev.get("ts_utc", "")[:19].replace("T", " ")
            close_p = f"{ev.get('close_price', 0.0):.2f}"
            cdir = f"{ev.get('cusum_dir', 0):+d}"
            z_val = f"{ev.get('z_dist', 0.0):+.2f}"
            side_val = str(ev.get("primary_side", 0))
            mp = f"{ev['meta_prob']:.4f}" if ev.get("meta_prob") is not None else "null"
            st = str(ev.get("order_status", "UNKNOWN"))[:25]
            print(f"{ts:<20} | {close_p:<10} | {cdir:<10} | {z_val:<8} | {side_val:<5} | {mp:<10} | {st:<25}")
        print("-" * len(e_header) + "\n")
    else:
        print("RECENT 3 BAR SIGNAL EVALUATIONS (DIAGNOSTIC TRACE):")
        print("No SHADOW_SIGNAL_EVALUATED events recorded yet.\n")


if __name__ == "__main__":
    path_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    generate_status_report(path_arg)
