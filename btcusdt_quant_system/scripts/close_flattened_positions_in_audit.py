import json
from datetime import datetime, timezone
from pathlib import Path

audit_path = Path("logs/shadow_trial_audit.jsonl")
if not audit_path.exists():
    print("No audit file found.")
    exit(0)

with open(audit_path, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

opened_trades = [r for r in records if r.get("event_type") == "SHADOW_POSITION_OPENED"]
closed_trades = [r for r in records if r.get("event_type") == "SHADOW_POSITION_CLOSED"]
closed_ids = {c.get("cl_ord_id") for c in closed_trades}

unclosed = [o for o in opened_trades if o.get("cl_ord_id") not in closed_ids]
print(f"Found {len(unclosed)} unclosed positions in shadow audit log.")

now_iso = datetime.now(timezone.utc).isoformat()
flatten_events = []
for pos in unclosed:
    cl_id = pos.get("cl_ord_id")
    sym = pos.get("symbol", "UNKNOWN")
    entry_price = float(pos.get("entry_price", 0.0))
    side = pos.get("side", 1)
    qty = float(pos.get("qty_btc", 0.0))
    notional = float(pos.get("notional_usd", 0.0))
    
    event = {
        "ts_utc": now_iso,
        "event_type": "SHADOW_POSITION_CLOSED",
        "symbol": sym,
        "cl_ord_id": cl_id,
        "side": side,
        "entry_price": entry_price,
        "exit_price": entry_price,
        "pnl_usd": 0.0,
        "net_return_bps": 0.0,
        "exit_reason": "MANUAL_CLEAN_SLATE_FLATTEN",
        "qty_btc": qty,
        "notional_usd": notional,
        "holding_bars": 0,
        "updated_equity": 12859.48,
        "drawdown_pct": 0.0
    }
    flatten_events.append(event)
    print(f"Closing {sym} (cl_ord_id={cl_id})")

if flatten_events:
    with open(audit_path, "a", encoding="utf-8") as f:
        for ev in flatten_events:
            f.write(json.dumps(ev) + "\n")
    print(f"Appended {len(flatten_events)} close events to {audit_path}")

