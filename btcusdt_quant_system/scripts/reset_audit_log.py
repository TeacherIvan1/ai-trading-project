import json
import datetime
from pathlib import Path

log_path = Path(__file__).resolve().parent.parent / "logs" / "shadow_trial_audit.jsonl"
log_path.parent.mkdir(parents=True, exist_ok=True)

record = {
    "ts_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "event_type": "BOT_SESSION_STARTED",
    "version": "2.5.0",
    "capital_cap_per_symbol": 1000.0,
    "max_concurrent_positions": 4,
}

with open(log_path, "w", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")

print(f"[RESET AUDIT LOG] Successfully wrote {log_path} with record: {record}")
