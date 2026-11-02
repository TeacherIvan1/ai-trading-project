import json

with open("logs/shadow_trial_audit.jsonl", "r", encoding="utf-8") as f:
    events = [json.loads(l) for l in f if "TESTNET_ORDER_ROUTED" in l]

print(f"Total TESTNET_ORDER_ROUTED events: {len(events)}")
for ev in events[-3:]:
    print("------------------------------------------")
    print("Symbol:", ev.get("symbol"))
    print("OrderID:", ev.get("orderId"))
    print("Status:", ev.get("status"))
    print("slOrderId:", ev.get("slOrderId"))
    print("tpOrderId:", ev.get("tpOrderId"))
    print("ptPrice:", ev.get("ptPrice"))
    print("slPrice:", ev.get("slPrice"))
    native = ev.get("raw_response", {}).get("native_tp_sl", {})
    print("SL Response:", native.get("sl_response"))
    print("TP Response:", native.get("tp_response"))

