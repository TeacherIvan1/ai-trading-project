import json
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:8050/api/state", timeout=5) as resp:
        data = json.loads(resp.read().decode())
        print("Bot Running:", data.get("bot_running"))
        print("Open Positions Count:", data.get("account_metrics", {}).get("open_positions"))
        print("Testnet Equity:", data.get("account_metrics", {}).get("testnet_equity"))
        print("Testnet Available:", data.get("account_metrics", {}).get("testnet_available"))
        in_pos = [c["symbol"] for c in data.get("active_cryptos", []) if c.get("in_position")]
        print("Active In-Position Cryptos:", in_pos)
        print("Total Trades:", data.get("account_metrics", {}).get("total_trades"))
except Exception as e:
    print("Error querying dashboard:", e)

