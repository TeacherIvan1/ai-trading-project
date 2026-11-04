import subprocess

service_content = """[Unit]
Description=Quant Trading Bot (Shadow Trial 10-Asset Universe)
After=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-trading-project/btcusdt_quant_system
Environment="PYTHONPATH=/home/ubuntu/ai-trading-project/btcusdt_quant_system"
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=/home/ubuntu/ai-trading-project/btcusdt_quant_system/.env
ExecStart=/home/ubuntu/ai-trading-project/btcusdt_quant_system/.venv/bin/python main_shadow_trial.py --testnet
Restart=always
RestartSec=5
StandardOutput=append:/home/ubuntu/ai-trading-project/btcusdt_quant_system/logs/daemon.log
StandardError=append:/home/ubuntu/ai-trading-project/btcusdt_quant_system/logs/daemon_err.log

[Install]
WantedBy=multi-user.target
"""

with open("/etc/systemd/system/quant-bot.service", "w") as f:
    f.write(service_content)

subprocess.run(["systemctl", "daemon-reload"], check=True)
subprocess.run(["systemctl", "restart", "quant-bot.service"], check=True)
print("Updated quant-bot.service and reloaded systemd successfully.")
