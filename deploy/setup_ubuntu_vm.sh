#!/usr/bin/env bash
# ==============================================================================
# Amazon Lightsail Ubuntu 24.04 Automated Bootstrap & 24/7 Deployment Script
# Instance: 512MB RAM, 1 vCPU | Public IP: 16.59.54.195 | User: ubuntu
# ==============================================================================
set -euo pipefail

echo "======================================================================"
echo " Starting Systematic Quant Bot & Dashboard Cloud VM Deployment"
echo " Time: $(date -u)"
echo " Host: $(hostname) | Kernel: $(uname -r)"
echo "======================================================================"

# ------------------------------------------------------------------------------
# 1. Enforce 2GB Swap File (Strictly Required for 512MB RAM Instances)
# ------------------------------------------------------------------------------
if ! swapon --show | grep -q "/swapfile"; then
    echo "[1/6] Configuring 2GB swap space to avoid OOM killer on 512MB instance..."
    if [ ! -f /swapfile ]; then
        sudo fallocate -l 2G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
    fi
    sudo swapon /swapfile
    if ! grep -q "/swapfile" /etc/fstab; then
        echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
    fi
    echo "  -> Swap enabled successfully:"
    free -h
else
    echo "[1/6] Swap is already active:"
    free -h
fi

# ------------------------------------------------------------------------------
# 2. Update System Packages
# ------------------------------------------------------------------------------
echo "[2/6] Updating apt packages and installing build dependencies..."
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git python3-venv python3-pip python3-dev build-essential curl ufw jq

# ------------------------------------------------------------------------------
# 3. Clone Repository & Setup Virtual Environment
# ------------------------------------------------------------------------------
REPO_DIR="/home/ubuntu/ai-trading-project"
SYS_DIR="$REPO_DIR/btcusdt_quant_system"

echo "[3/6] Fetching repository from GitHub into $REPO_DIR..."
if [ -d "$REPO_DIR/.git" ]; then
    echo "  -> Repository exists, updating to latest origin/main..."
    cd "$REPO_DIR"
    git fetch origin main
    git reset --hard origin/main
else
    echo "  -> Cloning repository..."
    git clone https://github.com/TeacherIvan1/ai-trading-project.git "$REPO_DIR"
    cd "$REPO_DIR"
fi

cd "$SYS_DIR"

echo "  -> Setting up Python virtual environment in $SYS_DIR/.venv..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Upgrade pip inside venv and install dependencies
source .venv/bin/activate
pip install --upgrade pip --no-cache-dir
echo "  -> Installing Python quantitative stack dependencies..."
pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# 4. Inject Environment Variables (.env) & Setup Log Directories
# ------------------------------------------------------------------------------
echo "[4/6] Injecting Binance Testnet credentials and configuring directories..."
mkdir -p "$SYS_DIR/logs" "$SYS_DIR/models"

cat << 'ENVFILE' > "$SYS_DIR/.env"
BINANCE_FUTURES_TESTNET_API_KEY="r6Zmj2j7qjpm1dNHW7pL4GqYDNIlcefWgv5MNWmjq90B5Bq1RAG20Ibviio2mDrZ"
BINANCE_FUTURES_TESTNET_API_SECRET="ujzmLoHr9BLRYoA6CT2xl0sbO1NBg686AJf9WIYna5chM9CGyj4OExLQ1WgLDiBb"
ENVFILE
chmod 600 "$SYS_DIR/.env"

# ------------------------------------------------------------------------------
# 5. Configure and Enable Systemd Services (24/7 Autostart & Watchdog)
# ------------------------------------------------------------------------------
echo "[5/6] Registering 24/7 systemd daemon services..."

# quant-bot.service
sudo bash -c "cat << 'SVCEOF' > /etc/systemd/system/quant-bot.service
[Unit]
Description=Quant Trading Bot (Shadow Trial 10-Asset Universe)
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-trading-project/btcusdt_quant_system
ExecStart=/home/ubuntu/ai-trading-project/btcusdt_quant_system/.venv/bin/python main_shadow_trial.py --testnet
Restart=always
RestartSec=5
StandardOutput=append:/home/ubuntu/ai-trading-project/btcusdt_quant_system/logs/daemon.log
StandardError=append:/home/ubuntu/ai-trading-project/btcusdt_quant_system/logs/daemon_err.log
Environment=PYTHONUNBUFFERED=1
Environment="PYTHONPATH=/home/ubuntu/ai-trading-project/btcusdt_quant_system"

[Install]
WantedBy=multi-user.target
SVCEOF"

# quant-dashboard.service
sudo bash -c "cat << 'SVCEOF' > /etc/systemd/system/quant-dashboard.service
[Unit]
Description=Quant Dashboard FastAPI & Uvicorn Service (Port 8050)
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-trading-project/btcusdt_quant_system
ExecStart=/home/ubuntu/ai-trading-project/btcusdt_quant_system/.venv/bin/python -m uvicorn dashboard.app:app --host 0.0.0.0 --port 8050
Restart=always
RestartSec=5
StandardOutput=append:/home/ubuntu/ai-trading-project/btcusdt_quant_system/logs/uvicorn.log
StandardError=append:/home/ubuntu/ai-trading-project/btcusdt_quant_system/logs/uvicorn_err.log
Environment=PYTHONUNBUFFERED=1
Environment="PYTHONPATH=/home/ubuntu/ai-trading-project/btcusdt_quant_system"

[Install]
WantedBy=multi-user.target
SVCEOF"

sudo systemctl daemon-reload
sudo systemctl enable quant-bot.service
sudo systemctl enable quant-dashboard.service

echo "  -> Starting systemd services..."
sudo systemctl restart quant-bot.service
sudo systemctl restart quant-dashboard.service

# ------------------------------------------------------------------------------
# 6. Verification and Health Check
# ------------------------------------------------------------------------------
echo "[6/6] Waiting 10 seconds for trading engine & dashboard initialization..."
sleep 10

echo "======================================================================"
echo " SYSTEMD DAEMON STATUS"
echo "======================================================================"
sudo systemctl status quant-bot.service --no-pager -n 5 || true
echo "----------------------------------------------------------------------"
sudo systemctl status quant-dashboard.service --no-pager -n 5 || true

echo "======================================================================"
echo " LOCALHOST:8050 API STATE QUERY"
echo "======================================================================"
curl -s http://127.0.0.1:8050/api/state | python3 -m json.tool | head -n 45 || curl -s http://127.0.0.1:8050/api/state

echo "======================================================================"
echo " CLOUD DEPLOYMENT COMPLETED SUCCESSFULLY!"
echo " Public Dashboard URL: http://16.59.54.195:8050"
echo " Bot Service:  sudo systemctl status quant-bot"
echo " Dashboard:    sudo systemctl status quant-dashboard"
echo " Live Logs:    tail -f /home/ubuntu/ai-trading-project/btcusdt_quant_system/logs/daemon.log"
echo "======================================================================"

