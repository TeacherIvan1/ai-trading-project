@echo off
cd /d O:\ai-trading-project\btcusdt_quant_system
start "" /B ".\.venv\Scripts\python.exe" -m uvicorn dashboard.app:app --host 127.0.0.1 --port 8050 > logs\dashboard.log 2>&1
timeout /t 2 /nobreak >nul
start http://localhost:8050
