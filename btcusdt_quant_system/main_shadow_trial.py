"""
main_shadow_trial.py
====================
Multi-Asset Live Zero-Risk Shadow Execution & Binance Futures Testnet Engine.
Supports the 10-asset major crypto universe:
["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]

Streams real-time L2 order book updates (@depth@100ms) and aggregated trades (@aggTrade).
Executes SEC 15c3-5 style synchronous pre-trade risk checks (<100us),
evaluates live Dollar Bars against the trained Meta-Labeling RandomForest model,
integrates TauricResearch / TradingAgents 5-phase multi-agent consensus pipeline,
and manages active paper positions with closed-loop Triple-Barrier exits.
Optionally routes acknowledged orders directly to Binance Futures Testnet via --testnet.
"""

import argparse
import asyncio
import json
import math
import os
import pickle
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import websockets

# Force line buffering on standard output and error so daemon.log updates immediately
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.orderbook_sync import LocalOrderBookManager
from core.data_bars import DollarBarAndMicrostructureEngine
from core.features_labels import generate_primary_signals
from execution.paper_engine import SynchronousPreTradeRiskAndPaperEngine, OrderIntent
from execution.binance_testnet_client import BinanceFuturesTestnetClient
from agents.trading_agents_bridge import trading_agents_pipeline

DEFAULT_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"
]

DEFAULT_THRESHOLDS = {
    "BTCUSDT": 20_000_000.0,
    "ETHUSDT": 20_000_000.0,
    "SOLUSDT": 5_000_000.0,
    "BNBUSDT": 5_000_000.0,
    "XRPUSDT": 5_000_000.0,
    "ADAUSDT": 5_000_000.0,
    "DOGEUSDT": 5_000_000.0,
    "AVAXUSDT": 5_000_000.0,
    "DOTUSDT": 5_000_000.0,
    "LINKUSDT": 5_000_000.0,
}

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


def load_universe_config() -> tuple:
    settings_file = PROJECT_ROOT / "config" / "settings.yaml"
    symbols = list(DEFAULT_UNIVERSE)
    thresholds = dict(DEFAULT_THRESHOLDS)
    if settings_file.exists():
        try:
            content = settings_file.read_text(encoding="utf-8")
            in_universe = False
            parsed_symbols = []
            for line in content.splitlines():
                stripped = line.strip()
                if "universe_symbols:" in stripped:
                    in_universe = True
                    continue
                if in_universe:
                    if stripped.startswith("- "):
                        sym = stripped.replace("- ", "").replace('"', '').replace("'", '').strip()
                        if sym:
                            parsed_symbols.append(sym)
                    elif stripped and not stripped.startswith("#"):
                        in_universe = False
            if parsed_symbols:
                symbols = parsed_symbols
        except Exception:
            pass
    return symbols, thresholds


class LiveShadowTrialRunner:
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        trained_rf_model=None,
        feature_cols: list = None,
        initial_equity: float = 2500.0,
        is_testnet: bool = False,
    ):
        cfg_symbols, cfg_thresholds = load_universe_config()
        self.symbols = [s.strip().upper() for s in (symbols or cfg_symbols)]
        self.thresholds = cfg_thresholds
        self.is_testnet = is_testnet

        # Check for environment override on dollar threshold
        env_threshold_override = os.environ.get("DOLLAR_BAR_THRESHOLD")
        if env_threshold_override:
            try:
                ov = float(env_threshold_override)
                for s in self.symbols:
                    self.thresholds[s] = ov if s in ("BTCUSDT", "ETHUSDT") else max(50000.0, ov * 0.25)
            except Exception:
                pass

        if self.is_testnet:
            self.rest_base_url = "https://testnet.binancefuture.com"
            self.ws_base_url = "wss://stream.binancefuture.com/stream?streams="
            self.testnet_client = BinanceFuturesTestnetClient()
            print(f"[INIT] Configured for Binance Futures TESTNET execution across {len(self.symbols)} symbols.", flush=True)
        else:
            self.rest_base_url = "https://fapi.binance.com"
            self.ws_base_url = "wss://fstream.binance.com/stream?streams="
            self.testnet_client = None
            print(f"[INIT] Configured for Live Shadow (Production Feed) across {len(self.symbols)} symbols.", flush=True)

        # Build multiplexed stream URL
        stream_names = []
        for s in self.symbols:
            s_low = s.lower()
            stream_names.append(f"{s_low}@depth@100ms")
            stream_names.append(f"{s_low}@aggTrade")
        self.ws_url = self.ws_base_url + "/".join(stream_names)

        # Per-symbol state
        self.ob_managers: Dict[str, LocalOrderBookManager] = {s: LocalOrderBookManager(s) for s in self.symbols}
        self.bar_engines: Dict[str, DollarBarAndMicrostructureEngine] = {
            s: DollarBarAndMicrostructureEngine(dollar_threshold=self.thresholds.get(s, 5_000_000.0))
            for s in self.symbols
        }
        self.risk_engines: Dict[str, SynchronousPreTradeRiskAndPaperEngine] = {
            s: SynchronousPreTradeRiskAndPaperEngine(
                initial_equity=initial_equity,
                pt_mult=1.5,
                sl_mult=1.5,
                max_horizon_bars=25,
                maker_fee_bps=2.0,
                taker_fee_bps=5.0
            )
            for s in self.symbols
        }

        # Directories and log files
        self.logs_dir = PROJECT_ROOT / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path = self.logs_dir / "shadow_trial_audit.jsonl"
        self.audit_log_path.touch(exist_ok=True)
        self.live_bars_csv_path = self.logs_dir / "live_dollar_bars.csv"
        if not self.live_bars_csv_path.exists():
            with open(self.live_bars_csv_path, "w", encoding="utf-8") as f:
                f.write("open_ts,close_ts,open,high,low,close,volume,dollar_volume,net_order_flow,ticks,symbol\n")

        # Load Meta-Labeling RF model
        models_dir = PROJECT_ROOT / "models"
        if trained_rf_model is None:
            model_path = models_dir / "ignition_meta_rf.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Trained model not found at {model_path}.")
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            self.model = trained_rf_model

        # Load warmup bars for BTCUSDT
        warmup_path = models_dir / "warmup_bars.parquet"
        btc_warmup_df = None
        if warmup_path.exists():
            try:
                df_w = pd.read_parquet(warmup_path)
                if len(df_w) >= 200:
                    btc_warmup_df = df_w.tail(300).reset_index(drop=True)
            except Exception:
                pass

        if btc_warmup_df is None:
            candidate_paths = [
                PROJECT_ROOT / "data" / "btcusdt_dollar_bars_20m_30d.parquet",
                PROJECT_ROOT / "data" / "btcusdt_dollar_bars_20m_30d.csv",
                PROJECT_ROOT / "data" / "btcusdt_dollar_bars_30d.parquet",
                PROJECT_ROOT / "data" / "btcusdt_dollar_bars_30d.csv",
            ]
            for cand in candidate_paths:
                if cand.exists():
                    try:
                        df_cand = pd.read_parquet(cand) if cand.suffix == ".parquet" else pd.read_csv(cand)
                        if len(df_cand) >= 200:
                            btc_warmup_df = df_cand.tail(300).reset_index(drop=True)
                            models_dir.mkdir(parents=True, exist_ok=True)
                            btc_warmup_df.to_parquet(warmup_path, index=False)
                            break
                    except Exception:
                        continue

        if btc_warmup_df is None:
            raise FileNotFoundError("Could not load or generate at least 200 historical warmup bars from models/ or data/ caches.")

        # Initialize warmup DataFrames per symbol
        self.bars_dfs: Dict[str, pd.DataFrame] = {}
        btc_ref_price = REF_PRICES.get("BTCUSDT", 80940.0)
        btc_threshold = self.thresholds.get("BTCUSDT", 20_000_000.0)

        for s in self.symbols:
            if s == "BTCUSDT":
                self.bars_dfs[s] = btc_warmup_df.copy()
            else:
                # Scale BTC warmup bars to the relative price scale of the asset
                s_ref = REF_PRICES.get(s, 100.0)
                price_ratio = s_ref / btc_ref_price
                vol_ratio = (self.thresholds.get(s, 5_000_000.0) / btc_threshold) / price_ratio

                scaled_df = btc_warmup_df.copy()
                scaled_df["open"] = scaled_df["open"] * price_ratio
                scaled_df["high"] = scaled_df["high"] * price_ratio
                scaled_df["low"] = scaled_df["low"] * price_ratio
                scaled_df["close"] = scaled_df["close"] * price_ratio
                scaled_df["volume"] = scaled_df["volume"] * vol_ratio
                scaled_df["dollar_volume"] = self.thresholds.get(s, 5_000_000.0)
                scaled_df["net_order_flow"] = scaled_df["net_order_flow"] * vol_ratio
                self.bars_dfs[s] = scaled_df

        self.current_bar_idx: Dict[str, int] = {s: len(self.bars_dfs[s]) for s in self.symbols}
        self.resync_counts: Dict[str, int] = {s: 0 for s in self.symbols}
        self.order_seq = 0

        self.last_prices: Dict[str, float] = {s: REF_PRICES.get(s, 100.0) for s in self.symbols}
        self.last_vols: Dict[str, float] = {s: 0.0015 for s in self.symbols}
        self.last_cusums: Dict[str, int] = {s: 0 for s in self.symbols}
        self.last_zdists: Dict[str, float] = {s: 0.0 for s in self.symbols}
        self.last_vpin_cdfs: Dict[str, float] = {s: 0.50 for s in self.symbols}

        self.feature_cols = feature_cols if feature_cols is not None else [
            "vpin", "vpin_cdf", "vpin_slope", "kyle_lambda_z", "ffd_close",
            "rolling_vol", "vol_regime_ratio", "flow_alignment", "abs_z_dist", "ret_autocorr"
        ]

        # Warmup indicators for all symbols
        for s in self.symbols:
            try:
                df_micro = self.bar_engines[s].compute_vpin_and_kyle_lambda(self.bars_dfs[s], window=50)
                df_sig = generate_primary_signals(df_micro, span=40, cusum_mult=1.35, mode="IGNITION")
                latest = df_sig.iloc[-1]
                self.last_prices[s] = float(latest["close"])
                self.last_vols[s] = float(latest.get("rolling_vol", 0.0015))
                self.last_cusums[s] = int(latest.get("cusum_dir", 0))
                self.last_zdists[s] = float(latest.get("z_dist", 0.0))
                self.last_vpin_cdfs[s] = float(latest.get("vpin_cdf", 0.50))
            except Exception as e:
                print(f"[WARMUP WARNING] Indicator warmup warning for {s}: {e}", flush=True)

        print(f"[WARMUP COMPLETE] Initialized {len(self.symbols)} assets. Ready for multi-stream ingestion.", flush=True)

    def _log_event(self, event_type: str, payload: dict):
        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **payload
        }
        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _record_live_dollar_bar_csv(self, symbol: str, bar: dict):
        line = (
            f"{bar['open_ts']},{bar['close_ts']},{bar['open']:.4f},{bar['high']:.4f},"
            f"{bar['low']:.4f},{bar['close']:.4f},{bar['volume']:.6f},{bar['dollar_volume']:.2f},"
            f"{bar['net_order_flow']:.6f},{bar['ticks']},{symbol}\n"
        )
        with open(self.live_bars_csv_path, "a", encoding="utf-8") as f:
            f.write(line)

    async def _fetch_rest_snapshot(self, symbol: str) -> dict:
        url = f"{self.rest_base_url}/fapi/v1/depth?symbol={symbol}&limit=1000"
        def _get():
            with urllib.request.urlopen(url, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        return await asyncio.to_thread(_get)

    async def _cancel_quotes_callback(self, symbol: str) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        print(f"[{now_iso}] [SAFETY] Cancelling resting shadow quotes for {symbol} during L2 resync.", flush=True)
        if self.testnet_client:
            try:
                res = await self.testnet_client.cancel_all_open_orders(symbol)
                self._log_event("TESTNET_CANCEL_ALL_OPEN_ORDERS", {"symbol": symbol, "result": res})
            except Exception as e:
                print(f"[{now_iso}] [TESTNET ERROR] Failed to cancel open orders for {symbol}: {e}", flush=True)

    async def _evaluate_new_dollar_bar(self, symbol: str, new_bar: dict):
        # 1. Log DOLLAR_BAR_CLOSED
        self._log_event("DOLLAR_BAR_CLOSED", {
            "symbol": symbol,
            "close": float(new_bar["close"]),
            "dollar_vol": float(new_bar.get("dollar_volume", 0.0)),
            "ticks": int(new_bar.get("ticks", 0))
        })

        try:
            bar_entry = {k: v for k, v in new_bar.items() if k != "symbol"}
            self.bars_dfs[symbol] = pd.concat([self.bars_dfs[symbol], pd.DataFrame([bar_entry])], ignore_index=True).tail(300)
            df_micro = self.bar_engines[symbol].compute_vpin_and_kyle_lambda(self.bars_dfs[symbol], window=50)
            df_sig = generate_primary_signals(df_micro, span=40, cusum_mult=1.35, mode="IGNITION")
            latest = df_sig.iloc[-1]

            close_price = float(latest["close"])
            primary_side = int(latest["primary_side"])
            cusum_dir = int(latest.get("cusum_dir", 0))
            z_dist = float(latest.get("z_dist", 0.0))
            rolling_vol = float(latest.get("rolling_vol", 0.0015))
            vpin_cdf = float(latest.get("vpin_cdf", 0.50))
            net_order_flow = float(new_bar.get("net_order_flow", 0.0))

            self.last_prices[symbol] = close_price
            self.last_vols[symbol] = rolling_vol
            self.last_cusums[symbol] = cusum_dir
            self.last_zdists[symbol] = z_dist
            self.last_vpin_cdfs[symbol] = vpin_cdf

            meta_prob = None
            order_status = "NO_PRIMARY_BREAKOUT"

            # TauricResearch / TradingAgents 5-Phase Deliberation
            deliberation = await trading_agents_pipeline.deliberate_asset(
                symbol=symbol,
                current_price=close_price,
                meta_prob=meta_prob,
                rolling_vol=rolling_vol,
                cusum_dir=cusum_dir,
                z_dist=z_dist,
                vpin_cdf=vpin_cdf,
                net_order_flow=net_order_flow,
            )

            if primary_side != 0:
                ob_manager = self.ob_managers[symbol]
                if not ob_manager.is_synced:
                    order_status = "ORDERBOOK_DESYNCED"
                else:
                    X_live = df_sig[self.feature_cols].iloc[[-1]].fillna(0.0)
                    meta_prob = float(self.model.predict_proba(X_live)[0, 1])

                    # Re-deliberate with computed meta_prob
                    deliberation = await trading_agents_pipeline.deliberate_asset(
                        symbol=symbol,
                        current_price=close_price,
                        meta_prob=meta_prob,
                        rolling_vol=rolling_vol,
                        cusum_dir=cusum_dir,
                        z_dist=z_dist,
                        vpin_cdf=vpin_cdf,
                        net_order_flow=net_order_flow,
                    )

                    # Check Risk Guardians & PM approval
                    if not deliberation.risk_verdict.approved:
                        order_status = f"VETOED_BY_RISK_GUARDIANS_{deliberation.risk_verdict.veto_reason}"
                    elif deliberation.final_verdict != "APPROVED":
                        order_status = "VETOED_BY_PORTFOLIO_MANAGER"
                    else:
                        best_bid = max(ob_manager.bids.keys()) if ob_manager.bids else 0.0
                        best_ask = min(ob_manager.asks.keys()) if ob_manager.asks else 0.0
                        mid_price = 0.5 * (best_bid + best_ask) if (best_bid and best_ask) else close_price
                        limit_price = best_bid if primary_side == 1 else best_ask

                        self.order_seq += 1
                        intent = OrderIntent(
                            cl_ord_id=f"SHADOW_{symbol}_{int(time.time())}_{self.order_seq}",
                            symbol=symbol,
                            side=primary_side,
                            limit_price=limit_price,
                            mid_price=mid_price,
                            meta_prob=meta_prob,
                            vpin_cdf=vpin_cdf,
                            rolling_vol=rolling_vol
                        )
                        decision = self.risk_engines[symbol].validate_and_route_shadow_order(intent, self.current_bar_idx[symbol])
                        dec_status = decision.get("status")

                        if dec_status in ("OPENED_SHADOW_POSITION", "ACKNOWLEDGED_PAPER"):
                            order_status = dec_status

                            if self.testnet_client:
                                try:
                                    qty = float(decision.get("qty_btc", 0.001))
                                    # Ensure Binance min notional ~$50 USDT
                                    if qty * limit_price < 50.0:
                                        raw_qty = 55.0 / limit_price
                                        precision = 3 if limit_price > 10.0 else (1 if limit_price > 1.0 else 0)
                                        qty = max(qty, round(raw_qty, precision))
                                    testnet_order = await self.testnet_client.place_order(
                                        symbol=symbol,
                                        side="BUY" if primary_side == 1 else "SELL",
                                        qty=qty,
                                        price=limit_price,
                                        order_type="LIMIT",
                                        time_in_force="GTC",
                                        client_order_id=intent.cl_ord_id
                                    )
                                    order_id = testnet_order.get("orderId")
                                    exch_status = testnet_order.get("status", "UNKNOWN")
                                    self._log_event("TESTNET_ORDER_ROUTED", {
                                        "orderId": order_id,
                                        "status": exch_status,
                                        "clientOrderId": intent.cl_ord_id,
                                        "symbol": symbol,
                                        "side": "BUY" if primary_side == 1 else "SELL",
                                        "price": limit_price,
                                        "qty": qty,
                                        "raw_response": testnet_order
                                    })
                                    print(f"[TESTNET ORDER ROUTED] {symbol} orderId={order_id} | status={exch_status} | price={limit_price:.2f} | qty={qty}", flush=True)
                                except Exception as e:
                                    print(f"[TESTNET ORDER ROUTE ERROR] {symbol}: {e}", flush=True)
                                    self._log_event("TESTNET_ORDER_ROUTED", {
                                        "orderId": None,
                                        "status": f"ERROR_{e}",
                                        "clientOrderId": intent.cl_ord_id,
                                        "symbol": symbol,
                                        "side": "BUY" if primary_side == 1 else "SELL",
                                        "price": limit_price,
                                        "qty": decision.get("qty_btc", 0.001)
                                    })
                        elif dec_status == "FILTERED":
                            order_status = "FILTERED_BY_META_PROB"
                        else:
                            order_status = f"{decision.get('status')}_{decision.get('reason', '')}"
        except Exception as err:
            close_price = float(new_bar.get("close", 0.0))
            primary_side = 0
            cusum_dir = 0
            z_dist = 0.0
            rolling_vol = 0.0
            meta_prob = None
            order_status = f"ERROR_{err}"
            print(f"[EVALUATION ERROR] {symbol}: {err}", flush=True)

        self._log_event("SHADOW_SIGNAL_EVALUATED", {
            "symbol": symbol,
            "close_price": close_price,
            "primary_side": primary_side,
            "cusum_dir": cusum_dir,
            "z_dist": round(z_dist, 4),
            "rolling_vol": round(rolling_vol, 6),
            "meta_prob": round(meta_prob, 4) if (primary_side != 0 and meta_prob is not None) else None,
            "order_status": order_status,
        })
        print(f"[BAR EVALUATION] {symbol:<8} Close={close_price:<9.2f} | CUSUM={cusum_dir:+d} | z_dist={z_dist:+.2f} | Side={primary_side} | Prob={meta_prob} | Status={order_status}", flush=True)

    async def _background_deliberation_loop(self):
        """Continuously refreshes TauricResearch multi-agent debates for all 10 assets."""
        while True:
            await asyncio.sleep(40)
            for s in self.symbols:
                try:
                    price = self.last_prices.get(s, 0.0)
                    if price > 0:
                        await trading_agents_pipeline.deliberate_asset(
                            symbol=s,
                            current_price=price,
                            meta_prob=None,
                            rolling_vol=self.last_vols.get(s, 0.0015),
                            cusum_dir=self.last_cusums.get(s, 0),
                            z_dist=self.last_zdists.get(s, 0.0),
                            vpin_cdf=self.last_vpin_cdfs.get(s, 0.50),
                            net_order_flow=0.0
                        )
                except Exception:
                    pass

    async def run_forever(self):
        bg_delib_task = asyncio.create_task(self._background_deliberation_loop())
        while True:
            try:
                mode_label = "TESTNET" if self.is_testnet else "PRODUCTION"
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=15) as ws:
                    print(f"[CONNECTED] Multiplexing {len(self.symbols)} assets L2 Diff Depth + trades ({mode_label})...", flush=True)
                    # Trigger initial sync tasks for all symbols
                    for s in self.symbols:
                        asyncio.create_task(
                            self.ob_managers[s].initialize_and_sync(self._fetch_rest_snapshot, self._cancel_quotes_callback)
                        )

                    async for raw_msg in ws:
                        msg = json.loads(raw_msg)
                        stream = msg.get("stream", "")
                        data = msg.get("data", {})

                        if "@" not in stream:
                            continue

                        symbol = stream.split("@")[0].upper()
                        if symbol not in self.symbols:
                            continue

                        if "depth" in stream:
                            event = {
                                "U": data["U"],
                                "u": data["u"],
                                "pu": data.get("pu", -1),
                                "b": data["b"],
                                "a": data["a"],
                            }
                            ok = self.ob_managers[symbol].process_realtime_event(event)
                            if not ok and not self.ob_managers[symbol].sync_in_progress:
                                self.resync_counts[symbol] += 1
                                self._log_event("L2_BOOK_RESYNC", {
                                    "symbol": symbol,
                                    "resync_count": self.resync_counts[symbol]
                                })
                                asyncio.create_task(
                                    self.ob_managers[symbol].initialize_and_sync(self._fetch_rest_snapshot, self._cancel_quotes_callback)
                                )
                        elif "aggTrade" in stream or "trade" in stream:
                            trade_price = float(data["p"])
                            trade_qty = float(data["q"])
                            trade_ts = int(data["T"])
                            is_buyer_maker = bool(data["m"])
                            self.last_prices[symbol] = trade_price

                            # 1. Check intrabar exits
                            closed_trade = self.risk_engines[symbol].check_intrabar_exits(
                                trade_price, self.current_bar_idx[symbol], is_bar_close=False
                            )
                            if closed_trade is not None:
                                self._log_event("SHADOW_POSITION_CLOSED", {**closed_trade, "symbol": symbol})
                                print(
                                    f"[TRADE CLOSED (INTRABAR)] Symbol={symbol} Reason={closed_trade['exit_reason']} | "
                                    f"ExitPrice={closed_trade['exit_price']} | NetReturn={closed_trade['net_return_bps']}bps | "
                                    f"PnL=${closed_trade['pnl_usd']:.2f} | Equity=${closed_trade['updated_equity']:.2f}",
                                    flush=True
                                )

                            # 2. Accumulate tick in Dollar Bar engine
                            bar = self.bar_engines[symbol].process_agg_trade(
                                timestamp_ms=trade_ts,
                                price=trade_price,
                                qty=trade_qty,
                                is_buyer_maker=is_buyer_maker
                            )
                            if bar is not None:
                                self.current_bar_idx[symbol] += 1
                                bar["symbol"] = symbol
                                self._record_live_dollar_bar_csv(symbol, bar)

                                # 3. Check exits on Dollar Bar close (vertical time barrier)
                                closed_trade_bar = self.risk_engines[symbol].check_intrabar_exits(
                                    float(bar["close"]), self.current_bar_idx[symbol], is_bar_close=True
                                )
                                if closed_trade_bar is not None:
                                    self._log_event("SHADOW_POSITION_CLOSED", {**closed_trade_bar, "symbol": symbol})
                                    print(
                                        f"[TRADE CLOSED (BAR CLOSE)] Symbol={symbol} Reason={closed_trade_bar['exit_reason']} | "
                                        f"ExitPrice={closed_trade_bar['exit_price']} | NetReturn={closed_trade_bar['net_return_bps']}bps | "
                                        f"PnL=${closed_trade_bar['pnl_usd']:.2f} | Equity=${closed_trade_bar['updated_equity']:.2f}",
                                        flush=True
                                    )

                                # 4. Evaluate new Dollar Bar
                                await self._evaluate_new_dollar_bar(symbol, bar)

            except Exception as e:
                self._log_event("WS_RECONNECT", {"error": str(e)})
                await asyncio.sleep(2.0)


def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Live Shadow & Binance Futures Testnet Execution Trial")
    parser.add_argument("--testnet", action="store_true", help="Route execution to Binance Futures Testnet")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols list (e.g. BTCUSDT,ETHUSDT)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    runner = LiveShadowTrialRunner(symbols=symbols, is_testnet=args.testnet)
    asyncio.run(runner.run_forever())


if __name__ == "__main__":
    main()
