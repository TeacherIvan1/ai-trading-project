"""
main_shadow_trial.py
====================
Multi-Asset Live Zero-Risk Shadow Execution & Binance Futures Testnet Engine.
Supports the 10-asset major crypto universe:
["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]

Guarantees 100% strict per-symbol state isolation across:
- LocalOrderBookManager
- DollarBarAndMicrostructureEngine ($5M for BTC/ETH, $1M for altcoins)
- Clean historical warmup bars fetched directly via Binance REST /fapi/v1/klines per symbol
- SynchronousPreTradeRiskAndPaperEngine
- TauricResearch / TradingAgents 5-Phase Consensus Deliberation
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
    "BTCUSDT": 5_000_000.0,
    "ETHUSDT": 5_000_000.0,
    "SOLUSDT": 1_000_000.0,
    "BNBUSDT": 1_000_000.0,
    "XRPUSDT": 1_000_000.0,
    "ADAUSDT": 1_000_000.0,
    "DOGEUSDT": 1_000_000.0,
    "AVAXUSDT": 1_000_000.0,
    "DOTUSDT": 1_000_000.0,
    "LINKUSDT": 1_000_000.0,
}

QTY_PRECISIONS = {
    "BTCUSDT": 3,
    "ETHUSDT": 3,
    "SOLUSDT": 2,
    "BNBUSDT": 2,
    "XRPUSDT": 1,
    "ADAUSDT": 0,
    "DOGEUSDT": 0,
    "AVAXUSDT": 1,
    "DOTUSDT": 1,
    "LINKUSDT": 2,
}


def format_order_qty(symbol: str, raw_qty: float) -> float:
    p = QTY_PRECISIONS.get(symbol, 2)
    return round(raw_qty, p) if p > 0 else float(max(1, int(raw_qty)))


class LiveShadowTrialRunner:
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        trained_rf_model=None,
        feature_cols: list = None,
        initial_equity: float = 2500.0,
        is_testnet: bool = False,
    ):
        self.symbols = [s.strip().upper() for s in (symbols or DEFAULT_UNIVERSE)]
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        self.is_testnet = is_testnet

        # Check for environment override on dollar threshold
        env_threshold_override = os.environ.get("DOLLAR_BAR_THRESHOLD")
        if env_threshold_override:
            try:
                ov = float(env_threshold_override)
                for s in self.symbols:
                    self.thresholds[s] = ov if s in ("BTCUSDT", "ETHUSDT") else max(50000.0, ov * 0.20)
            except Exception:
                pass

        if self.is_testnet:
            self.rest_base_url = "https://testnet.binancefuture.com"
            self.ws_base_url = "wss://stream.binancefuture.com/stream?streams="
            self.testnet_client = BinanceFuturesTestnetClient()
            print(f"[INIT] Binance Futures TESTNET execution across {len(self.symbols)} symbols.", flush=True)
        else:
            self.rest_base_url = "https://fapi.binance.com"
            self.ws_base_url = "wss://fstream.binance.com/stream?streams="
            self.testnet_client = None
            print(f"[INIT] Live Shadow (Production Feed) across {len(self.symbols)} symbols.", flush=True)

        # Build multiplexed stream URL
        stream_names = []
        for s in self.symbols:
            s_low = s.lower()
            stream_names.append(f"{s_low}@depth@100ms")
            stream_names.append(f"{s_low}@aggTrade")
        self.ws_url = self.ws_base_url + "/".join(stream_names)

        # 1. Isolate Per-Symbol State in Dictionaries
        self.ob_managers: Dict[str, LocalOrderBookManager] = {s: LocalOrderBookManager(s) for s in self.symbols}
        self.bar_engines: Dict[str, DollarBarAndMicrostructureEngine] = {
            s: DollarBarAndMicrostructureEngine(dollar_threshold=self.thresholds.get(s, 1_000_000.0))
            for s in self.symbols
        }
        self.risk_engine = SynchronousPreTradeRiskAndPaperEngine(
            initial_equity=initial_equity,
            pt_mult=1.5,
            sl_mult=1.5,
            max_horizon_bars=25,
            maker_fee_bps=2.0,
            taker_fee_bps=5.0
        )
        self.risk_engines = {s: self.risk_engine for s in self.symbols}

        # Setup logs
        self.logs_dir = PROJECT_ROOT / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path = self.logs_dir / "shadow_trial_audit.jsonl"
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

        # 2. Fetch clean symbol-specific historical warmup bars (200 klines per symbol)
        self.bars_df: Dict[str, pd.DataFrame] = {}
        self.bars_dfs = self.bars_df  # Alias for compatibility
        self.current_bar_idx: Dict[str, int] = {}
        self.resync_counts: Dict[str, int] = {s: 0 for s in self.symbols}
        self.order_seq = 0

        self.last_prices: Dict[str, float] = {}
        self.last_vols: Dict[str, float] = {}
        self.last_cusums: Dict[str, int] = {}
        self.last_zdists: Dict[str, float] = {}
        self.last_vpin_cdfs: Dict[str, float] = {}

        self.feature_cols = feature_cols if feature_cols is not None else [
            "vpin", "vpin_cdf", "vpin_slope", "kyle_lambda_z", "ffd_close",
            "rolling_vol", "vol_regime_ratio", "flow_alignment", "abs_z_dist", "ret_autocorr"
        ]

        print(f"[WARMUP] Fetching 200 clean 1m historical klines per symbol via Binance REST...", flush=True)
        for s in self.symbols:
            df_warmup = self._fetch_warmup_klines(s)
            self.bars_df[s] = df_warmup
            self.current_bar_idx[s] = len(df_warmup)

            # Pre-evaluate micro indicators & signals on isolated symbol data
            try:
                df_micro = self.bar_engines[s].compute_vpin_and_kyle_lambda(df_warmup, window=50)
                df_sig = generate_primary_signals(df_micro, span=40, cusum_mult=1.0, mode="IGNITION")
                latest = df_sig.iloc[-1]
                self.last_prices[s] = float(latest["close"])
                self.last_vols[s] = float(latest.get("rolling_vol", 0.0015))
                self.last_cusums[s] = int(latest.get("cusum_dir", 0))
                self.last_zdists[s] = float(latest.get("z_dist", 0.0))
                self.last_vpin_cdfs[s] = float(latest.get("vpin_cdf", 0.50))
                print(f"  -> {s:<8}: Close={self.last_prices[s]:<9.2f} Vol={self.last_vols[s]:.6f} z_dist={self.last_zdists[s]:+.2f}σ", flush=True)
            except Exception as e:
                print(f"[WARMUP ERROR] Failed indicators for {s}: {e}", flush=True)

        self._log_event("BOT_SESSION_STARTED", {
            "symbols": self.symbols,
            "mode": "TESTNET" if self.is_testnet else "PRODUCTION",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        print(f"[WARMUP COMPLETE] All 10 symbols initialized with clean price history.", flush=True)

    def _fetch_warmup_klines(self, symbol: str) -> pd.DataFrame:
        """Fetches 200 clean historical 1m klines from Binance REST."""
        urls = [
            f"{self.rest_base_url}/fapi/v1/klines?symbol={symbol}&interval=1m&limit=200",
            f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1m&limit=200"
        ]
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=6) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if isinstance(data, list) and len(data) >= 50:
                        rows = []
                        for k in data:
                            vol = float(k[5])
                            taker_buy_vol = float(k[9])
                            net_flow = 2.0 * taker_buy_vol - vol
                            rows.append({
                                "open_ts": int(k[0]),
                                "close_ts": int(k[6]),
                                "open": float(k[1]),
                                "high": float(k[2]),
                                "low": float(k[3]),
                                "close": float(k[4]),
                                "volume": vol,
                                "dollar_volume": float(k[7]),
                                "net_order_flow": net_flow,
                                "ticks": int(k[8]),
                            })
                        return pd.DataFrame(rows)
            except Exception:
                continue

        # Fallback to model warmup if available (BTC only)
        models_warmup = PROJECT_ROOT / "models" / "warmup_bars.parquet"
        if symbol == "BTCUSDT" and models_warmup.exists():
            return pd.read_parquet(models_warmup).tail(200).reset_index(drop=True)

        raise RuntimeError(f"Could not fetch historical warmup klines for {symbol}")

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

    async def _evaluate_new_dollar_bar(self, symbol: str, new_bar: dict, is_initial: bool = False):
        """
        Evaluates signals and executes trades strictly on self.bars_df[symbol].
        Wires TauricResearch/TradingAgents Consensus + IGNITION execution.
        """
        if not is_initial:
            self._log_event("DOLLAR_BAR_CLOSED", {
                "symbol": symbol,
                "close": float(new_bar["close"]),
                "dollar_vol": float(new_bar.get("dollar_volume", 0.0)),
                "ticks": int(new_bar.get("ticks", 0))
            })

        try:
            bar_entry = {k: v for k, v in new_bar.items() if k != "symbol"}
            if not is_initial:
                self.bars_df[symbol] = pd.concat([self.bars_df[symbol], pd.DataFrame([bar_entry])], ignore_index=True).tail(300)

            # Strictly evaluate on isolated symbol dataframe
            df_micro = self.bar_engines[symbol].compute_vpin_and_kyle_lambda(self.bars_df[symbol], window=50)
            df_sig = generate_primary_signals(df_micro, span=40, cusum_mult=1.0, mode="IGNITION")
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

            # Compute Meta-Labeling RF probability if model features exist
            meta_prob = None
            try:
                X_live = df_sig[self.feature_cols].iloc[[-1]].fillna(0.0)
                meta_prob = float(self.model.predict_proba(X_live)[0, 1])
            except Exception:
                meta_prob = None

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

            # Execution Gate Logic: EITHER
            # (a) primary_side != 0 and meta_prob >= 0.56
            # OR
            # (b) TradingAgents Phase 5 PM Sign-Off APPROVED (conviction >= 0.70) and cusum_dir aligns
            ign_trigger = (primary_side != 0 and meta_prob is not None and meta_prob >= 0.56)
            ta_trigger = (
                deliberation.final_verdict == "APPROVED"
                and deliberation.conviction_score >= 0.70
                and (
                    (deliberation.proposal.action == "BUY" and cusum_dir >= 0)
                    or (deliberation.proposal.action == "SELL" and cusum_dir <= 0)
                )
            )

            order_status = "NO_PRIMARY_BREAKOUT"
            if ign_trigger or ta_trigger:
                target_side = primary_side if ign_trigger else (1 if deliberation.proposal.action == "BUY" else -1)
                effective_prob = meta_prob if ign_trigger else deliberation.conviction_score
                trigger_reason = "IGNITION_BREAKOUT" if ign_trigger else "TRADING_AGENTS_PM_APPROVED"

                ob_manager = self.ob_managers[symbol]
                best_bid = max(ob_manager.bids.keys()) if ob_manager.bids else 0.0
                best_ask = min(ob_manager.asks.keys()) if ob_manager.asks else 0.0
                mid_price = 0.5 * (best_bid + best_ask) if (best_bid > 0 and best_ask > 0) else close_price
                limit_price = best_bid if (target_side == 1 and best_bid > 0) else (best_ask if (target_side == -1 and best_ask > 0) else close_price)

                self.order_seq += 1
                intent = OrderIntent(
                    cl_ord_id=f"SHADOW_{symbol}_{int(time.time())}_{self.order_seq}",
                    symbol=symbol,
                    side=target_side,
                    limit_price=limit_price,
                    mid_price=mid_price,
                    meta_prob=effective_prob,
                    vpin_cdf=vpin_cdf,
                    rolling_vol=rolling_vol
                )
                decision = self.risk_engine.validate_and_route_shadow_order(intent, self.current_bar_idx[symbol])
                dec_status = decision.get("status")

                if dec_status in ("OPENED_SHADOW_POSITION", "ACKNOWLEDGED_PAPER"):
                    order_status = dec_status

                    # Log SHADOW_POSITION_OPENED
                    self._log_event("SHADOW_POSITION_OPENED", {
                        "symbol": symbol,
                        "cl_ord_id": intent.cl_ord_id,
                        "side": target_side,
                        "entry_price": limit_price,
                        "pt_price": decision.get("pt_price"),
                        "sl_price": decision.get("sl_price"),
                        "qty": decision.get("qty_btc"),
                        "notional_usd": decision.get("notional_usd"),
                        "meta_prob": effective_prob,
                        "trigger_reason": trigger_reason,
                    })
                    print(
                        f"[POSITION OPENED] {symbol:<8} Side={'BUY' if target_side == 1 else 'SELL'} "
                        f"Price={limit_price:.2f} Notional=${decision.get('notional_usd', 0):.2f} "
                        f"Reason={trigger_reason}", flush=True
                    )

                    # Route order to Binance Futures Testnet if in testnet mode
                    if self.testnet_client:
                        try:
                            raw_qty = float(decision.get("qty_btc", 0.001))
                            if raw_qty * limit_price < 50.0:
                                raw_qty = 55.0 / limit_price
                            qty = format_order_qty(symbol, raw_qty)

                            testnet_order = await self.testnet_client.place_order(
                                symbol=symbol,
                                side="BUY" if target_side == 1 else "SELL",
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
                                "side": "BUY" if target_side == 1 else "SELL",
                                "price": limit_price,
                                "qty": qty,
                                "raw_response": testnet_order
                            })
                            print(f"[TESTNET ORDER ROUTED] {symbol} orderId={order_id} | status={exch_status} | price={limit_price:.2f} | qty={qty}", flush=True)
                        except Exception as e:
                            print(f"[TESTNET ROUTE ERROR] {symbol}: {e}", flush=True)
                            self._log_event("TESTNET_ORDER_ROUTED", {
                                "orderId": None,
                                "status": f"ERROR_{e}",
                                "clientOrderId": intent.cl_ord_id,
                                "symbol": symbol,
                                "side": "BUY" if target_side == 1 else "SELL",
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
            "meta_prob": round(meta_prob, 4) if (meta_prob is not None) else None,
            "order_status": order_status,
        })
        print(f"[BAR EVALUATION] {symbol:<8} Close={close_price:<9.2f} | CUSUM={cusum_dir:+d} | z_dist={z_dist:+.2f} | Side={primary_side} | Prob={meta_prob} | Status={order_status}", flush=True)

    async def trigger_initial_evaluations(self):
        """Evaluates all 10 warmed-up symbols immediately so qualifying signals execute on startup."""
        print(f"[STARTUP] Triggering initial evaluation across 10 warmed-up symbols...", flush=True)
        for s in self.symbols:
            try:
                if len(self.bars_df[s]) > 0:
                    latest_bar = self.bars_df[s].iloc[-1].to_dict()
                    await self._evaluate_new_dollar_bar(s, latest_bar, is_initial=True)
            except Exception as e:
                print(f"[STARTUP EVAL WARNING] {s}: {e}", flush=True)

    async def _background_deliberation_loop(self):
        """Continuously refreshes TradingAgents consensus theses across all symbols."""
        while True:
            await asyncio.sleep(45)
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
        asyncio.create_task(self._background_deliberation_loop())
        while True:
            try:
                mode_label = "TESTNET" if self.is_testnet else "PRODUCTION"
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=15) as ws:
                    print(f"[CONNECTED] Multiplexing {len(self.symbols)} assets L2 Diff Depth + trades ({mode_label})...", flush=True)
                    # Trigger initial orderbook snapshot sync
                    for s in self.symbols:
                        asyncio.create_task(
                            self.ob_managers[s].initialize_and_sync(self._fetch_rest_snapshot, self._cancel_quotes_callback)
                        )

                    # Trigger immediate startup evaluations across all 10 symbols
                    await asyncio.sleep(2.0)
                    await self.trigger_initial_evaluations()

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

                            # 1. Check intrabar exits isolated to this symbol
                            closed_trade = self.risk_engine.check_intrabar_exits(
                                trade_price, self.current_bar_idx[symbol], is_bar_close=False, symbol=symbol
                            )
                            if closed_trade is not None:
                                self._log_event("SHADOW_POSITION_CLOSED", {**closed_trade, "symbol": symbol})
                                print(
                                    f"[TRADE CLOSED (INTRABAR)] Symbol={symbol} Reason={closed_trade['exit_reason']} | "
                                    f"ExitPrice={closed_trade['exit_price']} | NetReturn={closed_trade['net_return_bps']}bps | "
                                    f"PnL=${closed_trade['pnl_usd']:.2f} | Equity=${closed_trade['updated_equity']:.2f}",
                                    flush=True
                                )

                            # 2. Accumulate tick in Dollar Bar engine for THIS symbol
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

                                # 3. Check exits on Dollar Bar close
                                closed_trade_bar = self.risk_engine.check_intrabar_exits(
                                    float(bar["close"]), self.current_bar_idx[symbol], is_bar_close=True, symbol=symbol
                                )
                                if closed_trade_bar is not None:
                                    self._log_event("SHADOW_POSITION_CLOSED", {**closed_trade_bar, "symbol": symbol})
                                    print(
                                        f"[TRADE CLOSED (BAR CLOSE)] Symbol={symbol} Reason={closed_trade_bar['exit_reason']} | "
                                        f"ExitPrice={closed_trade_bar['exit_price']} | NetReturn={closed_trade_bar['net_return_bps']}bps | "
                                        f"PnL=${closed_trade_bar['pnl_usd']:.2f} | Equity=${closed_trade_bar['updated_equity']:.2f}",
                                        flush=True
                                    )

                                # 4. Evaluate new Dollar Bar strictly for this symbol
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
