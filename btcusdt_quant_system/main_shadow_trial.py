"""
main_shadow_trial.py
====================
Live zero-risk shadow execution trial connected to Binance Futures (BTCUSDT).
Streams real-time L2 order book updates (@depth@100ms) and trades (@trade / @aggTrade).
Executes SEC 15c3-5 style synchronous pre-trade risk checks (<100us),
evaluates live Dollar Bars against the trained Meta-Labeling RandomForest model,
and manages active paper positions with closed-loop Triple-Barrier exits
(PT limit fills, SL stops, and vertical time horizon).
"""

import asyncio
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import websockets
import pandas as pd
import pickle

# Force line buffering on standard output and error so daemon.log updates immediately
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.orderbook_sync import LocalOrderBookManager
from core.data_bars import DollarBarAndMicrostructureEngine
from core.features_labels import generate_primary_signals
from execution.paper_engine import SynchronousPreTradeRiskAndPaperEngine, OrderIntent

BINANCE_WS_URL = "wss://fstream.binance.com/stream?streams=btcusdt@depth@100ms/btcusdt@trade/btcusdt@aggTrade"
BINANCE_REST_DEPTH = "https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=1000"


async def fetch_rest_snapshot(symbol: str) -> dict:
    def _get():
        with urllib.request.urlopen(BINANCE_REST_DEPTH, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    return await asyncio.to_thread(_get)


async def cancel_quotes_callback(symbol: str) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    print(f"[{now_iso}] [SAFETY] Cancelling resting shadow quotes for {symbol} during L2 resync.", flush=True)


class LiveShadowTrialRunner:
    def __init__(
        self,
        trained_rf_model=None,
        feature_cols: list = None,
        historical_warmup_bars: pd.DataFrame = None,
        initial_equity: float = 2500.0,
        dollar_bar_threshold: float = None,
    ):
        self.symbol = "BTCUSDT"
        self.ob_manager = LocalOrderBookManager(self.symbol)
        
        # Dollar Bar threshold (configurable via env var or argument; defaults to $200k for live formation or $20M)
        if dollar_bar_threshold is None:
            dollar_bar_threshold = float(os.environ.get("DOLLAR_BAR_THRESHOLD", 200_000.0))
        self.dollar_bar_threshold = dollar_bar_threshold
        self.bar_engine = DollarBarAndMicrostructureEngine(dollar_threshold=self.dollar_bar_threshold)
        
        self.risk_engine = SynchronousPreTradeRiskAndPaperEngine(
            initial_equity=initial_equity,
            pt_mult=1.5,
            sl_mult=1.5,
            max_horizon_bars=25,
            maker_fee_bps=2.0,
            taker_fee_bps=5.0
        )

        models_dir = PROJECT_ROOT / "models"
        if trained_rf_model is None:
            model_path = models_dir / "ignition_meta_rf.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Trained model not found at {model_path}. Run Task 1 calibration first.")
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            self.model = trained_rf_model

        # Step 1: Historical Warmup Loading (ensure at least 200 bars in self.bars_df)
        warmup_path = models_dir / "warmup_bars.parquet"
        loaded = False

        if historical_warmup_bars is not None and len(historical_warmup_bars) >= 200:
            self.bars_df = historical_warmup_bars.copy().tail(300).reset_index(drop=True)
            loaded = True
        elif warmup_path.exists():
            try:
                df_warmup = pd.read_parquet(warmup_path)
                if len(df_warmup) >= 200:
                    self.bars_df = df_warmup.tail(300).reset_index(drop=True)
                    loaded = True
            except Exception:
                loaded = False

        if not loaded:
            candidate_paths = [
                PROJECT_ROOT / "data" / "btcusdt_dollar_bars_20m_30d.parquet",
                PROJECT_ROOT / "data" / "btcusdt_dollar_bars_20m_30d.csv",
                PROJECT_ROOT / "data" / "btcusdt_dollar_bars_30d.parquet",
                PROJECT_ROOT / "data" / "btcusdt_dollar_bars_30d.csv",
            ]
            for cand_path in candidate_paths:
                if cand_path.exists():
                    try:
                        df_cand = pd.read_parquet(cand_path) if cand_path.suffix == ".parquet" else pd.read_csv(cand_path)
                        if len(df_cand) >= 200:
                            self.bars_df = df_cand.tail(300).reset_index(drop=True)
                            models_dir.mkdir(parents=True, exist_ok=True)
                            self.bars_df.to_parquet(warmup_path, index=False)
                            loaded = True
                            break
                    except Exception:
                        continue

        if not loaded:
            raise FileNotFoundError("Could not load or generate at least 200 historical warmup bars from models/ or data/ caches.")

        if len(self.bars_df) < 200:
            raise ValueError(f"Loaded insufficient warmup bars: {len(self.bars_df)} < 200")

        self.feature_cols = feature_cols if feature_cols is not None else [
            "vpin", "vpin_cdf", "vpin_slope", "kyle_lambda_z", "ffd_close",
            "rolling_vol", "vol_regime_ratio", "flow_alignment", "abs_z_dist", "ret_autocorr"
        ]
        self.current_bar_idx = len(self.bars_df)
        self.resync_count = 0
        self.order_seq = 0

        self.logs_dir = PROJECT_ROOT / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path = self.logs_dir / "shadow_trial_audit.jsonl"
        self.audit_log_path.touch(exist_ok=True)
        self.live_bars_csv_path = self.logs_dir / "live_dollar_bars.csv"
        if not self.live_bars_csv_path.exists():
            with open(self.live_bars_csv_path, "w", encoding="utf-8") as f:
                f.write("open_ts,close_ts,open,high,low,close,volume,dollar_volume,net_order_flow,ticks\n")

        # Warmup indicators and JIT compilation to avoid lag on first live bar
        try:
            df_micro_warmup = self.bar_engine.compute_vpin_and_kyle_lambda(self.bars_df, window=50)
            df_sig_warmup = generate_primary_signals(df_micro_warmup, span=40, cusum_mult=1.35, mode="IGNITION")
            print(f"[WARMUP COMPLETE] Loaded {len(self.bars_df)} warmup bars. Indicators initialized (NaN count in latest bar: {int(df_sig_warmup.iloc[-1].isna().sum())}).", flush=True)
        except Exception as e:
            print(f"[WARMUP WARNING] Indicator warmup initialization exception: {e}", flush=True)

    def _log_event(self, event_type: str, payload: dict):
        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **payload
        }
        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _record_live_dollar_bar_csv(self, bar: dict):
        line = (
            f"{bar['open_ts']},{bar['close_ts']},{bar['open']:.2f},{bar['high']:.2f},"
            f"{bar['low']:.2f},{bar['close']:.2f},{bar['volume']:.6f},{bar['dollar_volume']:.2f},"
            f"{bar['net_order_flow']:.6f},{bar['ticks']}\n"
        )
        with open(self.live_bars_csv_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _evaluate_new_dollar_bar(self, new_bar: dict):
        # 1. Log DOLLAR_BAR_CLOSED
        self._log_event("DOLLAR_BAR_CLOSED", {
            "close": float(new_bar["close"]),
            "dollar_vol": float(new_bar.get("dollar_volume", new_bar.get("notional", 0.0))),
            "ticks": int(new_bar.get("ticks", 0))
        })

        try:
            self.bars_df = pd.concat([self.bars_df, pd.DataFrame([new_bar])], ignore_index=True).tail(300)
            df_micro = self.bar_engine.compute_vpin_and_kyle_lambda(self.bars_df, window=50)
            df_sig = generate_primary_signals(df_micro, span=40, cusum_mult=1.35, mode="IGNITION")
            latest = df_sig.iloc[-1]

            close_price = float(latest["close"])
            primary_side = int(latest["primary_side"])
            cusum_dir = int(latest.get("cusum_dir", 0))
            z_dist = float(latest.get("z_dist", 0.0))
            rolling_vol = float(latest.get("rolling_vol", 0.0))

            meta_prob = None
            order_status = "NO_PRIMARY_BREAKOUT"

            if primary_side != 0:
                if not self.ob_manager.is_synced:
                    order_status = "ORDERBOOK_DESYNCED"
                else:
                    X_live = df_sig[self.feature_cols].iloc[[-1]].fillna(0.0)
                    meta_prob = float(self.model.predict_proba(X_live)[0, 1])

                    best_bid = max(self.ob_manager.bids.keys()) if self.ob_manager.bids else 0.0
                    best_ask = min(self.ob_manager.asks.keys()) if self.ob_manager.asks else 0.0
                    mid_price = 0.5 * (best_bid + best_ask) if (best_bid and best_ask) else close_price
                    limit_price = best_bid if primary_side == 1 else best_ask

                    self.order_seq += 1
                    intent = OrderIntent(
                        cl_ord_id=f"SHADOW_{int(time.time())}_{self.order_seq}",
                        symbol=self.symbol,
                        side=primary_side,
                        limit_price=limit_price,
                        mid_price=mid_price,
                        meta_prob=meta_prob,
                        vpin_cdf=float(latest.get("vpin_cdf", 0.5)),
                        rolling_vol=rolling_vol
                    )
                    decision = self.risk_engine.validate_and_route_shadow_order(intent, self.current_bar_idx)
                    if decision.get("status") == "OPENED_SHADOW_POSITION":
                        order_status = "OPENED_SHADOW_POSITION"
                    elif decision.get("status") == "FILTERED":
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
            print(f"[EVALUATION ERROR] {err}", flush=True)

        # Step 2: Always Log Diagnostic Signal Checks
        self._log_event("SHADOW_SIGNAL_EVALUATED", {
            "close_price": close_price,
            "primary_side": primary_side,
            "cusum_dir": cusum_dir,
            "z_dist": round(z_dist, 4),
            "rolling_vol": round(rolling_vol, 6),
            "meta_prob": round(meta_prob, 4) if (primary_side != 0 and meta_prob is not None) else None,
            "order_status": order_status,
        })
        print(f"[BAR EVALUATION] Close={close_price:.2f} | CUSUM={cusum_dir:+d} | z_dist={z_dist:+.2f} | Side={primary_side} | Prob={meta_prob} | Status={order_status}", flush=True)

    async def run_forever(self):
        while True:
            try:
                async with websockets.connect(BINANCE_WS_URL, ping_interval=20, ping_timeout=10) as ws:
                    print(f"[CONNECTED] Streaming live {self.symbol} L2 Diff Depth + trades...", flush=True)
                    sync_task = asyncio.create_task(
                        self.ob_manager.initialize_and_sync(fetch_rest_snapshot, cancel_quotes_callback)
                    )
                    async for raw_msg in ws:
                        msg = json.loads(raw_msg)
                        stream = msg.get("stream", "")
                        data = msg.get("data", {})

                        if "depth" in stream:
                            event = {
                                "U": data["U"],
                                "u": data["u"],
                                "pu": data.get("pu", -1),
                                "b": data["b"],
                                "a": data["a"],
                            }
                            ok = self.ob_manager.process_realtime_event(event)
                            if not ok and not self.ob_manager.sync_in_progress:
                                self.resync_count += 1
                                self._log_event("L2_BOOK_RESYNC", {"resync_count": self.resync_count})
                                sync_task = asyncio.create_task(
                                    self.ob_manager.initialize_and_sync(fetch_rest_snapshot, cancel_quotes_callback)
                                )
                        elif "trade" in stream or "aggTrade" in stream:
                            trade_price = float(data["p"])
                            trade_qty = float(data["q"])
                            trade_ts = int(data["T"])
                            is_buyer_maker = bool(data["m"])

                            # 1. Check intrabar exits on every live trade (is_bar_close=False)
                            closed_trade = self.risk_engine.check_intrabar_exits(
                                trade_price, self.current_bar_idx, is_bar_close=False
                            )
                            if closed_trade is not None:
                                self._log_event("SHADOW_POSITION_CLOSED", closed_trade)
                                print(
                                    f"[TRADE CLOSED (INTRABAR)] Reason={closed_trade['exit_reason']} | "
                                    f"ExitPrice={closed_trade['exit_price']} | NetReturn={closed_trade['net_return_bps']}bps | "
                                    f"PnL=${closed_trade['pnl_usd']:.2f} | Equity=${closed_trade['updated_equity']:.2f}",
                                    flush=True
                                )

                            # 2. Accumulate tick in Dollar Bar engine
                            bar = self.bar_engine.process_agg_trade(
                                timestamp_ms=trade_ts,
                                price=trade_price,
                                qty=trade_qty,
                                is_buyer_maker=is_buyer_maker
                            )
                            if bar is not None:
                                self.current_bar_idx += 1
                                self._record_live_dollar_bar_csv(bar)

                                # 3. Check exits on Dollar Bar close (vertical time barrier)
                                closed_trade_bar = self.risk_engine.check_intrabar_exits(
                                    float(bar["close"]), self.current_bar_idx, is_bar_close=True
                                )
                                if closed_trade_bar is not None:
                                    self._log_event("SHADOW_POSITION_CLOSED", closed_trade_bar)
                                    print(
                                        f"[TRADE CLOSED (BAR CLOSE)] Reason={closed_trade_bar['exit_reason']} | "
                                        f"ExitPrice={closed_trade_bar['exit_price']} | NetReturn={closed_trade_bar['net_return_bps']}bps | "
                                        f"PnL=${closed_trade_bar['pnl_usd']:.2f} | Equity=${closed_trade_bar['updated_equity']:.2f}",
                                        flush=True
                                    )

                                # 4. Evaluate new Dollar Bar for primary signal & shadow order routing
                                self._evaluate_new_dollar_bar(bar)

            except Exception as e:
                self._log_event("WS_RECONNECT", {"error": str(e)})
                await asyncio.sleep(2.0)


def main():
    runner = LiveShadowTrialRunner()
    asyncio.run(runner.run_forever())


if __name__ == "__main__":
    main()
