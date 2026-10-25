import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List, Dict, Optional

class DollarBarAndMicrostructureEngine:
    """
    Aggregates raw Binance Futures aggTrades into Information-Driven Dollar Bars
    and computes real-time VPIN order flow toxicity and Kyle's Lambda.
    """
    def __init__(self, dollar_threshold: float = 50_000_000.0, vpin_bucket_vol: float = 500.0, vpin_window: int = 50):
        self.dollar_threshold = dollar_threshold
        self.vpin_bucket_vol = vpin_bucket_vol
        self.vpin_window = vpin_window
        
        # Running Dollar Bar state
        self._reset_bar_state()
        self.completed_bars: List[Dict] = []

    def _reset_bar_state(self):
        self.open_p: Optional[float] = None
        self.high_p: float = -np.inf
        self.low_p: float = np.inf
        self.close_p: Optional[float] = None
        self.cum_vol: float = 0.0
        self.cum_dollar: float = 0.0
        self.signed_vol_imbalance: float = 0.0
        self.tick_count: int = 0
        self.open_ts: Optional[int] = None

    def process_agg_trade(self, timestamp_ms: int, price: float, qty: float, is_buyer_maker: bool) -> Optional[Dict]:
        """
        Processes a single Binance BTCUSDT aggTrade event.
        Note: On Binance, is_buyer_maker=True means the aggressor was a SELLER (hit the bid).
        """
        dollar_val = price * qty
        sign = -1.0 if is_buyer_maker else 1.0

        if self.open_p is None:
            self.open_p = price
            self.open_ts = timestamp_ms

        self.high_p = max(self.high_p, price)
        self.low_p = min(self.low_p, price)
        self.close_p = price
        self.cum_vol += qty
        self.cum_dollar += dollar_val
        self.signed_vol_imbalance += sign * qty
        self.tick_count += 1

        if self.cum_dollar >= self.dollar_threshold:
            bar = {
                "open_ts": self.open_ts,
                "close_ts": timestamp_ms,
                "open": self.open_p,
                "high": self.high_p,
                "low": self.low_p,
                "close": self.close_p,
                "volume": self.cum_vol,
                "dollar_volume": self.cum_dollar,
                "net_order_flow": self.signed_vol_imbalance,
                "ticks": self.tick_count,
            }
            self.completed_bars.append(bar)
            self._reset_bar_state()
            return bar
        return None

    @staticmethod
    def compute_vpin_and_kyle_lambda(bars_df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """
        Computes Bulk Volume Classification (BVC) VPIN and rolling Kyle's Lambda
        across completed Dollar Bars.
        """
        df = bars_df.copy()
        df["delta_p"] = df["close"].diff()
        df["sigma_dp"] = df["delta_p"].rolling(window=window, min_periods=10).std().replace(0, 1e-8)

        # Bulk Volume Classification (BVC): V_B = V * Phi(dP / sigma_dP)
        z_score = (df["delta_p"] / df["sigma_dp"]).fillna(0.0)
        df["v_buy"] = df["volume"] * norm.cdf(z_score)
        df["v_sell"] = df["volume"] - df["v_buy"]
        df["abs_imbalance"] = (df["v_buy"] - df["v_sell"]).abs()

        # VPIN = sum(|V_B - V_S|) / sum(V) over rolling window N
        rolling_imb = df["abs_imbalance"].rolling(window=window, min_periods=10).sum()
        rolling_vol = df["volume"].rolling(window=window, min_periods=10).sum()
        df["vpin"] = (rolling_imb / rolling_vol).clip(0.0, 1.0)
        df["vpin_cdf"] = df["vpin"].expanding(min_periods=window).rank(pct=True)

        # Empirical Kyle's Lambda proxy: |dP| / |Net Order Flow| smoothed over window
        abs_flow = df["net_order_flow"].abs().replace(0, 1e-8)
        df["kyle_lambda"] = (df["delta_p"].abs() / abs_flow).rolling(window=window, min_periods=10).mean()
        return df