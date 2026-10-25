import numpy as np
import pandas as pd
from numba import njit

def frac_diff_ffd(series: pd.Series, d: float = 0.35, thres: float = 1e-3) -> pd.Series:
    """Fixed-Width Window Fractional Differentiation (FFD) for stationarity with memory."""
    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        weights.append(w_k)
        k += 1
    weights = np.array(weights[::-1]).reshape(-1, 1)
    width = len(weights)
    vals = series.values
    out = np.full(len(vals), np.nan)
    for i in range(width - 1, len(vals)):
        window = vals[i - width + 1 : i + 1]
        if not np.isnan(window).any():
            out[i] = np.dot(weights.T, window)[0]
    return pd.Series(out, index=series.index)

@njit
def _cusum_signed_numba(diff_vals: np.ndarray, h_arr: np.ndarray) -> np.ndarray:
    """
    Signed Symmetric CUSUM filter:
    Returns +1 on upward structural break, -1 on downward structural break, 0 otherwise.
    """
    n = len(diff_vals)
    events = np.zeros(n, dtype=np.int8)
    s_pos = 0.0
    s_neg = 0.0
    for i in range(1, n):
        dy = diff_vals[i]
        if np.isnan(dy) or np.isnan(h_arr[i]) or h_arr[i] <= 0.0:
            continue
        s_pos = max(0.0, s_pos + dy)
        s_neg = min(0.0, s_neg + dy)
        if s_pos >= h_arr[i]:
            events[i] = 1
            s_pos = 0.0
            s_neg = 0.0
        elif -s_neg >= h_arr[i]:
            events[i] = -1
            s_pos = 0.0
            s_neg = 0.0
    return events

def generate_primary_signals(
    df: pd.DataFrame,
    span: int = 40,
    cusum_mult: float = 1.25,
    mode: str = "IGNITION"  # "IGNITION" (early breakout) or "REVERSION" (fade band extremes)
) -> pd.DataFrame:
    """
    Generates Primary Signals (+1 Long, -1 Short) on Signed CUSUM structural breaks:
      - IGNITION: Enters at the start of a CUSUM break when order flow confirms direction
                  and price is NOT yet overextended (< 1.75 sigma from EMA).
      - REVERSION: Fades exhausted CUSUM breaks that stretch beyond 1.75 sigma from EMA.
    """
    out = df.copy()
    log_p = np.log(out["close"])
    out["log_ret"] = log_p.diff()

    out["rolling_vol"] = out["log_ret"].ewm(span=span, min_periods=20).std()
    out["vol_regime_ratio"] = out["rolling_vol"] / out["log_ret"].ewm(span=span * 4, min_periods=40).std()

    out["ema_mid"] = out["close"].ewm(span=span, min_periods=20).mean()
    out["z_dist"] = (out["close"] - out["ema_mid"]) / (out["ema_mid"] * out["rolling_vol"]).replace(0, 1e-8)

    # Signed CUSUM structural break
    h_threshold = (out["rolling_vol"] * cusum_mult).clip(lower=0.0018)
    cusum_dir = _cusum_signed_numba(out["log_ret"].values, h_threshold.values)
    out["cusum_dir"] = cusum_dir

    # Order flow imbalance ratio in [-1, 1]
    out["ofi_ratio"] = out["net_order_flow"] / out["volume"].replace(0, 1e-8)
    out["ofi_smooth"] = out["ofi_ratio"].ewm(span=5, min_periods=1).mean()

    primary_side = np.zeros(len(out), dtype=np.int8)

    if mode == "IGNITION":
        # Enter early in the impulse (0.2 < |z_dist| < 1.75) when aggressive order flow confirms
        long_cond = (cusum_dir == 1) & (out["z_dist"] > 0.20) & (out["z_dist"] < 1.75) & (out["ofi_smooth"] > 0.05)
        short_cond = (cusum_dir == -1) & (out["z_dist"] < -0.20) & (out["z_dist"] > -1.75) & (out["ofi_smooth"] < -0.05)
        primary_side[long_cond] = 1
        primary_side[short_cond] = -1
    else:
        # REVERSION: Fade overextended structural breaks (|z_dist| >= 1.50) back toward EMA
        fade_short = (cusum_dir == 1) & (out["z_dist"] >= 1.50)
        fade_long = (cusum_dir == -1) & (out["z_dist"] <= -1.50)
        primary_side[fade_long] = 1
        primary_side[fade_short] = -1

    out["primary_side"] = primary_side

    # Stationary Microstructure Features for Meta-Labeling
    out["ffd_close"] = frac_diff_ffd(log_p, d=0.35, thres=1e-3)
    out["flow_alignment"] = out["primary_side"] * out["ofi_ratio"]
    out["abs_z_dist"] = out["z_dist"].abs()
    out["kyle_lambda_z"] = (
        (out["kyle_lambda"] - out["kyle_lambda"].rolling(80, min_periods=20).mean())
        / out["kyle_lambda"].rolling(80, min_periods=20).std().replace(0, 1e-8)
    )
    out["ret_autocorr"] = out["log_ret"].rolling(15, min_periods=10).corr(out["log_ret"].shift(1)).fillna(0.0)
    out["vpin_slope"] = out["vpin"].diff(5).fillna(0.0)

    return out

@njit
def apply_triple_barrier_meta_labels(
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    primary_sides: np.ndarray,
    rolling_vol: np.ndarray,
    pt_mult: float = 2.0,
    sl_mult: float = 1.25,
    max_horizon_bars: int = 40,
    roundtrip_fee: float = 0.00025
):
    """
    Path-dependent Triple-Barrier labeling without premature stall exits.
    Allows winning IGNITION breakouts to run to 2.0*sigma over a 40-bar horizon.
    """
    n = len(close_prices)
    meta_labels = np.full(n, -1, dtype=np.int8)
    realized_returns = np.zeros(n, dtype=np.float64)
    exit_indices = np.full(n, -1, dtype=np.int64)

    for i in range(n - 1):
        side = primary_sides[i]
        vol = rolling_vol[i]
        if side == 0 or np.isnan(vol) or vol <= 0.0:
            continue

        entry_p = close_prices[i]
        eff_vol = max(vol, 0.0022)
        pt_thresh = pt_mult * eff_vol
        sl_thresh = -sl_mult * eff_vol
        end_idx = min(n - 1, i + max_horizon_bars)

        hit_idx = end_idx
        gross_ret = 0.0

        for j in range(i + 1, end_idx + 1):
            if side == 1:
                up_exc = np.log(high_prices[j] / entry_p)
                dn_exc = np.log(low_prices[j] / entry_p)
            else:
                up_exc = -np.log(low_prices[j] / entry_p)
                dn_exc = -np.log(high_prices[j] / entry_p)

            if up_exc >= pt_thresh and dn_exc <= sl_thresh:
                hit_idx = j
                gross_ret = sl_thresh
                break
            elif up_exc >= pt_thresh:
                hit_idx = j
                gross_ret = pt_thresh
                break
            elif dn_exc <= sl_thresh:
                hit_idx = j
                gross_ret = sl_thresh
                break

            if j == end_idx:
                gross_ret = side * np.log(close_prices[j] / entry_p)

        net_ret = gross_ret - roundtrip_fee
        realized_returns[i] = net_ret
        meta_labels[i] = 1 if net_ret > 0.0 else 0
        exit_indices[i] = hit_idx

    return meta_labels, realized_returns, exit_indices