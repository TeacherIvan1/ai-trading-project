import time
import math
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

@dataclass
class OrderIntent:
    cl_ord_id: str
    symbol: str
    side: int                # 1 = Long, -1 = Short
    limit_price: float
    mid_price: float
    meta_prob: float         # Output probability from Meta-Labeling RF
    vpin_cdf: float          # Current empirical CDF of VPIN
    rolling_vol: float

@dataclass
class ActivePaperPosition:
    cl_ord_id: str
    symbol: str
    side: int
    entry_price: float
    arrival_mid: float
    qty_btc: float
    notional_usd: float
    entry_bar_idx: int
    pt_price: float
    sl_price: float
    max_bar_idx: int
    meta_prob: float
    entry_ts_ms: int

class SynchronousPreTradeRiskAndPaperEngine:
    """
    Closed-loop Shadow Execution & Synchronous Pre-Trade Risk Engine (<100us).
    Tracks active paper positions across live Triple-Barrier thresholds (PT, SL, Vertical Time).
    """
    def __init__(
        self,
        initial_equity: float = 2500.0,
        max_fat_finger_pct: float = 0.005,
        max_daily_drawdown_pct: float = 0.03,
        pt_mult: float = 1.5,
        sl_mult: float = 1.5,
        max_horizon_bars: int = 25,
        maker_fee_bps: float = 2.0,
        taker_fee_bps: float = 5.0
    ):
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.high_water_mark = initial_equity
        self.max_fat_finger_pct = max_fat_finger_pct
        self.max_daily_drawdown_pct = max_daily_drawdown_pct

        self.pt_mult = pt_mult
        self.sl_mult = sl_mult
        self.max_horizon_bars = max_horizon_bars

        self.maker_fee = maker_fee_bps / 10_000.0
        self.taker_fee = taker_fee_bps / 10_000.0

        self.kill_switch_engaged: bool = False
        self.active_position: Optional[ActivePaperPosition] = None
        self.closed_trades: List[Dict] = []

    def compute_quarter_kelly_fraction(self, p_i: float, rolling_vol: float = 0.0025) -> float:
        """
        Enforces the Row 1 Meta-Labeling gate (p_i >= 0.56) with Volatility-Targeted
        Quarter-Kelly sizing so high-volatility bars do not over-leverage tail risk.
        """
        if p_i < 0.56:
            return 0.0
        # Base empirical edge for Row 1 (55.3% win rate on symmetric 1.5:1.5 barriers -> f* ~ 0.106)
        # Target a constant 25 bps (0.25%) portfolio volatility risk per trade, capped at 15% notional
        target_trade_vol_risk = 0.0025
        eff_vol = max(rolling_vol, 0.0020)
        vol_scaled_fraction = target_trade_vol_risk / (1.5 * eff_vol)
        return max(0.05, min(vol_scaled_fraction * 0.25, 0.15))

    def validate_and_route_shadow_order(self, intent: OrderIntent, current_bar_idx: int) -> dict:
        t0_ns = time.perf_counter_ns()

        # 0. Reject if a position is already open (single-position state machine)
        if self.active_position is not None:
            return {"status": "SKIPPED", "reason": "POSITION_ALREADY_OPEN"}

        # 1. Out-of-band Kill Switch Check
        drawdown = (self.high_water_mark - self.equity) / self.high_water_mark
        if drawdown >= self.max_daily_drawdown_pct or self.kill_switch_engaged:
            self.kill_switch_engaged = True
            return {"status": "REJECTED", "reason": "KILL_SWITCH_DRAWDOWN_BREACH"}

        # 2. VPIN 99th Percentile Order Flow Toxicity Circuit Breaker
        if intent.vpin_cdf >= 0.99:
            return {"status": "REJECTED", "reason": "VPIN_TOXICITY_99TH_PERCENTILE"}

        # 3. Meta-Labeling Confidence Gate & Quarter-Kelly Sizing
        alloc_frac = self.compute_quarter_kelly_fraction(intent.meta_prob, intent.rolling_vol)
        if alloc_frac <= 0.0:
            return {"status": "FILTERED", "reason": f"LOW_META_PROB ({intent.meta_prob:.3f} < 0.56)"}

        # 4. Fat-Finger Price Collar Check
        price_dev = abs(intent.limit_price - intent.mid_price) / intent.mid_price
        if price_dev > self.max_fat_finger_pct:
            return {"status": "REJECTED", "reason": "FAT_FINGER_PRICE_COLLAR"}

        notional_usd = self.equity * alloc_frac
        qty_btc = round(notional_usd / intent.limit_price, 3)
        latency_us = (time.perf_counter_ns() - t0_ns) / 1000.0

        if qty_btc * intent.limit_price < 5.0:
            return {"status": "REJECTED", "reason": "BELOW_MIN_NOTIONAL"}

        eff_vol = max(intent.rolling_vol, 0.0022)
        if intent.side == 1:
            pt_price = intent.limit_price * math.exp(self.pt_mult * eff_vol)
            sl_price = intent.limit_price * math.exp(-self.sl_mult * eff_vol)
        else:
            pt_price = intent.limit_price * math.exp(-self.pt_mult * eff_vol)
            sl_price = intent.limit_price * math.exp(self.sl_mult * eff_vol)

        self.active_position = ActivePaperPosition(
            cl_ord_id=intent.cl_ord_id,
            symbol=intent.symbol,
            side=intent.side,
            entry_price=intent.limit_price,
            arrival_mid=intent.mid_price,
            qty_btc=qty_btc,
            notional_usd=qty_btc * intent.limit_price,
            entry_bar_idx=current_bar_idx,
            pt_price=pt_price,
            sl_price=sl_price,
            max_bar_idx=current_bar_idx + self.max_horizon_bars,
            meta_prob=intent.meta_prob,
            entry_ts_ms=int(time.time() * 1000)
        )

        return {
            "status": "OPENED_SHADOW_POSITION",
            "cl_ord_id": intent.cl_ord_id,
            "side": intent.side,
            "entry_price": intent.limit_price,
            "pt_price": round(pt_price, 2),
            "sl_price": round(sl_price, 2),
            "qty_btc": qty_btc,
            "notional_usd": round(qty_btc * intent.limit_price, 2),
            "kelly_fraction": round(alloc_frac, 4),
            "risk_latency_us": round(latency_us, 2)
        }

    def check_intrabar_exits(self, trade_price: float, current_bar_idx: int, is_bar_close: bool = False) -> Optional[Dict]:
        """
        Evaluates live aggTrade / trade prices and Dollar Bar closes against active Triple-Barrier limits.
        """
        pos = self.active_position
        if pos is None:
            return None

        exit_reason = None
        exit_price = trade_price
        fee_rate = self.maker_fee  # Entry assumed at passive top-of-book limit

        # 1. Check Horizontal Profit-Taking (PT) and Stop-Loss (SL) barriers on every tick
        if pos.side == 1:
            if trade_price >= pos.pt_price:
                exit_reason = "PROFIT_TAKE_BARRIER"
                exit_price = pos.pt_price
                fee_rate += self.maker_fee  # Resting take-profit limit fill
            elif trade_price <= pos.sl_price:
                exit_reason = "STOP_LOSS_BARRIER"
                exit_price = pos.sl_price
                fee_rate += self.taker_fee  # Market/stop-market exit
        else:
            if trade_price <= pos.pt_price:
                exit_reason = "PROFIT_TAKE_BARRIER"
                exit_price = pos.pt_price
                fee_rate += self.maker_fee
            elif trade_price >= pos.sl_price:
                exit_reason = "STOP_LOSS_BARRIER"
                exit_price = pos.sl_price
                fee_rate += self.taker_fee

        # 2. Check Vertical Expiration on Dollar Bar closes
        if exit_reason is None and is_bar_close:
            if current_bar_idx >= pos.max_bar_idx:
                exit_reason = "VERTICAL_TIME_BARRIER"
                fee_rate += self.taker_fee

        if exit_reason is None:
            return None

        # Compute Net PnL & Update Paper Equity
        signed_gross_ret = pos.side * ((exit_price - pos.entry_price) / pos.entry_price)
        net_ret = signed_gross_ret - fee_rate
        pnl_usd = pos.notional_usd * net_ret
        self.equity += pnl_usd
        self.high_water_mark = max(self.high_water_mark, self.equity)

        trade_summary = {
            **asdict(pos),
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "exit_bar_idx": current_bar_idx,
            "net_return_bps": round(net_ret * 10_000, 2),
            "pnl_usd": round(pnl_usd, 4),
            "updated_equity": round(self.equity, 2),
            "drawdown_pct": round((self.high_water_mark - self.equity) / self.high_water_mark * 100, 3)
        }
        self.closed_trades.append(trade_summary)
        self.active_position = None
        return trade_summary
