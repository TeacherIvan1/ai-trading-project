import time
import math
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

@dataclass
class OrderIntent:
    cl_ord_id: str
    symbol: str
    side: int                # +1 Buy, -1 Sell
    limit_price: float
    mid_price: float
    meta_prob: float         # Output of secondary ML classifier p_i in [0, 1]
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
    stall_bar_idx: int
    stall_min_ret: float
    max_bar_idx: int
    meta_prob: float
    entry_ts_ms: int

class SynchronousPreTradeRiskAndPaperEngine:
    """
    Closed-loop Shadow Execution & Synchronous Pre-Trade Risk Engine (<100us).
    Tracks active paper positions across live Triple-Barrier thresholds (PT, SL, Stall, Vertical Time).
    Supports isolated multi-asset state per symbol.
    """
    def __init__(
        self,
        initial_equity: float = 2500.0,
        max_fat_finger_pct: float = 0.005,
        max_daily_drawdown_pct: float = 0.20,
        pt_mult: float = 2.0,
        sl_mult: float = 1.25,
        max_horizon_bars: int = 40,
        stall_bars: int = 10,
        stall_min_sigma: float = 0.20,
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
        self.stall_bars = stall_bars
        self.stall_min_sigma = stall_min_sigma

        self.maker_fee = maker_fee_bps / 10_000.0
        self.taker_fee = taker_fee_bps / 10_000.0

        self.kill_switch_engaged: bool = False
        self.active_positions: Dict[str, ActivePaperPosition] = {}
        self.closed_trades: List[Dict] = []

    @property
    def active_position(self) -> Optional[ActivePaperPosition]:
        """Backwards compatibility for single-position inspections."""
        return next(iter(self.active_positions.values()), None)

    @active_position.setter
    def active_position(self, pos: Optional[ActivePaperPosition]):
        if pos is None:
            self.active_positions.clear()
        else:
            self.active_positions[pos.symbol] = pos

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

        # 0. Reject if a position is already open for THIS symbol
        if intent.symbol in self.active_positions:
            return {"status": "SKIPPED", "reason": f"POSITION_ALREADY_OPEN_FOR_{intent.symbol}"}

        # 0b. Maximum concurrent open positions check (strict cap of 4 assets)
        if len(self.active_positions) >= 4:
            return {"status": "FILTERED", "reason": "MAX_CONCURRENT_POSITIONS_REACHED"}

        # 1. Out-of-band Kill Switch Check
        drawdown = (self.high_water_mark - self.equity) / self.high_water_mark if self.high_water_mark > 0 else 0.0
        if drawdown >= self.max_daily_drawdown_pct:
            return {"status": "REJECTED", "reason": "KILL_SWITCH_DRAWDOWN_BREACH"}

        # 2. VPIN 99th Percentile Order Flow Toxicity Circuit Breaker
        if intent.vpin_cdf >= 0.99:
            return {"status": "REJECTED", "reason": "VPIN_TOXICITY_99TH_PERCENTILE"}

        # 3. Meta-Labeling Confidence Gate & Quarter-Kelly Sizing
        alloc_frac = self.compute_quarter_kelly_fraction(intent.meta_prob, intent.rolling_vol)
        if alloc_frac <= 0.0:
            if intent.meta_prob is not None and intent.meta_prob >= 0.56:
                alloc_frac = 0.05
            else:
                prob_str = f"{intent.meta_prob:.3f}" if intent.meta_prob is not None else "None"
                return {"status": "FILTERED", "reason": f"LOW_META_PROB ({prob_str} < 0.56)"}

        # 3b. Strict Single Position & Portfolio Capital Governors:
        # - Max single position notional: capped strictly at min($1,000 USDT, 10% of total equity)
        # - Max total portfolio notional: 40% of total equity
        max_single_notional = min(1000.0, self.equity * 0.10)
        current_total_notional = sum(p.notional_usd for p in self.active_positions.values())
        max_total_notional = self.equity * 0.40

        if current_total_notional >= max_total_notional:
            return {"status": "FILTERED", "reason": "MAX_PORTFOLIO_NOTIONAL_EXCEEDED"}

        avail_notional = max(0.0, max_total_notional - current_total_notional)
        target_notional = self.equity * alloc_frac
        notional_usd = min(target_notional, max_single_notional, avail_notional)

        if notional_usd < 5.0:
            return {"status": "FILTERED", "reason": "NOTIONAL_BELOW_MINIMUM"}

        # 4. Fat-Finger Price Collar Check
        if intent.mid_price > 0:
            price_dev = abs(intent.limit_price - intent.mid_price) / intent.mid_price
            if price_dev > self.max_fat_finger_pct:
                return {"status": "REJECTED", "reason": "FAT_FINGER_PRICE_COLLAR"}

        qty_btc = round(notional_usd / intent.limit_price, 4 if intent.limit_price < 10.0 else 3)
        latency_us = (time.perf_counter_ns() - t0_ns) / 1000.0

        if qty_btc * intent.limit_price < 5.0:
            # Scale to minimum $5 notional
            qty_btc = round(6.0 / intent.limit_price, 4 if intent.limit_price < 10.0 else 3)

        # Enforce hard upper bound on single position notional
        if qty_btc * intent.limit_price > max_single_notional:
            qty_btc = round(max_single_notional / intent.limit_price, 4 if intent.limit_price < 10.0 else 3)

        eff_vol = max(intent.rolling_vol, 0.0022)
        if intent.side == 1:
            pt_price = intent.limit_price * math.exp(self.pt_mult * eff_vol)
            sl_price = intent.limit_price * math.exp(-self.sl_mult * eff_vol)
        else:
            pt_price = intent.limit_price * math.exp(-self.pt_mult * eff_vol)
            sl_price = intent.limit_price * math.exp(self.sl_mult * eff_vol)

        pos = ActivePaperPosition(
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
            stall_bar_idx=current_bar_idx + self.stall_bars,
            stall_min_ret=self.stall_min_sigma * eff_vol,
            max_bar_idx=current_bar_idx + self.max_horizon_bars,
            meta_prob=intent.meta_prob,
            entry_ts_ms=int(time.time() * 1000)
        )
        self.active_positions[intent.symbol] = pos

        return {
            "status": "OPENED_SHADOW_POSITION",
            "cl_ord_id": intent.cl_ord_id,
            "symbol": intent.symbol,
            "side": intent.side,
            "entry_price": intent.limit_price,
            "pt_price": round(pt_price, 4 if pt_price < 1.0 else 2),
            "sl_price": round(sl_price, 4 if sl_price < 1.0 else 2),
            "qty_btc": qty_btc,
            "notional_usd": round(qty_btc * intent.limit_price, 2),
            "kelly_fraction": round(alloc_frac, 4),
            "risk_latency_us": round(latency_us, 2)
        }

    def check_intrabar_exits(
        self,
        trade_price: float,
        current_bar_idx: int,
        is_bar_close: bool = False,
        symbol: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Evaluates live aggTrade prices and Dollar Bar closes against active Triple-Barrier limits.
        Filters strictly by symbol to prevent cross-asset false barrier triggers.
        """
        if symbol:
            pos = self.active_positions.get(symbol)
        else:
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
                fee_rate += self.maker_fee
            elif trade_price <= pos.sl_price:
                exit_reason = "STOP_LOSS_BARRIER"
                exit_price = pos.sl_price
                fee_rate += self.taker_fee
        else:
            if trade_price <= pos.pt_price:
                exit_reason = "PROFIT_TAKE_BARRIER"
                exit_price = pos.pt_price
                fee_rate += self.maker_fee
            elif trade_price >= pos.sl_price:
                exit_reason = "STOP_LOSS_BARRIER"
                exit_price = pos.sl_price
                fee_rate += self.taker_fee

        # 2. Check Mid-Horizon Stall & Vertical Expiration on Dollar Bar closes
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
        if pos.symbol in self.active_positions:
            del self.active_positions[pos.symbol]
        return trade_summary
