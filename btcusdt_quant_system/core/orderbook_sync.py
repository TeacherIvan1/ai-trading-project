import asyncio
from collections import deque
from typing import Dict, Deque, Any

class LocalOrderBookManager:
    """
    Zero-drift Level 2 Order Book state machine specifically engineered for
    Binance USD-M Futures (fapi) using 'pu' (previous final update ID) chaining.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.local_update_id: int = -1
        self.event_buffer: Deque[Dict[str, Any]] = deque(maxlen=5000)
        self.is_synced: bool = False
        self.sync_in_progress: bool = False

    def _apply_levels(self, side_book: Dict[float, float], updates: list) -> None:
        for price_str, qty_str in updates:
            price, qty = float(price_str), float(qty_str)
            if qty == 0.0:
                side_book.pop(price, None)
            else:
                side_book[price] = qty

    def _is_book_crossed(self) -> bool:
        if not self.bids or not self.asks:
            return False
        return max(self.bids.keys()) >= min(self.asks.keys())

    async def initialize_and_sync(self, fetch_rest_snapshot_fn, cancel_quotes_fn) -> bool:
        """Executes the Binance Futures snapshot-bridge synchronization protocol."""
        if self.sync_in_progress:
            return False
        self.sync_in_progress = True
        self.is_synced = False

        try:
            await cancel_quotes_fn(self.symbol)

            # Wait until at least one diff-depth frame is buffered
            while not self.event_buffer:
                await asyncio.sleep(0.02)

            first_buffered_U = self.event_buffer[0]["U"]
            while True:
                snapshot = await fetch_rest_snapshot_fn(self.symbol)
                snapshot_id = int(snapshot["lastUpdateId"])
                if snapshot_id >= first_buffered_U or len(self.event_buffer) > 20:
                    break
                await asyncio.sleep(0.05)

            self.bids = {float(p): float(q) for p, q in snapshot["bids"] if float(q) > 0.0}
            self.asks = {float(p): float(q) for p, q in snapshot["asks"] if float(q) > 0.0}
            self.local_update_id = snapshot_id

            bridged = False
            while self.event_buffer:
                event = self.event_buffer.popleft()
                U, u = int(event["U"]), int(event["u"])
                pu = int(event.get("pu", -1))

                # Step 1: Drop any event whose final update ID is older than the snapshot
                if u < self.local_update_id:
                    continue

                # Step 2: Bridge event must straddle U <= lastUpdateId <= u (Binance Futures spec)
                if not bridged:
                    if U <= self.local_update_id <= u or pu == self.local_update_id:
                        bridged = True
                    elif U > self.local_update_id:
                        # Snapshot fell slightly behind buffer; accept if uncrossed or retry
                        return False
                else:
                    # Step 3: Subsequent events on Binance Futures must satisfy pu == previous u
                    if pu != -1 and pu != self.local_update_id:
                        return False
                    elif pu == -1 and U != self.local_update_id + 1:
                        return False

                self._apply_levels(self.bids, event["b"])
                self._apply_levels(self.asks, event["a"])
                self.local_update_id = u

            if self._is_book_crossed():
                return False

            self.is_synced = True
            return True
        finally:
            self.sync_in_progress = False

    def process_realtime_event(self, event: Dict[str, Any]) -> bool:
        """
        Processes live Binance Futures depthUpdate events.
        Validates pu == self.local_update_id (or Spot fallback U == local_update_id + 1).
        """
        if not self.is_synced:
            self.event_buffer.append(event)
            return False

        U, u = int(event["U"]), int(event["u"])
        pu = int(event.get("pu", -1))

        if u <= self.local_update_id:
            return True

        # Validate continuity via Binance Futures 'pu' field
        if pu != -1:
            if pu != self.local_update_id:
                self.is_synced = False
                self.event_buffer.clear()
                self.event_buffer.append(event)
                return False
        else:
            if U != self.local_update_id + 1:
                self.is_synced = False
                self.event_buffer.clear()
                self.event_buffer.append(event)
                return False

        self._apply_levels(self.bids, event["b"])
        self._apply_levels(self.asks, event["a"])
        self.local_update_id = u

        if self._is_book_crossed():
            self.is_synced = False
            self.event_buffer.clear()
            return False

        return True