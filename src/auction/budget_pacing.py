"""
Budget Pacing System.

This module manages advertiser budgets and ensures campaigns are paced
throughout the day to avoid spending all budget too quickly.

Key features:
1. Daily budget tracking
2. Real-time spend monitoring
3. Pacing multipliers to throttle delivery
4. Budget forecasting
5. Auto-bidding adjustments

Based on:
- "Budget Pacing for Targeted Online Advertisements at LinkedIn"
- Meta's budget pacing algorithms
- Google's budget optimization
"""
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from collections import defaultdict
import math


@dataclass
class BudgetConfig:
    """
    Budget configuration for an advertiser campaign.

    Attributes:
        advertiser_id: Unique advertiser identifier
        daily_budget: Daily budget limit in dollars
        total_budget: Total campaign budget
        start_date: Campaign start date
        end_date: Campaign end date
        pacing_strategy: How to pace budget ('even', 'asap', 'smart')
        min_spend_rate: Minimum fraction of budget to spend per hour (0-1)
        max_spend_rate: Maximum fraction of budget to spend per hour (0-1)
    """
    advertiser_id: str
    daily_budget: float
    total_budget: float
    start_date: datetime
    end_date: datetime
    pacing_strategy: str = 'even'  # 'even', 'asap', or 'smart'
    min_spend_rate: float = 0.02  # 2% per hour minimum
    max_spend_rate: float = 0.15  # 15% per hour maximum


@dataclass
class BudgetState:
    """
    Current budget state for an advertiser.

    Attributes:
        advertiser_id: Advertiser ID
        daily_budget: Daily budget
        spent_today: Amount spent today
        reserved: Amount reserved (pending charges)
        last_reset: When daily budget was last reset
        total_spent: Total amount spent in campaign
        pacing_multiplier: Current pacing multiplier (0-1)
        is_active: Whether budget is active
    """
    advertiser_id: str
    daily_budget: float
    spent_today: float = 0.0
    reserved: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)
    total_spent: float = 0.0
    pacing_multiplier: float = 1.0
    is_active: bool = True

    def remaining_budget(self) -> float:
        """Get remaining budget for today."""
        return max(0, self.daily_budget - self.spent_today - self.reserved)

    def utilization(self) -> float:
        """Get budget utilization (0-1)."""
        if self.daily_budget <= 0:
            return 0.0
        return min(1.0, (self.spent_today + self.reserved) / self.daily_budget)


class BudgetTracker:
    """
    Tracks and manages budgets for all advertisers in real-time.

    Thread-safe implementation using locks for concurrent access.
    """

    def __init__(self):
        self.budgets: Dict[str, BudgetState] = {}
        self.configs: Dict[str, BudgetConfig] = {}
        self.lock = threading.Lock()
        self._spend_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

    def register_budget(self, config: BudgetConfig):
        """
        Register a new budget configuration.

        Args:
            config: Budget configuration
        """
        with self.lock:
            self.configs[config.advertiser_id] = config
            self.budgets[config.advertiser_id] = BudgetState(
                advertiser_id=config.advertiser_id,
                daily_budget=config.daily_budget
            )

    def has_budget(self, advertiser_id: str) -> bool:
        """
        Check if advertiser has remaining budget.

        Args:
            advertiser_id: Advertiser ID

        Returns:
            True if budget available
        """
        with self.lock:
            if advertiser_id not in self.budgets:
                return False

            state = self.budgets[advertiser_id]
            return state.is_active and state.remaining_budget() > 0

    def get_remaining_budget(self, advertiser_id: str) -> float:
        """Get remaining budget for advertiser."""
        with self.lock:
            if advertiser_id not in self.budgets:
                return 0.0
            return self.budgets[advertiser_id].remaining_budget()

    def reserve_budget(self, advertiser_id: str, amount: float) -> bool:
        """
        Reserve budget (before impression is served).

        Args:
            advertiser_id: Advertiser ID
            amount: Amount to reserve

        Returns:
            True if reservation successful
        """
        with self.lock:
            if advertiser_id not in self.budgets:
                return False

            state = self.budgets[advertiser_id]

            if not state.is_active or state.remaining_budget() < amount:
                return False

            state.reserved += amount
            return True

    def charge(self, advertiser_id: str, amount: float):
        """
        Charge advertiser (after impression/click).

        Args:
            advertiser_id: Advertiser ID
            amount: Amount to charge
        """
        with self.lock:
            if advertiser_id not in self.budgets:
                return

            state = self.budgets[advertiser_id]

            # Move from reserved to spent
            if state.reserved >= amount:
                state.reserved -= amount

            state.spent_today += amount
            state.total_spent += amount

            # Record spend
            self._spend_history[advertiser_id].append((datetime.now(), amount))

            # Check if budget exhausted
            if state.remaining_budget() <= 0:
                state.is_active = False

    def release_reservation(self, advertiser_id: str, amount: float):
        """
        Release reserved budget (if impression not served).

        Args:
            advertiser_id: Advertiser ID
            amount: Amount to release
        """
        with self.lock:
            if advertiser_id not in self.budgets:
                return

            state = self.budgets[advertiser_id]
            state.reserved = max(0, state.reserved - amount)

    def reset_daily_budgets(self):
        """Reset all daily budgets (called at midnight)."""
        with self.lock:
            now = datetime.now()

            for advertiser_id, state in self.budgets.items():
                # Only reset if it's a new day
                if (now - state.last_reset).days >= 1:
                    state.spent_today = 0.0
                    state.reserved = 0.0
                    state.last_reset = now
                    state.is_active = True
                    state.pacing_multiplier = 1.0

    def get_budget_state(self, advertiser_id: str) -> Optional[BudgetState]:
        """Get current budget state."""
        with self.lock:
            return self.budgets.get(advertiser_id)

    def get_all_states(self) -> Dict[str, BudgetState]:
        """Get all budget states."""
        with self.lock:
            return dict(self.budgets)


class BudgetPacer:
    """
    Computes pacing multipliers to evenly distribute budget over time.

    The pacing multiplier is applied to bids to throttle delivery when
    spending too fast, or boost delivery when falling behind.

    Algorithm:
    1. Compute expected spend rate (budget / hours_remaining)
    2. Compare to actual spend rate
    3. Adjust pacing multiplier accordingly
    """

    def __init__(
        self,
        budget_tracker: BudgetTracker,
        update_interval_seconds: int = 300  # Update every 5 minutes
    ):
        self.budget_tracker = budget_tracker
        self.update_interval_seconds = update_interval_seconds
        self._last_update = datetime.now()

    def compute_pacing_multiplier(
        self,
        advertiser_id: str,
        current_time: Optional[datetime] = None
    ) -> float:
        """
        Compute pacing multiplier for advertiser.

        The multiplier is in range [0, 2]:
        - < 1: Spending too fast, throttle
        - = 1: On track
        - > 1: Spending too slow, boost

        Args:
            advertiser_id: Advertiser ID
            current_time: Current time (for testing)

        Returns:
            Pacing multiplier (0-2)
        """
        if current_time is None:
            current_time = datetime.now()

        state = self.budget_tracker.get_budget_state(advertiser_id)
        config = self.budget_tracker.configs.get(advertiser_id)

        if not state or not config:
            return 1.0

        # Hours elapsed today
        hours_elapsed = (current_time - state.last_reset).total_seconds() / 3600
        hours_elapsed = max(0.1, hours_elapsed)  # Avoid division by zero

        # Hours remaining today
        hours_in_day = 24
        hours_remaining = max(0.1, hours_in_day - hours_elapsed)

        # Expected spend by now (even pacing)
        expected_spend = state.daily_budget * (hours_elapsed / hours_in_day)

        # Actual spend
        actual_spend = state.spent_today + state.reserved

        # Pacing strategies
        if config.pacing_strategy == 'even':
            # Even pacing throughout the day
            if actual_spend >= expected_spend:
                # Spending too fast
                spend_rate = actual_spend / hours_elapsed
                target_rate = state.remaining_budget() / hours_remaining
                multiplier = target_rate / max(spend_rate, 0.001)
            else:
                # Spending too slow, boost
                multiplier = min(2.0, 1.0 + (expected_spend - actual_spend) / state.daily_budget)

        elif config.pacing_strategy == 'asap':
            # Spend as fast as possible
            multiplier = 2.0 if state.remaining_budget() > 0 else 0.0

        elif config.pacing_strategy == 'smart':
            # Smart pacing based on historical performance
            # Use exponential smoothing
            deviation = (actual_spend - expected_spend) / state.daily_budget
            multiplier = 1.0 - 0.5 * deviation  # Adjust by half the deviation

        else:
            multiplier = 1.0

        # Clamp to reasonable range
        multiplier = max(0.0, min(2.0, multiplier))

        # Update state
        state.pacing_multiplier = multiplier

        return multiplier

    def apply_pacing_to_bid(
        self,
        advertiser_id: str,
        original_bid: float
    ) -> float:
        """
        Apply pacing multiplier to bid.

        Args:
            advertiser_id: Advertiser ID
            original_bid: Original bid amount

        Returns:
            Adjusted bid amount
        """
        multiplier = self.compute_pacing_multiplier(advertiser_id)
        return original_bid * multiplier

    def update_all_pacing_multipliers(self):
        """Update pacing multipliers for all advertisers."""
        now = datetime.now()

        # Only update if enough time has passed
        if (now - self._last_update).total_seconds() < self.update_interval_seconds:
            return

        for advertiser_id in self.budget_tracker.budgets.keys():
            self.compute_pacing_multiplier(advertiser_id, now)

        self._last_update = now

    def get_spend_forecast(
        self,
        advertiser_id: str,
        hours_ahead: int = 24
    ) -> Dict[str, float]:
        """
        Forecast spend for the next N hours.

        Args:
            advertiser_id: Advertiser ID
            hours_ahead: Hours to forecast

        Returns:
            Dictionary with forecast metrics
        """
        state = self.budget_tracker.get_budget_state(advertiser_id)

        if not state:
            return {}

        # Simple linear forecast based on current spend rate
        now = datetime.now()
        hours_elapsed = (now - state.last_reset).total_seconds() / 3600
        hours_elapsed = max(0.1, hours_elapsed)

        current_rate = state.spent_today / hours_elapsed  # $ per hour
        forecast_spend = state.spent_today + current_rate * hours_ahead

        return {
            'current_spend': state.spent_today,
            'current_rate': current_rate,
            'forecast_spend': min(forecast_spend, state.daily_budget),
            'forecast_remaining': max(0, state.daily_budget - forecast_spend),
            'hours_to_depletion': state.remaining_budget() / max(current_rate, 0.001)
        }


class AutoBidder:
    """
    Automatic bid adjustment based on performance and budget.

    Adjusts bids to:
    1. Meet budget pacing goals
    2. Optimize for target CPA/ROAS
    3. React to competition
    """

    def __init__(
        self,
        budget_tracker: BudgetTracker,
        budget_pacer: BudgetPacer
    ):
        self.budget_tracker = budget_tracker
        self.budget_pacer = budget_pacer

    def compute_optimal_bid(
        self,
        advertiser_id: str,
        base_bid: float,
        click_prob: float,
        conversion_prob: float,
        target_cpa: Optional[float] = None,
        target_roas: Optional[float] = None
    ) -> float:
        """
        Compute optimal bid considering budget pacing and performance goals.

        Args:
            advertiser_id: Advertiser ID
            base_bid: Base bid from advertiser
            click_prob: Predicted click probability
            conversion_prob: Predicted conversion probability
            target_cpa: Target cost per acquisition (optional)
            target_roas: Target return on ad spend (optional)

        Returns:
            Optimal bid amount
        """
        # Start with base bid
        optimal_bid = base_bid

        # Apply budget pacing
        pacing_multiplier = self.budget_pacer.compute_pacing_multiplier(advertiser_id)
        optimal_bid *= pacing_multiplier

        # Apply performance-based adjustment
        if target_cpa is not None and conversion_prob > 0:
            # Bid should be: target_cpa * conversion_prob
            performance_bid = target_cpa * conversion_prob
            optimal_bid = min(optimal_bid, performance_bid)

        if target_roas is not None and conversion_prob > 0:
            # Assuming average order value (AOV) is known
            # bid = (AOV / target_roas) * conversion_prob
            # For now, use as a scaling factor
            roas_factor = 1.0 / max(target_roas, 1.0)
            optimal_bid *= roas_factor

        return max(0.01, optimal_bid)  # Minimum bid floor
