"""Auction module for VCG auction and budget pacing."""
from .vcg_auction import (
    VCGAuction,
    RealTimeAuctionEngine,
    Bid,
    AuctionOutcome,
    create_bids_from_candidates
)
from .budget_pacing import (
    BudgetTracker,
    BudgetPacer,
    AutoBidder,
    BudgetConfig,
    BudgetState
)

__all__ = [
    'VCGAuction',
    'RealTimeAuctionEngine',
    'Bid',
    'AuctionOutcome',
    'create_bids_from_candidates',
    'BudgetTracker',
    'BudgetPacer',
    'AutoBidder',
    'BudgetConfig',
    'BudgetState'
]
