"""
Vickrey-Clarke-Groves (VCG) Auction Mechanism.

This implements the economic engine of the ads system. The VCG auction is:
- Strategy-proof (truthful bidding is optimal)
- Efficient (maximizes social welfare)
- Individual rational (no negative utilities)

The auction works as follows:
1. Advertisers submit bids (max CPM/CPC they're willing to pay)
2. System predicts click/conversion probabilities
3. Expected values are computed: bid * P(action)
4. Winners selected by highest expected values
5. Prices charged using VCG formula (second-price generalization)

Based on:
- Vickrey-Clarke-Groves mechanism theory
- Google AdWords auction design
- Meta's ad auction system
"""
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Bid:
    """
    Represents an advertiser's bid for an ad.

    Attributes:
        ad_id: Ad identifier
        advertiser_id: Advertiser identifier
        bid_price: Bid price (CPM or CPC)
        bid_type: Type of bid ('cpm' or 'cpc')
        quality_score: Ad quality score (0-1)
        click_prob: Predicted click probability
        conversion_prob: Predicted conversion probability
    """
    ad_id: str
    advertiser_id: str
    bid_price: float
    bid_type: str = 'cpm'  # 'cpm' or 'cpc'
    quality_score: float = 1.0
    click_prob: float = 0.0
    conversion_prob: float = 0.0

    def compute_expected_value(self) -> float:
        """
        Compute expected value for ranking.

        For CPM: bid_price * quality_score
        For CPC: bid_price * click_prob * quality_score
        """
        if self.bid_type == 'cpm':
            return self.bid_price * self.quality_score
        elif self.bid_type == 'cpc':
            return self.bid_price * self.click_prob * self.quality_score
        else:
            raise ValueError(f"Unknown bid type: {self.bid_type}")


@dataclass
class AuctionOutcome:
    """
    Outcome of the auction for a winning bid.

    Attributes:
        ad_id: Winning ad ID
        advertiser_id: Advertiser ID
        bid_price: Original bid
        vcg_price: VCG price to charge
        expected_value: Expected value used for ranking
        rank: Position in auction (1 = highest)
        payment_type: How payment is charged ('impression' or 'click')
    """
    ad_id: str
    advertiser_id: str
    bid_price: float
    vcg_price: float
    expected_value: float
    rank: int
    payment_type: str

    def to_dict(self) -> Dict:
        return {
            'ad_id': self.ad_id,
            'advertiser_id': self.advertiser_id,
            'bid_price': self.bid_price,
            'vcg_price': self.vcg_price,
            'expected_value': self.expected_value,
            'rank': self.rank,
            'payment_type': self.payment_type
        }


class VCGAuction:
    """
    VCG Auction Engine.

    The VCG mechanism computes prices such that each winner pays their
    "externality" - the harm their presence causes to other advertisers.

    For ad position i, the VCG price is:
    payment_i = (value_{i+1} - value_{i+2}) / quality_i + (value_{i+2} - value_{i+3}) / quality_i + ...

    Simplified for ads: payment_i ≈ (next_best_value - your_incremental_value) / your_quality
    """

    def __init__(
        self,
        reserve_price: float = 0.01,  # Minimum bid
        use_quality_score: bool = True,
        quality_weight: float = 0.5  # Weight for quality vs bid
    ):
        """
        Initialize VCG auction.

        Args:
            reserve_price: Minimum acceptable bid
            use_quality_score: Whether to include quality score in ranking
            quality_weight: Weight for quality (0 = bid only, 1 = quality only)
        """
        self.reserve_price = reserve_price
        self.use_quality_score = use_quality_score
        self.quality_weight = quality_weight

    def run_auction(
        self,
        bids: List[Bid],
        num_slots: int = 1
    ) -> List[AuctionOutcome]:
        """
        Run VCG auction for multiple ad slots.

        Args:
            bids: List of bids from advertisers
            num_slots: Number of ad slots available

        Returns:
            List of auction outcomes for winners
        """
        if not bids:
            return []

        # Filter by reserve price
        valid_bids = [b for b in bids if b.bid_price >= self.reserve_price]

        if not valid_bids:
            return []

        # Compute expected values
        for bid in valid_bids:
            bid.expected_value = bid.compute_expected_value()

        # Sort by expected value (descending)
        sorted_bids = sorted(valid_bids, key=lambda b: b.expected_value, reverse=True)

        # Select winners
        winners = sorted_bids[:num_slots]

        # Compute VCG prices
        outcomes = []
        for i, winner in enumerate(winners):
            vcg_price = self._compute_vcg_price(sorted_bids, i)

            outcome = AuctionOutcome(
                ad_id=winner.ad_id,
                advertiser_id=winner.advertiser_id,
                bid_price=winner.bid_price,
                vcg_price=vcg_price,
                expected_value=winner.expected_value,
                rank=i + 1,
                payment_type='impression' if winner.bid_type == 'cpm' else 'click'
            )
            outcomes.append(outcome)

        return outcomes

    def _compute_vcg_price(
        self,
        sorted_bids: List[Bid],
        winner_index: int
    ) -> float:
        """
        Compute VCG price for a winner.

        Simplified VCG for ads:
        - If you're in position i, you pay what the person in position i+1 would have to bid
          to achieve your expected value, given your quality score

        Formula: price = next_bid * (next_quality / your_quality)

        This is the "Generalized Second Price" (GSP) approximation of VCG.
        """
        winner = sorted_bids[winner_index]

        if winner_index + 1 < len(sorted_bids):
            next_best = sorted_bids[winner_index + 1]

            # GSP: pay next person's bid adjusted by quality ratio
            if self.use_quality_score and winner.quality_score > 0:
                # For CPC bids, we need to account for click probability
                if winner.bid_type == 'cpc':
                    vcg_price = (next_best.expected_value / (winner.click_prob * winner.quality_score))
                else:
                    vcg_price = (next_best.expected_value / winner.quality_score)

                # Price should be at least reserve and at most their own bid
                vcg_price = max(self.reserve_price, min(vcg_price, winner.bid_price))
            else:
                # Without quality, just use next bid (standard second-price)
                vcg_price = next_best.bid_price
        else:
            # No competition, pay reserve price
            vcg_price = self.reserve_price

        return vcg_price

    def run_auction_batch(
        self,
        bids_list: List[List[Bid]],
        num_slots: int = 1
    ) -> List[List[AuctionOutcome]]:
        """
        Run auctions for multiple requests in parallel.

        Args:
            bids_list: List of bid lists (one per request)
            num_slots: Number of slots per auction

        Returns:
            List of outcomes lists
        """
        return [self.run_auction(bids, num_slots) for bids in bids_list]


class RealTimeAuctionEngine:
    """
    Real-time auction engine that integrates with ranking model.

    This combines:
    1. Ranked candidates from retrieval + ranking
    2. Advertiser bids
    3. Budget constraints
    4. VCG auction mechanism
    """

    def __init__(
        self,
        vcg_auction: VCGAuction,
        budget_tracker: Optional['BudgetTracker'] = None
    ):
        self.vcg_auction = vcg_auction
        self.budget_tracker = budget_tracker

    def run_real_time_auction(
        self,
        candidates: List[Dict],  # List of {ad_id, advertiser_id, ranking_scores, ...}
        advertiser_bids: Dict[str, Dict],  # advertiser_id -> {bid_price, bid_type, ...}
        num_slots: int = 3
    ) -> List[AuctionOutcome]:
        """
        Run real-time auction for ad serving.

        Args:
            candidates: Ranked candidates from retrieval/ranking system
            advertiser_bids: Current bids from advertisers
            num_slots: Number of ad slots to fill

        Returns:
            List of winning auction outcomes
        """
        # Create Bid objects
        bids = []
        for candidate in candidates:
            advertiser_id = candidate['advertiser_id']

            if advertiser_id not in advertiser_bids:
                continue

            # Check budget if tracker available
            if self.budget_tracker:
                if not self.budget_tracker.has_budget(advertiser_id):
                    continue  # Skip if out of budget

            bid_info = advertiser_bids[advertiser_id]

            bid = Bid(
                ad_id=candidate['ad_id'],
                advertiser_id=advertiser_id,
                bid_price=bid_info['bid_price'],
                bid_type=bid_info.get('bid_type', 'cpm'),
                quality_score=candidate.get('quality_score', 1.0),
                click_prob=candidate.get('click_prob', 0.01),
                conversion_prob=candidate.get('conversion_prob', 0.001)
            )
            bids.append(bid)

        # Run VCG auction
        outcomes = self.vcg_auction.run_auction(bids, num_slots)

        # Update budgets
        if self.budget_tracker:
            for outcome in outcomes:
                self.budget_tracker.reserve_budget(
                    outcome.advertiser_id,
                    outcome.vcg_price
                )

        return outcomes

    def process_impression(
        self,
        outcome: AuctionOutcome,
        clicked: bool = False,
        converted: bool = False
    ) -> float:
        """
        Process an impression and compute payment.

        Args:
            outcome: Auction outcome
            clicked: Whether user clicked
            converted: Whether user converted

        Returns:
            Amount to charge
        """
        if outcome.payment_type == 'impression':
            # CPM: charge per impression
            payment = outcome.vcg_price / 1000  # CPM -> per impression
        elif outcome.payment_type == 'click':
            # CPC: only charge if clicked
            payment = outcome.vcg_price if clicked else 0.0
        else:
            payment = 0.0

        # Update budget tracker
        if self.budget_tracker:
            self.budget_tracker.charge(outcome.advertiser_id, payment)

        return payment


def create_bids_from_candidates(
    candidates: List[Dict],
    ranking_scores: Dict[str, Dict[str, float]],
    advertiser_bids: Dict[str, Dict]
) -> List[Bid]:
    """
    Utility function to create Bid objects from candidates.

    Args:
        candidates: List of candidate ads
        ranking_scores: Ranking model scores per ad
        advertiser_bids: Advertiser bid information

    Returns:
        List of Bid objects
    """
    bids = []

    for candidate in candidates:
        ad_id = candidate['ad_id']
        advertiser_id = candidate['advertiser_id']

        if advertiser_id not in advertiser_bids:
            continue

        bid_info = advertiser_bids[advertiser_id]
        scores = ranking_scores.get(ad_id, {})

        bid = Bid(
            ad_id=ad_id,
            advertiser_id=advertiser_id,
            bid_price=bid_info['bid_price'],
            bid_type=bid_info.get('bid_type', 'cpm'),
            quality_score=candidate.get('quality_score', 1.0),
            click_prob=scores.get('click_prob', 0.01),
            conversion_prob=scores.get('conversion_prob', 0.001)
        )
        bids.append(bid)

    return bids
