"""
Data Schemas for the Generative Ads System.

This module defines all core data structures used throughout the system.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import torch
from datetime import datetime


class AdCategory(Enum):
    """Ad category enumeration."""
    FASHION = "fashion"
    ELECTRONICS = "electronics"
    HOME = "home"
    AUTOMOTIVE = "automotive"
    FOOD = "food"
    TRAVEL = "travel"
    ENTERTAINMENT = "entertainment"
    FINANCE = "finance"
    HEALTH = "health"
    OTHER = "other"


class DeviceType(Enum):
    """Device type enumeration."""
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"
    TV = "tv"


class InteractionType(Enum):
    """User interaction type."""
    CLICK = "click"
    VIEW = "view"
    CONVERSION = "conversion"
    SKIP = "skip"


@dataclass
class Ad:
    """
    Representation of an advertisement.

    Attributes:
        ad_id: Unique identifier for the ad
        advertiser_id: ID of the advertiser
        title: Ad title text
        description: Ad description text
        image_url: URL to ad image/creative
        category: Ad category
        keywords: List of keywords
        bid_price: Advertiser's bid price (CPM or CPC)
        daily_budget: Daily budget limit
        semantic_id: Tuple of semantic IDs from RQ-VAE encoder (set after encoding)
        embedding: Dense embedding vector (optional)
        metadata: Additional metadata
    """
    ad_id: str
    advertiser_id: str
    title: str
    description: str
    image_url: Optional[str] = None
    category: AdCategory = AdCategory.OTHER
    keywords: List[str] = field(default_factory=list)
    bid_price: float = 0.0
    daily_budget: float = 1000.0
    semantic_id: Optional[Tuple[int, ...]] = None
    embedding: Optional[torch.Tensor] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'ad_id': self.ad_id,
            'advertiser_id': self.advertiser_id,
            'title': self.title,
            'description': self.description,
            'image_url': self.image_url,
            'category': self.category.value,
            'keywords': self.keywords,
            'bid_price': self.bid_price,
            'daily_budget': self.daily_budget,
            'semantic_id': self.semantic_id,
            'metadata': self.metadata
        }


@dataclass
class User:
    """
    User profile representation.

    Attributes:
        user_id: Unique user identifier
        demographics: Demographic information (age, gender, location, etc.)
        interests: List of interest categories
        interaction_history: List of past interactions
        device_type: Primary device type
        metadata: Additional metadata
    """
    user_id: str
    demographics: Dict = field(default_factory=dict)
    interests: List[str] = field(default_factory=list)
    interaction_history: List['Interaction'] = field(default_factory=list)
    device_type: DeviceType = DeviceType.MOBILE
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'user_id': self.user_id,
            'demographics': self.demographics,
            'interests': self.interests,
            'device_type': self.device_type.value,
            'metadata': self.metadata
        }


@dataclass
class Interaction:
    """
    User-Ad interaction event.

    Attributes:
        user_id: User who performed the interaction
        ad_id: Ad that was interacted with
        semantic_id: Semantic ID of the ad (if available)
        interaction_type: Type of interaction
        timestamp: When the interaction occurred
        context: Contextual information (time of day, device, etc.)
        value: Value of the interaction (e.g., conversion amount)
    """
    user_id: str
    ad_id: str
    semantic_id: Optional[Tuple[int, ...]]
    interaction_type: InteractionType
    timestamp: datetime
    context: Dict = field(default_factory=dict)
    value: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'user_id': self.user_id,
            'ad_id': self.ad_id,
            'semantic_id': self.semantic_id,
            'interaction_type': self.interaction_type.value,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'value': self.value
        }


@dataclass
class RetrievalRequest:
    """
    Request for ad retrieval.

    Attributes:
        user_id: User requesting ads
        user_history: Sequence of recently interacted ad semantic IDs
        context: Context information (device, time, location, etc.)
        num_candidates: Number of candidates to retrieve
        demographics: User demographic features
    """
    user_id: str
    user_history: List[Tuple[int, ...]]  # List of semantic IDs
    context: Dict = field(default_factory=dict)
    num_candidates: int = 50
    demographics: Dict = field(default_factory=dict)


@dataclass
class RankingRequest:
    """
    Request for ranking retrieved candidates.

    Attributes:
        user_id: User ID
        user_features: User feature vector
        candidate_ads: List of candidate ads to rank
        context: Context features
    """
    user_id: str
    user_features: torch.Tensor
    candidate_ads: List[Ad]
    context: Dict = field(default_factory=dict)


@dataclass
class RankingResult:
    """
    Result of ranking model.

    Attributes:
        ad_id: Ad identifier
        click_prob: Predicted probability of click
        conversion_prob: Predicted probability of conversion
        skip_prob: Predicted probability of skip
        combined_score: Combined utility score
    """
    ad_id: str
    click_prob: float
    conversion_prob: float
    skip_prob: float
    combined_score: float

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'ad_id': self.ad_id,
            'click_prob': self.click_prob,
            'conversion_prob': self.conversion_prob,
            'skip_prob': self.skip_prob,
            'combined_score': self.combined_score
        }


@dataclass
class AuctionResult:
    """
    Result of the ad auction.

    Attributes:
        ad_id: Winning ad ID
        advertiser_id: Advertiser ID
        bid_price: Original bid price
        final_price: Final price to charge (VCG)
        expected_value: Expected value (bid * predicted probability)
        rank: Position in the auction
    """
    ad_id: str
    advertiser_id: str
    bid_price: float
    final_price: float
    expected_value: float
    rank: int

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'ad_id': self.ad_id,
            'advertiser_id': self.advertiser_id,
            'bid_price': self.bid_price,
            'final_price': self.final_price,
            'expected_value': self.expected_value,
            'rank': self.rank
        }


@dataclass
class TrainingExample:
    """
    Training example for model training.

    Attributes:
        user_history: Sequence of semantic IDs from user history
        target_semantic_id: Target ad semantic ID to predict
        user_features: User demographic and context features
        ad_features: Ad features
        labels: Ground truth labels (click, conversion, etc.)
        weight: Example weight for importance sampling
    """
    user_history: List[Tuple[int, ...]]
    target_semantic_id: Tuple[int, ...]
    user_features: torch.Tensor
    ad_features: torch.Tensor
    labels: Dict[str, float]  # {'click': 0/1, 'conversion': 0/1, etc.}
    weight: float = 1.0
