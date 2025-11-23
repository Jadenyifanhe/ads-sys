"""
Outcome-Conditioned Generative Retrieval (Pinterest PinRec-style).

This module implements outcome conditioning to enable business goal balancing.
The model learns to generate ads optimized for different outcome metrics like
clicks, conversions, saves, etc.

Based on:
- "PinRec: Outcome-Conditioned, Multi-Token Generative Retrieval" (Pinterest, 2024)
- https://arxiv.org/abs/2504.10507
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class OutcomeWeights:
    """
    Outcome weights for conditioning generation.

    These weights specify how much to optimize for each business metric.
    They should sum to 1.0 for proper normalization.

    Attributes:
        click: Weight for click metric (0-1)
        conversion: Weight for conversion metric (0-1)
        save: Weight for save/bookmark metric (0-1)
        share: Weight for share metric (0-1)
        view_time: Weight for view time metric (0-1)
    """
    click: float = 0.4
    conversion: float = 0.3
    save: float = 0.15
    share: float = 0.10
    view_time: float = 0.05

    def to_tensor(self, device: torch.device = None) -> torch.Tensor:
        """Convert to normalized tensor."""
        weights = torch.tensor([
            self.click,
            self.conversion,
            self.save,
            self.share,
            self.view_time
        ], dtype=torch.float32)

        # Normalize to sum to 1
        weights = weights / weights.sum()

        if device is not None:
            weights = weights.to(device)

        return weights

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'click': self.click,
            'conversion': self.conversion,
            'save': self.save,
            'share': self.share,
            'view_time': self.view_time
        }

    @classmethod
    def from_dict(cls, weights_dict: Dict[str, float]) -> 'OutcomeWeights':
        """Create from dictionary."""
        return cls(
            click=weights_dict.get('click', 0.4),
            conversion=weights_dict.get('conversion', 0.3),
            save=weights_dict.get('save', 0.15),
            share=weights_dict.get('share', 0.10),
            view_time=weights_dict.get('view_time', 0.05)
        )

    @classmethod
    def click_optimized(cls) -> 'OutcomeWeights':
        """Preset for click optimization."""
        return cls(click=0.7, conversion=0.2, save=0.05, share=0.03, view_time=0.02)

    @classmethod
    def conversion_optimized(cls) -> 'OutcomeWeights':
        """Preset for conversion optimization."""
        return cls(click=0.2, conversion=0.6, save=0.1, share=0.05, view_time=0.05)

    @classmethod
    def engagement_optimized(cls) -> 'OutcomeWeights':
        """Preset for engagement optimization."""
        return cls(click=0.25, conversion=0.15, save=0.25, share=0.25, view_time=0.1)

    @classmethod
    def balanced(cls) -> 'OutcomeWeights':
        """Balanced preset."""
        return cls(click=0.3, conversion=0.3, save=0.2, share=0.1, view_time=0.1)


class OutcomeEncoder(nn.Module):
    """
    Encodes outcome weights into a dense representation.

    This module processes the outcome weights and creates a conditioning
    vector that guides the generative model toward desired outcomes.
    """

    def __init__(
        self,
        num_outcomes: int = 5,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_outcomes = num_outcomes
        self.embed_dim = embed_dim

        # MLP to process outcome weights
        self.mlp = nn.Sequential(
            nn.Linear(num_outcomes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, outcome_weights: torch.Tensor) -> torch.Tensor:
        """
        Encode outcome weights.

        Args:
            outcome_weights: Outcome weight vector [batch, num_outcomes]

        Returns:
            Encoded outcome vector [batch, embed_dim]
        """
        return self.mlp(outcome_weights)


class OutcomeConditionedEncoder(nn.Module):
    """
    User history encoder with outcome conditioning.

    Extends the standard user history encoder to include outcome conditioning,
    allowing the model to optimize for different business metrics.
    """

    def __init__(
        self,
        num_quantizers: int = 4,
        codebook_size: int = 256,
        embed_dim: int = 512,
        num_demographics: int = 50,
        max_history_len: int = 100,
        num_layers: int = 6,
        num_heads: int = 8,
        num_outcomes: int = 5,
        outcome_embed_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim

        # Semantic ID embeddings
        self.semantic_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, embed_dim // num_quantizers)
            for _ in range(num_quantizers)
        ])

        # Position embedding
        self.position_embedding = nn.Embedding(max_history_len, embed_dim)

        # Demographic embeddings
        self.demographic_embedding = nn.Embedding(num_demographics, 64)

        # Outcome encoder
        self.outcome_encoder = OutcomeEncoder(
            num_outcomes=num_outcomes,
            embed_dim=outcome_embed_dim,
            hidden_dim=128,
            dropout=dropout
        )

        # Context MLP (includes demographics, context features, and outcome encoding)
        self.context_mlp = nn.Sequential(
            nn.Linear(64 + 32 + outcome_embed_dim, embed_dim),  # demo + context + outcome
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        semantic_ids: torch.Tensor,  # [batch, history_len, num_quantizers]
        demographic_ids: torch.Tensor,  # [batch, num_demo_features]
        context_features: torch.Tensor,  # [batch, context_dim]
        outcome_weights: torch.Tensor,  # [batch, num_outcomes]
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode user history with outcome conditioning.

        Args:
            semantic_ids: Historical semantic IDs
            demographic_ids: User demographics
            context_features: Context features
            outcome_weights: Outcome weight vector
            attention_mask: Attention mask

        Returns:
            Encoded representation [batch, history_len + 1, embed_dim]
        """
        batch_size, history_len, _ = semantic_ids.shape

        # Embed semantic IDs
        semantic_embeds = []
        for i in range(self.num_quantizers):
            embeds = self.semantic_embeddings[i](semantic_ids[:, :, i])
            semantic_embeds.append(embeds)
        semantic_embeds = torch.cat(semantic_embeds, dim=-1)  # [batch, history_len, embed_dim]

        # Add position embeddings
        positions = torch.arange(history_len, device=semantic_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(positions)
        semantic_embeds = semantic_embeds + position_embeds

        # Encode outcome weights
        outcome_embed = self.outcome_encoder(outcome_weights)  # [batch, outcome_embed_dim]

        # Process demographics and context with outcomes
        demo_embeds = self.demographic_embedding(demographic_ids).mean(dim=1)  # [batch, 64]
        context_input = torch.cat([demo_embeds, context_features, outcome_embed], dim=-1)
        context_token = self.context_mlp(context_input).unsqueeze(1)  # [batch, 1, embed_dim]

        # Concatenate context token (with outcome info) with history
        x = torch.cat([context_token, semantic_embeds], dim=1)

        # Create attention mask
        if attention_mask is not None:
            context_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=torch.bool)
            attention_mask = torch.cat([context_mask, attention_mask], dim=1)
        else:
            attention_mask = None

        # Encode with transformer
        encoded = self.transformer(
            x,
            src_key_padding_mask=~attention_mask if attention_mask is not None else None
        )

        return encoded


def create_outcome_weights_from_strategy(strategy: str = 'balanced') -> OutcomeWeights:
    """
    Create outcome weights from a strategy name.

    Args:
        strategy: One of 'balanced', 'click', 'conversion', 'engagement'

    Returns:
        OutcomeWeights instance
    """
    strategies = {
        'balanced': OutcomeWeights.balanced,
        'click': OutcomeWeights.click_optimized,
        'conversion': OutcomeWeights.conversion_optimized,
        'engagement': OutcomeWeights.engagement_optimized
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")

    return strategies[strategy]()


def sample_outcome_weights_for_training(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Sample diverse outcome weights for training.

    During training, we want the model to see various outcome weight combinations
    so it can learn to condition on different business goals.

    Args:
        batch_size: Number of samples to generate
        device: Device for tensors

    Returns:
        Outcome weight tensors [batch_size, num_outcomes]
    """
    # Sample from Dirichlet distribution for valid probability distributions
    alpha = torch.ones(5) * 2.0  # Concentration parameter
    weights = torch.distributions.Dirichlet(alpha).sample((batch_size,))

    return weights.to(device)
