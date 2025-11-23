"""
Deep Learning Recommendation Model (DLRM) with Multi-Task Learning.

This implements the "Lattice" ranking layer that scores retrieved ad candidates.
The model predicts multiple objectives simultaneously:
- P(Click)
- P(Conversion)
- P(Skip)
- Expected View Time

Architecture:
- User Tower: Processes user features
- Ad Tower: Processes ad features
- Feature Interaction: Cross-architecture for user-ad interaction
- Multi-Task Heads: Separate heads for each prediction task

Based on:
- "Deep Learning Recommendation Model for Personalization and Recommendation Systems" (DLRM)
- "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts" (MMoE)
- Meta's production ranking systems
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class MLPLayer(nn.Module):
    """Multi-layer perceptron with batch normalization and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FeatureInteraction(nn.Module):
    """
    Feature interaction layer using explicit feature crosses.

    Implements both:
    1. Dot product interactions (efficient)
    2. Learned cross features (expressive)
    """

    def __init__(
        self,
        num_features: int,
        embed_dim: int,
        interaction_type: str = 'dot'  # 'dot' or 'learned'
    ):
        super().__init__()

        self.num_features = num_features
        self.embed_dim = embed_dim
        self.interaction_type = interaction_type

        if interaction_type == 'learned':
            # Learn interaction weights
            num_interactions = num_features * (num_features - 1) // 2
            self.interaction_weights = nn.Parameter(
                torch.randn(num_interactions, embed_dim, embed_dim)
            )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute feature interactions.

        Args:
            embeddings: Feature embeddings [batch, num_features, embed_dim]

        Returns:
            Interaction features [batch, num_interactions, embed_dim]
        """
        batch_size, num_features, embed_dim = embeddings.shape
        interactions = []

        idx = 0
        for i in range(num_features):
            for j in range(i + 1, num_features):
                if self.interaction_type == 'dot':
                    # Element-wise product
                    interaction = embeddings[:, i, :] * embeddings[:, j, :]
                else:
                    # Learned bilinear interaction
                    # x_i^T W x_j
                    x_i = embeddings[:, i, :].unsqueeze(1)  # [batch, 1, embed_dim]
                    x_j = embeddings[:, j, :].unsqueeze(2)  # [batch, embed_dim, 1]
                    W = self.interaction_weights[idx]  # [embed_dim, embed_dim]
                    interaction = torch.bmm(
                        torch.bmm(x_i, W.unsqueeze(0).expand(batch_size, -1, -1)),
                        x_j
                    ).squeeze()  # [batch, 1]
                    interaction = interaction.unsqueeze(1).expand(-1, embed_dim)

                interactions.append(interaction)
                idx += 1

        if interactions:
            interactions = torch.stack(interactions, dim=1)  # [batch, num_interactions, embed_dim]
        else:
            interactions = torch.zeros(batch_size, 0, embed_dim, device=embeddings.device)

        return interactions


class UserTower(nn.Module):
    """
    User feature tower.

    Processes user features:
    - Demographics (age, gender, location, etc.)
    - Behavioral features (interaction history stats)
    - Contextual features (time, device, etc.)
    """

    def __init__(
        self,
        num_categorical_features: int,
        categorical_vocab_sizes: List[int],
        num_continuous_features: int,
        embed_dim: int = 64,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        # Categorical feature embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for vocab_size in categorical_vocab_sizes
        ])

        # Continuous feature processing
        self.continuous_mlp = nn.Sequential(
            nn.Linear(num_continuous_features, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Combine all features
        total_embed_dim = embed_dim * (num_categorical_features + 1)

        self.mlp = MLPLayer(
            input_dim=total_embed_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        )

    def forward(
        self,
        categorical_features: torch.Tensor,  # [batch, num_categorical]
        continuous_features: torch.Tensor  # [batch, num_continuous]
    ) -> torch.Tensor:
        """
        Process user features.

        Returns:
            User embedding [batch, output_dim]
        """
        # Embed categorical features
        cat_embeds = []
        for i, embedding in enumerate(self.embeddings):
            cat_embeds.append(embedding(categorical_features[:, i]))

        # Process continuous features
        cont_embed = self.continuous_mlp(continuous_features)

        # Concatenate all embeddings
        all_embeds = cat_embeds + [cont_embed]
        x = torch.cat(all_embeds, dim=-1)

        # MLP
        user_embed = self.mlp(x)

        return user_embed


class AdTower(nn.Module):
    """
    Ad feature tower.

    Processes ad features:
    - Semantic embedding (from RQ-VAE)
    - Category, keywords
    - Ad quality scores
    - Advertiser features
    """

    def __init__(
        self,
        semantic_embed_dim: int = 256,
        num_categorical_features: int = 5,
        categorical_vocab_sizes: List[int] = [100, 1000, 50, 20, 10],
        num_continuous_features: int = 10,
        embed_dim: int = 64,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        # Categorical feature embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for vocab_size in categorical_vocab_sizes
        ])

        # Semantic embedding projection
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Continuous feature processing
        self.continuous_mlp = nn.Sequential(
            nn.Linear(num_continuous_features, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Combine all features
        total_embed_dim = embed_dim * (num_categorical_features + 2)  # +2 for semantic and continuous

        self.mlp = MLPLayer(
            input_dim=total_embed_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        )

    def forward(
        self,
        semantic_embeddings: torch.Tensor,  # [batch, semantic_embed_dim]
        categorical_features: torch.Tensor,  # [batch, num_categorical]
        continuous_features: torch.Tensor  # [batch, num_continuous]
    ) -> torch.Tensor:
        """
        Process ad features.

        Returns:
            Ad embedding [batch, output_dim]
        """
        # Embed categorical features
        cat_embeds = []
        for i, embedding in enumerate(self.embeddings):
            cat_embeds.append(embedding(categorical_features[:, i]))

        # Process semantic embedding
        semantic_embed = self.semantic_proj(semantic_embeddings)

        # Process continuous features
        cont_embed = self.continuous_mlp(continuous_features)

        # Concatenate all embeddings
        all_embeds = cat_embeds + [semantic_embed, cont_embed]
        x = torch.cat(all_embeds, dim=-1)

        # MLP
        ad_embed = self.mlp(x)

        return ad_embed


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head with shared bottom layers.

    Uses MMoE (Multi-gate Mixture-of-Experts) architecture.
    Each task has its own expert mixture for task-specific learning.
    """

    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        num_experts: int = 4,
        expert_dim: int = 128,
        tower_dims: List[int] = [64, 32],
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.num_experts = num_experts

        # Shared experts
        self.experts = nn.ModuleList([
            MLPLayer(
                input_dim=input_dim,
                hidden_dims=[expert_dim],
                output_dim=expert_dim,
                dropout=dropout
            )
            for _ in range(num_experts)
        ])

        # Gating networks (one per task)
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts)
            for _ in range(num_tasks)
        ])

        # Task-specific towers
        self.towers = nn.ModuleList([
            MLPLayer(
                input_dim=expert_dim,
                hidden_dims=tower_dims,
                output_dim=1,
                dropout=dropout
            )
            for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Multi-task forward pass.

        Args:
            x: Combined user-ad features [batch, input_dim]

        Returns:
            Dictionary with task predictions
        """
        batch_size = x.shape[0]

        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, expert_dim]

        # Task-specific predictions
        task_outputs = {}
        task_names = ['click', 'conversion', 'skip', 'view_time']

        for task_idx in range(self.num_tasks):
            # Compute gate weights for this task
            gate_logits = self.gates[task_idx](x)  # [batch, num_experts]
            gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1)  # [batch, num_experts, 1]

            # Weighted combination of experts
            task_input = (expert_outputs * gate_weights).sum(dim=1)  # [batch, expert_dim]

            # Task-specific tower
            task_output = self.towers[task_idx](task_input).squeeze(-1)  # [batch]

            task_outputs[task_names[task_idx]] = task_output

        return task_outputs


class DLRM(nn.Module):
    """
    Complete Deep Learning Recommendation Model with Multi-Task Learning.

    This is the unified ranking layer that scores retrieved ad candidates.
    """

    def __init__(
        self,
        # User tower config
        num_user_categorical: int = 10,
        user_categorical_vocab_sizes: List[int] = None,
        num_user_continuous: int = 20,
        # Ad tower config
        semantic_embed_dim: int = 256,
        num_ad_categorical: int = 5,
        ad_categorical_vocab_sizes: List[int] = None,
        num_ad_continuous: int = 10,
        # Architecture config
        tower_embed_dim: int = 64,
        tower_hidden_dims: List[int] = [256, 128],
        tower_output_dim: int = 128,
        # Multi-task config
        num_tasks: int = 4,
        num_experts: int = 4,
        expert_dim: int = 128,
        # Other
        dropout: float = 0.1
    ):
        super().__init__()

        if user_categorical_vocab_sizes is None:
            user_categorical_vocab_sizes = [100] * num_user_categorical

        if ad_categorical_vocab_sizes is None:
            ad_categorical_vocab_sizes = [100] * num_ad_categorical

        # User tower
        self.user_tower = UserTower(
            num_categorical_features=num_user_categorical,
            categorical_vocab_sizes=user_categorical_vocab_sizes,
            num_continuous_features=num_user_continuous,
            embed_dim=tower_embed_dim,
            hidden_dims=tower_hidden_dims,
            output_dim=tower_output_dim,
            dropout=dropout
        )

        # Ad tower
        self.ad_tower = AdTower(
            semantic_embed_dim=semantic_embed_dim,
            num_categorical_features=num_ad_categorical,
            categorical_vocab_sizes=ad_categorical_vocab_sizes,
            num_continuous_features=num_ad_continuous,
            embed_dim=tower_embed_dim,
            hidden_dims=tower_hidden_dims,
            output_dim=tower_output_dim,
            dropout=dropout
        )

        # Feature interaction
        self.interaction = FeatureInteraction(
            num_features=2,  # user and ad
            embed_dim=tower_output_dim,
            interaction_type='dot'
        )

        # Combine features for multi-task head
        # user_embed + ad_embed + interaction
        combined_dim = tower_output_dim * 3

        # Multi-task head
        self.multi_task_head = MultiTaskHead(
            input_dim=combined_dim,
            num_tasks=num_tasks,
            num_experts=num_experts,
            expert_dim=expert_dim,
            tower_dims=[64, 32],
            dropout=dropout
        )

    def forward(
        self,
        # User features
        user_categorical: torch.Tensor,
        user_continuous: torch.Tensor,
        # Ad features
        ad_semantic_embeddings: torch.Tensor,
        ad_categorical: torch.Tensor,
        ad_continuous: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Dictionary with predictions for each task:
            - click: P(click)
            - conversion: P(conversion)
            - skip: P(skip)
            - view_time: Expected view time
        """
        # User tower
        user_embed = self.user_tower(user_categorical, user_continuous)

        # Ad tower
        ad_embed = self.ad_tower(ad_semantic_embeddings, ad_categorical, ad_continuous)

        # Feature interactions
        embeddings = torch.stack([user_embed, ad_embed], dim=1)  # [batch, 2, embed_dim]
        interactions = self.interaction(embeddings)  # [batch, 1, embed_dim]
        interactions = interactions.squeeze(1)  # [batch, embed_dim]

        # Combine all features
        combined = torch.cat([user_embed, ad_embed, interactions], dim=-1)

        # Multi-task predictions
        predictions = self.multi_task_head(combined)

        # Apply activations
        predictions['click'] = torch.sigmoid(predictions['click'])
        predictions['conversion'] = torch.sigmoid(predictions['conversion'])
        predictions['skip'] = torch.sigmoid(predictions['skip'])
        predictions['view_time'] = F.softplus(predictions['view_time'])  # Non-negative

        return predictions

    def predict_batch(
        self,
        user_categorical: torch.Tensor,
        user_continuous: torch.Tensor,
        ad_semantic_embeddings: torch.Tensor,
        ad_categorical: torch.Tensor,
        ad_continuous: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Batch prediction (same as forward but with eval mode).
        """
        self.eval()
        with torch.no_grad():
            return self.forward(
                user_categorical,
                user_continuous,
                ad_semantic_embeddings,
                ad_categorical,
                ad_continuous
            )


def compute_ranking_loss(
    predictions: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    task_weights: Optional[Dict[str, float]] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute multi-task ranking loss.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        task_weights: Optional weights for each task

    Returns:
        Tuple of (total_loss, loss_dict)
    """
    if task_weights is None:
        task_weights = {
            'click': 1.0,
            'conversion': 2.0,  # Higher weight for conversions
            'skip': 0.5,
            'view_time': 0.5
        }

    losses = {}

    # Binary cross-entropy for click, conversion, skip
    for task in ['click', 'conversion', 'skip']:
        if task in predictions and task in labels:
            losses[task] = F.binary_cross_entropy(
                predictions[task],
                labels[task],
                reduction='mean'
            )

    # MSE for view_time
    if 'view_time' in predictions and 'view_time' in labels:
        losses['view_time'] = F.mse_loss(
            predictions['view_time'],
            labels['view_time'],
            reduction='mean'
        )

    # Weighted total loss
    total_loss = sum(task_weights[task] * loss for task, loss in losses.items())

    # Loss dict
    loss_dict = {
        'total_loss': total_loss.item(),
        **{f'{task}_loss': loss.item() for task, loss in losses.items()}
    }

    return total_loss, loss_dict
