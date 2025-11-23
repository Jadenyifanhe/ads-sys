"""
Residual Quantized Variational Autoencoder (RQ-VAE) for Semantic Indexing.

This module implements the RQ-VAE architecture that converts ads into discrete
semantic IDs. The RQ-VAE uses multiple levels of vector quantization to create
a hierarchical semantic representation.

Based on:
- "Residual Vector Quantization" (Lee et al.)
- "RQ-VAE: Vector Quantized Variational Autoencoder with Residual Quantization"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from vector_quantize_pytorch import ResidualVQ


class AdEncoder(nn.Module):
    """
    Encoder network that converts ad features into dense embeddings.

    This network processes:
    - Text features (title + description)
    - Image features (from pre-trained vision model)
    - Categorical features (category, keywords)
    - Metadata features

    Architecture:
        - Text: BERT-style encoder or frozen CLIP text encoder
        - Image: ResNet or ViT features from frozen CLIP image encoder
        - Categorical: Embedding layers
        - Fusion: Multi-layer perceptron with residual connections
    """

    def __init__(
        self,
        text_dim: int = 768,  # BERT/CLIP dimension
        image_dim: int = 512,  # Image feature dimension
        category_vocab_size: int = 100,
        category_embed_dim: int = 64,
        hidden_dims: List[int] = [1024, 512, 256],
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.text_dim = text_dim
        self.image_dim = image_dim
        self.output_dim = output_dim

        # Category embedding
        self.category_embedding = nn.Embedding(category_vocab_size, category_embed_dim)

        # Calculate total input dimension
        input_dim = text_dim + image_dim + category_embed_dim

        # MLP layers with residual connections
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output projection
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))

        self.mlp = nn.Sequential(*layers)

        # Residual projection if input_dim != output_dim
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(
        self,
        text_features: torch.Tensor,  # [batch, text_dim]
        image_features: torch.Tensor,  # [batch, image_dim]
        category_ids: torch.Tensor,  # [batch]
    ) -> torch.Tensor:
        """
        Encode ad features into dense embedding.

        Args:
            text_features: Text features from BERT/CLIP
            image_features: Image features from ResNet/ViT
            category_ids: Category ID indices

        Returns:
            Dense embedding of shape [batch, output_dim]
        """
        # Get category embeddings
        category_embed = self.category_embedding(category_ids)

        # Concatenate all features
        x = torch.cat([text_features, image_features, category_embed], dim=-1)

        # Pass through MLP with residual connection
        residual = self.residual_proj(x)
        x = self.mlp(x) + residual

        # L2 normalize
        x = F.normalize(x, p=2, dim=-1)

        return x


class AdDecoder(nn.Module):
    """
    Decoder network that reconstructs ad features from quantized embeddings.

    This is used during training to ensure the quantized representations
    retain semantic information about the original ads.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: List[int] = [256, 512, 1024],
        text_dim: int = 768,
        image_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output heads for different modalities
        self.text_head = nn.Linear(prev_dim, text_dim)
        self.image_head = nn.Linear(prev_dim, image_dim)

    def forward(self, quantized_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode quantized embedding back to feature space.

        Args:
            quantized_embedding: Quantized embedding [batch, input_dim]

        Returns:
            Tuple of (text_features, image_features)
        """
        x = self.mlp(quantized_embedding)

        text_recon = self.text_head(x)
        image_recon = self.image_head(x)

        return text_recon, image_recon


class RQVAE(nn.Module):
    """
    Complete RQ-VAE model for semantic indexing.

    This model:
    1. Encodes ads into dense embeddings
    2. Quantizes embeddings into discrete codes using Residual Vector Quantization
    3. Decodes back to feature space (for training)

    The discrete codes form the "Semantic IDs" that the generative retrieval
    model will learn to generate.

    Args:
        text_dim: Dimension of text features
        image_dim: Dimension of image features
        embedding_dim: Dimension of continuous embeddings
        num_quantizers: Number of quantization levels (depth of semantic ID)
        codebook_size: Size of each codebook (vocabulary at each level)
        commitment_weight: Weight for commitment loss
        kmeans_init: Whether to initialize codebooks with k-means
        kmeans_iters: Number of k-means iterations
    """

    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 512,
        category_vocab_size: int = 100,
        embedding_dim: int = 256,
        num_quantizers: int = 4,  # Creates semantic IDs like (12, 56, 99, 3)
        codebook_size: int = 256,  # Each position can be 0-255
        commitment_weight: float = 1.0,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        **encoder_kwargs
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size

        # Encoder: Ad features -> Dense embedding
        self.encoder = AdEncoder(
            text_dim=text_dim,
            image_dim=image_dim,
            category_vocab_size=category_vocab_size,
            output_dim=embedding_dim,
            **encoder_kwargs
        )

        # Residual Vector Quantizer: Dense -> Discrete codes
        self.quantizer = ResidualVQ(
            dim=embedding_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            threshold_ema_dead_code=2,  # Replace dead codes
        )

        # Decoder: Quantized embedding -> Reconstructed features
        self.decoder = AdDecoder(
            input_dim=embedding_dim,
            text_dim=text_dim,
            image_dim=image_dim
        )

    def encode(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Encode ad features to dense embedding."""
        return self.encoder(text_features, image_features, category_ids)

    def quantize(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous embeddings to discrete codes.

        Args:
            embeddings: Continuous embeddings [batch, embedding_dim]

        Returns:
            Tuple of:
            - quantized: Quantized embeddings [batch, embedding_dim]
            - indices: Discrete semantic IDs [batch, num_quantizers]
            - commit_loss: Commitment loss
        """
        quantized, indices, commit_loss = self.quantizer(embeddings)
        return quantized, indices, commit_loss

    def decode(self, quantized_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode quantized embedding back to feature space."""
        return self.decoder(quantized_embedding)

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode -> quantize -> decode.

        Args:
            text_features: Text features [batch, text_dim]
            image_features: Image features [batch, image_dim]
            category_ids: Category IDs [batch]

        Returns:
            Dictionary containing:
            - semantic_ids: Discrete codes [batch, num_quantizers]
            - quantized: Quantized embeddings [batch, embedding_dim]
            - text_recon: Reconstructed text features
            - image_recon: Reconstructed image features
            - commit_loss: Commitment loss
        """
        # Encode
        embeddings = self.encode(text_features, image_features, category_ids)

        # Quantize
        quantized, semantic_ids, commit_loss = self.quantize(embeddings)

        # Decode
        text_recon, image_recon = self.decode(quantized)

        return {
            'semantic_ids': semantic_ids,
            'quantized': quantized,
            'embeddings': embeddings,
            'text_recon': text_recon,
            'image_recon': image_recon,
            'commit_loss': commit_loss
        }

    def get_semantic_id(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get semantic ID for an ad (inference mode).

        Args:
            text_features: Text features [batch, text_dim]
            image_features: Image features [batch, image_dim]
            category_ids: Category IDs [batch]

        Returns:
            Semantic IDs [batch, num_quantizers]
        """
        with torch.no_grad():
            embeddings = self.encode(text_features, image_features, category_ids)
            _, semantic_ids, _ = self.quantize(embeddings)
            return semantic_ids

    def lookup_by_semantic_id(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup quantized embeddings by semantic ID.

        Args:
            semantic_ids: Semantic IDs [batch, num_quantizers]

        Returns:
            Quantized embeddings [batch, embedding_dim]
        """
        # Use the quantizer's get_output_from_indices method
        quantized = self.quantizer.get_output_from_indices(semantic_ids)
        return quantized


def compute_rqvae_loss(
    outputs: Dict[str, torch.Tensor],
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    text_weight: float = 1.0,
    image_weight: float = 1.0,
    commit_weight: float = 0.25
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute RQ-VAE training loss.

    Loss = Reconstruction Loss + Commitment Loss

    Args:
        outputs: Model outputs dictionary
        text_features: Original text features
        image_features: Original image features
        text_weight: Weight for text reconstruction
        image_weight: Weight for image reconstruction
        commit_weight: Weight for commitment loss

    Returns:
        Tuple of (total_loss, loss_dict)
    """
    # Reconstruction losses (MSE)
    text_recon_loss = F.mse_loss(outputs['text_recon'], text_features)
    image_recon_loss = F.mse_loss(outputs['image_recon'], image_features)

    # Commitment loss (from quantizer)
    commit_loss = outputs['commit_loss']

    # Total loss
    total_loss = (
        text_weight * text_recon_loss +
        image_weight * image_recon_loss +
        commit_weight * commit_loss
    )

    # Loss breakdown
    loss_dict = {
        'total_loss': total_loss.item(),
        'text_recon_loss': text_recon_loss.item(),
        'image_recon_loss': image_recon_loss.item(),
        'commit_loss': commit_loss.item()
    }

    return total_loss, loss_dict
