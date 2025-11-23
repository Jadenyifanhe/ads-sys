"""
Generative Retrieval Model (TIGER/DSI-style).

This module implements the core generative retrieval engine that learns to
autoregressively generate semantic IDs of ads that a user should see next.

Architecture: Encoder-Decoder Transformer (T5-style)
- Encoder: Processes user history and context
- Decoder: Autoregressively generates semantic ID tokens

Key Innovation: Constrained Beam Search
- Uses Prefix Trie to ensure only valid semantic IDs are generated
- Prevents hallucination of non-existent ads

Based on:
- "Autoregressive Entity Retrieval" (TIGER)
- "Transformer Memory as a Differentiable Search Index" (DSI)
- "Recommender Systems with Generative Retrieval" (HSTU)
"""
import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration, T5EncoderModel
from typing import List, Tuple, Optional, Dict
import torch.nn.functional as F
from .prefix_trie import SemanticIDTrie


class UserHistoryEncoder(nn.Module):
    """
    Encodes user interaction history into a sequence representation.

    Takes:
    - Sequence of semantic IDs from user's past interactions
    - User demographic features
    - Contextual features (time, device, etc.)

    Produces:
    - Encoded representation for the decoder to condition on
    """

    def __init__(
        self,
        num_quantizers: int = 4,
        codebook_size: int = 256,
        embed_dim: int = 512,
        num_demographics: int = 50,
        demographic_embed_dim: int = 64,
        max_history_len: int = 100,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim

        # Embedding for each position in semantic ID
        # We create separate embeddings for each quantizer level
        self.semantic_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, embed_dim // num_quantizers)
            for _ in range(num_quantizers)
        ])

        # Position embedding for sequence
        self.position_embedding = nn.Embedding(max_history_len, embed_dim)

        # Demographic embeddings
        self.demographic_embedding = nn.Embedding(num_demographics, demographic_embed_dim)

        # Context features MLP
        self.context_mlp = nn.Sequential(
            nn.Linear(demographic_embed_dim + 32, embed_dim),  # 32 for context features
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
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode user history.

        Args:
            semantic_ids: Historical semantic IDs [batch, history_len, num_quantizers]
            demographic_ids: User demographic feature IDs [batch, num_demo_features]
            context_features: Contextual features [batch, context_dim]
            attention_mask: Attention mask [batch, history_len]

        Returns:
            Encoded representation [batch, history_len + 1, embed_dim]
            (+1 for the prepended context token)
        """
        batch_size, history_len, _ = semantic_ids.shape

        # Embed semantic IDs: concatenate embeddings from each quantizer level
        semantic_embeds = []
        for i in range(self.num_quantizers):
            embeds = self.semantic_embeddings[i](semantic_ids[:, :, i])
            semantic_embeds.append(embeds)
        semantic_embeds = torch.cat(semantic_embeds, dim=-1)  # [batch, history_len, embed_dim]

        # Add position embeddings
        positions = torch.arange(history_len, device=semantic_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(positions)
        semantic_embeds = semantic_embeds + position_embeds

        # Process demographics and context
        demo_embeds = self.demographic_embedding(demographic_ids).mean(dim=1)  # [batch, demo_embed_dim]
        context_input = torch.cat([demo_embeds, context_features], dim=-1)  # [batch, demo_embed_dim + context_dim]
        context_token = self.context_mlp(context_input).unsqueeze(1)  # [batch, 1, embed_dim]

        # Concatenate context token with history
        x = torch.cat([context_token, semantic_embeds], dim=1)  # [batch, history_len + 1, embed_dim]

        # Create attention mask
        if attention_mask is not None:
            # Prepend 1 for context token
            context_mask = torch.ones(batch_size, 1, device=attention_mask.device)
            attention_mask = torch.cat([context_mask, attention_mask], dim=1)
            # Convert to transformer mask format
            attention_mask = attention_mask.bool()
        else:
            attention_mask = None

        # Encode with transformer
        encoded = self.transformer(x, src_key_padding_mask=~attention_mask if attention_mask is not None else None)

        return encoded


class SemanticIDDecoder(nn.Module):
    """
    Autoregressive decoder that generates semantic IDs token by token.

    Uses Transformer decoder with causal attention.
    At each step, predicts the next token in the semantic ID.
    """

    def __init__(
        self,
        num_quantizers: int = 4,
        codebook_size: int = 256,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim

        # Output embedding (same as encoder for weight tying)
        self.output_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, embed_dim // num_quantizers)
            for _ in range(num_quantizers)
        ])

        # Position embedding for decoder
        self.position_embedding = nn.Embedding(num_quantizers, embed_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection for each position
        self.output_projection = nn.Linear(embed_dim, codebook_size)

    def forward(
        self,
        encoder_output: torch.Tensor,  # [batch, seq_len, embed_dim]
        target_semantic_ids: Optional[torch.Tensor] = None,  # [batch, num_quantizers] (for training)
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            encoder_output: Output from encoder [batch, seq_len, embed_dim]
            target_semantic_ids: Target semantic IDs [batch, num_quantizers]
            encoder_attention_mask: Mask for encoder output

        Returns:
            Logits for each position [batch, num_quantizers, codebook_size]
        """
        if target_semantic_ids is None:
            raise ValueError("target_semantic_ids required for training")

        batch_size = target_semantic_ids.shape[0]

        # Embed target (teacher forcing)
        target_embeds = []
        for i in range(self.num_quantizers):
            embeds = self.output_embeddings[i](target_semantic_ids[:, i])
            target_embeds.append(embeds)
        target_embeds = torch.stack(target_embeds, dim=1)  # [batch, num_quantizers, embed_dim/num_quantizers]

        # Reshape to [batch, num_quantizers, embed_dim]
        target_embeds = target_embeds.reshape(batch_size, self.num_quantizers, -1)
        if target_embeds.shape[-1] != self.embed_dim:
            target_embeds = F.pad(target_embeds, (0, self.embed_dim - target_embeds.shape[-1]))

        # Add position embeddings
        positions = torch.arange(self.num_quantizers, device=target_semantic_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(positions)
        target_embeds = target_embeds + position_embeds

        # Create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            self.num_quantizers,
            device=target_semantic_ids.device
        )

        # Decode
        decoded = self.transformer_decoder(
            target_embeds,
            encoder_output,
            tgt_mask=causal_mask,
            memory_key_padding_mask=~encoder_attention_mask if encoder_attention_mask is not None else None
        )

        # Project to vocabulary
        logits = self.output_projection(decoded)  # [batch, num_quantizers, codebook_size]

        return logits

    def generate_step(
        self,
        encoder_output: torch.Tensor,
        generated_so_far: torch.Tensor,  # [batch, current_len]
        position: int,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate next token (for inference).

        Args:
            encoder_output: Encoder output [batch, seq_len, embed_dim]
            generated_so_far: Already generated tokens [batch, current_len]
            position: Current position being generated
            encoder_attention_mask: Encoder mask

        Returns:
            Logits for next token [batch, codebook_size]
        """
        batch_size = generated_so_far.shape[0]
        current_len = generated_so_far.shape[1]

        # Embed generated tokens so far
        if current_len > 0:
            target_embeds = []
            for i in range(current_len):
                embeds = self.output_embeddings[i](generated_so_far[:, i])
                target_embeds.append(embeds)
            target_embeds = torch.stack(target_embeds, dim=1)  # [batch, current_len, embed_dim/num_q]
            target_embeds = target_embeds.reshape(batch_size, current_len, -1)
            if target_embeds.shape[-1] != self.embed_dim:
                target_embeds = F.pad(target_embeds, (0, self.embed_dim - target_embeds.shape[-1]))

            # Add position embeddings
            positions = torch.arange(current_len, device=generated_so_far.device).unsqueeze(0)
            position_embeds = self.position_embedding(positions)
            target_embeds = target_embeds + position_embeds
        else:
            # Start with zero embedding
            target_embeds = torch.zeros(batch_size, 1, self.embed_dim, device=encoder_output.device)

        # Create causal mask
        if current_len > 0:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                current_len,
                device=generated_so_far.device
            )
        else:
            causal_mask = None

        # Decode
        decoded = self.transformer_decoder(
            target_embeds,
            encoder_output,
            tgt_mask=causal_mask,
            memory_key_padding_mask=~encoder_attention_mask if encoder_attention_mask is not None else None
        )

        # Project last position
        logits = self.output_projection(decoded[:, -1, :])  # [batch, codebook_size]

        return logits


class GenerativeRetrievalModel(nn.Module):
    """
    Complete Generative Retrieval Model.

    Combines encoder and decoder for end-to-end training.
    """

    def __init__(
        self,
        num_quantizers: int = 4,
        codebook_size: int = 256,
        embed_dim: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        num_demographics: int = 50,
        max_history_len: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size

        self.encoder = UserHistoryEncoder(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            embed_dim=embed_dim,
            num_demographics=num_demographics,
            max_history_len=max_history_len,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        self.decoder = SemanticIDDecoder(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            embed_dim=embed_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(
        self,
        semantic_ids: torch.Tensor,
        demographic_ids: torch.Tensor,
        context_features: torch.Tensor,
        target_semantic_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Returns:
            Logits [batch, num_quantizers, codebook_size]
        """
        # Encode
        encoded = self.encoder(semantic_ids, demographic_ids, context_features, attention_mask)

        # Create encoder attention mask
        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            context_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=torch.bool)
            encoder_mask = torch.cat([context_mask, attention_mask], dim=1)
        else:
            encoder_mask = None

        # Decode
        logits = self.decoder(encoded, target_semantic_ids, encoder_mask)

        return logits

    @torch.no_grad()
    def generate(
        self,
        semantic_ids: torch.Tensor,
        demographic_ids: torch.Tensor,
        context_features: torch.Tensor,
        trie: SemanticIDTrie,
        num_beams: int = 10,
        num_return: int = 50,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate semantic IDs using constrained beam search.

        Args:
            semantic_ids: User history semantic IDs
            demographic_ids: User demographic IDs
            context_features: Context features
            trie: Prefix trie for constraint
            num_beams: Number of beams
            num_return: Number of results to return
            attention_mask: Attention mask

        Returns:
            Tuple of (generated_ids, scores)
            - generated_ids: [batch, num_return, num_quantizers]
            - scores: [batch, num_return]
        """
        # Encode
        encoded = self.encoder(semantic_ids, demographic_ids, context_features, attention_mask)

        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            context_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=torch.bool)
            encoder_mask = torch.cat([context_mask, attention_mask], dim=1)
        else:
            encoder_mask = None

        # Constrained beam search
        generated_ids, scores = self._constrained_beam_search(
            encoded,
            encoder_mask,
            trie,
            num_beams,
            num_return
        )

        return generated_ids, scores

    def _constrained_beam_search(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        trie: SemanticIDTrie,
        num_beams: int,
        num_return: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constrained beam search using the prefix trie.

        At each step, only tokens that lead to valid semantic IDs are considered.
        """
        batch_size = encoder_output.shape[0]
        device = encoder_output.device

        # Initialize beams: [batch, num_beams, current_len]
        beams = torch.zeros(batch_size, num_beams, 0, dtype=torch.long, device=device)
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        completed = [[] for _ in range(batch_size)]  # Store completed beams

        # Expand encoder output for beams
        encoder_output = encoder_output.unsqueeze(1).expand(-1, num_beams, -1, -1)
        encoder_output = encoder_output.reshape(batch_size * num_beams, *encoder_output.shape[2:])

        if encoder_mask is not None:
            encoder_mask = encoder_mask.unsqueeze(1).expand(-1, num_beams, -1)
            encoder_mask = encoder_mask.reshape(batch_size * num_beams, -1)

        # Generate each position
        for pos in range(self.num_quantizers):
            # Reshape beams for model
            beams_flat = beams.reshape(batch_size * num_beams, -1)

            # Get logits for next position
            logits = self.decoder.generate_step(
                encoder_output,
                beams_flat,
                pos,
                encoder_mask
            )  # [batch * num_beams, codebook_size]

            logits = logits.reshape(batch_size, num_beams, self.codebook_size)
            log_probs = F.log_softmax(logits, dim=-1)

            # Apply trie constraints
            for b in range(batch_size):
                for beam_idx in range(num_beams):
                    prefix = tuple(beams[b, beam_idx].tolist())
                    valid_tokens = trie.get_valid_next_tokens(prefix)

                    if not valid_tokens:
                        # Dead beam - set to very negative
                        log_probs[b, beam_idx, :] = float('-inf')
                    else:
                        # Mask invalid tokens
                        mask = torch.ones(self.codebook_size, dtype=torch.bool, device=device)
                        mask[valid_tokens] = False
                        log_probs[b, beam_idx, mask] = float('-inf')

            # Compute beam scores
            # [batch, num_beams, codebook_size]
            candidate_scores = beam_scores.unsqueeze(-1) + log_probs

            # Flatten and get top-k
            candidate_scores_flat = candidate_scores.reshape(batch_size, -1)  # [batch, num_beams * codebook_size]
            top_scores, top_indices = candidate_scores_flat.topk(num_beams, dim=-1)  # [batch, num_beams]

            # Compute which beam and which token
            beam_indices = top_indices // self.codebook_size  # [batch, num_beams]
            token_indices = top_indices % self.codebook_size  # [batch, num_beams]

            # Update beams
            new_beams = []
            new_scores = []

            for b in range(batch_size):
                batch_beams = []
                batch_scores = []

                for i in range(num_beams):
                    beam_idx = beam_indices[b, i]
                    token_idx = token_indices[b, i]
                    score = top_scores[b, i]

                    # Get previous beam
                    prev_beam = beams[b, beam_idx]

                    # Append new token
                    new_beam = torch.cat([prev_beam, token_idx.unsqueeze(0)])

                    # Check if complete
                    if pos == self.num_quantizers - 1:
                        # This is a complete semantic ID
                        completed[b].append((new_beam, score))
                    else:
                        batch_beams.append(new_beam)
                        batch_scores.append(score)

                # Pad if needed
                while len(batch_beams) < num_beams:
                    batch_beams.append(torch.zeros(pos + 1, dtype=torch.long, device=device))
                    batch_scores.append(torch.tensor(float('-inf'), device=device))

                new_beams.append(torch.stack(batch_beams[:num_beams]))
                new_scores.append(torch.stack(batch_scores[:num_beams]))

            beams = torch.stack(new_beams)  # [batch, num_beams, pos + 1]
            beam_scores = torch.stack(new_scores)  # [batch, num_beams]

        # Collect top-k completed beams
        result_ids = []
        result_scores = []

        for b in range(batch_size):
            # Sort completed beams by score
            completed[b].sort(key=lambda x: x[1], reverse=True)

            batch_ids = []
            batch_scores = []

            for beam, score in completed[b][:num_return]:
                batch_ids.append(beam)
                batch_scores.append(score)

            # Pad if needed
            while len(batch_ids) < num_return:
                batch_ids.append(torch.zeros(self.num_quantizers, dtype=torch.long, device=device))
                batch_scores.append(torch.tensor(float('-inf'), device=device))

            result_ids.append(torch.stack(batch_ids[:num_return]))
            result_scores.append(torch.stack(batch_scores[:num_return]))

        result_ids = torch.stack(result_ids)  # [batch, num_return, num_quantizers]
        result_scores = torch.stack(result_scores)  # [batch, num_return]

        return result_ids, result_scores


def compute_generative_loss(
    logits: torch.Tensor,
    target_semantic_ids: torch.Tensor,
    label_smoothing: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute cross-entropy loss for generative retrieval.

    Args:
        logits: Predicted logits [batch, num_quantizers, codebook_size]
        target_semantic_ids: Target semantic IDs [batch, num_quantizers]
        label_smoothing: Label smoothing factor

    Returns:
        Tuple of (loss, loss_dict)
    """
    batch_size, num_quantizers, codebook_size = logits.shape

    # Reshape for cross entropy
    logits = logits.reshape(-1, codebook_size)
    targets = target_semantic_ids.reshape(-1)

    # Compute cross entropy with label smoothing
    loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)

    # Compute accuracy per position
    predictions = logits.argmax(dim=-1).reshape(batch_size, num_quantizers)
    accuracy_per_pos = (predictions == target_semantic_ids).float().mean(dim=0)

    # Exact match accuracy (all positions correct)
    exact_match = (predictions == target_semantic_ids).all(dim=1).float().mean()

    loss_dict = {
        'loss': loss.item(),
        'exact_match_acc': exact_match.item(),
        'per_position_acc': accuracy_per_pos.tolist()
    }

    return loss, loss_dict
