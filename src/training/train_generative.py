"""
Training pipeline for Generative Retrieval Model.

This script trains the autoregressive model to generate semantic IDs.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generative_retrieval import GenerativeRetrievalModel, compute_generative_loss


class UserAdDataset(Dataset):
    """
    Dataset for user-ad interactions.

    Each sample represents a user session and target ad to predict.
    """

    def __init__(
        self,
        user_histories: List[List[Tuple[int, ...]]],  # List of semantic ID sequences
        target_semantic_ids: List[Tuple[int, ...]],  # Target semantic IDs
        demographic_ids: np.ndarray,  # User demographic features
        context_features: np.ndarray,  # Context features
        max_history_len: int = 50
    ):
        self.user_histories = user_histories
        self.target_semantic_ids = target_semantic_ids
        self.demographic_ids = torch.LongTensor(demographic_ids)
        self.context_features = torch.FloatTensor(context_features)
        self.max_history_len = max_history_len

    def __len__(self):
        return len(self.user_histories)

    def __getitem__(self, idx):
        history = self.user_histories[idx][-self.max_history_len:]  # Truncate
        target = self.target_semantic_ids[idx]

        # Pad history if needed
        num_quantizers = len(history[0]) if history else 4
        while len(history) < self.max_history_len:
            history = [(0,) * num_quantizers] + history  # Prepend padding

        history_tensor = torch.LongTensor(history)
        target_tensor = torch.LongTensor(target)

        return {
            'semantic_ids': history_tensor,
            'demographic_ids': self.demographic_ids[idx],
            'context_features': self.context_features[idx],
            'target_semantic_ids': target_tensor,
            'history_len': min(len(self.user_histories[idx]), self.max_history_len)
        }


class GenerativeModelTrainer:
    """Trainer for Generative Retrieval Model."""

    def __init__(
        self,
        model: GenerativeRetrievalModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = './checkpoints/generative'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.best_val_accuracy = 0.0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        total_exact_match = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch in pbar:
            semantic_ids = batch['semantic_ids'].to(self.device)
            demographic_ids = batch['demographic_ids'].to(self.device)
            context_features = batch['context_features'].to(self.device)
            target_semantic_ids = batch['target_semantic_ids'].to(self.device)

            # Create attention mask
            history_lens = batch['history_len']
            max_len = semantic_ids.size(1)
            attention_mask = torch.zeros(semantic_ids.size(0), max_len, dtype=torch.bool)
            for i, length in enumerate(history_lens):
                attention_mask[i, -length:] = True
            attention_mask = attention_mask.to(self.device)

            # Forward
            logits = self.model(
                semantic_ids,
                demographic_ids,
                context_features,
                target_semantic_ids,
                attention_mask
            )

            # Compute loss
            loss, loss_dict = compute_generative_loss(
                logits,
                target_semantic_ids,
                label_smoothing=0.1
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Metrics
            total_loss += loss_dict['loss']
            total_exact_match += loss_dict['exact_match_acc']
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss_dict['loss']:.4f}",
                'acc': f"{loss_dict['exact_match_acc']:.3f}"
            })

        return {
            'loss': total_loss / num_batches,
            'exact_match_acc': total_exact_match / num_batches
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0
        total_exact_match = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc='Validation'):
            semantic_ids = batch['semantic_ids'].to(self.device)
            demographic_ids = batch['demographic_ids'].to(self.device)
            context_features = batch['context_features'].to(self.device)
            target_semantic_ids = batch['target_semantic_ids'].to(self.device)

            history_lens = batch['history_len']
            max_len = semantic_ids.size(1)
            attention_mask = torch.zeros(semantic_ids.size(0), max_len, dtype=torch.bool)
            for i, length in enumerate(history_lens):
                attention_mask[i, -length:] = True
            attention_mask = attention_mask.to(self.device)

            logits = self.model(
                semantic_ids,
                demographic_ids,
                context_features,
                target_semantic_ids,
                attention_mask
            )

            loss, loss_dict = compute_generative_loss(logits, target_semantic_ids)

            total_loss += loss_dict['loss']
            total_exact_match += loss_dict['exact_match_acc']
            num_batches += 1

        return {
            'val_loss': total_loss / num_batches,
            'val_exact_match_acc': total_exact_match / num_batches
        }

    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, checkpoint_path)

    def train(self, num_epochs: int):
        """Full training loop."""
        print(f"Training Generative Retrieval Model for {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train Acc: {train_metrics['exact_match_acc']:.3f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"  Val Acc: {val_metrics['val_exact_match_acc']:.3f}")

            if epoch % 5 == 0:
                all_metrics = {**train_metrics, **val_metrics}
                self.save_checkpoint(epoch, all_metrics)

            if val_metrics and val_metrics['val_exact_match_acc'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['val_exact_match_acc']
                best_path = os.path.join(self.output_dir, 'best_model.pt')
                torch.save(self.model.state_dict(), best_path)
                print(f"  New best model! Acc: {self.best_val_accuracy:.3f}")

        print("\nTraining completed!")
