"""
Training pipeline for DLRM Ranking Model.

This script trains the multi-task ranking model.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ranking import DLRM, compute_ranking_loss


class RankingDataset(Dataset):
    """Dataset for ranking training."""

    def __init__(
        self,
        user_categorical: np.ndarray,
        user_continuous: np.ndarray,
        ad_semantic_embeddings: np.ndarray,
        ad_categorical: np.ndarray,
        ad_continuous: np.ndarray,
        labels: Dict[str, np.ndarray]  # {click, conversion, skip, view_time}
    ):
        self.user_categorical = torch.LongTensor(user_categorical)
        self.user_continuous = torch.FloatTensor(user_continuous)
        self.ad_semantic_embeddings = torch.FloatTensor(ad_semantic_embeddings)
        self.ad_categorical = torch.LongTensor(ad_categorical)
        self.ad_continuous = torch.FloatTensor(ad_continuous)
        self.labels = {k: torch.FloatTensor(v) for k, v in labels.items()}

    def __len__(self):
        return len(self.user_categorical)

    def __getitem__(self, idx):
        return {
            'user_categorical': self.user_categorical[idx],
            'user_continuous': self.user_continuous[idx],
            'ad_semantic_embeddings': self.ad_semantic_embeddings[idx],
            'ad_categorical': self.ad_categorical[idx],
            'ad_continuous': self.ad_continuous[idx],
            'labels': {k: v[idx] for k, v in self.labels.items()}
        }


class RankingTrainer:
    """Trainer for DLRM Ranking Model."""

    def __init__(
        self,
        model: DLRM,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = './checkpoints/ranking'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_losses = {'total_loss': 0}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch in pbar:
            # Move to device
            user_cat = batch['user_categorical'].to(self.device)
            user_cont = batch['user_continuous'].to(self.device)
            ad_sem = batch['ad_semantic_embeddings'].to(self.device)
            ad_cat = batch['ad_categorical'].to(self.device)
            ad_cont = batch['ad_continuous'].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch['labels'].items()}

            # Forward
            predictions = self.model(user_cat, user_cont, ad_sem, ad_cat, ad_cont)

            # Compute loss
            loss, loss_dict = compute_ranking_loss(predictions, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate
            for k, v in loss_dict.items():
                if k not in total_losses:
                    total_losses[k] = 0
                total_losses[k] += v

            num_batches += 1

            pbar.set_postfix({'loss': f"{loss_dict['total_loss']:.4f}"})

        return {k: v / num_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_losses = {'val_total_loss': 0}
        num_batches = 0

        for batch in tqdm(self.val_loader, desc='Validation'):
            user_cat = batch['user_categorical'].to(self.device)
            user_cont = batch['user_continuous'].to(self.device)
            ad_sem = batch['ad_semantic_embeddings'].to(self.device)
            ad_cat = batch['ad_categorical'].to(self.device)
            ad_cont = batch['ad_continuous'].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch['labels'].items()}

            predictions = self.model(user_cat, user_cont, ad_sem, ad_cat, ad_cont)
            loss, loss_dict = compute_ranking_loss(predictions, labels)

            total_losses['val_total_loss'] += loss_dict['total_loss']
            num_batches += 1

        return {k: v / num_batches for k, v in total_losses.items()}

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
        print(f"Training DLRM Ranking Model for {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['val_total_loss']:.4f}")

            if epoch % 5 == 0:
                all_metrics = {**train_metrics, **val_metrics}
                self.save_checkpoint(epoch, all_metrics)

            if val_metrics and val_metrics['val_total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_total_loss']
                best_path = os.path.join(self.output_dir, 'best_model.pt')
                torch.save(self.model.state_dict(), best_path)
                print(f"  New best model! Loss: {self.best_val_loss:.4f}")

        print("\nTraining completed!")
