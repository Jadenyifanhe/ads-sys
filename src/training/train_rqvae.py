"""
Training pipeline for RQ-VAE Semantic Indexer.

This script trains the RQ-VAE model to convert ads into discrete semantic IDs.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_indexer import RQVAE, compute_rqvae_loss


class AdDataset(Dataset):
    """
    Dataset for ad features.

    Each sample contains:
    - text_features: Pre-computed text embeddings (from BERT/CLIP)
    - image_features: Pre-computed image embeddings (from ResNet/CLIP)
    - category_ids: Category ID
    - ad_id: Ad identifier
    """

    def __init__(
        self,
        text_features: np.ndarray,
        image_features: np.ndarray,
        category_ids: np.ndarray,
        ad_ids: List[str]
    ):
        assert len(text_features) == len(image_features) == len(category_ids) == len(ad_ids)

        self.text_features = torch.FloatTensor(text_features)
        self.image_features = torch.FloatTensor(image_features)
        self.category_ids = torch.LongTensor(category_ids)
        self.ad_ids = ad_ids

    def __len__(self):
        return len(self.ad_ids)

    def __getitem__(self, idx):
        return {
            'text_features': self.text_features[idx],
            'image_features': self.image_features[idx],
            'category_ids': self.category_ids[idx],
            'ad_id': self.ad_ids[idx]
        }


class RQVAETrainer:
    """Trainer for RQ-VAE model."""

    def __init__(
        self,
        model: RQVAE,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = './checkpoints/rqvae'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Optimizer
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 100  # 100 epochs
        )

        self.best_val_loss = float('inf')
        self.global_step = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        total_text_recon = 0
        total_image_recon = 0
        total_commit = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch in pbar:
            text_features = batch['text_features'].to(self.device)
            image_features = batch['image_features'].to(self.device)
            category_ids = batch['category_ids'].to(self.device)

            # Forward
            outputs = self.model(text_features, image_features, category_ids)

            # Compute loss
            loss, loss_dict = compute_rqvae_loss(
                outputs,
                text_features,
                image_features
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Accumulate metrics
            total_loss += loss_dict['total_loss']
            total_text_recon += loss_dict['text_recon_loss']
            total_image_recon += loss_dict['image_recon_loss']
            total_commit += loss_dict['commit_loss']
            num_batches += 1

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'text': f"{loss_dict['text_recon_loss']:.4f}",
                'image': f"{loss_dict['image_recon_loss']:.4f}"
            })

        return {
            'loss': total_loss / num_batches,
            'text_recon_loss': total_text_recon / num_batches,
            'image_recon_loss': total_image_recon / num_batches,
            'commit_loss': total_commit / num_batches
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc='Validation'):
            text_features = batch['text_features'].to(self.device)
            image_features = batch['image_features'].to(self.device)
            category_ids = batch['category_ids'].to(self.device)

            outputs = self.model(text_features, image_features, category_ids)
            loss, loss_dict = compute_rqvae_loss(outputs, text_features, image_features)

            total_loss += loss_dict['total_loss']
            num_batches += 1

        return {'val_loss': total_loss / num_batches}

    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'global_step': self.global_step
        }, checkpoint_path)

        print(f"Checkpoint saved to {checkpoint_path}")

    def train(self, num_epochs: int):
        """Full training loop."""
        print(f"Training RQ-VAE for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Print metrics
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")

            # Save checkpoint
            if epoch % 5 == 0:
                all_metrics = {**train_metrics, **val_metrics}
                self.save_checkpoint(epoch, all_metrics)

            # Save best model
            if val_metrics and val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                best_path = os.path.join(self.output_dir, 'best_model.pt')
                torch.save(self.model.state_dict(), best_path)
                print(f"  New best model saved! Val Loss: {self.best_val_loss:.4f}")

        print("\nTraining completed!")


def main():
    """Main training function."""
    # Configuration
    config = {
        'text_dim': 768,
        'image_dim': 512,
        'category_vocab_size': 100,
        'embedding_dim': 256,
        'num_quantizers': 4,
        'codebook_size': 256,
        'batch_size': 128,
        'learning_rate': 1e-4,
        'num_epochs': 50
    }

    # Create model
    model = RQVAE(
        text_dim=config['text_dim'],
        image_dim=config['image_dim'],
        category_vocab_size=config['category_vocab_size'],
        embedding_dim=config['embedding_dim'],
        num_quantizers=config['num_quantizers'],
        codebook_size=config['codebook_size']
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # TODO: Load actual data
    # For now, create dummy data
    num_samples = 10000
    dummy_text = np.random.randn(num_samples, config['text_dim'])
    dummy_image = np.random.randn(num_samples, config['image_dim'])
    dummy_categories = np.random.randint(0, config['category_vocab_size'], num_samples)
    dummy_ids = [f"ad_{i}" for i in range(num_samples)]

    # Create datasets
    train_dataset = AdDataset(
        dummy_text[:8000],
        dummy_image[:8000],
        dummy_categories[:8000],
        dummy_ids[:8000]
    )

    val_dataset = AdDataset(
        dummy_text[8000:],
        dummy_image[8000:],
        dummy_categories[8000:],
        dummy_ids[8000:]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Create trainer
    trainer = RQVAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate']
    )

    # Train
    trainer.train(config['num_epochs'])


if __name__ == '__main__':
    main()
