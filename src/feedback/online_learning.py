"""
Online Learning and Feedback Loop Infrastructure.

This module implements real-time model updates based on user feedback.
The system continuously learns from:
- Click events
- Conversion events
- Skip events
- View time

Updates are applied within minutes of user interactions to keep models fresh.

Architecture:
1. Event Stream: Captures all user-ad interactions
2. Feature Store: Maintains up-to-date features
3. Online Trainer: Incremental model updates
4. Model Registry: Version management and deployment

Based on:
- Meta's real-time ML infrastructure
- LinkedIn's "Pro-ML" online learning system
- Twitter's online learning framework
"""
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading
import queue
import torch
import torch.nn as nn
from collections import deque, defaultdict
import pickle


@dataclass
class FeedbackEvent:
    """
    User feedback event.

    Attributes:
        event_id: Unique event identifier
        user_id: User who generated the event
        ad_id: Ad that was shown
        event_type: Type of event ('impression', 'click', 'conversion', 'skip')
        timestamp: When event occurred
        context: Additional context (device, location, etc.)
        value: Value of the event (conversion amount, view time, etc.)
        prediction_scores: Original model predictions for this impression
    """
    event_id: str
    user_id: str
    ad_id: str
    event_type: str  # 'impression', 'click', 'conversion', 'skip'
    timestamp: datetime
    context: Dict = field(default_factory=dict)
    value: float = 0.0
    prediction_scores: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'user_id': self.user_id,
            'ad_id': self.ad_id,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'value': self.value,
            'prediction_scores': self.prediction_scores
        }


class EventBuffer:
    """
    Thread-safe circular buffer for storing recent events.

    Used for:
    1. Online training on recent data
    2. Metrics computation
    3. Debugging and monitoring
    """

    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add(self, event: FeedbackEvent):
        """Add event to buffer."""
        with self.lock:
            self.buffer.append(event)

    def get_recent(self, n: int) -> List[FeedbackEvent]:
        """Get n most recent events."""
        with self.lock:
            return list(self.buffer)[-n:]

    def get_all(self) -> List[FeedbackEvent]:
        """Get all events in buffer."""
        with self.lock:
            return list(self.buffer)

    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()


class FeatureStore:
    """
    Real-time feature store for user and ad features.

    Maintains:
    - User interaction history (recent clicks, conversions)
    - Ad performance statistics (CTR, CVR)
    - Real-time aggregations
    """

    def __init__(self):
        self.user_features: Dict[str, Dict] = defaultdict(dict)
        self.ad_features: Dict[str, Dict] = defaultdict(dict)
        self.lock = threading.Lock()

        # Statistics
        self.ad_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'impressions': 0, 'clicks': 0, 'conversions': 0, 'skips': 0}
        )

    def update_from_event(self, event: FeedbackEvent):
        """Update features based on feedback event."""
        with self.lock:
            # Update ad statistics
            if event.event_type == 'impression':
                self.ad_stats[event.ad_id]['impressions'] += 1
            elif event.event_type == 'click':
                self.ad_stats[event.ad_id]['clicks'] += 1
            elif event.event_type == 'conversion':
                self.ad_stats[event.ad_id]['conversions'] += 1
            elif event.event_type == 'skip':
                self.ad_stats[event.ad_id]['skips'] += 1

            # Update user features
            if 'recent_interactions' not in self.user_features[event.user_id]:
                self.user_features[event.user_id]['recent_interactions'] = deque(maxlen=100)

            self.user_features[event.user_id]['recent_interactions'].append({
                'ad_id': event.ad_id,
                'event_type': event.event_type,
                'timestamp': event.timestamp
            })

    def get_user_features(self, user_id: str) -> Dict:
        """Get features for a user."""
        with self.lock:
            return dict(self.user_features.get(user_id, {}))

    def get_ad_features(self, ad_id: str) -> Dict:
        """Get features for an ad."""
        with self.lock:
            stats = self.ad_stats[ad_id]
            impressions = max(stats['impressions'], 1)

            return {
                'impressions': stats['impressions'],
                'clicks': stats['clicks'],
                'conversions': stats['conversions'],
                'skips': stats['skips'],
                'ctr': stats['clicks'] / impressions,
                'cvr': stats['conversions'] / impressions,
                'skip_rate': stats['skips'] / impressions
            }

    def save(self, filepath: str):
        """Save feature store to disk."""
        with self.lock:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'user_features': dict(self.user_features),
                    'ad_features': dict(self.ad_features),
                    'ad_stats': dict(self.ad_stats)
                }, f)

    def load(self, filepath: str):
        """Load feature store from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        with self.lock:
            self.user_features = defaultdict(dict, data['user_features'])
            self.ad_features = defaultdict(dict, data['ad_features'])
            self.ad_stats = defaultdict(
                lambda: {'impressions': 0, 'clicks': 0, 'conversions': 0, 'skips': 0},
                data['ad_stats']
            )


class OnlineTrainer:
    """
    Online model trainer that performs incremental updates.

    Uses mini-batch gradient descent on recent events to update models
    without full retraining.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        batch_size: int = 64,
        update_frequency: int = 100,  # Update every N events
        device: str = 'cpu'
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.device = device

        self.event_count = 0
        self.training_buffer: List[Dict] = []
        self.lock = threading.Lock()

    def add_training_example(self, example: Dict):
        """
        Add training example from feedback event.

        Args:
            example: Dictionary with 'features' and 'labels'
        """
        with self.lock:
            self.training_buffer.append(example)
            self.event_count += 1

            # Trigger update if buffer is full
            if len(self.training_buffer) >= self.batch_size:
                self._update_model()

    def _update_model(self):
        """Perform incremental model update."""
        if not self.training_buffer:
            return

        # Sample batch
        batch = self.training_buffer[:self.batch_size]
        self.training_buffer = self.training_buffer[self.batch_size:]

        # Convert to tensors
        try:
            features = torch.stack([ex['features'] for ex in batch]).to(self.device)
            labels = {
                task: torch.stack([ex['labels'][task] for ex in batch]).to(self.device)
                for task in batch[0]['labels'].keys()
            }

            # Forward pass
            self.model.train()
            predictions = self.model(features)

            # Compute loss
            loss = self.loss_fn(predictions, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        except Exception as e:
            print(f"Online training error: {e}")

    def force_update(self):
        """Force an immediate model update."""
        with self.lock:
            self._update_model()

    def get_stats(self) -> Dict:
        """Get training statistics."""
        with self.lock:
            return {
                'event_count': self.event_count,
                'buffer_size': len(self.training_buffer)
            }


class FeedbackLoop:
    """
    Complete feedback loop system.

    Coordinates:
    1. Event collection
    2. Feature store updates
    3. Online model training
    4. Metrics tracking
    """

    def __init__(
        self,
        event_buffer_size: int = 10000,
        feature_store: Optional[FeatureStore] = None,
        online_trainer: Optional[OnlineTrainer] = None
    ):
        self.event_buffer = EventBuffer(max_size=event_buffer_size)
        self.feature_store = feature_store or FeatureStore()
        self.online_trainer = online_trainer

        # Event queue for async processing
        self.event_queue = queue.Queue()
        self.running = False
        self.worker_thread = None

        # Metrics
        self.metrics = defaultdict(lambda: defaultdict(int))
        self.metrics_lock = threading.Lock()

    def start(self):
        """Start the feedback loop worker."""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._process_events, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """Stop the feedback loop worker."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

    def record_event(self, event: FeedbackEvent):
        """
        Record a feedback event.

        Args:
            event: Feedback event to record
        """
        # Add to queue for async processing
        self.event_queue.put(event)

        # Update metrics
        with self.metrics_lock:
            self.metrics[event.event_type]['count'] += 1
            self.metrics[event.event_type]['total_value'] += event.value

    def _process_events(self):
        """Worker thread to process events."""
        while self.running:
            try:
                # Get event with timeout
                event = self.event_queue.get(timeout=1)

                # Add to buffer
                self.event_buffer.add(event)

                # Update feature store
                self.feature_store.update_from_event(event)

                # Create training example and send to online trainer
                if self.online_trainer:
                    training_example = self._create_training_example(event)
                    if training_example:
                        self.online_trainer.add_training_example(training_example)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Event processing error: {e}")

    def _create_training_example(self, event: FeedbackEvent) -> Optional[Dict]:
        """
        Convert feedback event to training example.

        Args:
            event: Feedback event

        Returns:
            Training example dictionary or None
        """
        # This would be implemented based on your specific model architecture
        # For now, return None as placeholder
        return None

    def get_metrics(self) -> Dict:
        """Get current metrics."""
        with self.metrics_lock:
            return dict(self.metrics)

    def get_recent_events(self, n: int = 100) -> List[FeedbackEvent]:
        """Get recent events."""
        return self.event_buffer.get_recent(n)

    def compute_online_metrics(self) -> Dict[str, float]:
        """
        Compute online performance metrics.

        Returns:
            Dictionary with CTR, CVR, etc.
        """
        events = self.event_buffer.get_all()

        if not events:
            return {}

        impressions = sum(1 for e in events if e.event_type == 'impression')
        clicks = sum(1 for e in events if e.event_type == 'click')
        conversions = sum(1 for e in events if e.event_type == 'conversion')
        skips = sum(1 for e in events if e.event_type == 'skip')

        metrics = {
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'skips': skips,
            'ctr': clicks / max(impressions, 1),
            'cvr': conversions / max(impressions, 1),
            'skip_rate': skips / max(impressions, 1)
        }

        # Prediction calibration
        if events[0].prediction_scores:
            predicted_ctrs = []
            actual_clicks = []

            for event in events:
                if event.prediction_scores and 'click' in event.prediction_scores:
                    predicted_ctrs.append(event.prediction_scores['click'])
                    actual_clicks.append(1.0 if event.event_type == 'click' else 0.0)

            if predicted_ctrs:
                avg_predicted = sum(predicted_ctrs) / len(predicted_ctrs)
                avg_actual = sum(actual_clicks) / len(actual_clicks)
                metrics['prediction_bias'] = avg_predicted - avg_actual

        return metrics


class ModelRegistry:
    """
    Registry for managing model versions.

    Tracks:
    - Model versions
    - Performance metrics
    - Deployment status
    """

    def __init__(self, registry_path: str = './model_registry'):
        self.registry_path = registry_path
        self.versions: Dict[str, Dict] = {}
        self.active_version: Optional[str] = None

    def register_model(
        self,
        version: str,
        model_path: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None
    ):
        """Register a new model version."""
        self.versions[version] = {
            'version': version,
            'model_path': model_path,
            'metrics': metrics,
            'metadata': metadata or {},
            'registered_at': datetime.now(),
            'is_active': False
        }

    def activate_version(self, version: str):
        """Activate a model version."""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")

        # Deactivate current
        if self.active_version:
            self.versions[self.active_version]['is_active'] = False

        # Activate new
        self.versions[version]['is_active'] = True
        self.active_version = version

    def get_active_version(self) -> Optional[str]:
        """Get currently active version."""
        return self.active_version

    def get_version_info(self, version: str) -> Optional[Dict]:
        """Get information about a version."""
        return self.versions.get(version)

    def list_versions(self) -> List[Dict]:
        """List all versions."""
        return list(self.versions.values())
