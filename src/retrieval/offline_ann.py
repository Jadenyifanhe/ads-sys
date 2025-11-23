"""
Offline Approximate Nearest Neighbor (ANN) Indexing.

This module implements Pinterest's offline ANN approach for 70-80% cost reduction.
Instead of computing ANN search in real-time, we:
1. Precompute query embeddings offline
2. Run batch ANN search
3. Store results in cache
4. Online serving just does lookup

Based on:
- Pinterest Engineering: "Unlocking Efficient Ad Retrieval: Offline ANN"
- https://medium.com/pinterest-engineering/unlocking-efficient-ad-retrieval-offline-approximate-nearest-neighbors-in-pinterest-ads-6fccc131ac14
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import faiss
import pickle
import redis
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UserContext:
    """
    User context for clustering and caching.

    Attributes:
        demographics: User demographic features
        behavior_vector: Aggregated behavior features
        context_features: Time, device, location, etc.
        outcome_weights: Business objective weights
    """
    demographics: np.ndarray
    behavior_vector: np.ndarray
    context_features: np.ndarray
    outcome_weights: Optional[np.ndarray] = None

    def to_hash(self) -> str:
        """
        Create a hash key for this context.

        We use clustering/bucketing to group similar contexts together.
        """
        # Discretize continuous features for better cache hits
        demo_bucket = self._bucket_features(self.demographics, num_buckets=10)
        behavior_bucket = self._bucket_features(self.behavior_vector, num_buckets=20)
        context_bucket = self._bucket_features(self.context_features, num_buckets=5)

        # Combine into string
        key_parts = [
            f"d{demo_bucket}",
            f"b{behavior_bucket}",
            f"c{context_bucket}"
        ]

        if self.outcome_weights is not None:
            outcome_bucket = self._bucket_features(self.outcome_weights, num_buckets=5)
            key_parts.append(f"o{outcome_bucket}")

        key_string = "_".join(key_parts)

        # Hash for fixed-length key
        return hashlib.md5(key_string.encode()).hexdigest()

    @staticmethod
    def _bucket_features(features: np.ndarray, num_buckets: int) -> str:
        """Bucket continuous features into discrete bins."""
        # Normalize to [0, 1]
        features = (features - features.min()) / (features.max() - features.min() + 1e-8)

        # Discretize
        buckets = (features * num_buckets).astype(int)
        buckets = np.clip(buckets, 0, num_buckets - 1)

        return "".join(map(str, buckets))


@dataclass
class OfflineANNResult:
    """
    Result from offline ANN search.

    Attributes:
        ad_ids: List of candidate ad IDs
        scores: Similarity scores
        semantic_ids: Semantic IDs for the ads
        timestamp: When this result was computed
        ttl: Time-to-live in seconds
    """
    ad_ids: List[str]
    scores: np.ndarray
    semantic_ids: List[Tuple[int, ...]]
    timestamp: datetime = field(default_factory=datetime.now)
    ttl: int = 3600  # 1 hour default

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'ad_ids': self.ad_ids,
            'scores': self.scores.tolist(),
            'semantic_ids': [list(sid) for sid in self.semantic_ids],
            'timestamp': self.timestamp.isoformat(),
            'ttl': self.ttl
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'OfflineANNResult':
        """Deserialize from dictionary."""
        return cls(
            ad_ids=data['ad_ids'],
            scores=np.array(data['scores']),
            semantic_ids=[tuple(sid) for sid in data['semantic_ids']],
            timestamp=datetime.fromisoformat(data['timestamp']),
            ttl=data['ttl']
        )


class FaissIndexBuilder:
    """
    Builds and manages Faiss IVF-HNSW index.

    IVF (Inverted File): Partitions the space for faster search
    HNSW (Hierarchical Navigable Small World): Graph-based ANN for high recall
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        nlist: int = 1000,  # Number of IVF partitions
        m: int = 32,  # HNSW parameter
        metric: str = 'inner_product'  # or 'l2'
    ):
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        self.m = m
        self.metric = metric
        self.index: Optional[faiss.Index] = None

    def build_index(
        self,
        embeddings: np.ndarray,
        use_gpu: bool = False
    ) -> faiss.Index:
        """
        Build Faiss IVF-HNSW index.

        Args:
            embeddings: Ad embeddings [num_ads, embedding_dim]
            use_gpu: Whether to use GPU for index building

        Returns:
            Faiss index
        """
        num_ads = embeddings.shape[0]

        logger.info(f"Building Faiss IVF-HNSW index for {num_ads} ads...")

        # Create quantizer (HNSW)
        quantizer = faiss.IndexHNSWFlat(self.embedding_dim, self.m)

        # Create IVF index
        if self.metric == 'inner_product':
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                self.nlist,
                faiss.METRIC_INNER_PRODUCT
            )
        else:
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                self.nlist,
                faiss.METRIC_L2
            )

        # Move to GPU if requested
        if use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Using GPU for index building")
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self.index
            )

        # Train index
        logger.info("Training index...")
        self.index.train(embeddings.astype(np.float32))

        # Add vectors
        logger.info("Adding vectors to index...")
        self.index.add(embeddings.astype(np.float32))

        # Set search parameters
        self.index.nprobe = 10  # Number of clusters to visit during search

        logger.info("Index build complete!")

        return self.index

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.

        Args:
            query_embeddings: Query embeddings [num_queries, embedding_dim]
            k: Number of neighbors to return

        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            raise ValueError("Index not built yet!")

        query_embeddings = query_embeddings.astype(np.float32)
        distances, indices = self.index.search(query_embeddings, k)

        return distances, indices

    def save(self, filepath: str):
        """Save index to disk."""
        if self.index is None:
            raise ValueError("No index to save!")

        faiss.write_index(self.index, filepath)
        logger.info(f"Index saved to {filepath}")

    def load(self, filepath: str):
        """Load index from disk."""
        self.index = faiss.read_index(filepath)
        logger.info(f"Index loaded from {filepath}")


class OfflineANNCache:
    """
    Manages cached ANN results using Redis.

    Provides fast O(1) lookup for precomputed ANN results.
    """

    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        redis_db: int = 0,
        key_prefix: str = 'offline_ann:'
    ):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False
        )
        self.key_prefix = key_prefix

    def get(self, context_hash: str) -> Optional[OfflineANNResult]:
        """
        Retrieve cached result.

        Args:
            context_hash: Hash of user context

        Returns:
            OfflineANNResult if found, None otherwise
        """
        key = f"{self.key_prefix}{context_hash}"

        try:
            data = self.redis_client.get(key)

            if data is None:
                return None

            result_dict = pickle.loads(data)
            return OfflineANNResult.from_dict(result_dict)

        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None

    def set(
        self,
        context_hash: str,
        result: OfflineANNResult
    ):
        """
        Store result in cache.

        Args:
            context_hash: Hash of user context
            result: ANN result to store
        """
        key = f"{self.key_prefix}{context_hash}"

        try:
            data = pickle.dumps(result.to_dict())
            self.redis_client.setex(key, result.ttl, data)

        except Exception as e:
            logger.error(f"Error storing in cache: {e}")

    def delete(self, context_hash: str):
        """Delete cached result."""
        key = f"{self.key_prefix}{context_hash}"
        self.redis_client.delete(key)

    def clear_all(self):
        """Clear all cached results (use with caution!)."""
        pattern = f"{self.key_prefix}*"
        for key in self.redis_client.scan_iter(match=pattern):
            self.redis_client.delete(key)

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        pattern = f"{self.key_prefix}*"
        num_keys = sum(1 for _ in self.redis_client.scan_iter(match=pattern))

        info = self.redis_client.info('stats')

        return {
            'num_cached_contexts': num_keys,
            'cache_hits': info.get('keyspace_hits', 0),
            'cache_misses': info.get('keyspace_misses', 0),
            'hit_rate': info.get('keyspace_hits', 0) / max(
                info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1
            )
        }


class OfflineANNSystem:
    """
    Complete offline ANN system.

    Orchestrates index building, batch search, and caching.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        cache: Optional[OfflineANNCache] = None,
        index_builder: Optional[FaissIndexBuilder] = None
    ):
        self.embedding_dim = embedding_dim
        self.cache = cache or OfflineANNCache()
        self.index_builder = index_builder or FaissIndexBuilder(embedding_dim)

        # Mapping from index position to ad metadata
        self.ad_id_mapping: List[str] = []
        self.semantic_id_mapping: List[Tuple[int, ...]] = []

    def build_offline_index(
        self,
        ad_embeddings: np.ndarray,
        ad_ids: List[str],
        semantic_ids: List[Tuple[int, ...]],
        use_gpu: bool = False
    ):
        """
        Build the offline index.

        Args:
            ad_embeddings: Embeddings for all ads [num_ads, embedding_dim]
            ad_ids: List of ad IDs
            semantic_ids: List of semantic IDs
            use_gpu: Whether to use GPU
        """
        logger.info("Building offline ANN index...")

        # Store mappings
        self.ad_id_mapping = ad_ids
        self.semantic_id_mapping = semantic_ids

        # Build index
        self.index_builder.build_index(ad_embeddings, use_gpu=use_gpu)

        logger.info(f"Index built for {len(ad_ids)} ads")

    def precompute_for_contexts(
        self,
        contexts: List[UserContext],
        context_embeddings: np.ndarray,
        k: int = 100,
        ttl: int = 3600
    ):
        """
        Precompute ANN results for a batch of contexts.

        This is the "offline" part - run this in a batch job.

        Args:
            contexts: List of user contexts
            context_embeddings: Embeddings for contexts [num_contexts, embedding_dim]
            k: Number of neighbors per context
            ttl: Cache time-to-live in seconds
        """
        logger.info(f"Precomputing ANN for {len(contexts)} contexts...")

        # Batch search
        distances, indices = self.index_builder.search(context_embeddings, k=k)

        # Store results in cache
        for i, context in enumerate(contexts):
            context_hash = context.to_hash()

            # Get ad IDs and semantic IDs for this context
            ad_ids = [self.ad_id_mapping[idx] for idx in indices[i]]
            scores = distances[i]
            semantic_ids = [self.semantic_id_mapping[idx] for idx in indices[i]]

            result = OfflineANNResult(
                ad_ids=ad_ids,
                scores=scores,
                semantic_ids=semantic_ids,
                ttl=ttl
            )

            self.cache.set(context_hash, result)

        logger.info(f"Precomputed and cached {len(contexts)} context results")

    def retrieve_online(
        self,
        context: UserContext,
        fallback_to_search: bool = True
    ) -> Optional[OfflineANNResult]:
        """
        Online retrieval (fast lookup).

        This is the "online" part - called during serving.

        Args:
            context: User context
            fallback_to_search: If cache miss, fall back to real-time search

        Returns:
            OfflineANNResult or None
        """
        context_hash = context.to_hash()

        # Try cache first
        result = self.cache.get(context_hash)

        if result is not None:
            logger.debug(f"Cache hit for context {context_hash}")
            return result

        logger.debug(f"Cache miss for context {context_hash}")

        # Fallback to real-time search if needed
        if fallback_to_search and self.index_builder.index is not None:
            logger.warning("Cache miss - falling back to real-time ANN search")

            # Compute embedding for this context (this is expensive!)
            # In production, you'd have a model to do this
            # For now, we assume it's passed in separately
            return None  # Placeholder

        return None

    def save_index(self, filepath: str):
        """Save index to disk."""
        self.index_builder.save(filepath)

        # Save mappings
        mapping_file = filepath + '.mappings.pkl'
        with open(mapping_file, 'wb') as f:
            pickle.dump({
                'ad_ids': self.ad_id_mapping,
                'semantic_ids': self.semantic_id_mapping
            }, f)

    def load_index(self, filepath: str):
        """Load index from disk."""
        self.index_builder.load(filepath)

        # Load mappings
        mapping_file = filepath + '.mappings.pkl'
        with open(mapping_file, 'rb') as f:
            mappings = pickle.load(f)
            self.ad_id_mapping = mappings['ad_ids']
            self.semantic_id_mapping = mappings['semantic_ids']


def estimate_cost_savings(
    num_queries_per_second: int,
    num_contexts: int,
    online_search_latency_ms: float = 50,
    cache_lookup_latency_ms: float = 2,
    online_search_cost_per_1k: float = 0.10,
    cache_lookup_cost_per_1k: float = 0.001
) -> Dict:
    """
    Estimate cost savings from offline ANN.

    Args:
        num_queries_per_second: QPS
        num_contexts: Number of unique contexts to cache
        online_search_latency_ms: Latency of online ANN search
        cache_lookup_latency_ms: Latency of cache lookup
        online_search_cost_per_1k: Cost per 1000 online searches
        cache_lookup_cost_per_1k: Cost per 1000 cache lookups

    Returns:
        Dictionary with cost analysis
    """
    # Daily queries
    daily_queries = num_queries_per_second * 86400

    # Assume 80% cache hit rate
    cache_hit_rate = 0.8

    # Cached queries
    cached_queries = daily_queries * cache_hit_rate
    online_queries = daily_queries * (1 - cache_hit_rate)

    # Costs
    original_cost = (daily_queries / 1000) * online_search_cost_per_1k
    new_cost = (
        (cached_queries / 1000) * cache_lookup_cost_per_1k +
        (online_queries / 1000) * online_search_cost_per_1k
    )

    savings = original_cost - new_cost
    savings_percentage = (savings / original_cost) * 100

    # Latency
    original_latency = online_search_latency_ms
    new_latency = (
        cache_hit_rate * cache_lookup_latency_ms +
        (1 - cache_hit_rate) * online_search_latency_ms
    )

    return {
        'daily_queries': daily_queries,
        'cache_hit_rate': cache_hit_rate,
        'original_daily_cost': original_cost,
        'new_daily_cost': new_cost,
        'daily_savings': savings,
        'savings_percentage': savings_percentage,
        'original_latency_ms': original_latency,
        'new_latency_ms': new_latency,
        'latency_improvement': (original_latency - new_latency) / original_latency * 100
    }
