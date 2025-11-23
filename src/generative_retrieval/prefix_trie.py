"""
Prefix Trie for Constrained Beam Search.

This data structure is critical for preventing hallucination in the generative
retrieval model. It ensures that the model can only generate semantic IDs that
actually exist in the ad corpus.

During autoregressive generation, at each step, the trie restricts the next
token to only those that lead to valid complete semantic IDs.
"""
from typing import List, Tuple, Set, Optional, Dict
import pickle
from collections import defaultdict
import torch


class TrieNode:
    """
    Node in the Prefix Trie.

    Each node represents a position in the semantic ID sequence.
    Children represent valid next tokens at this position.
    """

    def __init__(self, token: Optional[int] = None):
        self.token = token
        self.children: Dict[int, 'TrieNode'] = {}
        self.is_terminal = False  # True if this completes a valid semantic ID
        self.ad_id: Optional[str] = None  # Store ad_id if terminal

    def add_child(self, token: int) -> 'TrieNode':
        """Add a child node with given token."""
        if token not in self.children:
            self.children[token] = TrieNode(token)
        return self.children[token]

    def get_child(self, token: int) -> Optional['TrieNode']:
        """Get child node for given token."""
        return self.children.get(token)

    def get_valid_next_tokens(self) -> List[int]:
        """Get list of valid next tokens from this node."""
        return list(self.children.keys())

    def is_valid_next_token(self, token: int) -> bool:
        """Check if token is valid from this node."""
        return token in self.children


class SemanticIDTrie:
    """
    Trie data structure for storing all valid semantic IDs.

    This trie:
    1. Stores all valid semantic ID sequences in the ad corpus
    2. Enables O(1) lookup of valid next tokens during generation
    3. Supports efficient batch operations for constrained beam search
    4. Maps complete sequences back to ad IDs

    Example:
        If we have ads with semantic IDs:
        - Ad1: (12, 56, 99)
        - Ad2: (12, 56, 100)
        - Ad3: (12, 78, 45)

        The trie structure:
        root
        └─ 12
           ├─ 56
           │  ├─ 99  [terminal, ad_id=Ad1]
           │  └─ 100 [terminal, ad_id=Ad2]
           └─ 78
              └─ 45  [terminal, ad_id=Ad3]

        At position after (12,), valid next tokens are [56, 78]
        At position after (12, 56,), valid next tokens are [99, 100]
    """

    def __init__(self, num_quantizers: int, codebook_size: int):
        """
        Initialize the trie.

        Args:
            num_quantizers: Depth of semantic IDs (e.g., 4)
            codebook_size: Vocabulary size at each position (e.g., 256)
        """
        self.root = TrieNode()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self._size = 0  # Number of semantic IDs stored

    def insert(self, semantic_id: Tuple[int, ...], ad_id: str):
        """
        Insert a semantic ID into the trie.

        Args:
            semantic_id: Tuple of integers representing the semantic ID
            ad_id: Associated ad ID for this semantic ID
        """
        assert len(semantic_id) == self.num_quantizers, \
            f"Expected semantic_id of length {self.num_quantizers}, got {len(semantic_id)}"

        node = self.root
        for token in semantic_id:
            assert 0 <= token < self.codebook_size, \
                f"Token {token} out of range [0, {self.codebook_size})"
            node = node.add_child(token)

        # Mark terminal and store ad_id
        if not node.is_terminal:
            node.is_terminal = True
            node.ad_id = ad_id
            self._size += 1

    def insert_batch(self, semantic_ids: List[Tuple[int, ...]], ad_ids: List[str]):
        """
        Insert multiple semantic IDs.

        Args:
            semantic_ids: List of semantic ID tuples
            ad_ids: List of corresponding ad IDs
        """
        assert len(semantic_ids) == len(ad_ids)
        for semantic_id, ad_id in zip(semantic_ids, ad_ids):
            self.insert(semantic_id, ad_id)

    def search(self, semantic_id: Tuple[int, ...]) -> Optional[str]:
        """
        Search for a semantic ID in the trie.

        Args:
            semantic_id: Semantic ID to search for

        Returns:
            ad_id if found, None otherwise
        """
        node = self.root
        for token in semantic_id:
            node = node.get_child(token)
            if node is None:
                return None

        return node.ad_id if node.is_terminal else None

    def get_valid_next_tokens(self, prefix: Tuple[int, ...]) -> List[int]:
        """
        Get valid next tokens given a prefix.

        Args:
            prefix: Partial semantic ID sequence

        Returns:
            List of valid next token IDs
        """
        node = self.root

        # Navigate to the node corresponding to the prefix
        for token in prefix:
            node = node.get_child(token)
            if node is None:
                return []  # Invalid prefix

        return node.get_valid_next_tokens()

    def get_valid_next_tokens_batch(
        self,
        prefixes: torch.Tensor
    ) -> List[List[int]]:
        """
        Get valid next tokens for a batch of prefixes.

        Args:
            prefixes: Tensor of shape [batch_size, prefix_length]

        Returns:
            List of valid next token lists for each prefix
        """
        batch_size = prefixes.shape[0]
        result = []

        for i in range(batch_size):
            prefix = tuple(prefixes[i].tolist())
            valid_tokens = self.get_valid_next_tokens(prefix)
            result.append(valid_tokens)

        return result

    def create_constraint_mask(
        self,
        prefixes: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create a constraint mask for beam search.

        Args:
            prefixes: Current prefixes [batch_size, prefix_length]
            device: Device for the mask tensor

        Returns:
            Boolean mask of shape [batch_size, codebook_size]
            where True indicates a valid next token
        """
        batch_size = prefixes.shape[0]
        mask = torch.zeros(batch_size, self.codebook_size, dtype=torch.bool, device=device)

        for i in range(batch_size):
            prefix = tuple(prefixes[i].tolist())
            valid_tokens = self.get_valid_next_tokens(prefix)
            if valid_tokens:
                mask[i, valid_tokens] = True

        return mask

    def is_complete(self, semantic_id: Tuple[int, ...]) -> bool:
        """
        Check if a semantic ID is complete and valid.

        Args:
            semantic_id: Semantic ID to check

        Returns:
            True if the semantic ID is complete and exists in the trie
        """
        if len(semantic_id) != self.num_quantizers:
            return False

        node = self.root
        for token in semantic_id:
            node = node.get_child(token)
            if node is None:
                return False

        return node.is_terminal

    def get_all_semantic_ids(self) -> List[Tuple[Tuple[int, ...], str]]:
        """
        Get all semantic IDs and their corresponding ad IDs.

        Returns:
            List of (semantic_id, ad_id) tuples
        """
        result = []

        def dfs(node: TrieNode, path: List[int]):
            if node.is_terminal:
                result.append((tuple(path), node.ad_id))

            for token, child in node.children.items():
                dfs(child, path + [token])

        dfs(self.root, [])
        return result

    def save(self, filepath: str):
        """Save trie to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'root': self.root,
                'num_quantizers': self.num_quantizers,
                'codebook_size': self.codebook_size,
                'size': self._size
            }, f)

    @classmethod
    def load(cls, filepath: str) -> 'SemanticIDTrie':
        """Load trie from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        trie = cls(data['num_quantizers'], data['codebook_size'])
        trie.root = data['root']
        trie._size = data['size']
        return trie

    def __len__(self) -> int:
        """Return number of semantic IDs in the trie."""
        return self._size

    def get_stats(self) -> Dict:
        """Get statistics about the trie."""
        stats = {
            'total_semantic_ids': self._size,
            'num_quantizers': self.num_quantizers,
            'codebook_size': self.codebook_size,
            'branching_factors': self._compute_branching_factors()
        }
        return stats

    def _compute_branching_factors(self) -> List[float]:
        """
        Compute average branching factor at each level.

        This helps understand the semantic tree structure.
        High branching = diverse vocabulary at that level.
        Low branching = similar semantic clustering.
        """
        level_stats = defaultdict(list)

        def dfs(node: TrieNode, depth: int):
            if node.children:
                level_stats[depth].append(len(node.children))
                for child in node.children.values():
                    dfs(child, depth + 1)

        dfs(self.root, 0)

        branching_factors = []
        for i in range(self.num_quantizers):
            if i in level_stats:
                avg_branching = sum(level_stats[i]) / len(level_stats[i])
                branching_factors.append(avg_branching)
            else:
                branching_factors.append(0.0)

        return branching_factors


def build_trie_from_ads(
    semantic_ids: List[Tuple[int, ...]],
    ad_ids: List[str],
    num_quantizers: int,
    codebook_size: int
) -> SemanticIDTrie:
    """
    Utility function to build a trie from a list of ads.

    Args:
        semantic_ids: List of semantic ID tuples
        ad_ids: List of ad IDs
        num_quantizers: Number of quantization levels
        codebook_size: Codebook vocabulary size

    Returns:
        Populated SemanticIDTrie
    """
    trie = SemanticIDTrie(num_quantizers, codebook_size)
    trie.insert_batch(semantic_ids, ad_ids)
    return trie
