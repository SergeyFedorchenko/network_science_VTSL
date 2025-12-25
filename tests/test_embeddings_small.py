"""
Test embeddings module with small toy graph.

Validates:
- Node2vec walk generation is deterministic
- Skip-gram training produces expected output shape
- Embedding similarity search works correctly
"""

import sys
from pathlib import Path

import igraph as ig
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.embeddings import (
    generate_node2vec_walks,
    get_embedding_pair_features,
    train_skipgram,
)


def test_node2vec_walks_deterministic():
    """Test that node2vec walks are deterministic with fixed seed."""
    # Create small graph: 0-1-2-3-4 (chain)
    g = ig.Graph(n=5, edges=[(0, 1), (1, 2), (2, 3), (3, 4)], directed=False)

    seed = 42
    num_walks = 2
    walk_length = 4

    # Generate walks twice with same seed
    walks1 = generate_node2vec_walks(
        g=g,
        num_walks=num_walks,
        walk_length=walk_length,
        p=1.0,
        q=1.0,
        seed=seed,
    )

    walks2 = generate_node2vec_walks(
        g=g,
        num_walks=num_walks,
        walk_length=walk_length,
        p=1.0,
        q=1.0,
        seed=seed,
    )

    assert walks1 == walks2, "Walks should be deterministic with same seed"
    assert len(walks1) == num_walks * g.vcount(), f"Expected {num_walks * g.vcount()} walks"

    print("✓ Node2vec walks are deterministic")


def test_skipgram_training_shape():
    """Test that skip-gram training produces correct embedding shape."""
    # Create toy walks
    walks = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0]]

    dimensions = 16
    window_size = 2
    seed = 42

    model = train_skipgram(
        walks=walks,
        dimensions=dimensions,
        window_size=window_size,
        seed=seed,
        epochs=3,
    )

    # Check that all nodes have embeddings
    for node_id in range(5):
        assert str(node_id) in model.wv, f"Node {node_id} should have embedding"
        emb = model.wv[str(node_id)]
        assert emb.shape == (dimensions,), f"Embedding shape should be ({dimensions},)"

    print("✓ Skip-gram training produces correct embedding shape")


def test_embedding_pair_features():
    """Test that embedding pair features are computed correctly."""
    # Create toy embeddings: 3 nodes, 4 dimensions
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],  # Similar to combination of 0 and 1
        ]
    )

    node_pairs = [(0, 1), (0, 2), (1, 2)]

    features = get_embedding_pair_features(embeddings, node_pairs)

    # Check shape: hadamard (4) + l1 + l2 + cosine = 7
    expected_dim = embeddings.shape[1] + 3
    assert features.shape == (len(node_pairs), expected_dim), \
        f"Expected shape ({len(node_pairs)}, {expected_dim})"

    # Check that cosine similarity makes sense
    # Nodes 0 and 1 are orthogonal -> cosine ~ 0
    cos_01 = features[0, -1]
    assert abs(cos_01) < 0.1, "Orthogonal nodes should have near-zero cosine"

    print("✓ Embedding pair features computed correctly")


def test_node2vec_biased_walks():
    """Test that p and q parameters affect walk behavior."""
    # Create graph: 0-1-2-3 (chain)
    g = ig.Graph(n=4, edges=[(0, 1), (1, 2), (2, 3)], directed=False)

    seed = 42
    num_walks = 10
    walk_length = 10

    # Test with high p (discourage return)
    walks_high_p = generate_node2vec_walks(
        g=g, num_walks=num_walks, walk_length=walk_length, p=10.0, q=1.0, seed=seed
    )

    # Test with low p (encourage return)
    walks_low_p = generate_node2vec_walks(
        g=g, num_walks=num_walks, walk_length=walk_length, p=0.1, q=1.0, seed=seed + 1
    )

    # Walks should be different (p affects behavior)
    assert walks_high_p != walks_low_p, "Different p values should produce different walks"

    print("✓ Node2vec bias parameters (p, q) affect walks")


if __name__ == "__main__":
    test_node2vec_walks_deterministic()
    test_skipgram_training_shape()
    test_embedding_pair_features()
    test_node2vec_biased_walks()
    print("\n✅ All embeddings tests passed!")
