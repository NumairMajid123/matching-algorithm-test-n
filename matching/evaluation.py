"""
NDCG calculation for evaluating ranking quality.

NDCG (Normalized Discounted Cumulative Gain) is a standard metric for
evaluating ranking algorithms. It takes into account both relevance and position.
"""

import numpy as np


def calculate_ndcg_at_k(predicted_ranks, ideal_ranks, k=10):
    """
    Calculate NDCG@k (Normalized Discounted Cumulative Gain at k).
    
    Args:
        predicted_ranks: List of property IDs in predicted order (highest score first)
            e.g. [123, 456, 789, ...]
        ideal_ranks: List of property IDs in ideal order (ground truth)
            e.g. [789, 123, 456, ...] where 789 is rank 1 (best match)
        k: Number of top results to evaluate (default: 10)
    
    Returns:
        float: NDCG@k value (0-1, where 1 is perfect)
    
    Example:
        ideal_ranks = [789, 123, 456]  # 789 is best, 123 is second best, etc.
        predicted_ranks = [123, 456, 789, ...]  # Algorithm ranked 123 first
        
        NDCG will be < 1.0 because the algorithm didn't rank 789 (best) first.
    """
    if not ideal_ranks:
        return 0.0
    
    # Create relevance scores
    # Higher rank in ideal_ranks = higher relevance
    # Rank 1 (index 0) gets highest relevance
    relevance = {}
    for i, prop_id in enumerate(ideal_ranks):
        relevance[prop_id] = len(ideal_ranks) - i  # Rank 1 = len(ideal), Rank 2 = len(ideal)-1, etc.
    
    # Calculate DCG (Discounted Cumulative Gain) for predicted ranking
    dcg = 0.0
    for i, prop_id in enumerate(predicted_ranks[:k]):
        rel = relevance.get(prop_id, 0)  # Relevance for this property (0 if not in ideal_ranks)
        # Discounted gain: relevance / log2(position + 1)
        # +1 because position is 0-indexed, +1 more to avoid log2(1) = 0
        dcg += rel / np.log2(i + 2)
    
    # Calculate IDCG (Ideal DCG) - what DCG would be if ranking was perfect
    idcg = 0.0
    # Sort relevance values in descending order (highest first)
    sorted_relevance = sorted(relevance.values(), reverse=True)
    for i, rel in enumerate(sorted_relevance[:k]):
        idcg += rel / np.log2(i + 2)
    
    # Normalize: DCG / IDCG
    if idcg == 0:
        return 0.0
    
    return dcg / idcg
