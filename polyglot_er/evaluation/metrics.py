"""
Precision, recall, and F1 metrics for cross-lingual entity resolution.

Two evaluation modes:
  - Cluster-level: compare predicted clusters to ground-truth clusters
  - Pairwise: compare all predicted co-reference pairs to gold pairs
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


def _cluster_to_pairs(clusters: List[List[Any]]) -> Set[Tuple[Any, Any]]:
    """Convert a list of clusters to a set of sorted element pairs."""
    pairs: Set[Tuple[Any, Any]] = set()
    for cluster in clusters:
        items = sorted(str(x) for x in cluster)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pairs.add((items[i], items[j]))
    return pairs


def pairwise_precision_recall_f1(
    predicted_clusters: List[List[Any]],
    gold_clusters: List[List[Any]],
) -> Dict[str, float]:
    """
    Compute pairwise precision, recall, and F1.

    Each cluster is treated as a set of element IDs. The method converts
    clusters to all-pairs, then computes standard classification metrics
    on the pair sets.

    Args:
        predicted_clusters: List of predicted entity clusters (each cluster
            is a list of entity identifiers)
        gold_clusters: List of ground-truth clusters

    Returns:
        Dict with keys "precision", "recall", "f1"
    """
    pred_pairs = _cluster_to_pairs(predicted_clusters)
    gold_pairs = _cluster_to_pairs(gold_clusters)

    if not pred_pairs and not gold_pairs:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gold_pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_pairs & gold_pairs)
    precision = tp / len(pred_pairs) if pred_pairs else 0.0
    recall = tp / len(gold_pairs) if gold_pairs else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def cluster_precision_recall_f1(
    predicted_clusters: List[List[Any]],
    gold_clusters: List[List[Any]],
) -> Dict[str, float]:
    """
    Cluster-level precision, recall, and F1 (set-matching).

    For each predicted cluster we find the best-matching gold cluster
    (by Jaccard) and accumulate TP/FP/FN.

    Args:
        predicted_clusters: List of predicted entity clusters
        gold_clusters: List of ground-truth clusters

    Returns:
        Dict with keys "precision", "recall", "f1"
    """
    gold_sets = [frozenset(str(x) for x in c) for c in gold_clusters]
    pred_sets = [frozenset(str(x) for x in c) for c in predicted_clusters]

    if not pred_sets and not gold_sets:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = fp = fn = 0
    matched_gold: Set[int] = set()

    for pred in pred_sets:
        best_overlap = 0
        best_idx = -1
        for idx, gold in enumerate(gold_sets):
            overlap = len(pred & gold)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
        if best_idx >= 0 and best_overlap > 0:
            gold = gold_sets[best_idx]
            tp += best_overlap
            fp += len(pred) - best_overlap
            fn += len(gold) - best_overlap
            matched_gold.add(best_idx)
        else:
            fp += len(pred)

    for idx, gold in enumerate(gold_sets):
        if idx not in matched_gold:
            fn += len(gold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}
