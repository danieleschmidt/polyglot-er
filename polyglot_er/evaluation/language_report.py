"""
Per-language-pair breakdown of cross-lingual entity resolution performance.

Reports precision, recall, and F1 for each language pair encountered in the
dataset (e.g. EN-RU, EN-ZH, EN-AR, RU-ZH, etc.).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .metrics import pairwise_precision_recall_f1


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
EntityRecord = Dict[str, Any]
ClusterList = List[List[str]]   # list of clusters, each cluster is list of entity ids


# ---------------------------------------------------------------------------
# LanguageReport
# ---------------------------------------------------------------------------

class LanguageReport:
    """
    Generate per-language-pair precision/recall/F1 breakdowns.

    Usage::

        report = LanguageReport(records, predicted_clusters, gold_clusters)
        breakdown = report.compute()
        # breakdown = {
        #   "en-ru": {"precision": 0.9, "recall": 0.85, "f1": 0.87},
        #   "en-zh": {...},
        #   "overall": {...},
        # }

    Args:
        records: List of entity record dicts with at least keys "id" and "lang"
        predicted_clusters: List of predicted clusters (each cluster = list of entity ids)
        gold_clusters: List of gold clusters
    """

    def __init__(
        self,
        records: List[EntityRecord],
        predicted_clusters: ClusterList,
        gold_clusters: ClusterList,
    ):
        self.records = records
        self.predicted_clusters = predicted_clusters
        self.gold_clusters = gold_clusters
        self._id_to_lang: Dict[str, str] = {r["id"]: r["lang"] for r in records}

    def _get_lang_pair(self, id_a: str, id_b: str) -> Optional[str]:
        """Return normalized language pair key 'xx-yy' (sorted)."""
        lang_a = self._id_to_lang.get(id_a, "?")
        lang_b = self._id_to_lang.get(id_b, "?")
        if lang_a == lang_b:
            return f"{lang_a}-{lang_b}"
        return "-".join(sorted([lang_a, lang_b]))

    def _clusters_for_pair(
        self,
        clusters: ClusterList,
        lang_a: str,
        lang_b: str,
    ) -> List[List[str]]:
        """
        Filter clusters to only include records from the given language pair.

        For each cluster, we keep only the members whose lang is lang_a or lang_b.
        Clusters with fewer than 2 members after filtering are dropped.
        """
        target_langs = {lang_a, lang_b}
        filtered = []
        for cluster in clusters:
            members = [
                eid for eid in cluster
                if self._id_to_lang.get(eid, "") in target_langs
            ]
            if len(members) >= 2:
                filtered.append(members)
        return filtered

    def _all_language_pairs(self) -> List[Tuple[str, str]]:
        """Enumerate all language pairs present in the gold clusters."""
        langs = set(self._id_to_lang.values())
        langs_sorted = sorted(langs)
        pairs = []
        for i, la in enumerate(langs_sorted):
            for lb in langs_sorted[i + 1 :]:
                pairs.append((la, lb))
        # Include same-language pairs
        for la in langs_sorted:
            pairs.append((la, la))
        return pairs

    def compute(self) -> Dict[str, Dict[str, float]]:
        """
        Compute per-language-pair and overall precision/recall/F1.

        Returns:
            Dict mapping language pair key → metrics dict.
            The key "overall" contains aggregate metrics across all pairs.
        """
        results: Dict[str, Dict[str, float]] = {}

        for lang_a, lang_b in self._all_language_pairs():
            pair_key = f"{lang_a}-{lang_b}"
            pred_filtered = self._clusters_for_pair(self.predicted_clusters, lang_a, lang_b)
            gold_filtered = self._clusters_for_pair(self.gold_clusters, lang_a, lang_b)
            if not gold_filtered:
                continue
            metrics = pairwise_precision_recall_f1(pred_filtered, gold_filtered)
            results[pair_key] = metrics

        # Overall metrics across all records
        results["overall"] = pairwise_precision_recall_f1(
            self.predicted_clusters, self.gold_clusters
        )

        return results

    def summary(self) -> str:
        """Return a human-readable summary string."""
        breakdown = self.compute()
        lines = ["Cross-Lingual Entity Resolution — Language Pair Report", "=" * 55]
        overall = breakdown.pop("overall", None)
        for pair_key in sorted(breakdown):
            m = breakdown[pair_key]
            lines.append(
                f"  {pair_key:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}"
            )
        if overall:
            lines.append("-" * 55)
            lines.append(
                f"  {'overall':12s}  P={overall['precision']:.3f}  R={overall['recall']:.3f}  F1={overall['f1']:.3f}"
            )
        return "\n".join(lines)
