"""
CrossLingualResolver — main entry point for cross-lingual entity resolution.

Takes a list of entity records with language codes and returns clusters of
records that refer to the same real-world entity.

The resolver uses the 5-tier CascadeMatcher internally and applies
union-find (disjoint-set) clustering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .matchers.cascade import CascadeMatcher


# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set Union) for clustering
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int):
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def clusters(self, n: int) -> Dict[int, List[int]]:
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return groups


# ---------------------------------------------------------------------------
# CrossLingualResolver
# ---------------------------------------------------------------------------

class CrossLingualResolver:
    """
    Cross-lingual entity resolution pipeline.

    Given a list of entity records (each with at least a ``name`` and ``lang``
    field), groups them into clusters of records that represent the same
    real-world entity.

    Uses the 5-tier CascadeMatcher and union-find clustering.

    Args:
        tier2_threshold: Jaro-Winkler threshold for same-script fuzzy (Tier 2)
        tier3_threshold: Phonetic similarity threshold (Tier 3)
        tier4_threshold: Embedding cosine threshold (Tier 4)
        force_tfidf: Use TF-IDF embedding fallback instead of sentence-transformers
        verbose: Print progress to stdout

    Example::

        from polyglot_er import CrossLingualResolver

        entities = [
            {"id": "e1", "name": "Vladimir Putin", "lang": "en", "entity_type": "PER"},
            {"id": "e2", "name": "Владимир Путин", "lang": "ru", "entity_type": "PER"},
            {"id": "e3", "name": "Angela Merkel", "lang": "en", "entity_type": "PER"},
        ]

        resolver = CrossLingualResolver(force_tfidf=True)
        clusters = resolver.resolve(entities)
        # [["e1", "e2"], ["e3"]]
    """

    def __init__(
        self,
        tier2_threshold: float = 0.85,
        tier3_threshold: float = 0.82,
        tier4_threshold: float = 0.75,
        force_tfidf: bool = False,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self._cascade = CascadeMatcher(
            tier2_threshold=tier2_threshold,
            tier3_threshold=tier3_threshold,
            tier4_threshold=tier4_threshold,
            force_tfidf=force_tfidf,
        )

    def resolve(self, records: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Resolve entity records into co-reference clusters.

        Args:
            records: List of entity record dicts. Required keys: ``name``, ``lang``.
                     Optional keys: ``entity_type``, ``id``.

        Returns:
            List of clusters. Each cluster is a list of record ``id`` values
            (or sequential indices as strings if no ``id`` is present).
        """
        if not records:
            return []

        # Assign ids if missing
        ids = [str(r.get("id", i)) for i, r in enumerate(records)]
        n = len(records)
        uf = _UnionFind(n)

        comparisons = 0
        matches = 0

        for i in range(n):
            for j in range(i + 1, n):
                ra, rb = records[i], records[j]
                result = self._cascade.match(
                    name_a=ra["name"],
                    name_b=rb["name"],
                    lang_a=ra.get("lang", ""),
                    lang_b=rb.get("lang", ""),
                    entity_type_a=ra.get("entity_type", ""),
                    entity_type_b=rb.get("entity_type", ""),
                )
                comparisons += 1
                if result.is_match:
                    uf.union(i, j)
                    matches += 1

        if self.verbose:
            print(
                f"Resolved {n} records in {comparisons} comparisons → {matches} matches"
            )

        clusters_dict = uf.clusters(n)
        return [[ids[i] for i in sorted(members)] for members in clusters_dict.values()]

    def resolve_jsonl(self, input_path: Path) -> List[List[str]]:
        """
        Load records from a JSONL file and resolve.

        Args:
            input_path: Path to input .jsonl file

        Returns:
            List of clusters
        """
        records = []
        with open(input_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return self.resolve(records)

    def resolve_and_save(
        self,
        input_path: Path,
        output_path: Path,
    ) -> List[List[str]]:
        """
        Resolve JSONL input and write clusters to JSON output.

        Args:
            input_path: Path to input .jsonl file
            output_path: Path to output .json file

        Returns:
            List of clusters
        """
        clusters = self.resolve_jsonl(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump({"clusters": clusters, "count": len(clusters)}, fh, indent=2)
        if self.verbose:
            print(f"Wrote {len(clusters)} clusters to {output_path}")
        return clusters
