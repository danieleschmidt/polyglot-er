"""en↔ru pilot — design doc §5 of the P7 ground-truth pipeline.

Steps:
  1. Extract ~500 positive sitelink pairs via WikidataLoader.
  2. Run PhoneticMatcher (T3) and EmbeddingMatcher (T4) standalone.
  3. Report marginal-T3 metric: |T3 hits ∩ ¬T4 hits| / total.

Decision rule from design doc §5:
  T3 marginal ≥ 15%  → headline 23% claim on track, proceed to Stage 2.
  T3 marginal < 5%   → reframe paper before further empirical work.
  In between        → ambiguous; revisit with Mike before Stage 2.

Note on T4 backend: this pilot uses the TF-IDF character-n-gram fallback
(``force_tfidf=True``) so the result is reproducible without downloading
sentence-transformers. A follow-up run with sentence-transformers will
report the same metric under the production T4 backend.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from polyglot_er.datasets.wikidata import TYPE_BUCKETS, WikidataLoader
from polyglot_er.matchers.embedding import EmbeddingMatcher
from polyglot_er.matchers.phonetic import PhoneticMatcher


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pilot_en_ru")


# Same thresholds as the cascade defaults.
TIER3_THRESHOLD = 0.82
TIER4_THRESHOLD = 0.75


def extract_pairs(cache_root: Path, n_per_type: int = 50) -> Path:
    """Run the SPARQL pipeline for en↔ru with ~50 pairs per bucket."""
    loader = WikidataLoader()
    out = loader.extract_sitelink_pairs(
        "en", "ru", n_per_type=n_per_type, cache_root=cache_root
    )
    return out


def evaluate(pairs_path: Path) -> dict:
    """Run T3 and T4 standalone against each pair; bucket the outcomes."""
    pairs = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines()]
    logger.info("Evaluating %d pairs", len(pairs))

    t3 = PhoneticMatcher(threshold=TIER3_THRESHOLD)
    t4 = EmbeddingMatcher(threshold=TIER4_THRESHOLD, force_tfidf=True)

    # Bucket each pair by (t3_hit, t4_hit) outcome.
    outcomes: Counter[tuple[bool, bool]] = Counter()
    per_bucket: dict[str, Counter[tuple[bool, bool]]] = {}
    sample_misses: list[dict] = []

    for pair in pairs:
        l1 = pair["l1_title"]
        l2 = pair["l2_title"]
        bucket = pair["bucket"]

        t3_match = t3.match(l1, l2, lang_a="en", lang_b="ru").is_match
        t4_match = t4.match(l1, l2, lang_a="en", lang_b="ru").is_match

        outcomes[(t3_match, t4_match)] += 1
        per_bucket.setdefault(bucket, Counter())[(t3_match, t4_match)] += 1

        if not t3_match and not t4_match and len(sample_misses) < 10:
            sample_misses.append({"l1": l1, "l2": l2, "bucket": bucket})

    n = sum(outcomes.values())
    t3_only = outcomes[(True, False)]
    t4_only = outcomes[(False, True)]
    both = outcomes[(True, True)]
    neither = outcomes[(False, False)]

    return {
        "n_pairs": n,
        "t3_recall": (t3_only + both) / n if n else 0.0,
        "t4_recall": (t4_only + both) / n if n else 0.0,
        "t3_marginal_over_t4": t3_only / n if n else 0.0,
        "t4_marginal_over_t3": t4_only / n if n else 0.0,
        "both": both,
        "neither": neither,
        "t3_only": t3_only,
        "t4_only": t4_only,
        "per_bucket": {
            b: {
                "n": sum(c.values()),
                "t3_only": c[(True, False)],
                "t4_only": c[(False, True)],
                "both": c[(True, True)],
                "neither": c[(False, False)],
            }
            for b, c in per_bucket.items()
        },
        "sample_misses": sample_misses,
    }


def decision(t3_marginal: float) -> str:
    if t3_marginal >= 0.15:
        return "PROCEED to Stage 2 — headline 23% T3 claim on track"
    if t3_marginal < 0.05:
        return "REFRAME paper — T3 marginal below 5% kill threshold"
    return "AMBIGUOUS — revisit with Mike before Stage 2"


def main() -> None:
    cache_root = Path("data/wikidata_raw")
    pairs_path = extract_pairs(cache_root, n_per_type=50)
    results = evaluate(pairs_path)

    print()
    print("=" * 72)
    print(f"en↔ru pilot — n = {results['n_pairs']} pairs")
    print("=" * 72)
    print(f"  T3 recall (phonetic alone):       {results['t3_recall']:.1%}")
    print(f"  T4 recall (embedding alone):      {results['t4_recall']:.1%}")
    print(f"  T3 marginal over T4 (key metric): {results['t3_marginal_over_t4']:.1%}")
    print(f"  T4 marginal over T3:              {results['t4_marginal_over_t3']:.1%}")
    print(f"  Both T3 and T4 match:             {results['both']}")
    print(f"  Neither matches:                  {results['neither']}")
    print()
    print("Per-type-bucket breakdown:")
    print(f"  {'bucket':<14} {'n':>5} {'t3+t4':>6} {'t3only':>7} {'t4only':>7} {'miss':>5}")
    for bucket, c in results["per_bucket"].items():
        print(
            f"  {bucket:<14} {c['n']:>5} {c['both']:>6} {c['t3_only']:>7} {c['t4_only']:>7} {c['neither']:>5}"
        )
    print()
    print("Sample missed pairs (both T3 and T4 miss):")
    for m in results["sample_misses"]:
        print(f"  [{m['bucket']:<14}] {m['l1']!r}  <-->  {m['l2']!r}")
    print()
    print(f"DECISION: {decision(results['t3_marginal_over_t4'])}")
    print("=" * 72)

    # Persist for the §7 write-up.
    report_path = pairs_path.parent / "pilot_report.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote report to %s", report_path)


if __name__ == "__main__":
    main()
