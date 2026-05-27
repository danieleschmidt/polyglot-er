"""Per-bucket precision/recall/F1 across all 5 pairs.

The aggregate F1 numbers from scripts/precision_eval.py hide variation
across entity types. For paper §7, the more useful figure is per-bucket:
which entity type does each tier work on?

For each (language_pair, bucket, tier) tuple we report precision, recall,
F1, and the supporting TP/FP/TN/FN. Negatives are drawn from the same
hard-negative construction as precision_eval.py (same-bucket max-JW).
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from jellyfish import jaro_winkler_similarity

from polyglot_er.matchers.embedding import EmbeddingMatcher
from polyglot_er.matchers.phonetic import PhoneticMatcher
from polyglot_er.normalization.title_format import normalize_wiki_title


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("per_bucket_f1")


TIER3_THRESHOLD = 0.82
TIER4_THRESHOLD = 0.75


def build_same_bucket_negatives(positives: list[dict]) -> list[dict]:
    """Same construction as precision_eval.py: one negative per positive,
    same-bucket peer with maximum L1 Jaro-Winkler."""
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for p in positives:
        by_bucket[p["bucket"]].append(p)
    out = []
    for p in positives:
        peers = [q for q in by_bucket[p["bucket"]] if q["qid"] != p["qid"]]
        if not peers:
            continue
        best = max(peers, key=lambda q: jaro_winkler_similarity(p["l1_title"], q["l1_title"]))
        out.append(
            {
                "l1_title": p["l1_title"],
                "l2_title": best["l2_title"],
                "bucket": p["bucket"],
                "label": "negative",
            }
        )
    return out


def derive(c: Counter) -> dict:
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "precision": prec, "recall": rec, "f1": f1}


def evaluate_per_bucket(positives: list[dict], negatives: list[dict], l1: str, l2: str, t3, t4) -> dict:
    """Tally TP/FP/TN/FN per (bucket, tier) and return per-bucket metrics."""
    counters: dict[str, dict[str, Counter]] = defaultdict(
        lambda: {"t3": Counter(), "t4": Counter(), "combined": Counter()}
    )

    def score(pair: dict) -> tuple[bool, float]:
        a = normalize_wiki_title(pair["l1_title"], l1)
        b = normalize_wiki_title(pair["l2_title"], l2)
        t3_hit = t3.match(a, b, lang_a=l1, lang_b=l2).is_match
        t4_score = float(t4.score(a, b, lang_a=l1, lang_b=l2))
        return t3_hit, t4_score

    for p in positives:
        t3_hit, t4_score = score(p)
        t4_hit = t4_score >= TIER4_THRESHOLD
        combined = t3_hit or t4_hit
        c = counters[p["bucket"]]
        c["t3"]["TP" if t3_hit else "FN"] += 1
        c["t4"]["TP" if t4_hit else "FN"] += 1
        c["combined"]["TP" if combined else "FN"] += 1

    for n in negatives:
        t3_hit, t4_score = score(n)
        t4_hit = t4_score >= TIER4_THRESHOLD
        combined = t3_hit or t4_hit
        c = counters[n["bucket"]]
        c["t3"]["FP" if t3_hit else "TN"] += 1
        c["t4"]["FP" if t4_hit else "TN"] += 1
        c["combined"]["FP" if combined else "TN"] += 1

    return {
        bucket: {tier: derive(c) for tier, c in tier_counters.items()}
        for bucket, tier_counters in counters.items()
    }


def run_pair(l1: str, l2: str, cache_root: Path, t3, t4) -> dict:
    extract = cache_root / f"{l1}_{l2}" / "sitelinks.jsonl"
    positives = [json.loads(line) for line in extract.read_text(encoding="utf-8").splitlines()]
    negatives = build_same_bucket_negatives(positives)
    per_bucket = evaluate_per_bucket(positives, negatives, l1, l2, t3, t4)

    payload = {
        "language_pair": f"{l1}_{l2}",
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "per_bucket": per_bucket,
    }
    out = cache_root / f"{l1}_{l2}" / "per_bucket_f1.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _print_pair(pair_key: str, payload: dict) -> None:
    print(f"\n--- {pair_key}  (n+={payload['n_positives']}, n-={payload['n_negatives']}) ---")
    print(f"  {'bucket':<14} {'tier':<10} {'P':>6} {'R':>6} {'F1':>6}   counts")
    for bucket, tiers in sorted(payload["per_bucket"].items()):
        for tier_label in ("t3", "t4", "combined"):
            m = tiers[tier_label]
            print(
                f"  {bucket:<14} {tier_label:<10} {m['precision']:>6.1%} {m['recall']:>6.1%} {m['f1']:>6.1%}   "
                f"TP={m['TP']:>3} FP={m['FP']:>3} TN={m['TN']:>3} FN={m['FN']:>3}"
            )


def _print_combined_heatmap(summary: dict) -> None:
    """Combined-cascade F1 by (pair, bucket) — the heatmap-style summary."""
    # Find all buckets across all pairs.
    all_buckets = sorted({b for s in summary.values() for b in s["per_bucket"]})
    pair_keys = sorted(summary.keys())

    print()
    print("=" * 92)
    print("COMBINED-CASCADE F1 HEATMAP — rows = bucket, cols = language pair")
    print("=" * 92)
    print(f"  {'bucket':<14}  " + "  ".join(f"{k:>7}" for k in pair_keys))
    for bucket in all_buckets:
        cells = []
        for k in pair_keys:
            data = summary[k]["per_bucket"].get(bucket)
            if data is None:
                cells.append("    -- ")
            else:
                cells.append(f"{data['combined']['f1']:>6.1%} ")
        print(f"  {bucket:<14}  " + "  ".join(cells))
    print("=" * 92)


def _print_tier_winner(summary: dict) -> None:
    """For each (pair, bucket) call out which tier produced the highest F1."""
    pair_keys = sorted(summary.keys())
    all_buckets = sorted({b for s in summary.values() for b in s["per_bucket"]})
    print()
    print("=" * 92)
    print("WHICH TIER WINS PER BUCKET (highest F1; ties broken by precision)")
    print("=" * 92)
    print(f"  {'bucket':<14}  " + "  ".join(f"{k:>7}" for k in pair_keys))
    for bucket in all_buckets:
        cells = []
        for k in pair_keys:
            data = summary[k]["per_bucket"].get(bucket)
            if data is None:
                cells.append("    -- ")
                continue
            scores = [
                (data["t3"]["f1"],       data["t3"]["precision"],       "T3"),
                (data["t4"]["f1"],       data["t4"]["precision"],       "T4"),
                (data["combined"]["f1"], data["combined"]["precision"], "T3|T4"),
            ]
            winner = max(scores)[2]
            cells.append(f"{winner:>7} ")
        print(f"  {bucket:<14}  " + "  ".join(cells))
    print("=" * 92)


def main() -> None:
    cache_root = Path("data/wikidata_raw")
    pairs = [("en", "de"), ("en", "fr"), ("en", "ru"), ("en", "ar"), ("en", "zh")]
    t3 = PhoneticMatcher(threshold=TIER3_THRESHOLD)
    t4 = EmbeddingMatcher(threshold=0.0, force_tfidf=False)
    logger.info("T4 backend: %s", t4.backend_name)

    summary: dict[str, dict] = {}
    for l1, l2 in pairs:
        summary[f"{l1}_{l2}"] = run_pair(l1, l2, cache_root, t3, t4)

    for k, payload in summary.items():
        _print_pair(k, payload)

    _print_combined_heatmap(summary)
    _print_tier_winner(summary)

    out = cache_root / "per_bucket_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote per-bucket summary to %s", out)


if __name__ == "__main__":
    main()
