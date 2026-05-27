"""Precision-recall evaluation with hard negatives.

Per design doc §3.3, for each cached pair extract:
1. Treat each row as a POSITIVE pair (same QID across languages).
2. Build one HARD NEGATIVE per positive: for entity e with title (l1, l2)
   in bucket B, find another entity e' in the same bucket whose L1 title
   is most similar to e.l1, and create the negative pair (e.l1, e'.l2).
   This produces same-type-different-entity pairs with elevated surface
   similarity on the L1 side — the regime the cascade actually has to
   discriminate in production.
3. Run T3 (phonetic) and T4 (production embedding) against the merged
   positive+negative set and report precision, recall, F1, and TP/FP/TN/FN
   counts. Combined cascade is "T3 hit OR T4 score >= threshold".

Output: per-pair JSON dropped at data/wikidata_raw/{pair}/precision_eval.json,
plus a top-level summary at data/wikidata_raw/precision_summary.json.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from jellyfish import jaro_winkler_similarity

from polyglot_er.matchers.embedding import EmbeddingMatcher
from polyglot_er.matchers.phonetic import PhoneticMatcher
from polyglot_er.normalization.title_format import normalize_wiki_title


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("precision_eval")


TIER3_THRESHOLD = 0.82
TIER4_THRESHOLD = 0.75


def build_hard_negatives(positives: list[dict], lang_a: str, lang_b: str) -> list[dict]:
    """For each positive, find another entity in the same bucket with the
    highest Jaro-Winkler on the L1 title and pair it as a negative.

    Skips positives whose bucket contains only one entity (can't form a
    same-bucket negative). Returns at most one negative per positive.
    """
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for p in positives:
        by_bucket[p["bucket"]].append(p)

    negatives: list[dict] = []
    for p in positives:
        peers = [q for q in by_bucket[p["bucket"]] if q["qid"] != p["qid"]]
        if not peers:
            continue
        best = max(peers, key=lambda q: jaro_winkler_similarity(p["l1_title"], q["l1_title"]))
        jw = jaro_winkler_similarity(p["l1_title"], best["l1_title"])
        negatives.append(
            {
                "l1_title": p["l1_title"],          # original L1
                "l2_title": best["l2_title"],       # different entity's L2
                "bucket": p["bucket"],
                "positive_qid": p["qid"],
                "negative_qid": best["qid"],
                "l1_jw_to_source": float(jw),
                "label": "negative",
            }
        )
    return negatives


def evaluate(positives: list[dict], negatives: list[dict], l1: str, l2: str, t3, t4) -> dict:
    """Run T3 and T4 against the merged set; tally TP/FP/TN/FN."""
    def score(pair: dict) -> tuple[bool, float]:
        a = normalize_wiki_title(pair["l1_title"], l1)
        b = normalize_wiki_title(pair["l2_title"], l2)
        t3_hit = t3.match(a, b, lang_a=l1, lang_b=l2).is_match
        t4_score = float(t4.score(a, b, lang_a=l1, lang_b=l2))
        return t3_hit, t4_score

    tallies: dict[str, dict] = {
        "t3":       Counter(),
        "t4":       Counter(),
        "combined": Counter(),
    }
    # Score positives.
    for p in positives:
        t3_hit, t4_score = score(p)
        t4_hit = t4_score >= TIER4_THRESHOLD
        combined = t3_hit or t4_hit
        tallies["t3"]["TP" if t3_hit else "FN"] += 1
        tallies["t4"]["TP" if t4_hit else "FN"] += 1
        tallies["combined"]["TP" if combined else "FN"] += 1
    # Score negatives.
    for n in negatives:
        t3_hit, t4_score = score(n)
        t4_hit = t4_score >= TIER4_THRESHOLD
        combined = t3_hit or t4_hit
        tallies["t3"]["FP" if t3_hit else "TN"] += 1
        tallies["t4"]["FP" if t4_hit else "TN"] += 1
        tallies["combined"]["FP" if combined else "TN"] += 1

    def derive(c: Counter) -> dict:
        tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": prec, "recall": rec, "f1": f1,
        }

    return {tier: derive(c) for tier, c in tallies.items()}


def _print_row(label: str, metrics: dict) -> None:
    print(
        f"  {label:<10}  "
        f"P={metrics['precision']:.1%}  R={metrics['recall']:.1%}  F1={metrics['f1']:.1%}  "
        f"(TP={metrics['TP']:>3} FP={metrics['FP']:>3} TN={metrics['TN']:>3} FN={metrics['FN']:>3})"
    )


def run_pair(l1: str, l2: str, cache_root: Path, t3, t4) -> dict:
    extract = cache_root / f"{l1}_{l2}" / "sitelinks.jsonl"
    positives = [json.loads(line) for line in extract.read_text(encoding="utf-8").splitlines()]
    negatives = build_hard_negatives(positives, l1, l2)
    logger.info("[%s↔%s] %d positives + %d hard negatives", l1, l2, len(positives), len(negatives))

    results = evaluate(positives, negatives, l1, l2, t3, t4)

    print()
    print(f"--- {l1}↔{l2}  positives={len(positives)} negatives={len(negatives)} ---")
    _print_row("T3 only",   results["t3"])
    _print_row("T4 only",   results["t4"])
    _print_row("T3 or T4",  results["combined"])

    out = cache_root / f"{l1}_{l2}" / "precision_eval.json"
    payload = {
        "language_pair": f"{l1}_{l2}",
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "results": results,
        "negative_sample": negatives[:5],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main(pairs: Iterable[tuple[str, str]] = (
    ("en", "de"), ("en", "fr"), ("en", "ru"), ("en", "ar"),
    ("en", "zh"), ("en", "ja"), ("en", "ko"),
)) -> None:
    cache_root = Path("data/wikidata_raw")
    t3 = PhoneticMatcher(threshold=TIER3_THRESHOLD)
    t4 = EmbeddingMatcher(threshold=0.0, force_tfidf=False)
    logger.info("T4 backend: %s", t4.backend_name)

    summary: dict[str, dict] = {}
    print()
    print("=" * 84)
    print("Precision-recall evaluation with hard negatives (same-bucket, max JW on L1)")
    print("=" * 84)
    for l1, l2 in pairs:
        payload = run_pair(l1, l2, cache_root, t3, t4)
        summary[f"{l1}_{l2}"] = payload

    # Headline F1 table.
    print()
    print("=" * 84)
    print("HEADLINE F1 (combined cascade = T3 hit OR T4 >= 0.75)")
    print("=" * 84)
    print(f"  {'pair':<8} {'n+':>5} {'n-':>5}  {'precision':>10} {'recall':>10} {'F1':>8}")
    for key, payload in summary.items():
        c = payload["results"]["combined"]
        print(
            f"  {key:<8} {payload['n_positives']:>5} {payload['n_negatives']:>5}  "
            f"{c['precision']:>10.1%} {c['recall']:>10.1%} {c['f1']:>8.1%}"
        )
    print("=" * 84)

    out = cache_root / "precision_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote summary to %s", out)


if __name__ == "__main__":
    main()
