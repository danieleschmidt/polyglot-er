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
from polyglot_er.normalization.script_detect import is_same_script
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
    """Run T3 and T4 against the merged set; tally TP/FP/TN/FN.

    The ``combined`` tier reports the production cascade's script-dispatch
    gate (Tier 1.5): for same-script pairs the T3 hit is suppressed and the
    decision is T4-only; for cross-script pairs the decision is T3 hit OR T4
    hit. The ``combined_ungated`` tier reports the pre-gate disjunction for
    side-by-side comparison.
    """
    def score(pair: dict) -> tuple[bool, float, bool]:
        a = normalize_wiki_title(pair["l1_title"], l1)
        b = normalize_wiki_title(pair["l2_title"], l2)
        t3_hit = t3.match(a, b, lang_a=l1, lang_b=l2).is_match
        t4_score = float(t4.score(a, b, lang_a=l1, lang_b=l2))
        same_script = is_same_script(a, b)
        return t3_hit, t4_score, same_script

    tallies: dict[str, dict] = {
        "t3":                Counter(),
        "t4":                Counter(),
        "combined":          Counter(),  # script-dispatch-gated (production)
        "combined_ungated":  Counter(),  # pre-gate (legacy comparison)
    }
    # Score positives.
    for p in positives:
        t3_hit, t4_score, same_script = score(p)
        t4_hit = t4_score >= TIER4_THRESHOLD
        # Script-dispatch gate: T3 is suppressed for same-script pairs.
        gated_combined = (t3_hit and not same_script) or t4_hit
        ungated_combined = t3_hit or t4_hit
        tallies["t3"]["TP" if t3_hit else "FN"] += 1
        tallies["t4"]["TP" if t4_hit else "FN"] += 1
        tallies["combined"]["TP" if gated_combined else "FN"] += 1
        tallies["combined_ungated"]["TP" if ungated_combined else "FN"] += 1
    # Score negatives.
    for n in negatives:
        t3_hit, t4_score, same_script = score(n)
        t4_hit = t4_score >= TIER4_THRESHOLD
        gated_combined = (t3_hit and not same_script) or t4_hit
        ungated_combined = t3_hit or t4_hit
        tallies["t3"]["FP" if t3_hit else "TN"] += 1
        tallies["t4"]["FP" if t4_hit else "TN"] += 1
        tallies["combined"]["FP" if gated_combined else "TN"] += 1
        tallies["combined_ungated"]["FP" if ungated_combined else "TN"] += 1

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
    _print_row("T3 only",      results["t3"])
    _print_row("T4 only",      results["t4"])
    _print_row("combined",     results["combined"])
    _print_row("ungated",      results["combined_ungated"])

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
    print("=" * 96)
    print("HEADLINE F1 (combined = script-dispatch gated; same-script → T4 only, cross-script → T3 ∨ T4)")
    print("=" * 96)
    print(f"  {'pair':<8} {'n+':>5} {'n-':>5}  "
          f"{'gated P':>8} {'gated R':>8} {'gated F1':>9}   "
          f"{'ungated P':>10} {'ungated R':>10} {'ungated F1':>11}")
    for key, payload in summary.items():
        c = payload["results"]["combined"]
        u = payload["results"]["combined_ungated"]
        print(
            f"  {key:<8} {payload['n_positives']:>5} {payload['n_negatives']:>5}  "
            f"{c['precision']:>8.1%} {c['recall']:>8.1%} {c['f1']:>9.1%}   "
            f"{u['precision']:>10.1%} {u['recall']:>10.1%} {u['f1']:>11.1%}"
        )
    print("=" * 96)

    out = cache_root / "precision_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote summary to %s", out)


if __name__ == "__main__":
    main()
