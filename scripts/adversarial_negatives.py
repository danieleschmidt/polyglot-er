"""Adversarial precision eval — Levenshtein-based hard negatives.

The first-pass precision eval (scripts/precision_eval.py) built one hard
negative per positive: the same-bucket entity with maximum Jaro-Winkler on
the L1 surface form. That's a reasonable bar but not the design doc's actual
spec.

Design doc §3.3 specifies: "Hard negatives are derived from secondary SPARQL
queries that filter for similar surface forms (Levenshtein distance ≤ 3 in
transliterated space) but distinct QIDs." This script builds the local
equivalent — Levenshtein-nearest across the entire pair's corpus (cross-bucket
allowed) — and re-runs the precision eval against this stricter set.

For each positive p, we keep up to 3 negatives drawn from the corpus's
entities with the smallest Levenshtein distance to p.l1_title (Levenshtein
distance, not similarity). Cross-bucket is allowed because the design doc
does not require type matching, and the adversarial version of "Smith John"
should let us pull "Smith Johnson" regardless of whether the latter is a
person or an organization.

Output: data/wikidata_raw/{pair}/adversarial_eval.json per pair, plus a
top-level data/wikidata_raw/adversarial_summary.json.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from jellyfish import levenshtein_distance

from polyglot_er.matchers.embedding import EmbeddingMatcher
from polyglot_er.matchers.phonetic import PhoneticMatcher
from polyglot_er.normalization.title_format import normalize_wiki_title


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("adversarial_negatives")


TIER3_THRESHOLD = 0.82
TIER4_THRESHOLD = 0.75
NEGATIVES_PER_POSITIVE = 3
LEVENSHTEIN_CAP = 5  # skip negatives beyond this distance — they're not hard


def build_adversarial_negatives(positives: list[dict]) -> list[dict]:
    """For each positive, take up to NEGATIVES_PER_POSITIVE other entities
    with smallest Levenshtein distance to the source L1 (cross-bucket allowed).

    Pairs (positive.l1, negative.l2). Skips candidates whose Levenshtein
    distance exceeds LEVENSHTEIN_CAP — those aren't actually hard, they're
    just the closest in a sparse corpus, and we want the *adversarial*
    regime, not "best available."
    """
    out: list[dict] = []
    for p in positives:
        candidates = []
        for q in positives:
            if q["qid"] == p["qid"]:
                continue
            d = int(levenshtein_distance(p["l1_title"], q["l1_title"]))
            if d > LEVENSHTEIN_CAP:
                continue
            candidates.append((d, q))
        candidates.sort(key=lambda x: x[0])
        for d, neg in candidates[:NEGATIVES_PER_POSITIVE]:
            out.append(
                {
                    "l1_title": p["l1_title"],
                    "l2_title": neg["l2_title"],
                    "bucket_positive": p["bucket"],
                    "bucket_negative": neg["bucket"],
                    "positive_qid": p["qid"],
                    "negative_qid": neg["qid"],
                    "l1_levenshtein": d,
                    "label": "negative",
                }
            )
    return out


def evaluate(positives: list[dict], negatives: list[dict], l1: str, l2: str, t3, t4) -> dict:
    def score(pair: dict) -> tuple[bool, float]:
        a = normalize_wiki_title(pair["l1_title"], l1)
        b = normalize_wiki_title(pair["l2_title"], l2)
        t3_hit = t3.match(a, b, lang_a=l1, lang_b=l2).is_match
        t4_score = float(t4.score(a, b, lang_a=l1, lang_b=l2))
        return t3_hit, t4_score

    tallies: dict[str, Counter] = {"t3": Counter(), "t4": Counter(), "combined": Counter()}
    for p in positives:
        t3_hit, t4_score = score(p)
        t4_hit = t4_score >= TIER4_THRESHOLD
        combined = t3_hit or t4_hit
        tallies["t3"]["TP" if t3_hit else "FN"] += 1
        tallies["t4"]["TP" if t4_hit else "FN"] += 1
        tallies["combined"]["TP" if combined else "FN"] += 1
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
        return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
                "precision": prec, "recall": rec, "f1": f1}

    return {tier: derive(c) for tier, c in tallies.items()}


def _print_row(label: str, m: dict) -> None:
    print(
        f"  {label:<10}  P={m['precision']:.1%}  R={m['recall']:.1%}  F1={m['f1']:.1%}  "
        f"(TP={m['TP']:>3} FP={m['FP']:>3} TN={m['TN']:>4} FN={m['FN']:>3})"
    )


def run_pair(l1: str, l2: str, cache_root: Path, t3, t4) -> dict:
    extract = cache_root / f"{l1}_{l2}" / "sitelinks.jsonl"
    positives = [json.loads(line) for line in extract.read_text(encoding="utf-8").splitlines()]
    negatives = build_adversarial_negatives(positives)

    # Distribution of Levenshtein distances on the adversarial negatives.
    lev_hist = Counter(n["l1_levenshtein"] for n in negatives)

    logger.info(
        "[%s↔%s] %d positives + %d adversarial negatives (Lev distribution: %s)",
        l1, l2, len(positives), len(negatives), dict(sorted(lev_hist.items())),
    )
    results = evaluate(positives, negatives, l1, l2, t3, t4)

    print()
    print(
        f"--- {l1}↔{l2}  positives={len(positives)} adversarial_negatives={len(negatives)} "
        f"(Lev<=5; up to 3/positive) ---"
    )
    _print_row("T3 only",  results["t3"])
    _print_row("T4 only",  results["t4"])
    _print_row("T3 or T4", results["combined"])

    out = cache_root / f"{l1}_{l2}" / "adversarial_eval.json"
    payload = {
        "language_pair": f"{l1}_{l2}",
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "negative_levenshtein_distribution": dict(sorted(lev_hist.items())),
        "results": results,
        "negative_sample": negatives[:5],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    cache_root = Path("data/wikidata_raw")
    pairs = [("en", "de"), ("en", "fr"), ("en", "ru"), ("en", "ar"), ("en", "zh")]
    t3 = PhoneticMatcher(threshold=TIER3_THRESHOLD)
    t4 = EmbeddingMatcher(threshold=0.0, force_tfidf=False)
    logger.info("T4 backend: %s", t4.backend_name)

    summary: dict[str, dict] = {}
    print()
    print("=" * 88)
    print("Adversarial precision eval — Levenshtein-based hard negatives (Lev<=5, up to 3/+)")
    print("=" * 88)
    for l1, l2 in pairs:
        payload = run_pair(l1, l2, cache_root, t3, t4)
        summary[f"{l1}_{l2}"] = payload

    print()
    print("=" * 88)
    print("COMPARISON: first-pass (max-JW within bucket) vs adversarial (Levenshtein ≤ 5)")
    print("=" * 88)
    print(f"  {'pair':<8} {'first-pass F1':>14} {'adv F1':>10} {'Δ F1':>10} {'first prec':>11} {'adv prec':>10}")
    for key, payload in summary.items():
        first = json.loads((cache_root / key / "precision_eval.json").read_text(encoding="utf-8"))
        f1_a = payload["results"]["combined"]["f1"]
        f1_b = first["results"]["combined"]["f1"]
        p_a = payload["results"]["combined"]["precision"]
        p_b = first["results"]["combined"]["precision"]
        print(
            f"  {key:<8} {f1_b:>14.1%} {f1_a:>10.1%} {f1_a - f1_b:>+10.1%} "
            f"{p_b:>11.1%} {p_a:>10.1%}"
        )
    print("=" * 88)

    out = cache_root / "adversarial_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote summary to %s", out)


if __name__ == "__main__":
    main()
