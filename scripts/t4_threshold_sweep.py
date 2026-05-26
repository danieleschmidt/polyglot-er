"""T4 threshold sweep + T3-only inspection (production T4 backend).

Two artifacts come out of this script:

1. **T4-threshold sweep**: T4 recall, T3-marginal-over-T4, and combined-cascade
   recall computed across a grid of T4 thresholds in [0.50, 0.90]. The curve
   shows where T3's marginal contribution sits relative to the design doc
   thresholds (15% PROCEED, 5% REFRAME) as T4's threshold moves.

2. **T3-only inspection**: at the cascade default T4 threshold (0.75), the
   subset of pairs where T3 matched but production T4 missed. Each row is
   printed with its bucket, the actual cosine score T4 assigned, and the
   normalized titles — so we can see whether T3-only cases form a coherent
   class (e.g., highly-transliterated names) or are residual noise.

Both artifacts are appended to data/wikidata_raw/en_ru/pilot_report.json under
``t4_threshold_sweep`` and ``t3_only_inspection`` respectively.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from polyglot_er.matchers.embedding import EmbeddingMatcher
from polyglot_er.matchers.phonetic import PhoneticMatcher
from polyglot_er.normalization.title_format import normalize_wiki_title


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("t4_threshold_sweep")


TIER3_THRESHOLD = 0.82
T4_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85, 0.90]
DEFAULT_T4_THRESHOLD = 0.75


def score_all_pairs(pairs: list[dict]) -> list[dict]:
    """Compute T3 hit + T4 cosine score for every pair, once.

    Returns one record per pair with the normalized titles, T3 hit/miss,
    T4 cosine score (raw, threshold-independent), and the original bucket.
    """
    t3 = PhoneticMatcher(threshold=TIER3_THRESHOLD)
    # T4: instantiate once; query .score() for the threshold-independent cosine.
    t4 = EmbeddingMatcher(threshold=0.0, force_tfidf=False)
    logger.info("T4 backend: %s", t4.backend_name)

    scored: list[dict] = []
    for pair in pairs:
        l1 = normalize_wiki_title(pair["l1_title"], "en")
        l2 = normalize_wiki_title(pair["l2_title"], "ru")
        t3_hit = t3.match(l1, l2, lang_a="en", lang_b="ru").is_match
        t4_score = float(t4.score(l1, l2, lang_a="en", lang_b="ru"))
        scored.append(
            {
                "l1": l1,
                "l2": l2,
                "l1_raw": pair["l1_title"],
                "l2_raw": pair["l2_title"],
                "bucket": pair["bucket"],
                "qid": pair["qid"],
                "t3_hit": t3_hit,
                "t4_score": t4_score,
            }
        )
    return scored


def sweep(scored: list[dict], t4_thresholds: list[float]) -> list[dict]:
    """For each T4 threshold, compute the cascade metrics."""
    n = len(scored)
    rows = []
    for thr in t4_thresholds:
        t3_only = sum(1 for s in scored if s["t3_hit"] and s["t4_score"] < thr)
        t4_only = sum(1 for s in scored if not s["t3_hit"] and s["t4_score"] >= thr)
        both = sum(1 for s in scored if s["t3_hit"] and s["t4_score"] >= thr)
        neither = sum(1 for s in scored if not s["t3_hit"] and s["t4_score"] < thr)
        rows.append(
            {
                "t4_threshold": thr,
                "t3_recall": (t3_only + both) / n,
                "t4_recall": (t4_only + both) / n,
                "combined_recall": (n - neither) / n,
                "t3_marginal_over_t4": t3_only / n,
                "t4_marginal_over_t3": t4_only / n,
                "both": both,
                "t3_only": t3_only,
                "t4_only": t4_only,
                "neither": neither,
            }
        )
    return rows


def inspect_t3_only(scored: list[dict], t4_threshold: float) -> dict:
    """Look at the T3-yes / T4-no set at a fixed T4 threshold."""
    t3_only = [s for s in scored if s["t3_hit"] and s["t4_score"] < t4_threshold]
    by_bucket = Counter(s["bucket"] for s in t3_only)
    # Distribution of T4 scores among T3-only cases — were they close to threshold?
    scores = sorted(s["t4_score"] for s in t3_only)
    return {
        "n": len(t3_only),
        "by_bucket": dict(by_bucket),
        "t4_score_min": scores[0] if scores else None,
        "t4_score_median": scores[len(scores) // 2] if scores else None,
        "t4_score_max": scores[-1] if scores else None,
        "near_threshold": sum(1 for s in scored if s["t3_hit"]
                              and (t4_threshold - 0.05) <= s["t4_score"] < t4_threshold),
        "samples": [
            {
                "bucket": s["bucket"],
                "l1": s["l1"],
                "l2": s["l2"],
                "t4_score": round(s["t4_score"], 3),
            }
            for s in sorted(t3_only, key=lambda s: s["t4_score"])[:15]
        ],
    }


def main() -> None:
    extract = Path("data/wikidata_raw/en_ru/sitelinks.jsonl")
    pairs = [json.loads(line) for line in extract.read_text(encoding="utf-8").splitlines()]
    logger.info("Loaded %d pairs", len(pairs))

    logger.info("Scoring every pair (one pass) ...")
    scored = score_all_pairs(pairs)

    sweep_rows = sweep(scored, T4_THRESHOLDS)
    inspection = inspect_t3_only(scored, DEFAULT_T4_THRESHOLD)

    print()
    print("=" * 84)
    print(f"T4 threshold sweep — n = {len(scored)} pairs, production T4 backend")
    print("=" * 84)
    print(f"{'thr':>5} {'T4 rec':>8} {'T3 marg':>8} {'T4 marg':>8} {'comb rec':>9} {'both':>5} {'T3o':>4} {'T4o':>4} {'miss':>5}  decision")
    for row in sweep_rows:
        if row["t3_marginal_over_t4"] >= 0.15:
            d = "PROCEED"
        elif row["t3_marginal_over_t4"] < 0.05:
            d = "REFRAME"
        else:
            d = "AMBIG"
        marker = "  <-- cascade default" if row["t4_threshold"] == DEFAULT_T4_THRESHOLD else ""
        print(
            f"{row['t4_threshold']:>5.2f} {row['t4_recall']:>8.1%} {row['t3_marginal_over_t4']:>8.1%} "
            f"{row['t4_marginal_over_t3']:>8.1%} {row['combined_recall']:>9.1%} "
            f"{row['both']:>5} {row['t3_only']:>4} {row['t4_only']:>4} {row['neither']:>5}  {d}{marker}"
        )

    print()
    print("=" * 84)
    print(f"T3-only inspection at T4 threshold = {DEFAULT_T4_THRESHOLD}")
    print("=" * 84)
    print(f"  Count: {inspection['n']}")
    print(f"  Near threshold ({DEFAULT_T4_THRESHOLD - 0.05} <= T4 < {DEFAULT_T4_THRESHOLD}): {inspection['near_threshold']}")
    print(
        f"  T4 cosine: min={inspection['t4_score_min']:.3f} "
        f"median={inspection['t4_score_median']:.3f} max={inspection['t4_score_max']:.3f}"
    )
    print(f"  By bucket: {inspection['by_bucket']}")
    print()
    print(f"  Lowest-T4-score samples (truly orthogonal to T4):")
    for s in inspection["samples"]:
        print(f"    [{s['bucket']:<14}] T4={s['t4_score']:>5.3f}  {s['l1']!r}  <-->  {s['l2']!r}")
    print("=" * 84)

    report_path = Path("data/wikidata_raw/en_ru/pilot_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["t4_threshold_sweep"] = {
        "t4_thresholds": T4_THRESHOLDS,
        "rows": sweep_rows,
    }
    report["t3_only_inspection"] = inspection
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Appended sweep + inspection to %s", report_path)


if __name__ == "__main__":
    main()
