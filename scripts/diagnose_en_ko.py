"""Diagnose the en↔ko gap — why is Korean F1=57.7% when en↔ru is 83.2%?

Both are alphabetic-cross-script with working transliterators. The recall
gap (en_ru 75.7% vs en_ko 41.7%) shouldn't be regime-explained. Three
hypotheses to test:

H1: Over-syllabification reduces JW.
    The simplified RR splits Hangul syllables with spaces ("yul ri ya"
    vs English "Yulia"). Each space inserts a position-shifting boundary
    that Jaro-Winkler weighs against. Test by computing JW on whitespace-
    stripped Romanization.

H2: Korean Wikipedia uses more semantic translation than Russian.
    Inspect the miss-both set: count pairs where the Hangul title is a
    near-Romanization of the English (T3-recoverable in principle) vs
    pairs where the Hangul title is a completely different semantic
    rendering (Forbidden City / 古宫-style cases applied to Korean).

H3: Romanization quality issue — simplified RR doesn't apply consonant-
    cluster sandhi rules that full RR uses. Inspect the lowest-JW
    miss-both pairs and check whether full-RR (e.g., via the `korean-
    romanizer` package if installed) would match.

This script answers H1 and H2 from existing data; H3 is qualitative
inspection.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from jellyfish import jaro_winkler_similarity

from polyglot_er.matchers.embedding import EmbeddingMatcher
from polyglot_er.matchers.phonetic import PhoneticMatcher
from polyglot_er.normalization.title_format import normalize_wiki_title
from polyglot_er.normalization.transliterate import transliterate_to_latin


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("diagnose_en_ko")


TIER3_THRESHOLD = 0.82
TIER4_THRESHOLD = 0.75


def main() -> None:
    extract = Path("data/wikidata_raw/en_ko/sitelinks.jsonl")
    pairs = [json.loads(line) for line in extract.read_text(encoding="utf-8").splitlines()]
    logger.info("Loaded %d en↔ko pairs", len(pairs))

    t3 = PhoneticMatcher(threshold=TIER3_THRESHOLD)
    t4 = EmbeddingMatcher(threshold=0.0, force_tfidf=False)

    # Score every pair under multiple JW variants.
    rows = []
    for p in pairs:
        l1 = normalize_wiki_title(p["l1_title"], "en")
        l2 = normalize_wiki_title(p["l2_title"], "ko")
        l1_norm = l1.lower().replace(" ", "")
        # Standard RR (current cascade behavior).
        l2_rr = transliterate_to_latin(l2, lang="ko").lower()
        # H1 test: strip whitespace from RR to test if space-induced JW noise
        # is responsible for missed matches.
        l2_rr_nospace = l2_rr.replace(" ", "")
        jw_standard = float(jaro_winkler_similarity(l1.lower(), l2_rr))
        jw_nospace = float(jaro_winkler_similarity(l1_norm, l2_rr_nospace))
        t4_score = float(t4.score(l1, l2, lang_a="en", lang_b="ko"))
        rows.append(
            {
                "bucket": p["bucket"],
                "l1": l1,
                "l2": l2,
                "l2_rr": l2_rr,
                "l2_rr_nospace": l2_rr_nospace,
                "jw_standard": jw_standard,
                "jw_nospace": jw_nospace,
                "t4_score": t4_score,
                "t3_standard_hit": jw_standard >= TIER3_THRESHOLD,
                "t3_nospace_hit": jw_nospace >= TIER3_THRESHOLD,
                "t4_hit": t4_score >= TIER4_THRESHOLD,
            }
        )

    n = len(rows)

    # H1: would no-space tokenization recover more matches?
    t3_standard = sum(1 for r in rows if r["t3_standard_hit"])
    t3_nospace = sum(1 for r in rows if r["t3_nospace_hit"])
    t4_hits = sum(1 for r in rows if r["t4_hit"])
    standard_recovered = sum(1 for r in rows if r["t3_standard_hit"] and not r["t4_hit"])
    nospace_recovered = sum(1 for r in rows if r["t3_nospace_hit"] and not r["t4_hit"])
    miss_both_standard = sum(1 for r in rows if not r["t3_standard_hit"] and not r["t4_hit"])
    miss_both_nospace = sum(1 for r in rows if not r["t3_nospace_hit"] and not r["t4_hit"])

    print()
    print("=" * 84)
    print(f"H1: Does whitespace-stripping the RR output recover more matches?")
    print("=" * 84)
    print(f"  Standard RR (current):  T3 recall={t3_standard / n:.1%}  "
          f"T3-marg-over-T4={standard_recovered / n:.1%}  "
          f"miss-both={miss_both_standard / n:.1%}")
    print(f"  No-space RR variant:    T3 recall={t3_nospace / n:.1%}  "
          f"T3-marg-over-T4={nospace_recovered / n:.1%}  "
          f"miss-both={miss_both_nospace / n:.1%}")
    print(f"  Δ T3 recall: {(t3_nospace - t3_standard) / n:+.1%}")
    print(f"  Δ miss-both: {(miss_both_nospace - miss_both_standard) / n:+.1%}")

    # H2: characterize miss-both set
    miss_both = [r for r in rows if not r["t3_standard_hit"] and not r["t4_hit"]]
    # "Near-romanization" = miss-both but jw_standard or jw_nospace > 0.7 (just
    # below T3 threshold). These would be recoverable with a slightly looser
    # threshold or better romanization.
    near_recovery = [r for r in miss_both if max(r["jw_standard"], r["jw_nospace"]) >= 0.7]
    far_miss = [r for r in miss_both if max(r["jw_standard"], r["jw_nospace"]) < 0.5]

    print()
    print("=" * 84)
    print(f"H2: Miss-both set composition (n={len(miss_both)})")
    print("=" * 84)
    print(f"  Near-recovery (max JW >= 0.7, below T3=0.82): {len(near_recovery)} ({len(near_recovery)/len(miss_both):.1%})")
    print(f"  Far miss (max JW < 0.5): {len(far_miss)} ({len(far_miss)/len(miss_both):.1%})")
    print(f"  By bucket: {dict(Counter(r['bucket'] for r in miss_both))}")

    # H3: show lowest-JW samples (truly semantic translations, can't fix)
    print()
    print("  Lowest-JW miss-both samples (likely semantic translations):")
    far_samples = sorted(miss_both, key=lambda r: max(r["jw_standard"], r["jw_nospace"]))[:10]
    for s in far_samples:
        print(
            f"    [{s['bucket']:<14}] jw={max(s['jw_standard'], s['jw_nospace']):>5.3f} "
            f"T4={s['t4_score']:>5.3f}  {s['l1']!r}  <-->  {s['l2']!r}  "
            f"(rr={s['l2_rr']!r})"
        )

    # Near-recovery samples (would benefit from better RR or looser threshold)
    print()
    print("  Near-recovery miss-both samples (would T3-recover with looser threshold or better RR):")
    near_samples = sorted(near_recovery, key=lambda r: -max(r["jw_standard"], r["jw_nospace"]))[:10]
    for s in near_samples:
        print(
            f"    [{s['bucket']:<14}] jw_std={s['jw_standard']:>5.3f} jw_nosp={s['jw_nospace']:>5.3f} "
            f"T4={s['t4_score']:>5.3f}  {s['l1']!r}  <-->  {s['l2_rr']!r}"
        )

    # Persist
    report_path = Path("data/wikidata_raw/en_ko/diagnosis.json")
    report_path.write_text(json.dumps({
        "h1_whitespace_test": {
            "t3_recall_standard": t3_standard / n,
            "t3_recall_nospace": t3_nospace / n,
            "delta_t3_recall": (t3_nospace - t3_standard) / n,
            "delta_miss_both": (miss_both_nospace - miss_both_standard) / n,
        },
        "h2_miss_both_composition": {
            "n_miss_both": len(miss_both),
            "n_near_recovery": len(near_recovery),
            "n_far_miss": len(far_miss),
            "by_bucket": dict(Counter(r['bucket'] for r in miss_both)),
        },
        "far_miss_samples": [
            {k: r[k] for k in ["bucket", "l1", "l2", "l2_rr", "jw_standard", "t4_score"]}
            for r in far_samples
        ],
        "near_recovery_samples": [
            {k: r[k] for k in ["bucket", "l1", "l2", "l2_rr", "jw_standard", "jw_nospace", "t4_score"]}
            for r in near_samples
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote diagnosis to %s", report_path)


if __name__ == "__main__":
    main()
