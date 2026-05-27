"""Diagnose the en↔zh recall gap — why does T4 cap at 38% and T3 add nothing?

Three questions:

1. Does the cascade's transliterator (transliterate_to_latin) actually produce
   anything meaningful for Han characters? If it returns the original Han
   string or a near-empty string, T3's Jaro-Winkler step has nothing to align
   against — that alone explains why T3 catches ~nothing.

2. Of the miss-both pairs (neither T3 nor T4 at default thresholds), how many
   look like phonetic transliterations that should align in principle (e.g.
   'Coca-Cola' ↔ '可口可乐' = "ke-kou-ke-le", phonetic), versus semantically
   different titles ('Forbidden City' ↔ '故宫' = "Old Palace") versus
   structurally rewritten titles?

3. What feature would a "Tier 3.5 for logographic" need that the current
   PhoneticMatcher doesn't have? E.g., Pinyin conversion of Han, then
   Jaro-Winkler against the English. Test this by reading the T4 cosine
   scores of pairs that would match under a hypothetical Pinyin+JW tier.

Output is printed and appended to data/wikidata_raw/en_zh/pilot_report.json
under ``en_zh_diagnosis``.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from polyglot_er.matchers.embedding import EmbeddingMatcher
from polyglot_er.matchers.phonetic import PhoneticMatcher
from polyglot_er.normalization.transliterate import transliterate_to_latin
from polyglot_er.normalization.title_format import normalize_wiki_title


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("diagnose_en_zh")


TIER3_THRESHOLD = 0.82
TIER4_THRESHOLD = 0.75


def diagnose_transliteration(pairs: list[dict]) -> dict:
    """Question 1: what does transliterate_to_latin do to the Chinese titles?"""
    samples = []
    n_empty = 0
    n_passthrough = 0  # output == input (transliterator did nothing)
    n_changed = 0
    for pair in pairs[:50]:  # representative sample
        zh = pair["l2_title"]
        out = transliterate_to_latin(zh)
        if not out or not out.strip():
            n_empty += 1
        elif out == zh:
            n_passthrough += 1
        else:
            n_changed += 1
        if len(samples) < 15:
            samples.append({"zh": zh, "transliterated": out})
    return {
        "sample_size": min(50, len(pairs)),
        "n_empty": n_empty,
        "n_passthrough": n_passthrough,
        "n_changed": n_changed,
        "samples": samples,
    }


def categorize_miss_both(pairs: list[dict], t3, t4) -> dict:
    """Question 2: categorize miss-both pairs by visual inspection of the data."""
    miss_both = []
    for pair in pairs:
        l1 = normalize_wiki_title(pair["l1_title"], "en")
        l2 = normalize_wiki_title(pair["l2_title"], "zh")
        t3_hit = t3.match(l1, l2, lang_a="en", lang_b="zh").is_match
        t4_score = float(t4.score(l1, l2, lang_a="en", lang_b="zh"))
        if t3_hit or t4_score >= TIER4_THRESHOLD:
            continue
        miss_both.append(
            {
                "l1": l1,
                "l2": l2,
                "bucket": pair["bucket"],
                "t4_score": t4_score,
                "l1_has_ascii_proper_noun": all(c.isascii() for c in l1) and l1[:1].isupper(),
                "l2_has_any_ascii": any(c.isascii() and c.isalpha() for c in l2),
            }
        )
    return {
        "n_miss_both": len(miss_both),
        "n_l1_ascii_proper_noun": sum(1 for m in miss_both if m["l1_has_ascii_proper_noun"]),
        "n_l2_has_ascii": sum(1 for m in miss_both if m["l2_has_any_ascii"]),
        "by_bucket": dict(Counter(m["bucket"] for m in miss_both)),
        "samples_low_t4": [
            {k: m[k] for k in ["bucket", "l1", "l2", "t4_score"]}
            for m in sorted(miss_both, key=lambda m: m["t4_score"])[:15]
        ],
        "samples_near_threshold": [
            {k: m[k] for k in ["bucket", "l1", "l2", "t4_score"]}
            for m in sorted(miss_both, key=lambda m: -m["t4_score"])[:15]
        ],
    }


def main() -> None:
    extract = Path("data/wikidata_raw/en_zh/sitelinks.jsonl")
    pairs = [json.loads(line) for line in extract.read_text(encoding="utf-8").splitlines()]
    logger.info("Loaded %d en↔zh pairs", len(pairs))

    t3 = PhoneticMatcher(threshold=TIER3_THRESHOLD)
    t4 = EmbeddingMatcher(threshold=0.0, force_tfidf=False)

    # Q1: transliteration behavior
    transl = diagnose_transliteration(pairs)
    print()
    print("=" * 84)
    print("Q1: What does transliterate_to_latin do to Han characters?")
    print("=" * 84)
    print(
        f"  Sample size: {transl['sample_size']}  empty: {transl['n_empty']}  "
        f"passthrough (output==input): {transl['n_passthrough']}  changed: {transl['n_changed']}"
    )
    print("  Examples (Han → transliterated):")
    for s in transl["samples"]:
        marker = "  [PASSTHROUGH]" if s["zh"] == s["transliterated"] else ""
        print(f"    {s['zh']!r}  ->  {s['transliterated']!r}{marker}")

    # Q2: characterize the miss-both set
    cat = categorize_miss_both(pairs, t3, t4)
    print()
    print("=" * 84)
    print("Q2: What's in the miss-both set (T3 miss AND T4 < 0.75)?")
    print("=" * 84)
    print(f"  Total miss-both: {cat['n_miss_both']} / {len(pairs)}")
    print(f"  L1 is ASCII proper noun (single token in Latin): {cat['n_l1_ascii_proper_noun']}")
    print(f"  L2 contains any ASCII alpha (suggesting borrowed/Latin chars): {cat['n_l2_has_ascii']}")
    print(f"  By bucket: {cat['by_bucket']}")
    print()
    print("  Lowest-T4-score misses (most semantically dissimilar):")
    for s in cat["samples_low_t4"]:
        print(f"    [{s['bucket']:<14}] T4={s['t4_score']:>5.3f}  {s['l1']!r}  <-->  {s['l2']!r}")
    print()
    print("  Near-threshold misses (T4 close to 0.75; calibration sensitive):")
    for s in cat["samples_near_threshold"]:
        print(f"    [{s['bucket']:<14}] T4={s['t4_score']:>5.3f}  {s['l1']!r}  <-->  {s['l2']!r}")
    print("=" * 84)

    # Persist
    report_path = Path("data/wikidata_raw/en_zh/pilot_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["en_zh_diagnosis"] = {
        "q1_transliteration_behavior": transl,
        "q2_miss_both_categorization": cat,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Appended diagnosis to %s", report_path)


if __name__ == "__main__":
    main()
