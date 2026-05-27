"""en↔ja no-lang-hint robustness test.

The pykakasi fix in commit f7ed981 routed Japanese kanji through Hepburn
when lang="ja" was passed. In practice we may not always know the L2
language at match time. The transliterator's auto-detect heuristic falls
back to "kana present → Japanese; otherwise → Chinese (Mandarin pinyin)."

This script runs the en_ja precision eval with and without the lang hint
to measure how much of the pykakasi gain survives in the no-hint regime.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from polyglot_er.matchers.embedding import EmbeddingMatcher
from polyglot_er.matchers.phonetic import PhoneticMatcher
from polyglot_er.normalization.title_format import normalize_wiki_title
from polyglot_er.normalization.transliterate import transliterate_to_latin


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("test_no_lang_hint")


TIER3_THRESHOLD = 0.82
TIER4_THRESHOLD = 0.75


def build_same_bucket_negatives(positives: list[dict]) -> list[dict]:
    from jellyfish import jaro_winkler_similarity
    from collections import defaultdict
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


def evaluate(positives: list[dict], negatives: list[dict], lang_b_hint: str, t4) -> dict:
    """Run T3+T4 with lang_b_hint passed (or empty) to the matcher."""
    from polyglot_er.matchers.phonetic import PhoneticMatcher
    t3 = PhoneticMatcher(threshold=TIER3_THRESHOLD)

    tallies: dict[str, Counter] = {"t3": Counter(), "t4": Counter(), "combined": Counter()}
    for p in positives:
        a = normalize_wiki_title(p["l1_title"], "en")
        b = normalize_wiki_title(p["l2_title"], "ja")
        t3_hit = t3.match(a, b, lang_a="en", lang_b=lang_b_hint).is_match
        t4_score = float(t4.score(a, b, lang_a="en", lang_b="ja"))
        t4_hit = t4_score >= TIER4_THRESHOLD
        combined = t3_hit or t4_hit
        tallies["t3"]["TP" if t3_hit else "FN"] += 1
        tallies["t4"]["TP" if t4_hit else "FN"] += 1
        tallies["combined"]["TP" if combined else "FN"] += 1
    for n in negatives:
        a = normalize_wiki_title(n["l1_title"], "en")
        b = normalize_wiki_title(n["l2_title"], "ja")
        t3_hit = t3.match(a, b, lang_a="en", lang_b=lang_b_hint).is_match
        t4_score = float(t4.score(a, b, lang_a="en", lang_b="ja"))
        t4_hit = t4_score >= TIER4_THRESHOLD
        combined = t3_hit or t4_hit
        tallies["t3"]["FP" if t3_hit else "TN"] += 1
        tallies["t4"]["FP" if t4_hit else "TN"] += 1
        tallies["combined"]["FP" if combined else "TN"] += 1

    return {tier: derive(c) for tier, c in tallies.items()}


def _print_row(label: str, m: dict) -> None:
    print(
        f"  {label:<14}  P={m['precision']:.1%}  R={m['recall']:.1%}  F1={m['f1']:.1%}  "
        f"(TP={m['TP']:>3} FP={m['FP']:>3} TN={m['TN']:>3} FN={m['FN']:>3})"
    )


def main() -> None:
    extract = Path("data/wikidata_raw/en_ja/sitelinks.jsonl")
    positives = [json.loads(line) for line in extract.read_text(encoding="utf-8").splitlines()]
    negatives = build_same_bucket_negatives(positives)

    t4 = EmbeddingMatcher(threshold=0.0, force_tfidf=False)

    with_hint = evaluate(positives, negatives, lang_b_hint="ja", t4=t4)
    without_hint = evaluate(positives, negatives, lang_b_hint="", t4=t4)

    print()
    print("=" * 84)
    print(f"en↔ja no-lang-hint robustness  (n+={len(positives)}, n-={len(negatives)})")
    print("=" * 84)
    print(f"\nWith lang_b='ja' (explicit):")
    _print_row("T3", with_hint["t3"])
    _print_row("T4", with_hint["t4"])
    _print_row("Combined", with_hint["combined"])

    print(f"\nWithout lang_b='' (kana-auto-detect only):")
    _print_row("T3", without_hint["t3"])
    _print_row("T4", without_hint["t4"])
    _print_row("Combined", without_hint["combined"])

    print()
    print("Delta (no-hint - explicit-hint):")
    for tier in ("t3", "t4", "combined"):
        m_w = with_hint[tier]
        m_wo = without_hint[tier]
        print(
            f"  {tier:<10}  ΔP={m_wo['precision'] - m_w['precision']:+.1%}  "
            f"ΔR={m_wo['recall'] - m_w['recall']:+.1%}  "
            f"ΔF1={m_wo['f1'] - m_w['f1']:+.1%}  "
            f"(T3 hits flipped: TP {m_w['TP']}→{m_wo['TP']}, FN {m_w['FN']}→{m_wo['FN']})"
        )

    # Quick check: how many en_ja L2 titles have kana? That's the population
    # where auto-detection works without the hint.
    has_kana = sum(1 for p in positives if any("぀" <= c <= "ヿ" for c in p["l2_title"]))
    print()
    print(f"Of {len(positives)} en_ja L2 titles, {has_kana} ({has_kana/len(positives):.1%}) contain kana.")
    print(f"The remaining {len(positives)-has_kana} are pure-kanji titles — these are exactly the ones")
    print(f"that lose the pykakasi route without lang='ja' (default to pypinyin Mandarin readings).")

    # Persist
    out = Path("data/wikidata_raw/en_ja/no_lang_hint_test.json")
    out.write_text(json.dumps({
        "with_hint": with_hint,
        "without_hint": without_hint,
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "n_titles_with_kana": has_kana,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote test result to %s", out)


if __name__ == "__main__":
    main()
