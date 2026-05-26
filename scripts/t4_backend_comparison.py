"""T4 backend comparison — TF-IDF fallback vs. sentence-transformers.

Uses the cached en↔ru sitelink extract (375 pairs from scripts/pilot_en_ru.py).
Runs T3 + each T4 backend, with the title-format normalizer applied. The
question this script answers: does the production T4 backend compress
T3-marginal below the 23% headline-claim threshold?

Output is appended to data/wikidata_raw/en_ru/pilot_report.json under
``t4_backend_comparison``.
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
logger = logging.getLogger("t4_comparison")


TIER3_THRESHOLD = 0.82
TIER4_THRESHOLD = 0.75


def evaluate(pairs: list[dict], t4: EmbeddingMatcher) -> dict:
    """Run T3 + the given T4 against all pairs with the normalizer applied."""
    t3 = PhoneticMatcher(threshold=TIER3_THRESHOLD)

    outcomes: Counter[tuple[bool, bool]] = Counter()
    for pair in pairs:
        l1 = normalize_wiki_title(pair["l1_title"], "en")
        l2 = normalize_wiki_title(pair["l2_title"], "ru")
        t3_hit = t3.match(l1, l2, lang_a="en", lang_b="ru").is_match
        t4_hit = t4.match(l1, l2, lang_a="en", lang_b="ru").is_match
        outcomes[(t3_hit, t4_hit)] += 1

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
        "both": both,
        "neither": neither,
        "t3_only": t3_only,
        "t4_only": t4_only,
    }


def main() -> None:
    extract = Path("data/wikidata_raw/en_ru/sitelinks.jsonl")
    if not extract.exists():
        raise SystemExit(
            f"{extract} missing — run scripts/pilot_en_ru.py first to generate it."
        )
    pairs = [json.loads(line) for line in extract.read_text(encoding="utf-8").splitlines()]
    logger.info("Loaded %d pairs from cached extract", len(pairs))

    logger.info("Run 1: TF-IDF fallback T4")
    tfidf_t4 = EmbeddingMatcher(threshold=TIER4_THRESHOLD, force_tfidf=True)
    tfidf = evaluate(pairs, tfidf_t4)

    logger.info("Run 2: sentence-transformers T4 (may take 30-60s for first model load)")
    st_t4 = EmbeddingMatcher(threshold=TIER4_THRESHOLD, force_tfidf=False)
    backend = st_t4.backend_name
    logger.info("Active T4 backend: %s", backend)
    st = evaluate(pairs, st_t4)

    print()
    print("=" * 72)
    print(f"T4 backend comparison — n = {tfidf['n_pairs']} pairs (normalizer ON)")
    print("=" * 72)
    print(f"{'metric':<30} {'TF-IDF':>10} {'ST/' + backend:>22} {'Δ':>8}")
    for key, label in [
        ("t3_recall",          "T3 recall (phonetic)"),
        ("t4_recall",          "T4 recall (embedding)"),
        ("t3_marginal_over_t4","T3 marginal over T4"),
    ]:
        print(
            f"{label:<30} {tfidf[key]:>10.1%} {st[key]:>22.1%} {st[key] - tfidf[key]:>+8.1%}"
        )
    for key, label in [
        ("both",    "Both match"),
        ("t3_only", "T3 only"),
        ("t4_only", "T4 only"),
        ("neither", "Neither"),
    ]:
        print(
            f"{label:<30} {tfidf[key]:>10d} {st[key]:>22d} {st[key] - tfidf[key]:>+8d}"
        )
    print()
    decision = (
        "PROCEED — T3 marginal still clears 15%" if st["t3_marginal_over_t4"] >= 0.15
        else "REFRAME — T3 marginal collapsed under production T4"
        if st["t3_marginal_over_t4"] < 0.05
        else "AMBIGUOUS — revisit with Mike"
    )
    print(f"DECISION (T3 marginal under production T4 = {st['t3_marginal_over_t4']:.1%}): {decision}")
    print("=" * 72)

    # Merge into the existing pilot report.
    report_path = Path("data/wikidata_raw/en_ru/pilot_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["t4_backend_comparison"] = {
        "tfidf": tfidf,
        "sentence_transformers": st,
        "sentence_transformers_backend_name": backend,
        "decision": decision,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Appended T4-backend-comparison block to %s", report_path)


if __name__ == "__main__":
    main()
