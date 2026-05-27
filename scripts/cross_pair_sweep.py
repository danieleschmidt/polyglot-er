"""Cross-pair sweep — run the en↔ru pilot pattern on additional language pairs.

The en↔ru pilot (scripts/pilot_en_ru.py + scripts/t4_threshold_sweep.py)
established that T3 catches a coherent class of phonetic-transliteration
proper nouns that production T4 (multilingual MiniLM) systematically misses.
This script tests whether that finding generalizes by running the same
pattern on:

- en↔de (Latin ↔ Latin, "easy" same-script baseline): T3 marginal should be
  near zero because no transliteration is needed.
- en↔zh (Latin ↔ Han, "hardest" different-system case): T3 marginal should
  be high if the phonetic-proper-noun class holds across alphabets.

Output: one summary table per pair, plus a side-by-side comparison row.
Persisted under data/wikidata_raw/{pair}/pilot_report.json and merged into
a top-level data/wikidata_raw/cross_pair_summary.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from polyglot_er.datasets.wikidata import WikidataLoader
from polyglot_er.matchers.embedding import EmbeddingMatcher
from polyglot_er.matchers.phonetic import PhoneticMatcher
from polyglot_er.normalization.title_format import normalize_wiki_title


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cross_pair_sweep")


TIER3_THRESHOLD = 0.82
T4_THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
DEFAULT_T4_THRESHOLD = 0.75
N_PER_TYPE = 50

# Default pairs when run with no CLI args. Additional pairs can be specified
# via --pair l1 l2 (repeatable). See main().
DEFAULT_PAIRS = [
    ("en", "de"),  # same-script baseline (Latin/Latin)
    ("en", "zh"),  # different-system stress case (Latin/Han)
]


def extract(l1: str, l2: str, cache_root: Path) -> Path:
    """Extract sitelink pairs, reusing the cached JSONL if it already exists."""
    out_dir = cache_root / f"{l1}_{l2}"
    out_path = out_dir / "sitelinks.jsonl"
    if out_path.exists():
        logger.info("[%s↔%s] cached extract found at %s — skipping SPARQL", l1, l2, out_path)
        return out_path
    loader = WikidataLoader()
    return loader.extract_sitelink_pairs(l1, l2, n_per_type=N_PER_TYPE, cache_root=cache_root)


def score_all(pairs: list[dict], l1: str, l2: str, t3, t4) -> list[dict]:
    scored = []
    for pair in pairs:
        # Normalizer is language-agnostic in the current implementation; safe
        # to apply to all pairs even when only one side uses Last,First.
        a = normalize_wiki_title(pair["l1_title"], l1)
        b = normalize_wiki_title(pair["l2_title"], l2)
        scored.append(
            {
                "l1": a,
                "l2": b,
                "bucket": pair["bucket"],
                "t3_hit": t3.match(a, b, lang_a=l1, lang_b=l2).is_match,
                "t4_score": float(t4.score(a, b, lang_a=l1, lang_b=l2)),
            }
        )
    return scored


def sweep(scored: list[dict]) -> list[dict]:
    n = len(scored)
    rows = []
    for thr in T4_THRESHOLDS:
        t3_only = sum(1 for s in scored if s["t3_hit"] and s["t4_score"] < thr)
        t4_only = sum(1 for s in scored if not s["t3_hit"] and s["t4_score"] >= thr)
        both = sum(1 for s in scored if s["t3_hit"] and s["t4_score"] >= thr)
        neither = sum(1 for s in scored if not s["t3_hit"] and s["t4_score"] < thr)
        rows.append(
            {
                "t4_threshold": thr,
                "t3_recall": (t3_only + both) / n if n else 0.0,
                "t4_recall": (t4_only + both) / n if n else 0.0,
                "combined_recall": (n - neither) / n if n else 0.0,
                "t3_marginal_over_t4": t3_only / n if n else 0.0,
                "both": both,
                "t3_only": t3_only,
                "t4_only": t4_only,
                "neither": neither,
            }
        )
    return rows


def inspect_t3_only(scored: list[dict], thr: float) -> dict:
    t3_only = [s for s in scored if s["t3_hit"] and s["t4_score"] < thr]
    if not t3_only:
        return {"n": 0, "samples": [], "by_bucket": {}}
    scores = sorted(s["t4_score"] for s in t3_only)
    return {
        "n": len(t3_only),
        "t4_score_min": scores[0],
        "t4_score_median": scores[len(scores) // 2],
        "t4_score_max": scores[-1],
        "by_bucket": dict(Counter(s["bucket"] for s in t3_only)),
        "samples": [
            {
                "bucket": s["bucket"],
                "l1": s["l1"],
                "l2": s["l2"],
                "t4_score": round(s["t4_score"], 3),
            }
            for s in sorted(t3_only, key=lambda s: s["t4_score"])[:10]
        ],
    }


def run_pair(l1: str, l2: str, cache_root: Path, t3, t4) -> dict:
    pairs_path = extract(l1, l2, cache_root)
    raw = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines()]
    logger.info("[%s↔%s] extracted %d pairs", l1, l2, len(raw))
    scored = score_all(raw, l1, l2, t3, t4)
    sweep_rows = sweep(scored)
    inspection = inspect_t3_only(scored, DEFAULT_T4_THRESHOLD)

    # Persist per-pair report.
    report = {
        "language_pair": f"{l1}_{l2}",
        "n_pairs": len(scored),
        "t4_threshold_sweep": {"t4_thresholds": T4_THRESHOLDS, "rows": sweep_rows},
        "t3_only_inspection_at_default": inspection,
    }
    (pairs_path.parent / "pilot_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return report


def _print_sweep_table(label: str, n: int, rows: list[dict]) -> None:
    print(f"\n{label} (n={n})")
    print(f"  {'thr':>5} {'T4 rec':>8} {'T3 marg':>8} {'comb rec':>9}  decision")
    for r in rows:
        if r["t3_marginal_over_t4"] >= 0.15:
            d = "PROCEED"
        elif r["t3_marginal_over_t4"] < 0.05:
            d = "REFRAME"
        else:
            d = "AMBIG"
        marker = "  <-- default" if r["t4_threshold"] == DEFAULT_T4_THRESHOLD else ""
        print(
            f"  {r['t4_threshold']:>5.2f} {r['t4_recall']:>8.1%} {r['t3_marginal_over_t4']:>8.1%} "
            f"{r['combined_recall']:>9.1%}  {d}{marker}"
        )


def _print_inspection(label: str, inspection: dict) -> None:
    print(f"\n{label} T3-only set at T4 threshold {DEFAULT_T4_THRESHOLD}")
    if not inspection["n"]:
        print("  (empty)")
        return
    print(
        f"  n={inspection['n']}  T4 cosine min={inspection['t4_score_min']:.3f} "
        f"med={inspection['t4_score_median']:.3f} max={inspection['t4_score_max']:.3f}"
    )
    print(f"  By bucket: {inspection['by_bucket']}")
    for s in inspection["samples"]:
        print(f"    [{s['bucket']:<14}] T4={s['t4_score']:>5.3f}  {s['l1']!r}  <-->  {s['l2']!r}")


def _load_report_for_pair(cache_root: Path, l1: str, l2: str) -> dict | None:
    """Load a previously-written per-pair report, if it exists."""
    p = cache_root / f"{l1}_{l2}" / "pilot_report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _normalize_legacy_en_ru(report: dict) -> dict:
    """The original en_ru pilot report has a different top-level schema (it
    pre-dates cross_pair_sweep.py). Map it to the cross-pair report layout
    so the summary printer can handle it uniformly."""
    if "t4_threshold_sweep" in report and "language_pair" in report:
        return report
    return {
        "language_pair": "en_ru",
        "n_pairs": report.get("normalized", {}).get("n_pairs", 375),
        "t4_threshold_sweep": report.get("t4_threshold_sweep"),
        "t3_only_inspection_at_default": report.get("t3_only_inspection"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("L1", "L2"),
        action="append",
        help="Language pair to evaluate (e.g. --pair en fr). Repeatable. "
        "Defaults to en_de + en_zh.",
    )
    args = parser.parse_args()
    pairs_to_run = [tuple(p) for p in (args.pair or DEFAULT_PAIRS)]

    cache_root = Path("data/wikidata_raw")
    t3 = PhoneticMatcher(threshold=TIER3_THRESHOLD)
    t4 = EmbeddingMatcher(threshold=0.0, force_tfidf=False)
    logger.info("T4 backend: %s", t4.backend_name)
    logger.info("Pairs to run: %s", pairs_to_run)

    for l1, l2 in pairs_to_run:
        run_pair(l1, l2, cache_root, t3, t4)

    # Aggregate all per-pair reports into the summary, not just the ones we
    # ran this invocation. This makes the summary grow monotonically across
    # incremental runs rather than being overwritten each time.
    summary: dict[str, dict] = {}
    for pair_dir in sorted(cache_root.iterdir()):
        if not pair_dir.is_dir():
            continue
        report = _load_report_for_pair(cache_root, *pair_dir.name.split("_", 1))
        if report is None:
            continue
        summary[pair_dir.name] = _normalize_legacy_en_ru(report)

    print()
    print("=" * 84)
    print(f"Cross-pair sweep — production T4 (paraphrase-multilingual-MiniLM-L12-v2)")
    print("=" * 84)

    ordered_keys = sorted(summary.keys())
    for pair_key in ordered_keys:
        rpt = summary[pair_key]
        if not rpt.get("t4_threshold_sweep"):
            continue
        _print_sweep_table(
            f"--- {pair_key} ---", rpt["n_pairs"], rpt["t4_threshold_sweep"]["rows"]
        )
        _print_inspection(
            f"--- {pair_key} ---", rpt["t3_only_inspection_at_default"]
        )

    print()
    print("=" * 84)
    print(f"HEADLINE: T3 marginal at T4 threshold {DEFAULT_T4_THRESHOLD}")
    print("=" * 84)
    for pair_key in ordered_keys:
        rows = summary[pair_key].get("t4_threshold_sweep", {}).get("rows", [])
        for r in rows:
            if r["t4_threshold"] == DEFAULT_T4_THRESHOLD:
                n = summary[pair_key]["n_pairs"]
                print(
                    f"  {pair_key:>8}  n={n:>4}  "
                    f"T4={r['t4_recall']:.1%}  T3marg={r['t3_marginal_over_t4']:.1%}  "
                    f"combined={r['combined_recall']:.1%}"
                )
    print("=" * 84)

    out = cache_root / "cross_pair_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote cross-pair summary to %s (%d pairs)", out, len(summary))


if __name__ == "__main__":
    main()
