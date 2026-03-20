"""
polyglot-er CLI

Usage:
    polyglot-er resolve --input entities.jsonl --output clusters.json [options]

Options:
    --input PATH       Input JSONL file with entity records
    --output PATH      Output JSON file for resolved clusters
    --tier2 FLOAT      Jaro-Winkler threshold for same-script fuzzy (default: 0.85)
    --tier3 FLOAT      Phonetic threshold for cross-script (default: 0.82)
    --tier4 FLOAT      Embedding cosine threshold (default: 0.75)
    --force-tfidf      Use TF-IDF fallback instead of sentence-transformers
    --verbose          Print progress to stdout
    --help             Show this message

Example:
    polyglot-er resolve --input data/multilingual_entities.jsonl --output /tmp/clusters.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_resolve(args: argparse.Namespace) -> int:
    """Run the resolver and write output."""
    from polyglot_er import CrossLingualResolver

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    resolver = CrossLingualResolver(
        tier2_threshold=args.tier2,
        tier3_threshold=args.tier3,
        tier4_threshold=args.tier4,
        force_tfidf=args.force_tfidf,
        verbose=args.verbose,
    )

    clusters = resolver.resolve_and_save(input_path, output_path)

    if args.verbose:
        print(f"Resolved {sum(len(c) for c in clusters)} records into {len(clusters)} clusters.")
    else:
        print(f"Clusters: {len(clusters)} → {output_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polyglot-er",
        description="Cross-Lingual Entity Resolution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    resolve_parser = subparsers.add_parser(
        "resolve",
        help="Resolve entity records into co-reference clusters",
    )
    resolve_parser.add_argument("--input", required=True, help="Input JSONL file")
    resolve_parser.add_argument("--output", required=True, help="Output JSON file")
    resolve_parser.add_argument("--tier2", type=float, default=0.85)
    resolve_parser.add_argument("--tier3", type=float, default=0.82)
    resolve_parser.add_argument("--tier4", type=float, default=0.75)
    resolve_parser.add_argument(
        "--force-tfidf",
        action="store_true",
        default=False,
        help="Use TF-IDF fallback backend",
    )
    resolve_parser.add_argument(
        "--verbose", "-v", action="store_true", default=False
    )

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "resolve":
        return cmd_resolve(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
