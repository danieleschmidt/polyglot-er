"""Smoke test: hit the live Wikidata SPARQL endpoint with one small query
to confirm the WikidataLoader works end-to-end.

Run with: python scripts/sparql_smoke.py
"""

from __future__ import annotations

import logging

from polyglot_er.datasets.wikidata import WikidataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    loader = WikidataLoader()
    query = WikidataLoader.build_sitelink_query("en", "ru", "Q5", limit=3)
    rows = loader.fetch_by_sparql(query)
    print(f"Got {len(rows)} rows from live SPARQL endpoint:")
    for i, row in enumerate(rows, 1):
        print(f"  [{i}] {row.get('l1Title')!r} <-> {row.get('l2Title')!r}  ({row.get('entity')})")


if __name__ == "__main__":
    main()
