"""
WikidataLoader — stub for loading multilingual entity labels from Wikidata.

This module documents how to query Wikidata Q-nodes for multilingual labels
using the Wikidata REST API and SPARQL endpoint.  No network calls are made
in this stub — it's designed to be extended for production use.

Wikidata REST API
-----------------
Base URL: https://www.wikidata.org/w/api.php

Example: fetch all labels for Q7251 (Alan Turing)::

    import requests
    resp = requests.get(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbgetentities",
            "ids": "Q7251",
            "props": "labels",
            "format": "json",
        },
        headers={"User-Agent": "polyglot-er/0.1 (research)"},
    )
    data = resp.json()
    labels = data["entities"]["Q7251"]["labels"]
    # labels = {"en": {"language": "en", "value": "Alan Turing"}, "ru": {...}, ...}

SPARQL endpoint
---------------
URL: https://query.wikidata.org/sparql

Example SPARQL query for person labels::

    SELECT ?item ?itemLabel WHERE {
      ?item wdt:P31 wd:Q5.  # instance of human
      ?item wdt:P27 wd:Q30.  # citizen of United States
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en,ru,zh,ar". }
    }
    LIMIT 100
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class WikidataLoader:
    """
    Stub loader for Wikidata multilingual entity labels.

    In production, implement ``fetch_labels`` and ``fetch_by_sparql``
    using the ``requests`` library and the endpoints documented above.

    Args:
        user_agent: HTTP User-Agent string (Wikidata requires identification)
        languages: List of BCP-47 language codes to fetch
    """

    WIKIDATA_API = "https://www.wikidata.org/w/api.php"
    SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

    def __init__(
        self,
        user_agent: str = "polyglot-er/0.1 (research; contact: your@email.com)",
        languages: Optional[List[str]] = None,
    ):
        self.user_agent = user_agent
        self.languages = languages or ["en", "ru", "zh", "ar", "de", "fr"]

    def fetch_labels(self, qids: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Fetch multilingual labels for the given Q-node IDs.

        STUB — raises NotImplementedError. To implement::

            import requests
            resp = requests.get(
                self.WIKIDATA_API,
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(qids),
                    "props": "labels",
                    "languages": "|".join(self.languages),
                    "format": "json",
                },
                headers={"User-Agent": self.user_agent},
                timeout=10,
            )
            resp.raise_for_status()
            entities = resp.json().get("entities", {})
            result = {}
            for qid, entity in entities.items():
                result[qid] = {
                    lang: label_info["value"]
                    for lang, label_info in entity.get("labels", {}).items()
                }
            return result

        Args:
            qids: List of Wikidata Q-node identifiers (e.g. ["Q7251", "Q1339"])

        Returns:
            Dict mapping qid → {lang_code → label_string}
        """
        raise NotImplementedError(
            "WikidataLoader.fetch_labels is a stub. "
            "See the docstring for implementation instructions."
        )

    def fetch_by_sparql(self, sparql_query: str) -> List[Dict[str, Any]]:
        """
        Execute a SPARQL query against the Wikidata endpoint.

        STUB — raises NotImplementedError. To implement::

            import requests
            resp = requests.get(
                self.SPARQL_ENDPOINT,
                params={"query": sparql_query, "format": "json"},
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/sparql-results+json",
                },
                timeout=30,
            )
            resp.raise_for_status()
            bindings = resp.json()["results"]["bindings"]
            return [
                {k: v.get("value") for k, v in row.items()}
                for row in bindings
            ]

        Args:
            sparql_query: SPARQL SELECT query string

        Returns:
            List of row dicts {variable_name → value_string}
        """
        raise NotImplementedError(
            "WikidataLoader.fetch_by_sparql is a stub. "
            "See the docstring for implementation instructions."
        )

    @staticmethod
    def example_sparql_persons(languages: str = "en,ru,zh,ar", limit: int = 100) -> str:
        """
        Return an example SPARQL query for multilingual person labels.

        Args:
            languages: Comma-separated BCP-47 codes
            limit: Maximum number of results

        Returns:
            SPARQL query string
        """
        return f"""
SELECT DISTINCT ?item ?itemLabel WHERE {{
  ?item wdt:P31 wd:Q5.
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "{languages}".
  }}
}}
LIMIT {limit}
""".strip()
