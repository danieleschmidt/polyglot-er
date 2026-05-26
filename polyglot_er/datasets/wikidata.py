"""
WikidataLoader — multilingual entity-label loader with Wikidata REST and SPARQL.

Implements Stage 1 of the P7 ground-truth pipeline (see
`docgraph-publication/papers/emnlp2027-polyglot-er/wikidata_pipeline_design.md`):
bulk SPARQL extraction of cross-lingual sitelink pairs for each (L1, L2) language
pair, type-stratified per §3.2 of the design doc.

Network behavior
----------------
``WikidataLoader`` honors Wikidata's documented rate limit (5 queries per second
per IP, 60 seconds per query) via a token-bucket throttle and exponential backoff
on HTTP 429 / 503 responses. Public endpoints:

- REST: ``https://www.wikidata.org/w/api.php``
- SPARQL: ``https://query.wikidata.org/sparql``

Both endpoints require a non-anonymous ``User-Agent`` header. Set
``user_agent`` to a string that identifies the project and a contact address.

Cache layout
------------
``extract_sitelink_pairs`` writes one JSONL row per cross-lingual identity pair to
``data/wikidata_raw/{L1}_{L2}/sitelinks.jsonl``. Each row has the schema
documented in §3.1 of the design doc:
``{qid, l1_title, l2_title, wikidata_type, type_label, bucket}``.

For offline testing, the network layer is isolated in
``WikidataLoader._http_get`` and can be monkey-patched / injected.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import requests

logger = logging.getLogger(__name__)


WIKIDATA_API = "https://www.wikidata.org/w/api.php"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

DEFAULT_USER_AGENT = "polyglot-er/0.2 (research; https://github.com/danieleschmidt/polyglot-er)"

SPARQL_QPS_LIMIT = 5.0
SPARQL_TIMEOUT_SECONDS = 60

# Type buckets for stratified sampling per design doc §3.2.
# Map Wikidata QID → human-readable bucket name. Order matters: the per-pair
# extraction queries each bucket in this order, which gives deterministic
# JSONL ordering for downstream reproducibility.
TYPE_BUCKETS: tuple[tuple[str, str], ...] = (
    ("Q5", "person"),
    ("Q43229", "organization"),
    ("Q515", "city"),
    ("Q6256", "country"),
    ("Q11424", "film"),
    ("Q571", "book"),
    ("Q11173", "chemical"),
    ("Q838948", "work_of_art"),
    ("Q15089776", "event"),
    ("Q4830453", "business"),
)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


@dataclass
class _TokenBucket:
    """Monotonic-clock token bucket for the SPARQL endpoint."""

    rate: float  # tokens per second
    capacity: float
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def acquire(self, cost: float = 1.0, clock: Callable[[], float] = time.monotonic) -> None:
        """Block until ``cost`` tokens are available, then deduct them."""
        while True:
            now = clock()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens >= cost:
                self.tokens -= cost
                return
            time.sleep(max((cost - self.tokens) / self.rate, 0.01))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


@dataclass
class WikidataLoader:
    """Cross-lingual sitelink-pair loader backed by the Wikidata SPARQL endpoint.

    Args:
        user_agent: HTTP User-Agent string identifying the project + a contact.
        qps_limit: Maximum sustained queries per second (default 5.0,
            Wikidata's documented per-IP cap).
        sparql_timeout: Per-query timeout in seconds (default 60, the
            endpoint's hard limit).
        max_retries: Maximum exponential-backoff retries on transient errors.
        session: Optional pre-configured ``requests.Session`` for connection reuse.
    """

    user_agent: str = DEFAULT_USER_AGENT
    qps_limit: float = SPARQL_QPS_LIMIT
    sparql_timeout: int = SPARQL_TIMEOUT_SECONDS
    max_retries: int = 5
    session: Optional[requests.Session] = None
    _bucket: _TokenBucket = field(init=False)

    def __post_init__(self) -> None:
        self._bucket = _TokenBucket(rate=self.qps_limit, capacity=self.qps_limit)
        if self.session is None:
            self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

    # -- low-level HTTP --------------------------------------------------

    def _http_get(
        self,
        url: str,
        params: dict[str, Any],
        headers: Optional[dict[str, str]] = None,
    ) -> requests.Response:
        """Perform a rate-limited GET. Isolated so tests can monkey-patch it."""
        self._bucket.acquire()
        assert self.session is not None
        merged_headers = dict(self.session.headers)
        if headers:
            merged_headers.update(headers)
        return self.session.get(
            url, params=params, headers=merged_headers, timeout=self.sparql_timeout
        )

    def _sparql(self, query: str) -> list[dict[str, Any]]:
        """Execute ``query`` against the SPARQL endpoint with retry/backoff."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._http_get(
                    SPARQL_ENDPOINT,
                    params={"query": query, "format": "json"},
                    headers={"Accept": "application/sparql-results+json"},
                )
                if resp.status_code in (429, 503):
                    delay = 2**attempt
                    logger.warning(
                        "SPARQL endpoint returned %s; backing off %.1fs (attempt %d/%d)",
                        resp.status_code,
                        delay,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                bindings = resp.json()["results"]["bindings"]
                return [{k: v.get("value") for k, v in row.items()} for row in bindings]
            except (requests.RequestException, ValueError, KeyError) as exc:
                last_exc = exc
                delay = 2**attempt
                logger.warning(
                    "SPARQL request failed (%s); retrying in %.1fs (attempt %d/%d)",
                    exc,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"SPARQL request failed after {self.max_retries} retries"
        ) from last_exc

    # -- query construction ---------------------------------------------

    @staticmethod
    def build_sitelink_query(
        l1: str,
        l2: str,
        type_qid: str,
        limit: int = 1000,
        offset: int = 0,
    ) -> str:
        """Construct a per-type sitelink-pair SPARQL query.

        Matches §3.1 + §3.2 of the design doc: returns up to ``limit`` rows of
        ``(entity, l1Title, l2Title, type, typeLabel)`` for entities of
        ``wdt:P31 type_qid`` that have sitelinks in both ``l1`` and ``l2``.
        """
        return f"""SELECT ?entity ?l1Title ?l2Title ?type ?typeLabel WHERE {{
  ?l1Article schema:about ?entity ;
             schema:isPartOf <https://{l1}.wikipedia.org/> ;
             schema:name ?l1Title .
  ?l2Article schema:about ?entity ;
             schema:isPartOf <https://{l2}.wikipedia.org/> ;
             schema:name ?l2Title .
  ?entity wdt:P31 wd:{type_qid} .
  BIND(wd:{type_qid} AS ?type)
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en" .
    ?type rdfs:label ?typeLabel .
  }}
}}
LIMIT {limit}
OFFSET {offset}""".strip()

    # -- public entrypoints ---------------------------------------------

    def extract_sitelink_pairs(
        self,
        l1: str,
        l2: str,
        n_per_type: int = 1000,
        cache_root: Path | str = "data/wikidata_raw",
        type_buckets: Iterable[tuple[str, str]] = TYPE_BUCKETS,
    ) -> Path:
        """Extract type-stratified cross-lingual sitelink pairs for one language pair.

        For each (qid, bucket) in ``type_buckets``, queries the SPARQL endpoint
        for up to ``n_per_type`` entities that have sitelinks in both ``l1`` and
        ``l2``. Writes results as JSONL to
        ``{cache_root}/{l1}_{l2}/sitelinks.jsonl``.

        Each output row::

            {"qid": str, "l1_title": str, "l2_title": str,
             "wikidata_type": str, "type_label": str, "bucket": str}

        Returns the path to the JSONL file. Idempotent at the file level —
        re-running overwrites.
        """
        out_dir = Path(cache_root) / f"{l1}_{l2}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "sitelinks.jsonl"

        total = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for type_qid, bucket in type_buckets:
                logger.info("Querying %s↔%s bucket %s (%s)", l1, l2, bucket, type_qid)
                for row in self._paged_query(
                    lambda offset, qid=type_qid: self.build_sitelink_query(
                        l1, l2, qid, limit=min(n_per_type, 1000), offset=offset
                    ),
                    target=n_per_type,
                ):
                    record = {
                        "qid": _qid_from_uri(row.get("entity", "")),
                        "l1_title": row.get("l1Title", ""),
                        "l2_title": row.get("l2Title", ""),
                        "wikidata_type": _qid_from_uri(row.get("type", "")),
                        "type_label": row.get("typeLabel", ""),
                        "bucket": bucket,
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1

        logger.info("Wrote %d pairs to %s", total, out_path)
        return out_path

    def _paged_query(
        self,
        query_for_offset: Callable[[int], str],
        target: int,
        page_size: int = 1000,
    ) -> Iterator[dict[str, Any]]:
        """Yield up to ``target`` rows by paging the SPARQL endpoint.

        Stops when a page returns fewer rows than ``page_size`` (no more data)
        or when ``target`` rows have been emitted.
        """
        emitted = 0
        offset = 0
        while emitted < target:
            page = self._sparql(query_for_offset(offset))
            if not page:
                return
            for row in page:
                yield row
                emitted += 1
                if emitted >= target:
                    return
            if len(page) < page_size:
                return
            offset += page_size

    # -- REST label fetcher ---------------------------------------------

    def fetch_labels(
        self,
        qids: list[str],
        languages: Optional[list[str]] = None,
    ) -> dict[str, dict[str, str]]:
        """Fetch multilingual labels for ``qids`` via the Wikidata REST API.

        Returns ``{qid: {lang: label}}``.
        """
        if not qids:
            return {}
        langs = languages or ["en", "ru", "zh", "ar", "de", "fr"]
        resp = self._http_get(
            WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "ids": "|".join(qids),
                "props": "labels",
                "languages": "|".join(langs),
                "format": "json",
            },
        )
        resp.raise_for_status()
        entities = resp.json().get("entities", {})
        result: dict[str, dict[str, str]] = {}
        for qid, entity in entities.items():
            result[qid] = {
                lang: info["value"]
                for lang, info in entity.get("labels", {}).items()
            }
        return result

    def fetch_by_sparql(self, sparql_query: str) -> list[dict[str, Any]]:
        """Execute an arbitrary SPARQL SELECT query and return row dicts."""
        return self._sparql(sparql_query)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qid_from_uri(uri: str) -> str:
    """Extract a Wikidata QID from a full-URI binding."""
    if not uri:
        return ""
    return uri.rsplit("/", 1)[-1]
