"""Tests for the Wikidata SPARQL extraction module (P7 Stage 1).

The HTTP layer is mocked via monkey-patching ``_http_get`` so these tests
do not require network access. Tests cover query construction, response
parsing, rate-limiting, retry/backoff, paging, and JSONL output.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import pytest

from polyglot_er.datasets.wikidata import (
    SPARQL_ENDPOINT,
    TYPE_BUCKETS,
    WIKIDATA_API,
    WikidataLoader,
    _qid_from_uri,
    _TokenBucket,
)


class _FakeResponse:
    """Minimal stand-in for ``requests.Response`` used in monkey-patched tests."""

    def __init__(self, status_code: int = 200, payload: Optional[dict[str, Any]] = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")


def _sparql_bindings(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Wrap row dicts in the SPARQL JSON results envelope."""
    return {
        "results": {
            "bindings": [
                {k: {"type": "literal", "value": v} for k, v in row.items()}
                for row in rows
            ]
        }
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestQidFromUri:
    def test_extracts_qid_from_full_uri(self):
        assert _qid_from_uri("http://www.wikidata.org/entity/Q7251") == "Q7251"

    def test_returns_empty_for_empty_input(self):
        assert _qid_from_uri("") == ""

    def test_handles_uri_with_no_slashes(self):
        assert _qid_from_uri("Q5") == "Q5"


class TestTypeBuckets:
    def test_has_ten_buckets(self):
        # Design doc §3.2 specifies 10 type buckets.
        assert len(TYPE_BUCKETS) == 10

    def test_buckets_are_unique(self):
        qids = [qid for qid, _ in TYPE_BUCKETS]
        labels = [label for _, label in TYPE_BUCKETS]
        assert len(set(qids)) == len(qids)
        assert len(set(labels)) == len(labels)

    def test_buckets_use_canonical_qids(self):
        qid_map = dict(TYPE_BUCKETS)
        # Spot-check a few canonical QIDs from the design doc.
        assert qid_map["Q5"] == "person"
        assert qid_map["Q43229"] == "organization"
        assert qid_map["Q6256"] == "country"


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------


class TestTokenBucket:
    def test_starts_full(self):
        b = _TokenBucket(rate=5.0, capacity=5.0)
        # Five immediate acquires should succeed without blocking.
        for _ in range(5):
            b.acquire()

    def test_refills_over_time(self):
        b = _TokenBucket(rate=5.0, capacity=5.0)
        # Drain.
        for _ in range(5):
            b.acquire()
        # Simulate time passing via injected clock.
        fake_time = [time.monotonic() + 1.0]  # 1 second later → 5 new tokens
        b.acquire(clock=lambda: fake_time[0])

    def test_does_not_exceed_capacity(self):
        b = _TokenBucket(rate=5.0, capacity=5.0)
        # 10 seconds → would refill 50 tokens at rate 5, but capacity caps at 5.
        fake_time = [time.monotonic() + 10.0]
        b.acquire(clock=lambda: fake_time[0])
        # After one acquire, ≤ 4 tokens remain.
        assert b.tokens <= 4.0


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------


class TestBuildSitelinkQuery:
    def test_includes_both_language_codes(self):
        q = WikidataLoader.build_sitelink_query("en", "ru", "Q5")
        assert "https://en.wikipedia.org/" in q
        assert "https://ru.wikipedia.org/" in q

    def test_uses_provided_type_qid(self):
        q = WikidataLoader.build_sitelink_query("en", "de", "Q43229")
        assert "wd:Q43229" in q

    def test_includes_limit_and_offset(self):
        q = WikidataLoader.build_sitelink_query("en", "ru", "Q5", limit=500, offset=1000)
        assert "LIMIT 500" in q
        assert "OFFSET 1000" in q

    def test_requests_type_label(self):
        q = WikidataLoader.build_sitelink_query("en", "fr", "Q515")
        assert "wikibase:label" in q
        assert "?typeLabel" in q


# ---------------------------------------------------------------------------
# Loader: response parsing + retry
# ---------------------------------------------------------------------------


class TestSparqlParsing:
    def test_parses_bindings_to_row_dicts(self, monkeypatch):
        loader = WikidataLoader()
        rows = [
            {
                "entity": "http://www.wikidata.org/entity/Q7251",
                "l1Title": "Alan Turing",
                "l2Title": "Алан Тьюринг",
                "type": "http://www.wikidata.org/entity/Q5",
                "typeLabel": "human",
            }
        ]
        monkeypatch.setattr(
            loader,
            "_http_get",
            lambda url, params, headers=None: _FakeResponse(200, _sparql_bindings(rows)),
        )
        result = loader._sparql("SELECT * WHERE {}")
        assert len(result) == 1
        assert result[0]["entity"] == "http://www.wikidata.org/entity/Q7251"
        assert result[0]["l1Title"] == "Alan Turing"

    def test_retries_on_429_then_succeeds(self, monkeypatch):
        loader = WikidataLoader(max_retries=3)
        # Speed up backoff so tests stay quick.
        monkeypatch.setattr("polyglot_er.datasets.wikidata.time.sleep", lambda _: None)

        responses = [
            _FakeResponse(429, {}),
            _FakeResponse(200, _sparql_bindings([{"entity": "x"}])),
        ]
        calls = {"n": 0}

        def fake_get(url, params, headers=None):
            r = responses[calls["n"]]
            calls["n"] += 1
            return r

        monkeypatch.setattr(loader, "_http_get", fake_get)
        result = loader._sparql("SELECT * WHERE {}")
        assert calls["n"] == 2
        assert result == [{"entity": "x"}]

    def test_raises_after_exhausting_retries(self, monkeypatch):
        loader = WikidataLoader(max_retries=2)
        monkeypatch.setattr("polyglot_er.datasets.wikidata.time.sleep", lambda _: None)
        monkeypatch.setattr(
            loader,
            "_http_get",
            lambda url, params, headers=None: _FakeResponse(503, {}),
        )
        with pytest.raises(RuntimeError, match="failed after 2 retries"):
            loader._sparql("SELECT * WHERE {}")


# ---------------------------------------------------------------------------
# Paging
# ---------------------------------------------------------------------------


class TestPagedQuery:
    def test_stops_when_page_smaller_than_page_size(self, monkeypatch):
        loader = WikidataLoader()
        pages = [
            [{"entity": f"http://www.wikidata.org/entity/Q{i}"} for i in range(1000)],
            [{"entity": "http://www.wikidata.org/entity/Q9999"}],  # partial → terminate
        ]
        call_index = {"n": 0}

        def fake_sparql(query):
            page = pages[call_index["n"]]
            call_index["n"] += 1
            return page

        monkeypatch.setattr(loader, "_sparql", fake_sparql)
        rows = list(loader._paged_query(lambda offset: "Q", target=5000, page_size=1000))
        assert len(rows) == 1001
        assert call_index["n"] == 2

    def test_stops_at_target(self, monkeypatch):
        loader = WikidataLoader()
        monkeypatch.setattr(
            loader, "_sparql", lambda q: [{"entity": "x"} for _ in range(1000)]
        )
        rows = list(loader._paged_query(lambda offset: "Q", target=42, page_size=1000))
        assert len(rows) == 42


# ---------------------------------------------------------------------------
# Extract end-to-end
# ---------------------------------------------------------------------------


class TestExtractSitelinkPairs:
    def test_writes_jsonl_with_expected_schema(self, monkeypatch, tmp_path):
        loader = WikidataLoader()

        def fake_sparql(query):
            # Return one row per bucket call — each call uses a different
            # type_qid, recoverable from the query text.
            for qid, _ in TYPE_BUCKETS:
                if f"wd:{qid}" in query:
                    return [
                        {
                            "entity": f"http://www.wikidata.org/entity/Q_{qid}_1",
                            "l1Title": f"L1 title for {qid}",
                            "l2Title": f"L2 title for {qid}",
                            "type": f"http://www.wikidata.org/entity/{qid}",
                            "typeLabel": f"label_{qid}",
                        }
                    ]
            return []

        monkeypatch.setattr(loader, "_sparql", fake_sparql)
        out_path = loader.extract_sitelink_pairs(
            "en", "ru", n_per_type=2, cache_root=tmp_path
        )

        assert out_path == tmp_path / "en_ru" / "sitelinks.jsonl"
        rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 10  # one per bucket
        for row in rows:
            assert set(row.keys()) == {
                "qid",
                "l1_title",
                "l2_title",
                "wikidata_type",
                "type_label",
                "bucket",
            }
            assert row["qid"].startswith("Q_")
            assert row["wikidata_type"] in {qid for qid, _ in TYPE_BUCKETS}

    def test_preserves_bucket_order(self, monkeypatch, tmp_path):
        loader = WikidataLoader()

        def fake_sparql(query):
            for qid, _ in TYPE_BUCKETS:
                if f"wd:{qid}" in query:
                    return [{"entity": f"http://www.wikidata.org/entity/E_{qid}"}]
            return []

        monkeypatch.setattr(loader, "_sparql", fake_sparql)
        out_path = loader.extract_sitelink_pairs(
            "en", "de", n_per_type=1, cache_root=tmp_path
        )
        rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
        ordered_buckets = [b for _, b in TYPE_BUCKETS]
        assert [r["bucket"] for r in rows] == ordered_buckets


# ---------------------------------------------------------------------------
# REST labels
# ---------------------------------------------------------------------------


class TestFetchLabels:
    def test_returns_empty_for_no_qids(self):
        loader = WikidataLoader()
        assert loader.fetch_labels([]) == {}

    def test_parses_label_payload(self, monkeypatch):
        loader = WikidataLoader()
        payload = {
            "entities": {
                "Q7251": {
                    "labels": {
                        "en": {"language": "en", "value": "Alan Turing"},
                        "ru": {"language": "ru", "value": "Алан Тьюринг"},
                    }
                }
            }
        }
        monkeypatch.setattr(
            loader,
            "_http_get",
            lambda url, params, headers=None: _FakeResponse(200, payload),
        )
        result = loader.fetch_labels(["Q7251"])
        assert result == {
            "Q7251": {"en": "Alan Turing", "ru": "Алан Тьюринг"},
        }


# ---------------------------------------------------------------------------
# Endpoint constants
# ---------------------------------------------------------------------------


class TestEndpointConstants:
    def test_sparql_endpoint_is_https(self):
        assert SPARQL_ENDPOINT.startswith("https://")
        assert "query.wikidata.org" in SPARQL_ENDPOINT

    def test_rest_endpoint_is_https(self):
        assert WIKIDATA_API.startswith("https://")
        assert "wikidata.org" in WIKIDATA_API
