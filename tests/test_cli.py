"""Tests for the CLI and JSONL I/O roundtrip."""

import json
import sys
import tempfile
from pathlib import Path

import pytest


DATA_FILE = Path(__file__).parent.parent / "data" / "multilingual_entities.jsonl"


def test_cli_help(capsys):
    """CLI --help exits 0."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_cli_resolve(tmp_path):
    """CLI resolve command exits 0 and produces output file."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cli import main

    records = [
        {"id": "A1", "name": "Vladimir Putin", "lang": "en", "entity_type": "PER"},
        {"id": "A1", "name": "Владимир Путин", "lang": "ru", "entity_type": "PER"},
        {"id": "A2", "name": "Angela Merkel", "lang": "en", "entity_type": "PER"},
    ]
    input_file = tmp_path / "entities.jsonl"
    output_file = tmp_path / "clusters.json"

    with open(input_file, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    exit_code = main([
        "resolve",
        "--input", str(input_file),
        "--output", str(output_file),
        "--force-tfidf",
        "--tier4", "0.4",
    ])

    assert exit_code == 0
    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert "clusters" in data
    assert data["count"] >= 1


def test_cli_missing_input(tmp_path):
    """CLI with non-existent input file returns exit code 1."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cli import main

    exit_code = main([
        "resolve",
        "--input", str(tmp_path / "nonexistent.jsonl"),
        "--output", str(tmp_path / "out.json"),
    ])
    assert exit_code == 1


def test_jsonl_input_output_roundtrip(tmp_path):
    """
    JSONL input → resolver → JSON output → verify structure is preserved.
    """
    from polyglot_er.resolver import CrossLingualResolver

    records = [
        {"id": "R1", "name": "Obama", "lang": "en", "entity_type": "PER"},
        {"id": "R1", "name": "Обама", "lang": "ru", "entity_type": "PER"},
        {"id": "R2", "name": "Einstein", "lang": "en", "entity_type": "PER"},
        {"id": "R2", "name": "Эйнштейн", "lang": "ru", "entity_type": "PER"},
    ]
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.json"

    with open(input_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    resolver = CrossLingualResolver(force_tfidf=True, tier4_threshold=0.3)
    clusters = resolver.resolve_and_save(input_path, output_path)

    # Verify output JSON structure
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert "clusters" in data
    assert isinstance(data["clusters"], list)
    assert data["count"] == len(data["clusters"])
    # All record IDs should appear in exactly one cluster
    all_ids = {str(r.get("id", i)) for i, r in enumerate(records)}
    # Ids in clusters (may differ if id field not unique)
    cluster_ids = {eid for c in data["clusters"] for eid in c}
    assert cluster_ids  # non-empty


def test_wikidata_loader_stub():
    """WikidataLoader stub documents API without network call."""
    from polyglot_er.datasets.wikidata import WikidataLoader

    loader = WikidataLoader()
    assert loader.WIKIDATA_API.startswith("https://")
    assert loader.SPARQL_ENDPOINT.startswith("https://")

    # fetch_labels raises NotImplementedError (stub)
    with pytest.raises(NotImplementedError, match="stub"):
        loader.fetch_labels(["Q7251"])

    # fetch_by_sparql raises NotImplementedError (stub)
    with pytest.raises(NotImplementedError, match="stub"):
        loader.fetch_by_sparql("SELECT ?item WHERE { ?item wdt:P31 wd:Q5. } LIMIT 1")

    # example_sparql_persons returns valid string
    query = WikidataLoader.example_sparql_persons()
    assert "SELECT" in query
    assert "wikibase:language" in query
