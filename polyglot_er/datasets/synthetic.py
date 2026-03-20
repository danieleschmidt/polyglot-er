"""
Synthetic cross-lingual entity pair generator.

Produces a controlled test corpus of multilingual person name records where
the ground-truth clustering is known (same Q-id → same entity).

Each entity appears in 2–3 languages, covering EN, DE, RU, ZH, and AR.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

# ---------------------------------------------------------------------------
# Ground-truth multilingual entity corpus
# ---------------------------------------------------------------------------
# Format: each outer list is one entity cluster (same real-world person).
# Fields: id, name, lang, entity_type, aliases (optional list of strings)
# ---------------------------------------------------------------------------

_ENTITIES: List[List[Dict[str, Any]]] = [
    # Q1 — Vladimir Putin
    [
        {"id": "Q1", "name": "Vladimir Putin", "lang": "en", "entity_type": "PER", "aliases": ["V. Putin"]},
        {"id": "Q1", "name": "Wladimir Putin", "lang": "de", "entity_type": "PER", "aliases": []},
        {"id": "Q1", "name": "Владимир Путин", "lang": "ru", "entity_type": "PER", "aliases": ["Путин"]},
    ],
    # Q2 — Xi Jinping
    [
        {"id": "Q2", "name": "Xi Jinping", "lang": "en", "entity_type": "PER", "aliases": []},
        {"id": "Q2", "name": "習近平", "lang": "zh", "entity_type": "PER", "aliases": []},
        {"id": "Q2", "name": "Си Цзиньпин", "lang": "ru", "entity_type": "PER", "aliases": []},
    ],
    # Q3 — Angela Merkel
    [
        {"id": "Q3", "name": "Angela Merkel", "lang": "en", "entity_type": "PER", "aliases": []},
        {"id": "Q3", "name": "Ангела Меркель", "lang": "ru", "entity_type": "PER", "aliases": []},
        {"id": "Q3", "name": "أنغيلا ميركل", "lang": "ar", "entity_type": "PER", "aliases": []},
    ],
    # Q4 — Barack Obama
    [
        {"id": "Q4", "name": "Barack Obama", "lang": "en", "entity_type": "PER", "aliases": []},
        {"id": "Q4", "name": "Барак Обама", "lang": "ru", "entity_type": "PER", "aliases": []},
        {"id": "Q4", "name": "باراك أوباما", "lang": "ar", "entity_type": "PER", "aliases": []},
    ],
    # Q5 — Emmanuel Macron
    [
        {"id": "Q5", "name": "Emmanuel Macron", "lang": "en", "entity_type": "PER", "aliases": []},
        {"id": "Q5", "name": "Эмманюэль Макрон", "lang": "ru", "entity_type": "PER", "aliases": []},
    ],
    # Q6 — Mahatma Gandhi
    [
        {"id": "Q6", "name": "Mahatma Gandhi", "lang": "en", "entity_type": "PER", "aliases": ["M. Gandhi"]},
        {"id": "Q6", "name": "Махатма Ганди", "lang": "ru", "entity_type": "PER", "aliases": []},
        {"id": "Q6", "name": "مهاتما غاندي", "lang": "ar", "entity_type": "PER", "aliases": []},
    ],
    # Q7 — Nelson Mandela
    [
        {"id": "Q7", "name": "Nelson Mandela", "lang": "en", "entity_type": "PER", "aliases": []},
        {"id": "Q7", "name": "Нельсон Мандела", "lang": "ru", "entity_type": "PER", "aliases": []},
        {"id": "Q7", "name": "نيلسون مانديلا", "lang": "ar", "entity_type": "PER", "aliases": []},
    ],
    # Q8 — Albert Einstein
    [
        {"id": "Q8", "name": "Albert Einstein", "lang": "en", "entity_type": "PER", "aliases": []},
        {"id": "Q8", "name": "Альберт Эйнштейн", "lang": "ru", "entity_type": "PER", "aliases": []},
        {"id": "Q8", "name": "阿尔伯特·爱因斯坦", "lang": "zh", "entity_type": "PER", "aliases": []},
    ],
    # Q9 — Marie Curie
    [
        {"id": "Q9", "name": "Marie Curie", "lang": "en", "entity_type": "PER", "aliases": []},
        {"id": "Q9", "name": "Мария Кюри", "lang": "ru", "entity_type": "PER", "aliases": []},
        {"id": "Q9", "name": "ماري كوري", "lang": "ar", "entity_type": "PER", "aliases": []},
    ],
    # Q10 — Napoleon Bonaparte
    [
        {"id": "Q10", "name": "Napoleon Bonaparte", "lang": "en", "entity_type": "PER", "aliases": []},
        {"id": "Q10", "name": "Наполеон Бонапарт", "lang": "ru", "entity_type": "PER", "aliases": []},
        {"id": "Q10", "name": "نابليون بونابرت", "lang": "ar", "entity_type": "PER", "aliases": []},
        {"id": "Q10", "name": "拿破仑·波拿巴", "lang": "zh", "entity_type": "PER", "aliases": []},
    ],
]


def generate_synthetic_entities() -> List[Dict[str, Any]]:
    """
    Return a flat list of all synthetic entity records.

    Returns:
        List of dicts with keys: id, name, lang, entity_type, aliases
    """
    records = []
    for cluster in _ENTITIES:
        records.extend(cluster)
    return records


def get_ground_truth_clusters() -> Dict[str, List[str]]:
    """
    Return ground-truth cluster mapping.

    Returns:
        Dict mapping entity id (e.g. "Q1") to list of (name, lang) tuples
    """
    clusters: Dict[str, List] = {}
    for cluster in _ENTITIES:
        for record in cluster:
            qid = record["id"]
            if qid not in clusters:
                clusters[qid] = []
            clusters[qid].append((record["name"], record["lang"]))
    return clusters


def load_synthetic_entities(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Load entity records from a JSONL file.

    Args:
        path: Path to .jsonl file

    Yields:
        Entity record dicts
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_synthetic_entities(path: Path) -> None:
    """
    Write the synthetic entity corpus to a JSONL file.

    Args:
        path: Destination file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = generate_synthetic_entities()
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {path}")


if __name__ == "__main__":
    import sys

    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/multilingual_entities.jsonl")
    write_synthetic_entities(dest)
