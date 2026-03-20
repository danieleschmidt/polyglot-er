# polyglot-er 🌍

**Cross-Lingual Entity Resolution Pipeline**

Resolve the same real-world entity appearing with different name forms across multiple languages using a 5-tier fusion cascade extended with multilingual embeddings.

---

## The Problem

The same entity can appear in radically different surface forms depending on language and script:

| Entity | EN | DE | RU | ZH | AR |
|--------|----|----|----|----|-----|
| Vladimir Putin | Vladimir Putin | Wladimir Putin | Владимир Путин | 弗拉基米尔·普京 | فلاديمير بوتين |
| Angela Merkel | Angela Merkel | Angela Merkel | Ангела Меркель | 安格拉·默克尔 | أنغيلا ميركل |
| Napoleon Bonaparte | Napoleon Bonaparte | Napoleon Bonaparte | Наполеон Бонапарт | 拿破仑·波拿巴 | نابليون بونابرت |

Naive string matching fails. Even romanization-aware fuzzy matching struggles with Arabic and CJK scripts. **polyglot-er** solves this with a staged cascade that applies the cheapest matching tier first and escalates to multilingual embeddings only when needed.

---

## The 5-Tier Cascade

polyglot-er extends the [DocGraph entity fusion cascade](https://github.com/danieleschmidt/docgraph) with multilingual/cross-script support:

| Tier | Method | Condition | Threshold |
|------|--------|-----------|-----------|
| **0** | Unicode normalization + exact match | Always | exact |
| **1** | Entity type check | Skip cross-type pairs | strict |
| **2** | Same-script fuzzy (Jaro-Winkler) | Same script family | ≥ 0.85 |
| **3** | Cross-script phonetic (transliteration + Jaro-Winkler) | Different scripts | ≥ 0.82 |
| **4** | Multilingual embedding cosine | No earlier decision | ≥ 0.75 |

Tiers 0–3 are **O(1) per pair** and require no model loading. Tier 4 (embeddings) is only invoked when the cheaper tiers are inconclusive.

---

## Supported Script Families

- **Latin** — EN, DE, FR, ES, PT, and most European languages
- **Cyrillic** — RU, UK, BG, SR, and other Slavic languages  
- **CJK** — ZH (Mandarin Chinese), JA (Japanese Kanji), KO (Korean Hanja)
- **Arabic** — AR, FA (Farsi), UR (Urdu)
- **Devanagari** — HI, SA, and related South Asian scripts
- **Greek** — Modern and Classical Greek
- Other scripts pass through to the embedding tier

---

## Quick Start

### Installation

```bash
git clone https://github.com/danieleschmidt/polyglot-er
cd polyglot-er
pip install -e .
```

### Resolve a mixed-language entity list

```bash
# Using the bundled test data
python cli.py resolve \
  --input data/multilingual_entities.jsonl \
  --output /tmp/clusters.json \
  --verbose
```

Output `clusters.json`:
```json
{
  "clusters": [
    ["Q1_en_0", "Q1_de_1", "Q1_ru_2"],
    ["Q2_en_3", "Q2_zh_4", "Q2_ru_5"],
    ...
  ],
  "count": 10
}
```

### Python API

```python
from polyglot_er import CrossLingualResolver

entities = [
    {"id": "e1", "name": "Vladimir Putin", "lang": "en", "entity_type": "PER"},
    {"id": "e2", "name": "Владимир Путин", "lang": "ru", "entity_type": "PER"},
    {"id": "e3", "name": "弗拉基米尔·普京", "lang": "zh", "entity_type": "PER"},
    {"id": "e4", "name": "Angela Merkel", "lang": "en", "entity_type": "PER"},
]

resolver = CrossLingualResolver()
clusters = resolver.resolve(entities)
# [["e1", "e2", "e3"], ["e4"]]
```

### Per-language-pair evaluation

```python
from polyglot_er.evaluation.language_report import LanguageReport

records = [{"id": str(i), "lang": r["lang"]} for i, r in enumerate(entities)]
report = LanguageReport(records, predicted_clusters, gold_clusters)
print(report.summary())
```

```
Cross-Lingual Entity Resolution — Language Pair Report
=======================================================
  en-ru        P=0.900  R=0.875  F1=0.887
  en-zh        P=0.850  R=0.800  F1=0.824
  en-ar        P=0.820  R=0.780  F1=0.800
  overall      P=0.860  R=0.820  F1=0.840
```

---

## Using Real sentence-transformers Embeddings

For production-quality cross-lingual matching, install sentence-transformers:

```bash
pip install sentence-transformers
```

polyglot-er will automatically use `paraphrase-multilingual-MiniLM-L12-v2` (supports 50+ languages). No code changes required — the `EmbeddingMatcher` detects the library and falls back to TF-IDF character n-grams when it's absent.

To use a different model:

```python
from polyglot_er.matchers.embedding import EmbeddingMatcher

matcher = EmbeddingMatcher(model_name="paraphrase-multilingual-mpnet-base-v2")
result = matcher.match("Vladimir Putin", "弗拉基米尔·普京", lang_a="en", lang_b="zh")
print(result.score)  # ~0.87
```

---

## Extending to New Languages and Scripts

### Add a new script family

Edit `polyglot_er/normalization/script_detect.py`:

```python
class ScriptFamily(str, Enum):
    ...
    GEORGIAN = "Georgian"   # ka

def _char_script(char: str) -> ScriptFamily:
    ...
    if "GEORGIAN" in name_upper:
        return ScriptFamily.GEORGIAN
```

### Add transliteration for a new script

Edit `polyglot_er/normalization/transliterate.py`:

```python
_GEORGIAN_TO_LATIN = {"ა": "a", "ბ": "b", ...}

def transliterate_georgian(text: str) -> str:
    return "".join(_GEORGIAN_TO_LATIN.get(ch, ch) for ch in text)
```

Update `transliterate_to_latin` to dispatch on `ScriptFamily.GEORGIAN`.

### Load real multilingual data from Wikidata

See `polyglot_er/datasets/wikidata.py` for the documented stub with full
implementation instructions using the Wikidata REST API and SPARQL endpoint.

---

## Project Structure

```
polyglot_er/
├── __init__.py
├── resolver.py              — CrossLingualResolver (main entrypoint)
├── normalization/
│   ├── unicode_norm.py      — NFC/NFD, diacritic stripping
│   ├── script_detect.py     — Script family detection
│   └── transliterate.py     — Cyrillic/CJK/Arabic → Latin
├── matchers/
│   ├── base.py              — Abstract CrossLingualMatcher + MatchResult
│   ├── phonetic.py          — Cross-script Jaro-Winkler after transliteration
│   ├── embedding.py         — Multilingual cosine similarity (ST or TF-IDF)
│   └── cascade.py           — 5-tier cascade orchestrator
├── datasets/
│   ├── synthetic.py         — 30-record EN/RU/ZH/AR/DE test corpus
│   └── wikidata.py          — WikidataLoader stub
└── evaluation/
    ├── metrics.py           — Pairwise + cluster P/R/F1
    └── language_report.py   — Per-language-pair breakdown
cli.py                       — `polyglot-er resolve` CLI
data/
└── multilingual_entities.jsonl  — 30 synthetic entities, 10 real-world persons
tests/                       — 50 pytest tests
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## License

MIT
