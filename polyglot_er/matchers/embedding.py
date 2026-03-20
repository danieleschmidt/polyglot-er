"""
Multilingual embedding similarity matcher.

Primary backend: sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
Fallback backend: TF-IDF with character n-grams (no external dependencies)

The fallback is activated automatically when sentence-transformers is not
installed or when ``force_tfidf=True`` is passed to the constructor.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .base import CrossLingualMatcher, MatchResult


# ---------------------------------------------------------------------------
# TF-IDF fallback (character n-gram cosine similarity)
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int = 3) -> Dict[str, int]:
    """Extract character n-gram frequency dict from text."""
    text = f"#{text}#"  # boundary markers
    ngrams: Dict[str, int] = {}
    for i in range(len(text) - n + 1):
        gram = text[i : i + n]
        ngrams[gram] = ngrams.get(gram, 0) + 1
    return ngrams


def _cosine_tfidf(ngrams_a: Dict[str, int], ngrams_b: Dict[str, int]) -> float:
    """Cosine similarity between two n-gram frequency dicts."""
    if not ngrams_a or not ngrams_b:
        return 0.0
    dot = sum(ngrams_a.get(k, 0) * v for k, v in ngrams_b.items())
    mag_a = math.sqrt(sum(v * v for v in ngrams_a.values()))
    mag_b = math.sqrt(sum(v * v for v in ngrams_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class _TFIDFFallback:
    """Character n-gram TF-IDF backend (no external dependencies)."""

    def __init__(self, ngram_size: int = 3):
        self.ngram_size = ngram_size

    def encode(self, texts: List[str]):
        """Return n-gram dicts as 'embeddings' (list of dicts)."""
        return [_char_ngrams(t.lower(), self.ngram_size) for t in texts]

    def similarity(self, a, b) -> float:  # type: ignore[override]
        return _cosine_tfidf(a, b)


# ---------------------------------------------------------------------------
# sentence-transformers backend
# ---------------------------------------------------------------------------

def _load_sentence_transformer(model_name: str):
    """Try to load sentence-transformers model; return None on failure."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer(model_name)
        return model
    except Exception:
        return None


def _cosine_np(a, b) -> float:
    """Cosine similarity between two numpy-compatible vectors."""
    try:
        import numpy as np  # type: ignore
        a, b = np.array(a), np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0
    except ImportError:
        # Pure-Python dot product fallback
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))
        return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


class _SentenceTransformerBackend:
    """sentence-transformers backend wrapper."""

    def __init__(self, model):
        self._model = model

    def encode(self, texts: List[str]):
        return self._model.encode(texts, convert_to_numpy=True)

    def similarity(self, a, b) -> float:
        return _cosine_np(a, b)


# ---------------------------------------------------------------------------
# EmbeddingMatcher
# ---------------------------------------------------------------------------

#: Default multilingual model
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingMatcher(CrossLingualMatcher):
    """
    Multilingual embedding cosine similarity matcher.

    Automatically selects the best available backend:
      1. sentence-transformers (``paraphrase-multilingual-MiniLM-L12-v2``)
      2. TF-IDF character n-gram cosine (pure Python fallback)

    Args:
        model_name: sentence-transformers model name
        threshold: Minimum cosine similarity for a positive match
        force_tfidf: Force the TF-IDF fallback (useful for testing / CI)

    Example::

        matcher = EmbeddingMatcher()
        result = matcher.match("Vladimir Putin", "Владимир Путин", lang_a="en", lang_b="ru")
        print(result)  # MatchResult(match=True, score=..., tier=4, method='EmbeddingMatcher')
    """

    DEFAULT_THRESHOLD = 0.75

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: Optional[float] = None,
        force_tfidf: bool = False,
    ):
        super().__init__(threshold)
        self._model_name = model_name
        self._force_tfidf = force_tfidf
        self._backend = None  # lazy init

    def _get_backend(self):
        if self._backend is not None:
            return self._backend

        if not self._force_tfidf:
            model = _load_sentence_transformer(self._model_name)
            if model is not None:
                self._backend = _SentenceTransformerBackend(model)
                return self._backend

        self._backend = _TFIDFFallback()
        return self._backend

    @property
    def backend_name(self) -> str:
        """Name of the active backend (for debugging)."""
        backend = self._get_backend()
        return type(backend).__name__

    def score(self, name_a: str, name_b: str, lang_a: str = "", lang_b: str = "") -> float:
        """
        Compute multilingual embedding cosine similarity.

        Args:
            name_a: First entity name
            name_b: Second entity name
            lang_a: Language hint (passed to some backends)
            lang_b: Language hint (passed to some backends)

        Returns:
            Cosine similarity in [0, 1]
        """
        backend = self._get_backend()
        embeddings = backend.encode([name_a, name_b])
        return backend.similarity(embeddings[0], embeddings[1])

    def match(self, name_a, name_b, lang_a="", lang_b="", tier=None) -> MatchResult:
        s = self.score(name_a, name_b, lang_a, lang_b)
        return MatchResult(
            is_match=s >= self.threshold,
            score=s,
            tier=tier if tier is not None else 4,
            method="EmbeddingMatcher",
            details={"backend": self.backend_name},
        )
