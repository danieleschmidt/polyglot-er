"""Matcher classes: base, phonetic, embedding, and cascade."""

from .base import CrossLingualMatcher, MatchResult
from .phonetic import PhoneticMatcher
from .embedding import EmbeddingMatcher
from .cascade import CascadeMatcher

__all__ = [
    "CrossLingualMatcher",
    "MatchResult",
    "PhoneticMatcher",
    "EmbeddingMatcher",
    "CascadeMatcher",
]
