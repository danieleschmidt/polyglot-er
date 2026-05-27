"""Tests for matcher classes: phonetic, embedding (TF-IDF fallback), and cascade."""

import pytest

from polyglot_er.matchers.base import MatchResult, CrossLingualMatcher
from polyglot_er.matchers.phonetic import PhoneticMatcher
from polyglot_er.matchers.embedding import EmbeddingMatcher
from polyglot_er.matchers.cascade import CascadeMatcher


# ---------------------------------------------------------------------------
# MatchResult / base
# ---------------------------------------------------------------------------

def test_match_result_repr():
    r = MatchResult(is_match=True, score=0.9, tier=2, method="test")
    assert "True" in repr(r)
    assert "0.900" in repr(r)


def test_match_result_fields():
    r = MatchResult(is_match=False, score=0.5, tier=3, method="PhoneticMatcher", details={"k": "v"})
    assert r.is_match is False
    assert r.score == 0.5
    assert r.tier == 3
    assert r.details["k"] == "v"


# ---------------------------------------------------------------------------
# PhoneticMatcher
# ---------------------------------------------------------------------------

def test_same_script_fuzzy_match():
    """'Putin' vs 'Puttin' → should match via phonetic."""
    matcher = PhoneticMatcher(threshold=0.80)
    result = matcher.match("Putin", "Puttin")
    assert result.is_match, f"Expected match; score={result.score:.3f}"


def test_phonetic_exact_match():
    matcher = PhoneticMatcher()
    result = matcher.match("Putin", "Putin")
    assert result.is_match
    assert result.score == pytest.approx(1.0, abs=0.001)


def test_cross_script_phonetic_putin():
    """'Putin' vs transliterated 'Путин' should match."""
    matcher = PhoneticMatcher(threshold=0.75)
    result = matcher.match("Putin", "Путин")
    assert result.is_match, f"Cross-script phonetic failed; score={result.score:.3f}"


def test_cross_script_phonetic_clearly_different():
    """Completely unrelated names should not match."""
    matcher = PhoneticMatcher()
    result = matcher.match("Putin", "Меркель")
    assert result.score < 0.9


def test_phonetic_match_result_has_transliteration():
    matcher = PhoneticMatcher()
    result = matcher.match("Путин", "Putin")
    assert "transliterated_a" in result.details
    assert "transliterated_b" in result.details


# ---------------------------------------------------------------------------
# EmbeddingMatcher (forced TF-IDF for tests — no network required)
# ---------------------------------------------------------------------------

def test_embedding_fallback_tfidf():
    """TF-IDF fallback works without sentence-transformers installed."""
    matcher = EmbeddingMatcher(force_tfidf=True)
    assert matcher.backend_name == "_TFIDFFallback"
    score = matcher.score("Putin", "Putin")
    assert score == pytest.approx(1.0, abs=0.01)


def test_embedding_tfidf_similar():
    matcher = EmbeddingMatcher(force_tfidf=True)
    score_same = matcher.score("Vladimir Putin", "Vladimir Putin")
    score_diff = matcher.score("Vladimir Putin", "Angela Merkel")
    assert score_same > score_diff


def test_embedding_tfidf_cross_script_match():
    """TF-IDF may not achieve 0.75 for cross-script, but should not error."""
    matcher = EmbeddingMatcher(force_tfidf=True, threshold=0.75)
    result = matcher.match("Putin", "Путин")
    # Just verify it runs and returns a valid result
    assert isinstance(result.is_match, bool)
    assert 0.0 <= result.score <= 1.0


def test_embedding_match_result_has_backend():
    matcher = EmbeddingMatcher(force_tfidf=True)
    result = matcher.match("hello", "hello")
    assert "backend" in result.details


# ---------------------------------------------------------------------------
# CascadeMatcher
# ---------------------------------------------------------------------------

def test_cascade_exact_match():
    """Exact same name → Tier 0 (exact) match."""
    cascade = CascadeMatcher(force_tfidf=True)
    result = cascade.match("Putin", "Putin")
    assert result.is_match
    assert result.tier == 0


def test_cascade_type_mismatch():
    """Mismatched entity types → Tier 1 reject."""
    cascade = CascadeMatcher(force_tfidf=True)
    result = cascade.match(
        "Putin", "Putin",
        entity_type_a="PER", entity_type_b="ORG"
    )
    assert not result.is_match
    assert result.tier == 1


def test_cascade_same_language_fuzzy():
    """EN/EN pairs with similar names should match at Tier 2 or 3."""
    cascade = CascadeMatcher(force_tfidf=True)
    result = cascade.match(
        "Vladimir Putin", "Vladimir Puting",
        lang_a="en", lang_b="en",
        entity_type_a="PER", entity_type_b="PER"
    )
    # Close names in same script should match at tier 2 or phonetic tier 3
    assert result.is_match, f"Expected match; score={result.score:.3f} tier={result.tier}"
    assert result.tier in (0, 2, 3)


def test_cascade_cross_language_reaches_embedding():
    """EN/RU pair with different scripts should reach at least Tier 3 or 4."""
    cascade = CascadeMatcher(force_tfidf=True, tier4_threshold=0.3)
    result = cascade.match(
        "Barack Obama", "Барак Обама",
        lang_a="en", lang_b="ru",
        entity_type_a="PER", entity_type_b="PER"
    )
    # Phonetic or embedding tier
    assert result.tier in (3, 4), f"Expected tier 3/4; got {result.tier}"


def test_cascade_clearly_different_names():
    """Unrelated names should not match."""
    cascade = CascadeMatcher(force_tfidf=True)
    result = cascade.match("Vladimir Putin", "Marie Curie")
    assert not result.is_match


def test_cascade_empty_name():
    cascade = CascadeMatcher(force_tfidf=True)
    result = cascade.match("", "Putin")
    assert not result.is_match


def test_cascade_script_dispatch_gate_bypasses_t3_for_same_script():
    """Same-script pairs MUST NOT reach Tier 3 — the script-dispatch gate
    routes them straight to Tier 4 to avoid the F1-negative T3+T4 overlap
    on same-script regimes (§4.3 of the paper)."""
    cascade = CascadeMatcher(force_tfidf=True, tier4_threshold=0.99, tier2_threshold=0.99)
    # Two same-script Latin strings that share no surface form. Tier 2 won't
    # fire (jw < 0.99), Tier 4 won't fire (cosine < 0.99 with force_tfidf).
    # Without the gate, Tier 3 would transliterate (passthrough for Latin) and
    # apply Jaro-Winkler on the raw strings. With the gate, Tier 3 is skipped
    # and the cascade falls through to Tier 4, which is the final decider.
    result = cascade.match(
        "Bonjour le monde", "Bonjour le mondee",
        lang_a="fr", lang_b="fr",
    )
    # The decisive tier should NOT be 3.
    assert result.tier != 3, (
        f"Same-script pair was decided at Tier 3 (gate failed); "
        f"got tier={result.tier} score={result.score:.3f}"
    )


def test_cascade_script_dispatch_gate_allows_t3_for_cross_script():
    """Cross-script pairs MUST still flow through Tier 3."""
    cascade = CascadeMatcher(force_tfidf=True, tier4_threshold=0.99)
    result = cascade.match(
        "Putin", "Путин",
        lang_a="en", lang_b="ru",
        entity_type_a="PER", entity_type_b="PER",
    )
    # Cross-script — Tier 3 should be the decisive tier.
    assert result.tier == 3, (
        f"Cross-script pair did not reach Tier 3; got tier={result.tier}"
    )
    assert result.is_match


def test_cascade_threshold_adjustment():
    """Lower threshold should produce more matches."""
    cascade_strict = CascadeMatcher(force_tfidf=True, tier4_threshold=0.99)
    cascade_loose = CascadeMatcher(force_tfidf=True, tier4_threshold=0.1)
    strict_result = cascade_strict.match("Putin", "Puttin", lang_a="en", lang_b="en")
    loose_result = cascade_loose.match("Putin", "Puttin", lang_a="en", lang_b="en")
    # Loose threshold should be at least as permissive as strict
    assert loose_result.is_match or not strict_result.is_match
