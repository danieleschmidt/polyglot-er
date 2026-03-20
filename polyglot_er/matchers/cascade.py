"""
Cross-Lingual Entity Resolution Cascade

5-tier cascade mirroring the DocGraph entity fusion pipeline, extended with
multilingual/cross-script support:

  Tier 0 — Unicode normalization + lowercase
  Tier 1 — Entity type check (skip cross-type pairs)
  Tier 2 — Same-script fuzzy (Jaro-Winkler ≥ 0.85)
  Tier 3 — Cross-script phonetic (transliteration + Jaro-Winkler ≥ 0.82)
  Tier 4 — Multilingual embedding cosine ≥ 0.75

Each tier can emit a definitive answer (match / no-match) or abstain to let
the next tier decide. A pair that survives to Tier 4 without a definitive
early answer is resolved by the embedding matcher.
"""

from typing import Optional

from .base import CrossLingualMatcher, MatchResult
from .phonetic import PhoneticMatcher
from .embedding import EmbeddingMatcher
from ..normalization.unicode_norm import normalize_for_matching
from ..normalization.script_detect import detect_script, is_same_script

try:
    from jellyfish import jaro_winkler_similarity as _jaro_winkler
except ImportError:
    from .phonetic import _jaro_winkler  # type: ignore[attr-defined]


#: Jaro-Winkler threshold for same-script fuzzy matching (Tier 2)
TIER2_THRESHOLD = 0.85
#: Jaro-Winkler threshold for cross-script phonetic (Tier 3)
TIER3_THRESHOLD = 0.82
#: Cosine threshold for multilingual embeddings (Tier 4)
TIER4_THRESHOLD = 0.75


class CascadeMatcher(CrossLingualMatcher):
    """
    5-tier cross-lingual entity resolution cascade.

    Args:
        tier2_threshold: Jaro-Winkler threshold for same-script fuzzy (default 0.85)
        tier3_threshold: Phonetic threshold for cross-script (default 0.82)
        tier4_threshold: Embedding cosine threshold (default 0.75)
        force_tfidf: Force TF-IDF backend in EmbeddingMatcher (useful for testing)

    Example::

        cascade = CascadeMatcher()
        result = cascade.match(
            "Vladimir Putin", "Владимир Путин",
            entity_type_a="PER", entity_type_b="PER",
            lang_a="en", lang_b="ru"
        )
        print(result)
    """

    DEFAULT_THRESHOLD = TIER4_THRESHOLD

    def __init__(
        self,
        tier2_threshold: float = TIER2_THRESHOLD,
        tier3_threshold: float = TIER3_THRESHOLD,
        tier4_threshold: float = TIER4_THRESHOLD,
        force_tfidf: bool = False,
    ):
        super().__init__(threshold=tier4_threshold)
        self.tier2_threshold = tier2_threshold
        self.tier3_threshold = tier3_threshold
        self._phonetic = PhoneticMatcher(threshold=tier3_threshold)
        self._embedding = EmbeddingMatcher(threshold=tier4_threshold, force_tfidf=force_tfidf)

    def score(self, name_a: str, name_b: str, lang_a: str = "", lang_b: str = "") -> float:
        """Return highest score reached across all tiers."""
        result = self.match(name_a, name_b, lang_a=lang_a, lang_b=lang_b)
        return result.score

    def match(  # type: ignore[override]
        self,
        name_a: str,
        name_b: str,
        lang_a: str = "",
        lang_b: str = "",
        entity_type_a: str = "",
        entity_type_b: str = "",
        tier: Optional[int] = None,
    ) -> MatchResult:
        """
        Run the 5-tier cascade match.

        Args:
            name_a: First entity name
            name_b: Second entity name
            lang_a: BCP-47 language code for name_a
            lang_b: BCP-47 language code for name_b
            entity_type_a: NER type for name_a (e.g. "PER", "ORG")
            entity_type_b: NER type for name_b
            tier: Unused (cascade internally assigns tiers)

        Returns:
            MatchResult with the tier that made the final decision
        """
        # ------------------------------------------------------------------
        # Tier 0: Unicode normalization
        # ------------------------------------------------------------------
        norm_a = normalize_for_matching(name_a)
        norm_b = normalize_for_matching(name_b)

        if not norm_a or not norm_b:
            return MatchResult(is_match=False, score=0.0, tier=0, method="Cascade/empty")

        # ------------------------------------------------------------------
        # Tier 1: Entity type check (before exact match — cross-type is always rejected)
        # ------------------------------------------------------------------
        if entity_type_a and entity_type_b and entity_type_a != entity_type_b:
            return MatchResult(
                is_match=False,
                score=0.0,
                tier=1,
                method="Cascade/type-mismatch",
                details={"type_a": entity_type_a, "type_b": entity_type_b},
            )

        # Exact match after normalization → definitive match (after type check)
        if norm_a == norm_b:
            return MatchResult(is_match=True, score=1.0, tier=0, method="Cascade/exact")

        # ------------------------------------------------------------------
        # Tier 2: Same-script fuzzy (Jaro-Winkler)
        # ------------------------------------------------------------------
        same_script = is_same_script(name_a, name_b)
        if same_script:
            jw = float(_jaro_winkler(norm_a, norm_b))
            if jw >= self.tier2_threshold:
                return MatchResult(
                    is_match=True,
                    score=jw,
                    tier=2,
                    method="Cascade/same-script-fuzzy",
                    details={"jaro_winkler": jw},
                )
            # If same-script but below threshold, still check embedding
            # (don't hard-reject — two same-script names can be very different)

        # ------------------------------------------------------------------
        # Tier 3: Cross-script phonetic
        # ------------------------------------------------------------------
        phonetic_result = self._phonetic.match(name_a, name_b, lang_a, lang_b, tier=3)
        if phonetic_result.is_match:
            return phonetic_result

        # ------------------------------------------------------------------
        # Tier 4: Multilingual embedding
        # ------------------------------------------------------------------
        embedding_result = self._embedding.match(name_a, name_b, lang_a, lang_b, tier=4)
        return embedding_result

    @property
    def embedding_backend(self) -> str:
        """Name of the active embedding backend."""
        return self._embedding.backend_name
