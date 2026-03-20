"""
Phonetic similarity matcher for cross-lingual entity resolution.

Strategy:
  1. Transliterate both names to Latin script
  2. Apply Unicode normalization + lowercase
  3. Compute Jaro-Winkler similarity on the transliterated forms

This covers cross-script pairs such as "Putin" (EN) vs "Путин" (RU).
"""

from .base import CrossLingualMatcher, MatchResult
from ..normalization.transliterate import transliterate_to_latin
from ..normalization.unicode_norm import normalize_for_matching

try:
    from jellyfish import jaro_winkler_similarity as _jaro_winkler
except ImportError:
    # Pure-Python fallback implementation
    def _jaro_winkler(s1: str, s2: str) -> float:  # type: ignore[misc]
        """Simplified Jaro-Winkler (no prefix boost) as fallback."""
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        len_s1, len_s2 = len(s1), len(s2)
        match_dist = max(len_s1, len_s2) // 2 - 1
        if match_dist < 0:
            match_dist = 0
        s1_matches = [False] * len_s1
        s2_matches = [False] * len_s2
        matches = 0
        transpositions = 0
        for i, c1 in enumerate(s1):
            start = max(0, i - match_dist)
            end = min(i + match_dist + 1, len_s2)
            for j in range(start, end):
                if s2_matches[j] or c1 != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
        if matches == 0:
            return 0.0
        k = 0
        for i, matched in enumerate(s1_matches):
            if not matched:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        jaro = (
            matches / len_s1
            + matches / len_s2
            + (matches - transpositions / 2) / matches
        ) / 3
        # Winkler prefix bonus
        prefix = 0
        for i in range(min(4, len_s1, len_s2)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        return jaro + prefix * 0.1 * (1 - jaro)


class PhoneticMatcher(CrossLingualMatcher):
    """
    Cross-script phonetic similarity matcher.

    Transliterates both names to Latin, normalizes, then computes
    Jaro-Winkler similarity.

    Default threshold: 0.82 (slightly below the same-script Tier-2 threshold
    of 0.85 to allow for transliteration noise).
    """

    DEFAULT_THRESHOLD = 0.82

    def score(self, name_a: str, name_b: str, lang_a: str = "", lang_b: str = "") -> float:
        """
        Compute phonetic similarity after transliteration.

        Args:
            name_a: First entity name (any script)
            name_b: Second entity name (any script)
            lang_a: Language hint (unused; script auto-detected)
            lang_b: Language hint (unused; script auto-detected)

        Returns:
            Jaro-Winkler similarity score on transliterated forms
        """
        latin_a = normalize_for_matching(transliterate_to_latin(name_a))
        latin_b = normalize_for_matching(transliterate_to_latin(name_b))
        if not latin_a or not latin_b:
            return 0.0
        return float(_jaro_winkler(latin_a, latin_b))

    def match(self, name_a, name_b, lang_a="", lang_b="", tier=None) -> MatchResult:
        s = self.score(name_a, name_b, lang_a, lang_b)
        latin_a = normalize_for_matching(transliterate_to_latin(name_a))
        latin_b = normalize_for_matching(transliterate_to_latin(name_b))
        return MatchResult(
            is_match=s >= self.threshold,
            score=s,
            tier=tier if tier is not None else 3,
            method="PhoneticMatcher",
            details={"transliterated_a": latin_a, "transliterated_b": latin_b},
        )
