"""
Script family detection for multilingual entity resolution.

Detects the dominant script family of a text string, which informs
which matching tier to apply (same-script fuzzy vs. cross-script phonetic).
"""

import unicodedata
from enum import Enum, auto
from typing import Counter


class ScriptFamily(str, Enum):
    """Script family classification for entity name strings."""

    LATIN = "Latin"
    CYRILLIC = "Cyrillic"
    CJK = "CJK"           # Chinese ideographs + Japanese kanji + Japanese kana
    HANGUL = "Hangul"     # Korean Hangul (alphabetic syllabary)
    ARABIC = "Arabic"
    DEVANAGARI = "Devanagari"
    GREEK = "Greek"
    HEBREW = "Hebrew"
    OTHER = "Other"


def _char_script(char: str) -> ScriptFamily:
    """Map a single character to its script family."""
    try:
        name = unicodedata.name(char, "")
    except (TypeError, ValueError):
        return ScriptFamily.OTHER

    name_upper = name.upper()

    if "HANGUL" in name_upper:
        return ScriptFamily.HANGUL
    if "LATIN" in name_upper:
        return ScriptFamily.LATIN
    if "CYRILLIC" in name_upper:
        return ScriptFamily.CYRILLIC
    if any(tag in name_upper for tag in ("CJK", "KANGXI", "KATAKANA", "HIRAGANA")):
        return ScriptFamily.CJK
    if "ARABIC" in name_upper:
        return ScriptFamily.ARABIC
    if "DEVANAGARI" in name_upper:
        return ScriptFamily.DEVANAGARI
    if "GREEK" in name_upper:
        return ScriptFamily.GREEK
    if "HEBREW" in name_upper:
        return ScriptFamily.HEBREW
    return ScriptFamily.OTHER


# Scripts that the transliterator can convert to Latin. When a string mixes
# Latin with one of these, the non-Latin part is the side that needs work, so
# the script detector should report the non-Latin script for routing.
_TRANSLITERABLE_NON_LATIN = (
    ScriptFamily.CYRILLIC,
    ScriptFamily.CJK,
    ScriptFamily.HANGUL,
    ScriptFamily.ARABIC,
    ScriptFamily.DEVANAGARI,
    ScriptFamily.GREEK,
    ScriptFamily.HEBREW,
)


def detect_script(text: str) -> ScriptFamily:
    """
    Detect the dominant script family of a text string.

    Counts script family assignments for each non-whitespace character.
    When the text mixes Latin with a transliterable non-Latin script, the
    non-Latin script is reported — the cascade should route on the side
    that needs transliteration, not the side that doesn't. Falls back to
    OTHER for empty/punctuation.

    Args:
        text: Input string

    Returns:
        ScriptFamily enum value

    Examples:
        >>> detect_script("Vladimir Putin")
        <ScriptFamily.LATIN: 'Latin'>
        >>> detect_script("Владимир Путин")
        <ScriptFamily.CYRILLIC: 'Cyrillic'>
        >>> detect_script("弗拉基米尔·普京")
        <ScriptFamily.CJK: 'CJK'>
        >>> detect_script("فلاديمير بوتين")
        <ScriptFamily.ARABIC: 'Arabic'>
        >>> detect_script("율리야 리프니츠카야")
        <ScriptFamily.HANGUL: 'Hangul'>
        >>> detect_script("新少林寺/SHAOLIN")   # mixed Latin+CJK routes as CJK
        <ScriptFamily.CJK: 'CJK'>
    """
    if not text or not text.strip():
        return ScriptFamily.OTHER

    counts: Counter = Counter()
    for char in text:
        if char.isspace():
            continue
        script = _char_script(char)
        if script != ScriptFamily.OTHER:
            counts[script] += 1

    if not counts:
        return ScriptFamily.OTHER

    # If the string mixes Latin with a transliterable script, return the
    # transliterable one — that side is where the cascade has work to do.
    for non_latin in _TRANSLITERABLE_NON_LATIN:
        if counts.get(non_latin, 0) > 0 and counts.get(ScriptFamily.LATIN, 0) > 0:
            return non_latin

    return counts.most_common(1)[0][0]


def is_same_script(text_a: str, text_b: str) -> bool:
    """
    Return True if both strings belong to the same script family.

    Args:
        text_a: First string
        text_b: Second string

    Returns:
        bool
    """
    return detect_script(text_a) == detect_script(text_b)
