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
    CJK = "CJK"           # Chinese, Japanese Kanji, Korean Hanja
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


def detect_script(text: str) -> ScriptFamily:
    """
    Detect the dominant script family of a text string.

    Counts script family assignments for each non-whitespace character
    and returns the plurality winner. Falls back to OTHER for empty/punctuation.

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
