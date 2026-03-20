"""Tests for normalization modules: unicode_norm, script_detect, transliterate."""

import pytest

from polyglot_er.normalization.unicode_norm import (
    normalize_unicode,
    strip_diacritics,
    normalize_for_matching,
)
from polyglot_er.normalization.script_detect import (
    detect_script,
    is_same_script,
    ScriptFamily,
)
from polyglot_er.normalization.transliterate import (
    transliterate_cyrillic,
    transliterate_to_latin,
)


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------

def test_script_detection_latin():
    assert detect_script("Vladimir Putin") == ScriptFamily.LATIN
    assert detect_script("Angela Merkel") == ScriptFamily.LATIN
    assert detect_script("Barack Obama") == ScriptFamily.LATIN


def test_script_detection_cyrillic():
    assert detect_script("Владимир Путин") == ScriptFamily.CYRILLIC
    assert detect_script("Мария Кюри") == ScriptFamily.CYRILLIC


def test_script_detection_cjk():
    assert detect_script("習近平") == ScriptFamily.CJK
    assert detect_script("弗拉基米尔·普京") == ScriptFamily.CJK


def test_script_detection_arabic():
    assert detect_script("فلاديمير بوتين") == ScriptFamily.ARABIC
    assert detect_script("باراك أوباما") == ScriptFamily.ARABIC


def test_script_detection_empty():
    assert detect_script("") == ScriptFamily.OTHER
    assert detect_script("   ") == ScriptFamily.OTHER


def test_is_same_script_true():
    assert is_same_script("Putin", "Merkel") is True
    assert is_same_script("Путин", "Меркель") is True


def test_is_same_script_false():
    assert is_same_script("Putin", "Путин") is False
    assert is_same_script("Putin", "普京") is False


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------

def test_unicode_normalization_nfc():
    # NFC combines decomposed characters
    decomposed = "e\u0301"  # 'e' + combining acute accent
    composed = "\xe9"       # 'é' precomposed
    assert normalize_unicode(decomposed, "NFC") == composed


def test_unicode_normalization_decomposes():
    """normalize_unicode NFD decomposes precomposed characters."""
    composed = "é"          # precomposed
    decomposed = "e\u0301"  # 'e' + combining acute
    result = normalize_unicode(composed, "NFD")
    assert result == decomposed


def test_strip_diacritics_latin():
    assert strip_diacritics("Müller") == "Muller"
    assert strip_diacritics("résumé") == "resume"
    assert strip_diacritics("café") == "cafe"


def test_normalize_for_matching_whitespace():
    assert normalize_for_matching("  Vladimir  Putin  ") == "vladimir putin"


def test_normalize_for_matching_empty():
    assert normalize_for_matching("") == ""
    assert normalize_for_matching("   ") == ""


def test_normalize_for_matching_lowercase():
    assert normalize_for_matching("PUTIN") == "putin"


# ---------------------------------------------------------------------------
# Transliteration
# ---------------------------------------------------------------------------

def test_transliterate_cyrillic_putin():
    result = transliterate_cyrillic("Путин")
    # Should produce something close to "Putin"
    assert "utin" in result.lower() or "Putin" in result


def test_transliterate_cyrillic_full_name():
    result = transliterate_cyrillic("Владимир")
    assert result  # non-empty
    # All Cyrillic chars should be converted
    for ch in result:
        assert ord(ch) < 0x0400 or ch == " ", f"Cyrillic char remaining: {ch}"


def test_transliterate_to_latin_latin_passthrough():
    """Latin text should pass through unchanged."""
    assert transliterate_to_latin("Putin") == "Putin"
    assert transliterate_to_latin("Merkel") == "Merkel"


def test_transliterate_to_latin_cyrillic():
    result = transliterate_to_latin("Путин")
    # Should contain 'utin'
    assert "utin" in result.lower()


def test_transliterate_to_latin_arabic():
    result = transliterate_to_latin("بوتين")
    assert isinstance(result, str)
    assert len(result) > 0
