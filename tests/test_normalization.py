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


def test_script_detect_hangul():
    from polyglot_er.normalization.script_detect import detect_script, ScriptFamily

    assert detect_script("율리야 리프니츠카야") == ScriptFamily.HANGUL
    assert detect_script("샤미트 카치루") == ScriptFamily.HANGUL


def test_script_detect_mixed_latin_cjk_routes_as_cjk():
    """Mixed Latin+CJK strings should route as CJK so the kanji side gets
    transliterated. The pre-fix plurality vote classified anything with even
    one Latin character as LATIN, which silently dropped the CJK content from
    the transliteration pipeline."""
    from polyglot_er.normalization.script_detect import detect_script, ScriptFamily

    assert detect_script("新少林寺/SHAOLIN") == ScriptFamily.CJK
    assert detect_script("百度 (search engine)") == ScriptFamily.CJK


def test_transliterate_to_latin_hangul():
    """Hangul should be romanized via simplified RR, not pass through
    unchanged. The pre-fix behavior (passthrough) made T3 catch zero pairs
    in en↔ko evaluation."""
    out = transliterate_to_latin("율리야 리프니츠카야")
    assert "yul" in out
    # No Hangul codepoints should remain.
    assert not any("가" <= c <= "힣" for c in out), out
    # Output is alphanumeric + whitespace
    assert all(c.isalpha() or c.isspace() for c in out), out


def test_transliterate_hangul_no_internal_spaces():
    """Syllables within a word are concatenated without internal spaces.
    Per-syllable spacing (the v1 behavior) inserted artificial token
    boundaries that Jaro-Winkler penalized heavily — costing +16pp T3
    recall on en_ko. Source-string spaces between words are preserved."""
    from polyglot_er.normalization.transliterate import transliterate_hangul

    # One Hangul word → one continuous romanization, no internal spaces.
    assert transliterate_hangul("율리야") == "yulriya"
    # Two source words → two romanizations separated by the original space.
    assert " " in transliterate_hangul("율리야 리프니츠카야")


def test_transliterate_to_latin_cjk_uses_pinyin():
    """CJK ideographs should pinyin-transliterate, not fall through to codepoint hex.

    The pre-pypinyin path emitted ``x{codepoint_hex}`` for ideographs absent from
    the hand-maintained lookup table. That hex output broke downstream
    Jaro-Winkler matching for en↔zh. With pypinyin wired in, ideographs are
    converted to Hanyu Pinyin syllables (space-separated for word alignment).
    """
    # Simplified Chinese — Mao Zedong
    assert transliterate_to_latin("毛泽东") == "mao ze dong"
    # Traditional Chinese — Pfizer (transliterated as a phonetic borrowing)
    assert transliterate_to_latin("輝瑞") == "hui rui"
    # Mixed punctuation should be preserved
    assert transliterate_to_latin("鮑勃·巴雷特") == "bao bo·ba lei te"
    # No codepoint-hex leakage even on rare characters
    out = transliterate_to_latin("莎孚")
    assert "x" not in out, f"codepoint-hex leaked into pypinyin output: {out!r}"
    assert all(c.isalpha() or c.isspace() for c in out), out
