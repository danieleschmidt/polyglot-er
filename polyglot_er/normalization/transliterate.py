"""
Basic transliteration: convert Cyrillic, CJK, and Arabic characters
to approximate Latin phonetic representations.

These are simplified, heuristic transliterations intended to enable
cross-script fuzzy phonetic matching. The CJK path prefers ``pypinyin``
when installed (Hanyu Pinyin coverage of ~all CJK Unified Ideographs);
the built-in lookup table is a fallback for environments without it.
"""

from typing import Dict

try:
    from pypinyin import lazy_pinyin, Style  # type: ignore[import-not-found]
    _PYPINYIN_AVAILABLE = True
except ImportError:
    _PYPINYIN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Cyrillic → Latin (BGN/PCGN-inspired, simplified)
# ---------------------------------------------------------------------------
_CYRILLIC_TO_LATIN: Dict[str, str] = {
    "а": "a",  "б": "b",  "в": "v",  "г": "g",  "д": "d",
    "е": "ye", "ё": "yo", "ж": "zh", "з": "z",  "и": "i",
    "й": "y",  "к": "k",  "л": "l",  "м": "m",  "н": "n",
    "о": "o",  "п": "p",  "р": "r",  "с": "s",  "т": "t",
    "у": "u",  "ф": "f",  "х": "kh", "ц": "ts", "ч": "ch",
    "ш": "sh", "щ": "shch","ъ": "",   "ы": "y",  "ь": "",
    "э": "e",  "ю": "yu", "я": "ya",
    # Uppercase variants
    "А": "A",  "Б": "B",  "В": "V",  "Г": "G",  "Д": "D",
    "Е": "Ye", "Ё": "Yo", "Ж": "Zh", "З": "Z",  "И": "I",
    "Й": "Y",  "К": "K",  "Л": "L",  "М": "M",  "Н": "N",
    "О": "O",  "П": "P",  "Р": "R",  "С": "S",  "Т": "T",
    "У": "U",  "Ф": "F",  "Х": "Kh", "Ц": "Ts", "Ч": "Ch",
    "Ш": "Sh", "Щ": "Shch","Ъ": "",  "Ы": "Y",  "Ь": "",
    "Э": "E",  "Ю": "Yu", "Я": "Ya",
}

# ---------------------------------------------------------------------------
# CJK: Common name characters → Pinyin (highly simplified lookup table)
# Full Pinyin conversion requires a proper library; this covers common
# characters found in famous person names for demonstration purposes.
# ---------------------------------------------------------------------------
_CJK_TO_PINYIN: Dict[str, str] = {
    "弗": "fu",   "拉": "la",   "基": "ji",   "米": "mi",
    "尔": "er",   "普": "pu",   "京": "jing", "习": "xi",
    "近": "jin",  "平": "ping", "毛": "mao",  "泽": "ze",
    "东": "dong", "邓": "deng", "小": "xiao", "江": "jiang",
    "泽": "ze",   "民": "min",  "胡": "hu",   "锦": "jin",
    "涛": "tao",  "李": "li",   "克": "ke",   "强": "qiang",
    "王": "wang", "张": "zhang","刘": "liu",  "陈": "chen",
    "杨": "yang", "赵": "zhao", "黄": "huang","周": "zhou",
    "吴": "wu",   "徐": "xu",   "孙": "sun",  "马": "ma",
    "胡": "hu",   "朱": "zhu",  "林": "lin",  "何": "he",
    "郭": "guo",  "罗": "luo",  "梁": "liang","宋": "song",
    "唐": "tang", "蔡": "cai",  "金": "jin",  "韩": "han",
}

# ---------------------------------------------------------------------------
# Arabic → Latin (simplified, covering common person name letters)
# ---------------------------------------------------------------------------
_ARABIC_TO_LATIN: Dict[str, str] = {
    "ا": "a",  "ب": "b",  "ت": "t",  "ث": "th", "ج": "j",
    "ح": "h",  "خ": "kh", "د": "d",  "ذ": "dh", "ر": "r",
    "ز": "z",  "س": "s",  "ش": "sh", "ص": "s",  "ض": "d",
    "ط": "t",  "ظ": "z",  "ع": "a",  "غ": "gh", "ف": "f",
    "ق": "q",  "ك": "k",  "ل": "l",  "م": "m",  "ن": "n",
    "ه": "h",  "و": "w",  "ي": "y",  "ى": "a",  "ة": "a",
    "أ": "a",  "إ": "i",  "آ": "a",  "ؤ": "w",  "ئ": "y",
    "ء": "",   "ّ": "",   "َ": "",   "ِ": "",   "ُ": "",
    "ً": "",   "ٍ": "",   "ٌ": "",   "ْ": "",
    # Common Farsi/Persian extras
    "پ": "p",  "چ": "ch", "ژ": "zh", "گ": "g",
}


def transliterate_cyrillic(text: str) -> str:
    """
    Transliterate Cyrillic text to Latin.

    Args:
        text: String containing Cyrillic characters

    Returns:
        Approximate Latin phonetic representation

    Example:
        >>> transliterate_cyrillic("Путин")
        'Putin'
    """
    return "".join(_CYRILLIC_TO_LATIN.get(ch, ch) for ch in text)


def _is_cjk_ideograph(ch: str) -> bool:
    """True if ``ch`` is in the CJK Unified Ideographs ranges we transliterate."""
    return "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf"


def transliterate_cjk(text: str) -> str:
    """
    Transliterate CJK characters to Hanyu Pinyin.

    When ``pypinyin`` is installed, runs the full library for ideograph coverage
    (recommended). Without it, falls back to the small built-in lookup table
    and emits ``x{codepoint}`` for unknown ideographs.

    Pinyin syllables are joined with single spaces, so downstream Jaro-Winkler
    matching aligns word-by-word against romanized input.

    Args:
        text: String containing CJK characters

    Returns:
        Approximate Latin phonetic representation
    """
    if _PYPINYIN_AVAILABLE:
        return _transliterate_cjk_pypinyin(text)
    return _transliterate_cjk_fallback(text)


def _transliterate_cjk_pypinyin(text: str) -> str:
    """Primary CJK path: pypinyin for ideographs, passthrough for everything else.

    We process runs of ideograph vs. non-ideograph characters separately so
    non-CJK punctuation/whitespace is preserved verbatim.
    """
    out: list[str] = []
    buf: list[str] = []

    def flush_ideographs() -> None:
        if buf:
            syllables = lazy_pinyin("".join(buf), style=Style.NORMAL)
            out.append(" ".join(syllables))
            buf.clear()

    for ch in text:
        if _is_cjk_ideograph(ch):
            buf.append(ch)
        else:
            flush_ideographs()
            out.append(ch)
    flush_ideographs()
    return "".join(out)


def _transliterate_cjk_fallback(text: str) -> str:
    """Pre-pypinyin fallback: small hand-maintained lookup + codepoint-hex."""
    parts = []
    for ch in text:
        if ch in _CJK_TO_PINYIN:
            parts.append(_CJK_TO_PINYIN[ch])
        elif _is_cjk_ideograph(ch):
            parts.append(f"x{ord(ch):04x}")
        else:
            parts.append(ch)
    return "".join(parts)


def transliterate_arabic(text: str) -> str:
    """
    Transliterate Arabic text to Latin.

    Args:
        text: String containing Arabic characters

    Returns:
        Approximate Latin phonetic representation
    """
    return "".join(_ARABIC_TO_LATIN.get(ch, ch) for ch in text)


def transliterate_to_latin(text: str) -> str:
    """
    Auto-detect script and transliterate to Latin.

    Handles Cyrillic, CJK, and Arabic. Latin text is returned unchanged.

    Args:
        text: Input string in any supported script

    Returns:
        Latin-script representation (approximate)

    Examples:
        >>> transliterate_to_latin("Путин")
        'Putin'
        >>> transliterate_to_latin("Putin")
        'Putin'
    """
    from .script_detect import detect_script, ScriptFamily

    script = detect_script(text)
    if script == ScriptFamily.CYRILLIC:
        return transliterate_cyrillic(text)
    if script == ScriptFamily.CJK:
        return transliterate_cjk(text)
    if script == ScriptFamily.ARABIC:
        return transliterate_arabic(text)
    # LATIN, GREEK, DEVANAGARI, OTHER — return as-is
    return text
