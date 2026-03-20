"""
Unicode normalization for cross-lingual entity resolution.

Handles NFC/NFD normalization, diacritic stripping, and preprocessing
for cross-script comparison.
"""

import unicodedata
import re


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """
    Normalize Unicode text to the specified form.

    Args:
        text: Input string
        form: Unicode normalization form — "NFC", "NFD", "NFKC", or "NFKD"

    Returns:
        Normalized string

    Examples:
        >>> normalize_unicode("café", "NFC")
        'café'
        >>> normalize_unicode("café", "NFD")  # decomposed — 'e' + combining accent
        'café'
    """
    return unicodedata.normalize(form, text)


def strip_diacritics(text: str) -> str:
    """
    Strip combining diacritical marks from a string (Latin-centric).

    Decomposes to NFD first, then removes category 'Mn' (Mark, Nonspacing).

    Args:
        text: Input string

    Returns:
        String with diacritics removed

    Examples:
        >>> strip_diacritics("Müller")
        'Muller'
        >>> strip_diacritics("résumé")
        'resume'
    """
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def normalize_for_matching(text: str) -> str:
    """
    Full normalization pipeline for entity matching.

    Steps:
      1. NFC normalization
      2. Lowercase
      3. Strip leading/trailing whitespace
      4. Collapse internal whitespace
      5. Remove punctuation that is not part of names (hyphens kept)

    Args:
        text: Raw entity name string

    Returns:
        Normalized string suitable for comparison

    Examples:
        >>> normalize_for_matching("  Vladimir  Putin  ")
        'vladimir putin'
        >>> normalize_for_matching("O'Brien")
        "o'brien"
    """
    if not text:
        return ""
    text = normalize_unicode(text, "NFC")
    text = text.lower()
    text = text.strip()
    # Collapse internal whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove characters that are not letters, digits, space, hyphen, or apostrophe
    text = re.sub(r"[^\w\s\-']", "", text, flags=re.UNICODE)
    return text
