"""Wikipedia title-format normalization.

Two surface-form transforms that the upstream extraction layer applies before
passing names to the cascade matchers:

- :func:`swap_last_first` rewrites "Lastname, Firstname[ Patronymic]" → "Firstname
  Lastname". Russian Wikipedia titles person entities this way by convention
  ("Стивен, Элизабет"); English Wikipedia uses the natural "Firstname Lastname"
  order. The transform is conservative: it only fires when the title has exactly
  one comma followed by whitespace and the leading chunk does not itself contain
  whitespace, which is a strong signal of the "Last, First" convention rather
  than a stylistic comma.

- :func:`strip_disambig_parens` removes a trailing disambiguation parenthetical
  such as "Apple (company)" → "Apple" or "Fred (footballer, born 1979)" → "Fred".
  Both English and Russian Wikipedia use parenthetical disambiguation in the
  same positions, so the transform is language-agnostic.

These transforms are intentionally minimal — name-component parsing, honorifics
handling, and multi-language patronymic detection are out of scope. The
:func:`normalize_wiki_title` entrypoint composes both transforms and is the
function callers should use.
"""

from __future__ import annotations

import re

# A single comma followed by whitespace, with at least one non-comma non-whitespace
# character on either side. The leading chunk (the "Lastname") must not itself
# contain whitespace — this avoids firing on phrases like "Smith, John, the
# politician" or "Apple, Inc." where the comma is stylistic rather than a
# Last/First boundary.
_LAST_FIRST_RE = re.compile(r"^(\S+),\s+(\S.*)$")

# Trailing " (...)" disambiguation. The parenthetical is non-greedy and must be
# anchored at end-of-string after optional whitespace.
_TRAILING_PARENS_RE = re.compile(r"\s*\([^()]*\)\s*$")


def swap_last_first(title: str) -> str:
    """Rewrite "Lastname, Firstname[ Patronymic]" → "Firstname[ Patronymic] Lastname".

    Returns the original string unchanged when the title does not match the
    "Last, First" pattern (no comma, multi-word leading chunk, etc.).
    """
    m = _LAST_FIRST_RE.match(title)
    if not m:
        return title
    last, rest = m.group(1), m.group(2)
    return f"{rest} {last}"


def strip_disambig_parens(title: str) -> str:
    """Remove a trailing disambiguation parenthetical.

    "Apple (company)" → "Apple"
    "Fred (footballer, born 1979)" → "Fred"
    "Mercury" → "Mercury"  (unchanged; no parenthetical)
    """
    return _TRAILING_PARENS_RE.sub("", title)


def normalize_wiki_title(title: str, lang: str = "") -> str:
    """Apply the Wikipedia title-format transforms in order.

    The ``lang`` argument is reserved for future language-specific behavior;
    the current implementation applies both transforms regardless of language
    because the patterns they target are consistent across the Wikipedia
    editions that use them.
    """
    title = strip_disambig_parens(title)
    title = swap_last_first(title)
    return title
