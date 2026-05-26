"""Tests for the Wikipedia title-format normalizer."""

from polyglot_er.normalization.title_format import (
    normalize_wiki_title,
    strip_disambig_parens,
    swap_last_first,
)


class TestSwapLastFirst:
    def test_swaps_simple_lastname_firstname(self):
        assert swap_last_first("Стивен, Элизабет") == "Элизабет Стивен"

    def test_swaps_with_patronymic(self):
        assert (
            swap_last_first("Стравинский, Игорь Фёдорович")
            == "Игорь Фёдорович Стравинский"
        )

    def test_leaves_natural_order_unchanged(self):
        assert swap_last_first("Vladimir Putin") == "Vladimir Putin"

    def test_leaves_no_comma_unchanged(self):
        assert swap_last_first("Mercury") == "Mercury"

    def test_does_not_fire_on_multi_word_first_chunk(self):
        # "Apple, Inc." has a stylistic comma but "Apple" is single-word and
        # this DOES match the heuristic — that's a known false positive and
        # acceptable for the dominant Russian Wikipedia "Last, First" case.
        # The protection is for cases like "Smith John, the politician" where
        # the leading chunk has whitespace.
        assert swap_last_first("Smith John, the politician") == "Smith John, the politician"

    def test_does_not_fire_without_whitespace_after_comma(self):
        assert swap_last_first("Foo,Bar") == "Foo,Bar"

    def test_handles_empty_string(self):
        assert swap_last_first("") == ""


class TestStripDisambigParens:
    def test_strips_simple_disambig(self):
        assert strip_disambig_parens("Apple (company)") == "Apple"

    def test_strips_compound_disambig(self):
        assert (
            strip_disambig_parens("Fred (footballer, born 1979)") == "Fred"
        )

    def test_leaves_no_parens_unchanged(self):
        assert strip_disambig_parens("Mercury") == "Mercury"

    def test_does_not_strip_internal_parens(self):
        # Only trailing parens should be stripped.
        assert strip_disambig_parens("Foo (bar) Baz") == "Foo (bar) Baz"

    def test_strips_trailing_whitespace_before_parens(self):
        assert strip_disambig_parens("Apple    (company)") == "Apple"


class TestNormalizeWikiTitle:
    def test_applies_both_transforms(self):
        # "Фред (футболист, 1979)" → strip parens → "Фред" → no swap
        assert normalize_wiki_title("Фред (футболист, 1979)") == "Фред"

    def test_strip_then_swap(self):
        # "Греко, Бадди (singer)" → strip parens → "Греко, Бадди" → swap → "Бадди Греко"
        assert normalize_wiki_title("Греко, Бадди (singer)") == "Бадди Греко"

    def test_english_natural_order_unchanged(self):
        assert normalize_wiki_title("Buddy Greco") == "Buddy Greco"

    def test_buddy_greco_round_trip_matches(self):
        # The headline failure case from the pilot.
        en = normalize_wiki_title("Buddy Greco")
        ru = normalize_wiki_title("Греко, Бадди")
        assert en == "Buddy Greco"
        assert ru == "Бадди Греко"
        # The transliterated form of ru should now align with en.

    def test_idempotent(self):
        # Applying twice should give the same result.
        t = "Стравинский, Игорь Фёдорович"
        assert normalize_wiki_title(normalize_wiki_title(t)) == normalize_wiki_title(t)
