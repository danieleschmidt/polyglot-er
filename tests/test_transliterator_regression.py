"""Transliterator regression test (paper §4.6).

This file pins the *exact* Latin output emitted by
``polyglot_er.normalization.transliterate.transliterate_to_latin`` for a
representative set of inputs across Cyrillic, CJK Han (Chinese context), CJK
Han (Japanese context, kana-detected and lang-hinted), Hangul, Arabic, and
mixed-script Latin+CJK strings.

**Why exact-string assertions.** §4.6 of the polyglot-er paper documents four
silent transliterator failures the project caught only by cross-regime
anomaly detection (`pypinyin` codepoint-hex fallback, missing Hangul handler,
Japanese kanji given Mandarin readings, Hangul over-syllabification). Each
failure cost 10-22pp of Tier-3 marginal contribution before being detected.
The Gemini critique (2026-05-27) of the EMNLP draft raised
"Tool-Dependency Robustness Vulnerability" as a paper-level concern: rule-
based transliterators (`pypinyin`, `pykakasi`, `jamo` decomposition) can
change behavior across library versions, and a silent change in script-
handler output will silently degrade downstream cascade F1.

This test pins the expected output so that any future library or code
behavior change fails CI loudly rather than degrading recall numbers in
production. Inputs include the original engineering-bug cases recorded in
``PLAYBOOK_2026-05-27.md``: 百度, 安倍晋三, 율리야, مالمو, Малмë, and the
mixed-script 新少林寺/SHAOLIN string that the script detector previously
mis-routed.

When a transliterator library update legitimately changes the romanization
of a pinned input, update the expected string in the EXPECTED table below
AND record the change + before/after in the playbook so the §4.6 history
remains traceable.
"""

import pytest

from polyglot_er.normalization.transliterate import transliterate_to_latin


# ---------------------------------------------------------------------------
# Pinned (input, lang, expected_output) cases — exact-string regression set.
# ---------------------------------------------------------------------------
#
# Format: ``(label, input_text, lang_hint, expected_output)``.
# ``lang_hint`` is the BCP-47 hint passed to ``transliterate_to_latin``; ""
# means "no hint" (the script detector + kana-presence heuristic decide).
#
# The cases below cover six script families plus a Latin passthrough check
# and the mixed-script gotcha. Inputs marked (BUG-CASE) are the original
# engineering-failure cases the playbook documents.
PINNED_CASES = [
    # ----- Cyrillic ------------------------------------------------------
    ("cyr_putin",                "Путин",                "",   "Putin"),
    ("cyr_obama_two_words",      "Барак Обама",          "",   "Barak Obama"),
    ("cyr_malme_with_umlaut_e",  "Малмë",                "",   "Malmë"),

    # ----- CJK Han (Chinese context) -------------------------------------
    # 'baidu' was previously 'x767ex5ea6' (BUG-CASE — pre-pypinyin codepoint
    # fallback); pinned here to the post-fix pypinyin Hanyu Pinyin output.
    ("han_zh_baidu_BUG_CASE",    "百度",                 "zh", "bai du"),
    ("han_zh_xi_jinping",        "习近平",               "zh", "xi jin ping"),
    ("han_zh_pfizer_hui_rui",    "輝瑞",                 "zh", "hui rui"),

    # ----- CJK Han mixed with Latin (script-detector gotcha) -------------
    # 'SHAOLIN' is preserved verbatim; the kanji are pinyin-romanized. Before
    # the script-detector fix, this string was classified as LATIN and the
    # kanji were never transliterated.
    ("mixed_lat_cjk_shaolin",    "新少林寺/SHAOLIN",      "zh", "xin shao lin si/SHAOLIN"),
    ("mixed_lat_cjk_iphone",     "iPhone 12 Pro 蘋果",   "zh", "iPhone 12 Pro ping guo"),

    # ----- CJK Han (Japanese context, language-hinted) -------------------
    # 'abe shinzou' was previously 'an bei jin san' (BUG-CASE — kanji routed
    # through pypinyin Mandarin readings); pinned here to the post-fix
    # pykakasi Hepburn output.
    ("han_ja_abe_lang_hinted_BUG_CASE", "安倍晋三",       "ja", "abe shinzou"),

    # ----- CJK Han (Japanese context, kana auto-detection) ---------------
    # Kana presence in the string should trigger the Japanese handler even
    # without a lang hint.
    ("han_ja_kana_detected",     "東京タワー",           "",   "toukyou tawaa"),
    ("han_ja_kana_lang_hinted",  "東京タワー",           "ja", "toukyou tawaa"),

    # ----- CJK Han with no hint and no kana (heuristic falls back to zh) -
    # Documents the auto-detection limitation: pure-kanji Japanese names
    # without kana and without a lang hint will be transliterated as Mandarin.
    # This is intentional and is the reason the kana-detection heuristic
    # exists; pinned so any change to the heuristic is visible.
    ("han_no_hint_no_kana_zh_fallback", "安倍晋三",       "",   "an bei jin san"),

    # ----- Hangul --------------------------------------------------------
    # 'yulriya' was previously the input itself (BUG-CASE — no Hangul
    # handler); pinned here to the post-fix simplified-RR output.
    ("hangul_yuliya_single_BUG_CASE", "율리야",          "",   "yulriya"),
    # 'yulriya ripeunicheukaya' was previously 'yul ri ya ri peu ni cheu
    # ka ya' (BUG-CASE — per-syllable spacing penalized JW heavily); pinned
    # to the post-fix no-internal-space form.
    ("hangul_lipnitskaya_no_internal_spaces_BUG_CASE",
                                 "율리야 리프니츠카야",  "",   "yulriya ripeunicheukaya"),
    ("hangul_malmoe",            "말뫼",                 "",   "malmoe"),

    # ----- Arabic --------------------------------------------------------
    ("arabic_malmw",             "مالمو",                "",   "malmw"),
    ("arabic_putin_two_words",   "فلاديمير بوتين",       "",   "fladymyr bwtyn"),

    # ----- Latin passthrough --------------------------------------------
    ("latin_ascii_passthrough",  "Vladimir Putin",       "",   "Vladimir Putin"),
    ("latin_diacritic_preserved","Malmö",                "",   "Malmö"),

    # ----- Edge: empty string -------------------------------------------
    ("empty",                    "",                     "",   ""),
]


@pytest.mark.parametrize(
    "label,text,lang,expected",
    PINNED_CASES,
    ids=[c[0] for c in PINNED_CASES],
)
def test_transliterate_to_latin_pinned(
    label: str, text: str, lang: str, expected: str
) -> None:
    """Each pinned (input, lang) MUST emit the recorded Latin string.

    A failure here means a script handler's behavior has changed. Before
    updating EXPECTED, audit the change against §4.6 of the paper and the
    playbook; library updates that silently change CJK or Hangul output
    have historically cost 10-22pp of cascade recall.
    """
    actual = transliterate_to_latin(text, lang=lang)
    assert actual == expected, (
        f"Transliterator regression on '{label}'\n"
        f"  input    = {text!r}  (lang={lang!r})\n"
        f"  expected = {expected!r}\n"
        f"  actual   = {actual!r}\n"
        f"If this change is intentional, update PINNED_CASES and document "
        f"the before/after in PLAYBOOK_2026-05-27.md."
    )


def test_transliterator_regression_set_covers_all_handler_paths() -> None:
    """Sanity check that the pinned set exercises every script handler.

    If a future contributor adds a new script handler but forgets to pin
    expected output for it, the cascade can regress silently. This test
    documents which handler paths the regression set covers.
    """
    labels = {label for label, *_ in PINNED_CASES}
    required_coverage = {
        # Cyrillic handler
        "cyr_putin",
        # CJK Han (Chinese context, pypinyin path)
        "han_zh_baidu_BUG_CASE",
        # CJK Han (Japanese context, pykakasi path) — lang-hinted
        "han_ja_abe_lang_hinted_BUG_CASE",
        # CJK Han (Japanese context, kana-detection path)
        "han_ja_kana_detected",
        # CJK Han (no hint, no kana — Chinese fallback)
        "han_no_hint_no_kana_zh_fallback",
        # Hangul handler (single syllable)
        "hangul_yuliya_single_BUG_CASE",
        # Hangul handler (multi-word, no internal spaces)
        "hangul_lipnitskaya_no_internal_spaces_BUG_CASE",
        # Arabic handler
        "arabic_malmw",
        # Latin passthrough
        "latin_ascii_passthrough",
        # Mixed Latin+CJK (script-detector gotcha)
        "mixed_lat_cjk_shaolin",
    }
    missing = required_coverage - labels
    assert not missing, (
        f"Regression set missing handler coverage for: {sorted(missing)}. "
        f"Add pinned cases or update required_coverage."
    )
