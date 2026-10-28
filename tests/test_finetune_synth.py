"""Query-generation plan and reply validation (no network)."""

from __future__ import annotations

import json
from collections import Counter

from fedlex_bench.finetune.synth import LANGS, STYLES, plan_article, validate_reply

IDS = [f"220:{n}" for n in range(1, 301)]


def _reply(plan, texts):
    return json.dumps({"questions": [{"slot": i, "text": t} for i, t in enumerate(texts, 1)]})


def test_plan_is_deterministic_and_shown_language_is_left_out():
    for a in IDS:
        p = plan_article(a)
        assert p == plan_article(a)
        langs = [lang for lang, _ in p.slots]
        assert langs[:2] == ["en", "en"]
        assert set(langs[2:]) == set(LANGS) - {p.shown_lang}
        styles = [s for _, s in p.slots]
        assert len(set(styles)) == 4 and set(styles) <= set(STYLES)


def test_plan_spreads_languages_and_styles():
    plans = [plan_article(a) for a in IDS]
    assert all(c > 60 for c in Counter(p.shown_lang for p in plans).values())  # 300 articles
    assert all(c > 100 for c in Counter(s for p in plans for _, s in p.slots).values())


def test_validate_accepts_good_reply():
    p = plan_article("220:1")
    texts = [
        "notice period lease",
        "How long is the notice period for ending a lease?",
        "Mein Vermieter will mir kündigen, wie viel Zeit habe ich?",
        "Quels délais s'appliquent ?",
    ]
    assert validate_reply(_reply(p, texts), p) == (texts, "")


def test_validate_rejects_bad_replies():
    p = plan_article("220:1")
    good = [
        "a b",
        "How long is the notice period for a lease?",
        "Wie lange dauert die Frist wirklich?",
        "Quelle est la durée de la période ?",
    ]
    assert validate_reply("not json", p)[1] == "shape"
    assert validate_reply(_reply(p, good[:3]), p)[1] == "shape"
    assert (
        validate_reply(_reply(p, [*good[:3], "Was sagt Art. 257 über die Frist hier?"]), p)[1]
        == "id_leak"
    )
    assert (
        validate_reply(_reply(p, [*good[:3], "see SR 220 for the details of this"]), p)[1]
        == "id_leak"
    )
    assert validate_reply(_reply(p, [good[1], good[1], good[2], good[3]]), p)[1] == "duplicate"
    assert validate_reply(_reply(p, ["x", *good[1:]]), p)[1] == "length"
