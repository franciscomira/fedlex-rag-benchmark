from fedlex_bench.eval.calibration import agreement, draw_sample


def row(i, answerer, *, trap=False, status="answered", cit=("220:1",)):
    return {"id": i, "answerer": answerer, "is_trap": trap, "status": status,
            "citations": list(cit), "over_refusal": None if trap else status == "refused"}


def pool():
    rows = []
    for n in range(40):
        rows.append(row(f"a{n:02d}", "m1"))
        rows.append(row(f"a{n:02d}", "m2"))
    for n in range(5):
        rows.append(row(f"t{n}", "m1", trap=True, status="refused", cit=()))
    for n in range(40, 48):
        rows.append(row(f"a{n}", "m1", status="refused", cit=()))
    return rows


def test_draw_sample_is_deterministic_stratified_and_one_answer_per_item():
    s1 = draw_sample(pool(), seed=7)
    s2 = draw_sample(pool(), seed=7)
    assert [(r["id"], r["answerer"]) for r in s1] == [(r["id"], r["answerer"]) for r in s2]
    assert len(s1) == 30
    assert len({r["id"] for r in s1}) == 30
    assert sum(r["is_trap"] for r in s1) == 5
    assert sum((not r["is_trap"]) and (r["status"] == "refused" or not r["citations"]) for r in s1) == 5


def test_agreement_counts_fact_and_trap_labels():
    judge = {"a": [True, False], "b": [True], "t": True}
    human = {"a": [True, True], "b": [True], "t": False}
    r = agreement(judge, human)
    assert r["facts"] == {"n": 3, "agree": 2, "raw": 2 / 3, "kappa": r["facts"]["kappa"]}
    assert r["traps"] == {"n": 1, "agree": 0, "raw": 0.0}
    assert r["overall_raw"] == 2 / 4
