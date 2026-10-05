"""Run: python test_brain.py   (or pytest). The retrieval check loads the local bge-small model."""
from brain import EVAL_SET, Brain, evaluate

b = Brain(with_vectors=False)
DEAL = "D-101"


def test_financing_request_changes_over_time():
    f = b.financing(DEAL)
    assert [h["amount_musd"] for h in f["history"]] == [12.0, 15.0, 14.0]
    assert f["current"]["sources"] == ["outlook://msg-009"]
    assert len(f["history"][0]["sources"]) == 2  # email + CIM restating $12M are evidence, not a change


def test_lender_status_is_latest_per_lender():
    status = {r["lender"]: r["current"] for r in b.lenders_reviewed(DEAL)}
    assert status == {"Larkspur Bank": "declined", "Quillfeather Credit": "indication_of_interest",
                      "Basaltline Capital": "reviewing"}


def test_relevant_lenders_exclude_engaged_and_flag_declines():
    r = b.relevant_lenders(DEAL)
    assert [s["lender"] for s in r["suggested"]] == ["Tidewren Lending"]
    assert [s["lender"] for s in r["negative_signals"]] == ["Larkspur Bank"]


def test_people_and_no_cross_deal_leak():
    names = {p["name"] for p in b.who(DEAL)}
    assert names == {"Priya Raman", "Tom Becker", "Lena Ortiz", "Mark Ellis", "Jane Cho", "Owen Hale"}
    assert b.latest(DEAL)["recent"][0]["source_ref"] == "outlook://msg-009"


def test_every_edge_traces_to_a_real_source():
    refs = {d["source_ref"] for d in b.docs}
    assert b.edges and all(e["source_ref"] in refs for e in b.edges)


def test_retrieval_regression():
    # Guard, not a claim: 12 docs is too small to rank retrievers. Paraphrased relationship questions
    # ("which bank said no") miss in every mode; the REVIEWED edge answers them, which is the point.
    assert len(EVAL_SET) == 10
    assert evaluate(Brain())["hybrid"]["hit_at_k"] >= 0.8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
