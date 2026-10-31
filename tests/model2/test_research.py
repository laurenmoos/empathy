from empathy.model2.research import assess_research_gate


def test_research_gate_requires_three_seeds_paired_worlds_and_leakage_evidence():
    scores = {
        seed: {
            name: {"a": value, "b": value, "c": value}
            for name, value in [("full", 0.8), ("constant", 0.4), ("single_belief", 0.5)]
        }
        for seed in (7, 19, 31)
    }
    assert assess_research_gate(scores, leakage_suite_passed=True)["status"] == "established"
    assert assess_research_gate(scores, leakage_suite_passed=False)["status"] == "not_established"
    assert (
        assess_research_gate({7: scores[7]}, leakage_suite_passed=True)["status"]
        == "not_established"
    )
    scores[19]["full"]["a"] = 0.1
    scores[31]["single_belief"].pop("a")
    assert assess_research_gate(scores, leakage_suite_passed=True)["status"] == "not_established"
