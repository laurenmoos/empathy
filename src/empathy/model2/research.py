"""Registered, paired world-level research gate; never infer leakage-test success."""

import numpy as np

from .evaluate import world_interval

REGISTERED_SEEDS = (7, 19, 31)


def assess_research_gate(seed_scores, *, leakage_suite_passed: bool) -> dict:
    result = {
        "status": "not_established",
        "registered_seeds": list(REGISTERED_SEEDS),
        "leakage_suite_passed": leakage_suite_passed,
        "individual_seed_scores": seed_scores,
        "comparisons": {},
    }
    if set(seed_scores) != set(REGISTERED_SEEDS):
        result["reason"] = "registered three training seeds are required"
        return result
    required = {"full", "constant", "single_belief"}
    if any(not required.issubset(seed_scores[s]) for s in REGISTERED_SEEDS):
        result["reason"] = "missing registered controls"
        return result
    worlds = set(seed_scores[7]["full"])
    if len(worlds) < 2 or any(
        set(seed_scores[s][name]) != worlds for s in REGISTERED_SEEDS for name in required
    ):
        result["reason"] = "paired persistent world sets must match across all runs"
        return result
    if any(
        not np.isfinite(value) or not 0 <= value <= 1
        for s in REGISTERED_SEEDS
        for name in required
        for value in seed_scores[s][name].values()
    ):
        raise ValueError("agreement values must be finite and in [0,1]")
    for baseline in ("constant", "single_belief"):
        effects = {
            world: float(
                np.mean(
                    [
                        seed_scores[s]["full"][world] - seed_scores[s][baseline][world]
                        for s in REGISTERED_SEEDS
                    ]
                )
            )
            for world in sorted(worlds)
        }
        result["comparisons"][baseline] = {
            "mean_effect": float(np.mean(list(effects.values()))),
            "world_effects": effects,
            "interval": world_interval(list(effects.values())),
        }
    if leakage_suite_passed and all(
        item["interval"]["lower"] > 0 for item in result["comparisons"].values()
    ):
        result["status"] = "established"
    else:
        result["reason"] = "positive paired lower bounds and passed leakage suite are required"
    return result
