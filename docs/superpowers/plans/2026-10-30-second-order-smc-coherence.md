# Second-Order SMC Coherence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the live demo implement the README's belief-only second-order world search and show how many physical constraints survive from Model 1 beliefs into Model 2's recovered world.

**Architecture:** Add a deterministic sequential Monte Carlo world search whose only observation is the fixed four-event pre/post/difference belief history. Candidate operator compositions and parameters generate simulated trajectory belief fields; transport-shaped spatial discrepancy and signed belief-change discrepancy update particle weights across events. The live session evaluates the ranked world against hidden truth only after inference, producing a three-slot constraint-recovery score for the UI.

**Tech Stack:** Python 3.11, NumPy, PyTorch tensor contracts, HTML Canvas, vanilla JavaScript, pytest.

**Spec:** `README.md` — “Model 2 Thinking” and “Coherence Measure and Constraint Recovery.”

## Global Constraints

- Second-order inference receives only `BeliefInputs`; no video frames, initial state, world code, operator labels, or physical parameters may enter candidate scoring.
- The true world is used only after inference to evaluate recovery.
- Search is deterministic under a fixed seed and carries a ranked particle posterior across the valid event suffix.
- Live coherence covers the three adjustable operators: gravity, wind, and damping.
- A constraint is recovered when activity agrees and, when active, its normalized parameter is within `0.10`; coherence is recovered constraints divided by three.
- Preserve the existing single-canvas trajectory/belief visualization and muted parchment/green visual system.

## Review Focus

- Leakage: a changed hidden target with identical belief input must not change the SMC prediction.
- Degenerate beliefs: empty or uniform beliefs must return a finite normalized posterior rather than NaNs.
- Neutral worlds: inactive constraints must count as recovered only when the prediction also leaves them inactive.
- Determinism: same beliefs and seed must reproduce particles, weights, best world, and coherence.
- Performance: the default live particle/search budget must keep one episode interactive on CPU.

---

### Task 1: Belief-only sequential Monte Carlo world search

**Files:**
- Create: `src/perception_env/model2/smc.py`
- Create: `tests/model2/test_smc.py`

**Interfaces:**
- Produces: `SMCConfig`, `SMCResult`, `BeliefWorldSMC.infer(inputs: BeliefInputs) -> SMCResult`.
- `SMCResult` exposes ranked canonical world vectors, normalized weights, best world, and per-event discrepancies.

- [ ] Write failing tests for belief-only input, finite normalized weights on empty beliefs, same-seed determinism, and a synthetic curved-belief case ranking gravity above a neutral candidate.
- [ ] Run `.venv/bin/pytest tests/model2/test_smc.py -q` and confirm import failure.
- [ ] Implement deterministic particles over gravity/wind/damping composition and parameter supports, marginalizing a small fixed bank of entry states into candidate path fields.
- [ ] Implement downsampled entropic transport discrepancy for spatial shape plus signed-change discrepancy, sequential log-weight updates, stable normalization, and seeded resampling/jitter.
- [ ] Run `.venv/bin/pytest tests/model2/test_smc.py -q` and confirm pass.

### Task 2: Constraint recovery and live second-order payload

**Files:**
- Create: `src/perception_env/model2/coherence.py`
- Create: `tests/model2/test_live_coherence.py`
- Modify: `src/perception_env/live_thinking.py`
- Modify: `src/perception_env/live_inference.py`
- Modify: `tests/test_live_thinking.py`
- Modify: `tests/test_live_inference.py`

**Interfaces:**
- Produces: `ConstraintRecovery`, `evaluate_constraint_recovery(actual_flat, predicted_flat, *, tolerance=0.10)`.
- Live thought payload adds `constraint_recovery` with `recovered`, `total`, `score`, and one status per adjustable operator.
- Live second-order runtime uses `BeliefWorldSMC` and never receives the true world during inference or posterior updating.

- [ ] Write failing exact recovery tests for inactive agreement, activity mismatch, active parameter tolerance, and score arithmetic.
- [ ] Write a leakage test proving identical `BeliefInputs` yield identical guesses when only hidden truth changes.
- [ ] Implement constraint recovery against the three adjustable slots using normalized parameters.
- [ ] Integrate the SMC runtime into thinking sessions; evaluate hidden truth only after each guess and serialize recovery in each thought step.
- [ ] Run `.venv/bin/pytest tests/model2/test_live_coherence.py tests/test_live_thinking.py tests/test_live_inference.py -q` and confirm pass.

### Task 3: Understated coherence visualization

**Files:**
- Modify: `explorer.html`
- Modify: `tests/test_explorer.py`

**Interfaces:**
- Consumes: `thought_step.constraint_recovery`.
- Produces: a compact `coherenceScore`, `coherenceFill`, and `coherenceDetail` display within the existing second-order panel.

- [ ] Write failing explorer tests requiring “Coherence,” the recovered/total wording, a muted horizontal meter, and per-constraint recovered/missed row states without adding another canvas or panel.
- [ ] Implement payload validation and update the existing second-order readout with a thin desaturated meter, tabular score, and quiet status dots beside gravity/wind/damping.
- [ ] Clear coherence with the other second-order output on reset and parameter changes.
- [ ] Run `.venv/bin/pytest tests/test_explorer.py -q` and confirm pass.

### Task 4: End-to-end verification

**Files:**
- Modify as required by Tasks 1–3 only.

- [ ] Benchmark one live episode and assert the default CPU search stays within the interactive test budget.
- [ ] Run `.venv/bin/ruff check src tests` and confirm clean output.
- [ ] Run `.venv/bin/pytest -q` and confirm the full suite passes.
- [ ] Run `git diff --check` and inspect the final diff for raw-frame or hidden-world leakage into `BeliefWorldSMC.infer`.
