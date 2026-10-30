# Synchronized Learned Rollouts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Play held-out ground-truth rollouts, Model 1 belief revisions, and Model 2 world guesses on one timeline, with the guessed operator parameters visibly compared against the actual parameters.

**Architecture:** Extend the static learned-example export from one episode to a bank of held-out episodes so the browser remains honest and dependency-free. The interactive physics sandbox keeps its free sliders; the learned view selects a saved rollout, plays its ground-truth clip and belief update together, and renders a predicted-versus-actual parameter comparison from the saved learned outputs.

**Tech Stack:** Python 3.11, PyTorch, NumPy, static HTML/CSS/JavaScript, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-parameterized-occlusion-environment-design.md`

## Global Constraints

- Model 2 receives only Model 1 belief images; ground-truth frames and parameters remain evaluation-only.
- The browser must not imply that arbitrary slider settings run the trained networks.
- Learned rollouts must come from held-out Model 1 records and the saved Model 2 checkpoint.
- The page must run from the existing local static server without external dependencies.
- Actual and predicted values use the same `WorldSchema.v1()` slot order.

## Review Focus

- A malformed or missing rollout must not leave the page in a half-switched state.
- Rollouts with inactive operators must compare canonical zero parameters without inventing activity.
- Reset must rewind the ground-truth clip and belief phase together.
- Switching rollouts must clear the prior rollout's frame, belief, and comparison values.
- Narrow layouts must preserve the ground-truth → belief → world-guess reading order.

---

### Task 1: Export a bank of learned rollouts

**Files:**
- Modify: `src/perception_env/export_learned_example.py`
- Create: `tests/test_export_learned_example.py`
- Regenerate: `learned-example.json`

**Interfaces:**
- Consumes: held-out Model 1 truth/prediction records, Model 2 `best.pt`, and a sequence of example IDs.
- Produces: `export_learned_examples(model1_run, model2_run, output, example_ids) -> Path` and a `learned-demo-v2` payload with `rollouts[]`.

- [ ] **Step 1: Write failing schema, ordering, provenance, and belief-only inference tests.**
- [ ] **Step 2: Run `pytest tests/test_export_learned_example.py -q` and verify the missing bank exporter fails.**
- [ ] **Step 3: Implement batched export while retaining `export_learned_example(...)` as a one-ID compatibility wrapper.**
- [ ] **Step 4: Regenerate `learned-example.json` from four held-out examples and run the focused tests.**
- [ ] **Step 5: Commit the exporter and artifact.**

### Task 2: Synchronize the learned visual narrative

**Files:**
- Modify: `explorer.html`
- Modify: `tests/test_explorer.py`

**Interfaces:**
- Consumes: the `learned-demo-v2` rollout bank.
- Produces: a learned-rollout selector, one playhead shared by ground truth and beliefs, and a predicted-versus-actual parameter comparison.

- [ ] **Step 1: Write failing tests for simultaneous panel visibility, shared reset/playhead behavior, rollout selection, and actual-versus-guessed output.**
- [ ] **Step 2: Run the focused explorer tests and verify they fail for the absent synchronized UI.**
- [ ] **Step 3: Implement the learned view so ground truth, Model 1, and Model 2 are visible together; show pre-belief during occlusion and post-belief after reappearance.**
- [ ] **Step 4: Render actual and predicted active operators and numeric parameters side by side, clearly labeled as evaluation.**
- [ ] **Step 5: Run the focused tests and commit the browser update.**

### Task 3: Verify the complete experience

**Files:**
- Modify only if verification exposes a defect: `explorer.html`, `tests/test_explorer.py`, `src/perception_env/export_learned_example.py`

**Interfaces:**
- Consumes: the served explorer and regenerated rollout bank.
- Produces: verified desktop and narrow learned-rollout views with no browser errors.

- [ ] **Step 1: Serve the repository and inspect at least two rollouts from start through reappearance.**
- [ ] **Step 2: Verify the actual-versus-guessed values change when the rollout changes.**
- [ ] **Step 3: Verify the interactive sandbox remains independent and unchanged.**
- [ ] **Step 4: Run `pytest -q`, `ruff check .`, and `git diff --check`.**
- [ ] **Step 5: Commit any verification fix and report the learned/static limitation plainly.**
