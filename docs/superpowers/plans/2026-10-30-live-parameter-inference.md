# Live Parameter Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user set physical parameters, generate a fresh occluded rollout, and compare those ground-truth parameters with Model 2's guess from Model 1 beliefs.

**Architecture:** A local Python inference engine converts a validated parameter request into a `WorldCode`, simulates and renders one occlusion event, runs the saved Model 1 checkpoint on the pre/post clips, rasterizes its beliefs, and sends only belief tensors into Model 2. A standard-library HTTP server serves the existing page and a JSON `/api/infer` endpoint; the browser uses sliders to request and visualize each new run.

**Tech Stack:** Python 3.11, PyTorch, NumPy, standard-library `http.server`, static HTML/CSS/JavaScript, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-parameterized-occlusion-environment-design.md`

## Global Constraints

- Model 2 receives beliefs only, never frames, coordinates, masks, or ground-truth parameters.
- Browser controls expose physical acceleration and damping values; zero means the operator is inactive.
- Every response contains the exact canonical ground-truth vector and the decoded Model 2 vector in `WorldSchema.v1()` order.
- Invalid or non-reappearing parameter combinations return an actionable 422 response without stale browser results.
- Checkpoints load once when the local server starts, not per request.

## Review Focus

- Activity thresholds around the acceleration neutral interval and zero damping.
- Simulation settings that send the ball out of bounds or prevent reappearance.
- First-order leakage into Model 2.
- Concurrent requests sharing model objects.
- Server startup when checkpoints are absent or port 8765 is already occupied.

---

### Task 1: Implement live learned inference

**Files:**
- Create: `src/perception_env/live_inference.py`
- Create: `tests/test_live_inference.py`

**Interfaces:**
- Produces: `LiveParameters`, `LiveInferenceResult`, `world_from_parameters(...)`, and `LiveInferenceEngine.infer(...)`.

- [ ] Write failing tests for parameter canonicalization, occlusion rejection, belief-only Model 2 inputs, and response vectors.
- [ ] Implement deterministic simulation, Model 1 pre/post inference, belief rasterization, and Model 2 decoding.
- [ ] Run the focused tests and commit.

### Task 2: Add the local inference server

**Files:**
- Create: `src/perception_env/live_server.py`
- Create: `tests/test_live_server.py`
- Modify: `src/perception_env/cli.py`

**Interfaces:**
- Consumes: `LiveInferenceEngine`.
- Produces: static GET serving plus `POST /api/infer` and CLI command `perception-env live-demo`.

- [ ] Write failing request/response, validation, and 422 error tests with an injected engine.
- [ ] Implement the server and CLI wiring with checkpoints loaded once.
- [ ] Run focused tests and commit.

### Task 3: Replace fixed rollout selection with parameter controls

**Files:**
- Modify: `explorer.html`
- Modify: `tests/test_explorer.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `POST /api/infer` JSON.
- Produces: gravity, wind, and damping controls; run button; belief overlay; explicit ground-truth-versus-guessed table.

- [ ] Write failing UI contract tests for parameter controls, request payload, loading/error states, and comparison output.
- [ ] Implement the controls and remove the prerecorded rollout selector.
- [ ] Start the live server, verify two parameter settings visually, then run `pytest -q`, `ruff check .`, and `git diff --check`.
- [ ] Commit and document the one-command launch path.
