# Research Demo and Reproducibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship one reproducible command and one visual dashboard that demonstrate ground truth, Model 1 belief superpositions, Model 2 second-order world inference, iterative correction, and the true-versus-recovered complexity limit.

**Architecture:** A versioned experiment configuration composes the trained Model 1 and Model 2 checkpoints with the streaming evaluator. The run writes immutable arrays, checkpoints, metrics, and a manifest first; a separate renderer builds the browser dashboard exclusively from those saved artifacts.

**Tech Stack:** Python 3.11, PyTorch, NumPy, Pillow, Matplotlib, pytest, static HTML/CSS/JavaScript.

**Spec:** `docs/superpowers/specs/2026-10-30-parameterized-occlusion-environment-design.md`

## Global Constraints

- One ball and deterministic toy physics only.
- Model 2 receives belief histories and belief-derived summaries only; no raw frames, coordinates, masks, detector features, or direct outcome marker.
- Every result is reproducible from an explicit seed, configuration, dataset version, and checkpoint hash.
- Every dashboard panel is rendered from saved records without rerunning simulation or inference.
- Complexity claims are empirical and search-relative, never presented as a formal VC-dimension result.

## Review Focus

- Missing checkpoints or dataset files must fail with an actionable error before an output directory is replaced.
- A dashboard built from a different checkpoint must receive a different deterministic manifest hash.
- The published HTML must work from a local static server without external URLs.
- A quick CPU configuration must finish end to end and exercise both learned models.
- The production Model 2 export must reject first-order fields even when an ablation configuration enables them elsewhere.

---

### Task 1: Freeze the experiment bundle schema

**Files:**
- Create: `src/perception_env/experiment.py`
- Create: `tests/test_experiment.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: Model 1 and Model 2 checkpoint metadata, streaming episode records, and complexity reports from Subplans 2–5.
- Produces: `ExperimentConfig`, `ExperimentManifest`, `save_experiment_bundle(path, ...)`, and `load_experiment_bundle(path)`.

- [ ] **Step 1: Write failing round-trip, missing-input, and content-hash tests** in `tests/test_experiment.py`.
- [ ] **Step 2: Run `pytest tests/test_experiment.py -v` and verify the new imports fail.**
- [ ] **Step 3: Implement typed configuration, manifest validation, and content hashing** in `src/perception_env/experiment.py`.
- [ ] **Step 4: Run `pytest tests/test_experiment.py -v` and verify it passes.**
- [ ] **Step 5: Commit with `git commit -m "feat: add reproducible experiment bundles"`.**

### Task 2: Add the end-to-end experiment command

**Files:**
- Create: `src/perception_env/run_experiment.py`
- Create: `tests/test_run_experiment.py`
- Modify: `src/perception_env/cli.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, trained checkpoint loaders, `run_streaming_episode(...)`, and complexity evaluators.
- Produces: `run_experiment(config, output) -> ExperimentManifest` and CLI command `perception-env run-experiment`.

- [ ] **Step 1: Write failing tests for quick CPU execution, preflight validation, deterministic replay, and first-order leakage rejection.**
- [ ] **Step 2: Run `pytest tests/test_run_experiment.py -v` and verify failures name the missing command and function.**
- [ ] **Step 3: Implement the orchestration path** so arrays and metrics are saved before visual rendering begins.
- [ ] **Step 4: Run `pytest tests/test_run_experiment.py -v` and verify it passes.**
- [ ] **Step 5: Commit with `git commit -m "feat: run the learned perceptual loop end to end"`.**

### Task 3: Render the three-level visual narrative

**Files:**
- Create: `src/perception_env/dashboard.py`
- Create: `tests/test_dashboard.py`
- Create: `src/perception_env/assets/dashboard.css`
- Create: `src/perception_env/assets/dashboard.js`

**Interfaces:**
- Consumes: a loaded `ExperimentManifest` and saved arrays only.
- Produces: `render_dashboard(bundle_path, output_path) -> Path` with panels named `Ground truth`, `Model 1 · Belief superposition`, and `Model 2 · Second-order perception`.

- [ ] **Step 1: Write failing tests for the three visual roles, offline assets, saved-record-only rendering, and accessible image descriptions.**
- [ ] **Step 2: Run `pytest tests/test_dashboard.py -v` and verify the renderer is absent.**
- [ ] **Step 3: Implement the static dashboard renderer** with a streaming timeline and true-versus-recovered complexity plot.
- [ ] **Step 4: Run `pytest tests/test_dashboard.py -v` and inspect the quick-run page at desktop and narrow widths.**
- [ ] **Step 5: Commit with `git commit -m "feat: render the learned-model research dashboard"`.**

### Task 4: Freeze a reproducible quick benchmark

**Files:**
- Create: `configs/experiments/quick-v1.yaml`
- Create: `configs/experiments/benchmark-v1.yaml`
- Create: `tests/test_reproducibility.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: the full `run-experiment` and dashboard pipeline.
- Produces: documented commands for a small CPU smoke run and a longer benchmark run.

- [ ] **Step 1: Write a failing subprocess test that runs `quick-v1` twice and compares scientific content hashes.**
- [ ] **Step 2: Run `pytest tests/test_reproducibility.py -v` and verify the configurations are missing.**
- [ ] **Step 3: Add frozen configurations and concise run/inspect documentation.**
- [ ] **Step 4: Run `pytest -q`, `ruff check .`, and the quick command twice; verify all checks pass and hashes match.**
- [ ] **Step 5: Commit with `git commit -m "test: freeze the reproducible learned-model demo"`.**

