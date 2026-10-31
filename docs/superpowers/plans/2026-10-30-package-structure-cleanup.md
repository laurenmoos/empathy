# Package Structure Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove generated repository bulk and reorganize the remaining implementation into legible physics, reasoning, live, Model 1, and Model 2 subsystems without changing the demo's behavior.

**Architecture:** Preserve the one required learned checkpoint at a stable tracked path, remove the unused Model 2 checkpoint interface, then mechanically move flat modules into responsibility-based subpackages. Keep module responsibilities intact and update every source/test import rather than adding compatibility wrappers.

**Tech Stack:** Python 3.11, Hatchling, PyTorch, NumPy, pytest, Ruff

**Spec:** `docs/superpowers/specs/2026-10-30-package-structure-cleanup-design.md`

## Global Constraints

- Preserve `perception-env = perception_env.cli:main` and every current command.
- Preserve Pinball UI and live HTTP behavior.
- Preserve Model 1 and Model 2 training behavior.
- Remove only the unused `--model2-checkpoint` live option.
- Do not touch the user's uncommitted `README.md` or `docs/images/` content.
- Keep all six tracked YAML configuration presets.
- Do not add flat compatibility modules or opportunistic behavior changes.

## Review Focus

- A default `live-demo` launch must resolve a real Model 1 checkpoint after `artifacts/` is deleted.
- Root and nested CLI help must still work, with `--model2-checkpoint` absent only from `live-demo`.
- No source or test may import an old flat module path after the moves.
- The lazy live inference/thinking cycle must remain lazy and import successfully.
- A clean checkout without generated artifacts or caches must pass the complete suite.

---

### Task 1: Preserve the one runtime asset and remove the dead checkpoint option

**Files:**
- Create: `checkpoints/model1.pt`
- Modify: `src/perception_env/cli.py`
- Modify: `src/perception_env/live_inference.py` (later moved to `live/inference.py`)
- Modify: `src/perception_env/live_server.py` (later moved to `live/server.py`)
- Modify: `tests/test_cli.py`
- Modify: `tests/test_live_server.py`

**Interfaces:**
- Produces: `LiveInferenceEngine.from_checkpoint(model1_checkpoint: Path)`
- Produces: `run_live_server(host: str, port: int, root: Path, model1_checkpoint: Path)`
- Produces: default checkpoint path `checkpoints/model1.pt`

- [ ] **Step 1: Add failing CLI tests**

Assert that `live-demo` defaults to an existing `checkpoints/model1.pt`, exposes `--model1-checkpoint`, does not expose `--model2-checkpoint`, and delegates only the Model 1 path to `run_live_server`.

- [ ] **Step 2: Run the focused tests and confirm the old two-checkpoint interface fails them**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_cli.py tests/test_live_server.py`

- [ ] **Step 3: Preserve the checkpoint and simplify the live interface**

Copy the exact current Model 1 checkpoint bytes to `checkpoints/model1.pt`. Rename `from_checkpoints` to `from_checkpoint`, remove its discarded Model 2 argument, remove the CLI option, and update server construction.

- [ ] **Step 4: Run the focused tests**

Expected: all focused tests pass.

- [ ] **Step 5: Commit the runtime asset and interface cleanup**

Commit only the checkpoint, source, and focused tests; leave user-owned files unstaged.

---

### Task 2: Group flat source modules into coherent subpackages

**Files:**
- Create: `src/perception_env/physics/__init__.py`
- Move to `src/perception_env/physics/`: `config.py`, `world.py`, `operators.py`, `simulate.py`, `render.py`, `events.py`, `families.py`
- Create: `src/perception_env/reasoning/__init__.py`
- Move to `src/perception_env/reasoning/`: `beliefs.py`, `reachability.py`, `recovery.py`, `identifiability.py`, `shattering.py`, `coherence.py`, `records.py`, `visualize.py`
- Create: `src/perception_env/live/__init__.py`
- Move/rename to `src/perception_env/live/`: `live_inference.py` → `inference.py`, `live_thinking.py` → `thinking.py`, `live_server.py` → `server.py`
- Modify: `src/perception_env/cli.py`
- Modify: all affected modules under `src/perception_env/model1/` and `src/perception_env/model2/`
- Modify: affected tests under `tests/`
- Create: `tests/test_package_layout.py`

**Interfaces:**
- Produces: `perception_env.physics.*`, `perception_env.reasoning.*`, and `perception_env.live.*`
- Preserves: `perception_env.cli:main`, `perception_env.model1.*`, and `perception_env.model2.*`

- [ ] **Step 1: Add a failing package-layout test**

Assert the three new packages and their expected modules exist/import, the old flat source files do not exist, CLI construction succeeds, and the live inference/thinking modules import in either order.

- [ ] **Step 2: Run the layout test and confirm it fails before the moves**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_package_layout.py`

- [ ] **Step 3: Move modules and add minimal package initializers**

Use mechanical moves only. Each new `__init__.py` contains a package docstring and no eager re-exports.

- [ ] **Step 4: Update imports mechanically and exhaustively**

Update source, tests, lazy imports, monkeypatch targets, and `TYPE_CHECKING` imports to the new paths. Preserve the existing lazy cycle between live inference and thinking. Do not rewrite historical documents.

- [ ] **Step 5: Prove no stale imports remain**

Search source and tests for all 18 retired flat paths and old filenames; expect zero matches except intentional textual assertions in `test_package_layout.py`.

- [ ] **Step 6: Run focused subsystem tests**

Run the package-layout, physics/reasoning root tests, live tests, `tests/model1`, and `tests/model2` with pytest cache and bytecode disabled.

- [ ] **Step 7: Run Ruff and the complete test suite**

Run: `.venv/bin/python -m ruff check --no-cache src tests`

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`

- [ ] **Step 8: Commit the package reorganization**

Stage only moved source, affected tests, and package initializers.

---

### Task 3: Delete obsolete and generated bulk, then verify a clean checkout

**Files:**
- Delete: `learned-example.json`
- Delete ignored generated paths: `artifacts/`, caches, and empty `configs/streaming/`
- Delete after final review: ignored `.superpowers/`
- Keep: all tracked files under `configs/`

**Interfaces:**
- Consumes: tracked `checkpoints/model1.pt` from Task 1
- Produces: repository with no required runtime dependency under ignored output directories

- [ ] **Step 1: Verify `learned-example.json` has no runtime or test references**

Search source, tests, HTML, README, and project configuration; expect zero references.

- [ ] **Step 2: Delete the tracked unused fixture and exact generated directories**

Delete only the paths listed above. Do not touch `docs/images/`, `README.md`, YAML configs, or historical docs.

- [ ] **Step 3: Verify one-click startup dependencies**

Assert the CLI default checkpoint exists, load it through `LiveInferenceEngine.from_checkpoint`, create the server on an ephemeral port, and close it without serving indefinitely.

- [ ] **Step 4: Run final static and behavioral verification**

Run Ruff without cache, `git diff --check`, root/nested CLI help, the live-server smoke test, and the complete pytest suite with bytecode and pytest cache disabled.

- [ ] **Step 5: Commit tracked cleanup**

Commit `learned-example.json` deletion and any final test adjustment without staging user-owned changes.

- [ ] **Step 6: Request whole-change review, resolve substantive findings once, then remove generated review scratch**

After review is clean, delete only `.superpowers/` and any regenerated caches. Repeat cache-free Ruff and pytest verification on the exact final tree.
