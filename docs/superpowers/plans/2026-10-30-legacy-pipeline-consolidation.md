# Legacy Pipeline Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the superseded streaming/export prototype while preserving the current Pinball live demo and both learned-model pipelines.

**Architecture:** Retire one isolated top-level execution lane rather than merging focused Model 1, Model 2, simulation, and evaluation modules. The root CLI will expose `live-demo` as the interactive path and will no longer expose `stream-demo`; current live inference remains in `live_inference.py`, `live_thinking.py`, and `live_server.py`.

**Tech Stack:** Python 3.14, pytest, Ruff, vanilla HTML/JavaScript

**Spec:** `docs/superpowers/specs/2026-10-30-legacy-pipeline-consolidation-design.md`

## Global Constraints

- Preserve the Pinball interface and all `live-demo` API schemas and behavior.
- Preserve Model 1 and Model 2 training, evaluation, visualization, and package-local CLIs.
- Preserve simulation, reachability, shattering, identifiability, recovery, and coherence code.
- Do not modify or stage the user's existing `README.md` and `docs/images/` changes.
- Leave historical plan documents intact.
- Because the repository moved, run Python checks with `PYTHONPATH=/Users/laurenmoos/perception/src`, disable pytest's cache provider, and pass `--no-cache` to Ruff.

## Review Focus

- Root help must list `live-demo` and must not list retired `stream-demo`.
- `live-demo` must still construct its parser and delegate to `run_live_server` unchanged.
- No surviving source or test may import a deleted legacy module.
- The browser must continue to display runtime failures directly without the removed narration helper.
- Model 1's current `model1/stream_train.py` path must remain untouched and its tests must pass.

---

### Task 1: Remove the unused browser narration helper

**Files:**
- Delete: `explorer_state.js`
- Modify: `tests/test_explorer.py`

**Interfaces:**
- Consumes: inline error rendering through `statusMessage` in `explorer.html`
- Produces: an explorer test suite with no dependency on the retired `PerceptionUIState.frameNarration` helper

- [ ] **Step 1: Confirm the helper is not loaded by the product**

Run: `rg -n "explorer_state|PerceptionUIState|frameNarration" explorer.html src`

Expected: no product matches; only the helper file and its dedicated test refer to `frameNarration`.

- [ ] **Step 2: Remove the obsolete test dependency**

In `tests/test_explorer.py`, remove the `subprocess` import, `EXPLORER_STATE`, the two assertions that read the helper, and `test_frame_narration_preserves_stream_failures`. Keep `test_normal_playback_is_visual_without_narration_feeds`, which already protects the current inline error behavior.

- [ ] **Step 3: Delete the unused helper**

Delete `explorer_state.js` with `apply_patch`.

- [ ] **Step 4: Verify the explorer remains covered**

Run: `env PYTHONPATH=/Users/laurenmoos/perception/src .venv/bin/python -m pytest tests/test_explorer.py -q -p no:cacheprovider`

Expected: all remaining explorer tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_explorer.py explorer_state.js
git commit -m "refactor: remove obsolete explorer narration helper"
```

### Task 2: Retire the superseded streaming/export lane

**Files:**
- Modify: `src/perception_env/cli.py`
- Modify: `tests/test_cli.py`
- Delete: `src/perception_env/continual.py`
- Delete: `src/perception_env/streaming.py`
- Delete: `src/perception_env/stream_records.py`
- Delete: `src/perception_env/streaming_demo.py`
- Delete: `src/perception_env/export_learned_example.py`
- Delete: `configs/streaming/v1.yaml`
- Delete: `tests/test_continual.py`
- Delete: `tests/test_streaming.py`
- Delete: `tests/test_stream_records.py`
- Delete: `tests/test_streaming_demo.py`
- Delete: `tests/test_streaming_visuals.py`
- Delete: `tests/test_export_learned_example.py`

**Interfaces:**
- Consumes: root CLI parser `_parser()` and dispatcher `main(argv)`
- Produces: root CLI help containing `live-demo` but not `stream-demo`; no legacy module import surface

- [ ] **Step 1: Write the failing CLI retirement assertion**

Update `test_help_lists_all_research_commands` in `tests/test_cli.py` so its required commands include `live-demo` and exclude `stream-demo`, then add `assert "stream-demo" not in result.stdout`. Remove `test_stream_demo_command_builds_visual_timeline` and its now-unused `json` use only if no later test requires that import.

- [ ] **Step 2: Run the CLI test and verify the intended failure**

Run: `env PYTHONPATH=/Users/laurenmoos/perception/src .venv/bin/python -m pytest tests/test_cli.py::test_help_lists_all_research_commands -q -p no:cacheprovider`

Expected: FAIL because current help still contains `stream-demo`.

- [ ] **Step 3: Remove the retired CLI branch**

In `src/perception_env/cli.py`, remove the `stream-demo` parser and its `main()` dispatch branch. Do not alter the adjacent `live-demo` parser or dispatch.

- [ ] **Step 4: Delete the isolated legacy modules, configuration, and tests**

Delete every file listed for deletion in this task with `apply_patch`. Do not delete `src/perception_env/model1/stream_train.py` or `configs/model1/stream_v1.yaml`.

- [ ] **Step 5: Prove no surviving importer remains**

Run: `rg -n "perception_env\.(continual|streaming|stream_records|streaming_demo|export_learned_example)|from [.]?(continual|streaming|stream_records|streaming_demo|export_learned_example)" src tests`

Expected: no matches.

- [ ] **Step 6: Verify the current streaming and live paths**

Run: `env PYTHONPATH=/Users/laurenmoos/perception/src .venv/bin/python -m pytest tests/test_cli.py tests/test_live_inference.py tests/test_live_thinking.py tests/test_live_server.py tests/model1/test_stream_train.py -q -p no:cacheprovider`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/perception_env/cli.py tests/test_cli.py src/perception_env/continual.py src/perception_env/streaming.py src/perception_env/stream_records.py src/perception_env/streaming_demo.py src/perception_env/export_learned_example.py configs/streaming/v1.yaml tests/test_continual.py tests/test_streaming.py tests/test_stream_records.py tests/test_streaming_demo.py tests/test_streaming_visuals.py tests/test_export_learned_example.py
git commit -m "refactor: retire legacy streaming prototype"
```

### Task 3: Verify the consolidated package

**Files:**
- No production changes expected

**Interfaces:**
- Consumes: the complete surviving package and test suite
- Produces: evidence that the cleanup removed only the approved lane

- [ ] **Step 1: Recount the surviving source surface**

Run: `rg --files src/perception_env -g '*.py' | wc -l && wc -l $(rg --files src/perception_env -g '*.py') | tail -1`

Expected: five fewer Python modules and roughly 1,337 fewer Python source lines than the audit baseline.

- [ ] **Step 2: Run static checks**

Run: `.venv/bin/python -m ruff check --no-cache src tests`

Expected: `All checks passed!`

- [ ] **Step 3: Run the full suite**

Run: `env PYTHONPATH=/Users/laurenmoos/perception/src .venv/bin/python -m pytest -q -p no:cacheprovider`

Expected: all remaining tests pass with no failures or errors.

- [ ] **Step 4: Inspect the final diff and status**

Run: `git diff --check && git status --short && git diff --stat HEAD~2..HEAD`

Expected: only the approved retirement files, CLI/test edits, spec, and plan are included; `README.md` and `docs/images/` remain unstaged user changes.
