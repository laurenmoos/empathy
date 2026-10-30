# Coherence and Complexity Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how faithfully the streaming system preserves possible/impossible judgments over time and identify the first operator-family level whose empirical complexity the learned system no longer recovers.

**Architecture:** Every true and predicted world is evaluated on the same registered probe bank, producing event-indexed possibility-label tensors and family-level world-by-probe matrices. Temporal coherence is the mean true-versus-predicted possibility agreement over valid events and probes; outcome coverage and prediction stability are reported separately so they cannot conceal one another. Repeated, search-relative shattering estimates for true and recovered world banks receive seeded bootstrap intervals, then a configured rule locates the first family level below both scientific acceptance thresholds.

**Tech Stack:** Python 3.11, NumPy, existing reachability/shattering/identifiability modules, PyYAML, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-parameterized-occlusion-environment-design.md`

## Global Constraints

- All true-versus-recovered comparisons use the identical contexts, entry-state grids, probes, tolerances, search budget, and registered seeds.
- Ground-truth complexity is computed from physical-world possibility labels, never from Model 1 samples or observed outcome frequency.
- Recovered complexity is computed by decoding Model 2 predictions into valid `WorldCode` values and rerunning the same finite reachability oracle.
- The primary coherence measure is possibility-label agreement over time; exact parameter error is a secondary identifiable-only diagnostic.
- Every shattering result is named empirical and search-relative and records operator support, sampling methods and counts, uncertainty grid, tolerance, algorithm, budget, seeds, and interval method.
- Context-conditioned and global multi-context estimates are stored under different metric names and are never pooled.
- All aggregate reports and figures are regenerated from saved label arrays and metadata without rerunning models.

## Review Focus

- Empty valid-event masks and family levels with no accepted predictions must return explicit `insufficient_data`, not NaN-filled success reports.
- A recovered world that is parameter-different but possibility-equivalent must receive full possibility agreement while remaining ineligible for exact-parameter credit unless identifiable.
- Shattering recovery ratios must handle true dimension zero without division by zero or artificially perfect capacity claims.
- Bootstrap intervals and search results must be invariant to input row ordering for the same seed and registered world IDs.
- A threshold crossing at one level followed by apparent recovery at a harder level must still mark the first crossing and visibly report the non-monotonicity.

---

### Task 1: Define temporal perceptual coherence

**Files:**
- Create: `src/perception_env/coherence.py`
- Create: `tests/test_coherence.py`

**Interfaces:**
- Consumes: Boolean `true_labels` and `predicted_labels` shaped `[event, probe]`, Boolean `valid_events`, Boolean `outcome_covered`, event IDs, probe IDs, and family levels.
- Produces: `TemporalCoherenceReport` and `compute_temporal_coherence(...)`.

- [ ] **Step 1: Write failing metric tests**

  Add exact fixtures for perfect agreement (`coherence=1`), one disagreement among eight valid judgments (`coherence=0.875`), invalid-event exclusion, outcome coverage independent of label agreement, and prediction stability equal to one minus the fraction of adjacent valid predicted-label flips. Add an empty-mask test expecting status `insufficient_data` and no scalar score.

- [ ] **Step 2: Run the tests and verify RED**

  Run: `.venv/bin/pytest tests/test_coherence.py -q`

  Expected: FAIL because the coherence module is missing.

- [ ] **Step 3: Implement the report and metric**

  Add:

  `compute_temporal_coherence(true_labels: np.ndarray, predicted_labels: np.ndarray, valid_events: np.ndarray, outcome_covered: np.ndarray, event_ids: Sequence[str], probe_ids: Sequence[str]) -> TemporalCoherenceReport`.

  The report stores per-event agreement, aggregate agreement, coverage rate, adjacent-event stability, valid counts, and status. Validate shapes and Boolean dtypes; never coerce nonbinary floats silently.

- [ ] **Step 4: Run focused and full tests**

  Run: `.venv/bin/pytest tests/test_coherence.py -q && .venv/bin/pytest -q`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/perception_env/coherence.py tests/test_coherence.py
  git commit -m "feat: define temporal possibility coherence"
  ```

### Task 2: Build matched true and recovered possibility banks

**Files:**
- Create: `src/perception_env/recovery.py`
- Create: `tests/test_recovery.py`

**Interfaces:**
- Consumes: saved streaming predictions from Subplan 4, `WorldSchema`, `FamilyConfig`, registered contexts and `Probe` values, `enumerate_reachable`, and the valid world decoder from Subplan 3.
- Produces: `RecoveredWorldBank`, `MatchedPossibilityBank`, and `build_matched_possibility_bank(...)`.

- [ ] **Step 1: Write failing matched-bank tests**

  Add fixtures proving the true and predicted tensors share `[event, probe]` shape and probe identity, decoding failures are retained with `valid_events=False`, equivalent world codes get identical labels, and raw frames/belief images are neither accepted nor persisted by this evaluation API.

- [ ] **Step 2: Run the tests and verify RED**

  Run: `.venv/bin/pytest tests/test_recovery.py -q`

  Expected: FAIL because the recovery module is missing.

- [ ] **Step 3: Implement matched oracle evaluation**

  Add:

  `build_matched_possibility_bank(true_worlds: Sequence[WorldCode], predicted_worlds: Sequence[WorldCode | None], contexts: Sequence[EvaluationContext], probes: Sequence[Probe]) -> MatchedPossibilityBank`.

  `EvaluationContext` pins entry grid, simulation config, query steps, and context ID. Store labels plus world/probe/context IDs and full oracle metadata. Require each probe to match its context and use one oracle route for both banks.

- [ ] **Step 4: Run focused and full tests**

  Run: `.venv/bin/pytest tests/test_recovery.py -q && .venv/bin/pytest -q`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/perception_env/recovery.py tests/test_recovery.py
  git commit -m "feat: compare matched recovered possibilities"
  ```

### Task 3: Estimate true and recovered shattering with intervals

**Files:**
- Modify: `src/perception_env/shattering.py`
- Create: `tests/test_recovered_shattering.py`

**Interfaces:**
- Consumes: paired family-level `LabelMatrix` values from Task 2, registered search seeds, `max_n`, budget, and `same_context`.
- Produces: `ShatteringInterval`, `ComplexityRecoveryEstimate`, and `estimate_complexity_recovery(...)`.

- [ ] **Step 1: Write failing synthetic calibration tests**

  Use small Boolean classes with known dimensions 0, 1, and 2. Assert true and recovered searches share seed/budget metadata, intervals contain every repeated estimate, identical matrices have equal estimates, a collapsed recovered matrix has lower recovered dimension, and zero-dimensional true classes return ratio status `undefined_zero_true_dimension` rather than a numeric divide-by-zero result.

- [ ] **Step 2: Run the tests and verify RED**

  Run: `.venv/bin/pytest tests/test_recovered_shattering.py -q`

  Expected: FAIL because paired recovery estimates are missing.

- [ ] **Step 3: Implement repeated paired estimates**

  Add:

  `estimate_complexity_recovery(true_matrix: LabelMatrix, recovered_matrix: LabelMatrix, *, max_n: int, budget: int, search_seeds: tuple[int, ...], bootstrap_seed: int, bootstrap_samples: int, same_context: bool = True) -> ComplexityRecoveryEstimate`.

  Canonically sort rows by world ID before seeded work. Report repeated dimensions, median dimensions, percentile 95% intervals, recovered/true ratio when defined, matrix label agreement, metric name, and complete search metadata.

- [ ] **Step 4: Run focused and full tests**

  Run: `.venv/bin/pytest tests/test_recovered_shattering.py -q && .venv/bin/pytest -q`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/perception_env/shattering.py tests/test_recovered_shattering.py
  git commit -m "feat: estimate recovered shattering intervals"
  ```

### Task 4: Locate the operator-complexity capacity breakpoint

**Files:**
- Modify: `src/perception_env/coherence.py`
- Create: `tests/test_capacity_breakpoint.py`
- Create: `configs/benchmarks/coherence-v1.yaml`

**Interfaces:**
- Consumes: ordered family levels, one `TemporalCoherenceReport` and `ComplexityRecoveryEstimate` per level, plus `CapacityThresholds(min_coherence=0.90, min_shattering_ratio=0.80)`.
- Produces: `CapacityFrontier` and `find_capacity_breakpoint(...)`.

- [ ] **Step 1: Write failing threshold and edge-case tests**

  Test no crossing, crossing caused by coherence alone, crossing caused by complexity alone, simultaneous crossing, insufficient data, true dimension zero, and non-monotonic later recovery. Assert the first failing family remains the breakpoint and non-monotonic recovery is listed in `warnings`.

- [ ] **Step 2: Run the tests and verify RED**

  Run: `.venv/bin/pytest tests/test_capacity_breakpoint.py -q`

  Expected: FAIL because the frontier API is absent.

- [ ] **Step 3: Implement the configured decision rule**

  Add:

  `find_capacity_breakpoint(levels: Sequence[str], coherence: Mapping[str, TemporalCoherenceReport], complexity: Mapping[str, ComplexityRecoveryEstimate], thresholds: CapacityThresholds) -> CapacityFrontier`.

  A level passes only when aggregate coherence is at least 0.90 and the lower 95% bound of the defined shattering ratio is at least 0.80. Zero-true-dimension levels are descriptive and cannot cause a ratio failure. Keep raw values and reason codes in the result.

- [ ] **Step 4: Run focused and full tests**

  Run: `.venv/bin/pytest tests/test_capacity_breakpoint.py -q && .venv/bin/pytest -q`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/perception_env/coherence.py tests/test_capacity_breakpoint.py configs/benchmarks/coherence-v1.yaml
  git commit -m "feat: locate perceptual capacity breakpoint"
  ```

### Task 5: Persist benchmark reports and render scientific visuals

**Files:**
- Create: `src/perception_env/evaluation_records.py`
- Modify: `src/perception_env/visualize.py`
- Create: `tests/test_coherence_visuals.py`

**Interfaces:**
- Consumes: matched possibility banks, coherence reports, complexity estimates, capacity frontier, identifiability reports, and registered configurations.
- Produces: `save_evaluation_bundle`, `load_evaluation_bundle`, `render_coherence_trace`, `render_shattering_recovery_curve`, `render_capacity_frontier`, `render_recovery_label_matrices`, and `render_breakpoint_failure_gallery`.

- [ ] **Step 1: Write failing persistence and visual tests**

  Round-trip a two-level fixture and assert all numeric arrays, IDs, thresholds, seeds, and metadata survive. Render solely from the reloaded bundle and assert nonblank images with expected titles/legends. The failure gallery must include true/recovered beliefs and world codes only for events stored in the bundle.

- [ ] **Step 2: Run the tests and verify RED**

  Run: `.venv/bin/pytest tests/test_coherence_visuals.py -q`

  Expected: FAIL because evaluation records and renderers are absent.

- [ ] **Step 3: Implement complete saved evaluation artifacts**

  Save arrays in NPZ and JSON-safe metadata separately. Figures must show 95% intervals, search budget, family level, sample counts, and thresholds. True and recovered label matrices use the same row/column ordering and color scale. The capacity plot marks the first crossing and any later non-monotonic recovery.

- [ ] **Step 4: Run focused and full tests**

  Run: `.venv/bin/pytest tests/test_coherence_visuals.py -q && .venv/bin/pytest -q`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/perception_env/evaluation_records.py src/perception_env/visualize.py tests/test_coherence_visuals.py
  git commit -m "feat: visualize coherence and capacity recovery"
  ```

### Task 6: Add the end-to-end complexity benchmark command

**Files:**
- Modify: `src/perception_env/cli.py`
- Modify: `README.md`
- Create: `tests/test_coherence_cli.py`

**Interfaces:**
- Consumes: saved Subplan 4 runs across W0–W5, `configs/benchmarks/coherence-v1.yaml`, registered seeds, and Tasks 1–5.
- Produces: `perception-env evaluate-coherence --runs ... --config ... --output ...`, `summary.json`, canonical arrays, figures, manifest, and `index.html`.

- [ ] **Step 1: Write failing CLI tests**

  Test a tiny synthetic fixture and a `--quick` saved-stream fixture. Assert the CLI never loads raw models, validates that all runs share schema/probe/config versions, rejects overlapping input/output paths before deletion, and records thresholds, search budgets, input content hashes, seeds, confidence intervals, and breakpoint reason.

- [ ] **Step 2: Run the tests and verify RED**

  Run: `.venv/bin/pytest tests/test_coherence_cli.py -q`

  Expected: FAIL because the command is absent.

- [ ] **Step 3: Implement the evaluation command and documentation**

  The command reads saved prediction/world records, regenerates only physical reachability labels, computes all reports, saves the evaluation bundle, and renders the five canonical visual artifacts. Document that the number is an empirical, context- and budget-conditioned complexity recovery result, not a formal VC-dimension proof.

- [ ] **Step 4: Verify the complete stage**

  Run: `.venv/bin/pytest -q && .venv/bin/ruff check .`

  Run the quick command twice with the same inputs and compare output arrays and deterministic hashes.

  Expected: all tests pass and outputs match.

- [ ] **Step 5: Commit**

  ```bash
  git add src/perception_env/cli.py README.md tests/test_coherence_cli.py
  git commit -m "feat: benchmark temporal complexity recovery"
  ```

## Completion Gate

- Temporal coherence, outcome coverage, and prediction stability are independently visible for every family level.
- True and recovered empirical shattering estimates share the same probes, contexts, budgets, and seeds and include 95% intervals.
- The first failing operator-family level is reported with an auditable threshold reason.
- Exact parameter accuracy is scored only for identifiable examples; possibility-equivalent worlds receive correct possibility credit.
- Every headline number can be reconstructed from the saved evaluation bundle, and the full test suite plus Ruff pass.
