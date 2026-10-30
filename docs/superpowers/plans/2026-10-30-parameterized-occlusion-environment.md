# Parameterized Occlusion Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic one-ball research environment whose operator families, belief superpositions, possible/impossible judgments, and empirical shattering behavior are numerically testable and visually inspectable.

**Architecture:** A typed world code configures a small ordered operator bank and deterministic fixed-step simulator. Separate observation, reachability, belief, dataset, shattering, and visualization modules preserve the distinction between latent physics, first-order perception, and second-order belief inputs. Saved records are the shared boundary: every metric and visual is regenerated from them without rerunning simulation.

**Tech Stack:** Python 3.11, NumPy, PyTorch tensors at model boundaries, Pillow, PyYAML, Matplotlib, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-10-30-parameterized-occlusion-environment-design.md`

## Global Constraints

- One moving ball and static geometry only.
- Coordinates use normalized \([0,1]\) space and fixed-step semi-implicit Euler integration.
- Update order is acceleration, damping, velocity integration, position integration, then collision resolution.
- Occlusion changes observation only and must never mutate latent state.
- Model 2 records contain only four-event histories of pre-belief, post-belief, and belief difference; they contain no raw frames, ball coordinates, masks, detector features, or direct reappearance markers.
- World codes have fixed-width activity flags and normalized parameter blocks with a versioned field order.
- Active continuous-operator ranges exclude a neutral interval so active-zero is not silently equivalent to inactive.
- Canonical images are 64-by-64 grayscale; resolution remains configurable.
- Binary occupancy is canonical; density images are derived.
- Every numerical subsystem has a deterministic visual proof generated from saved records.
- All empirical shattering results are labeled search-relative and include sampling and search budgets.
- Samples without a fully hidden interval and visible reappearance are rejected with a recorded reason.
- A rollout that leaves \([0,1]^2\) without a configured boundary terminates and is invalid for benchmark generation.

## Review Focus

- Operator ordering at simultaneous force and collision events must be deterministic and agree with the documented update order; Task 2 adds a compound-event regression test.
- Active flags paired with neutral parameters must be rejected rather than producing ambiguous world codes; Task 1 adds schema validation tests.
- Empty or saturated reachable sets must rasterize and visualize without divide-by-zero or misleading density normalization; Task 4 adds both cases.
- Probe sets mixing contexts must never be reported as the primary context-conditioned shattering estimate; Task 6 tests metric naming and metadata.
- Saved records containing forbidden first-order fields in Model 2 inputs must fail validation; Task 5 adds an explicit leakage test.

---

## File map

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, pytest, and Ruff configuration. |
| `src/perception_env/config.py` | Typed global, rendering, uncertainty, and benchmark configurations. |
| `src/perception_env/world.py` | Operator slots, world-code validation, flattening, normalization, and versioning. |
| `src/perception_env/operators.py` | Pure per-step operator functions and documented execution order. |
| `src/perception_env/simulate.py` | Deterministic rollout, termination, and trajectory result types. |
| `src/perception_env/render.py` | Frame rendering, visibility masks, and occlusion. |
| `src/perception_env/events.py` | Occlusion interval and reappearance-event extraction. |
| `src/perception_env/reachability.py` | Hidden-entry enumeration, reachability labels, and endpoint rasterization. |
| `src/perception_env/beliefs.py` | Weighted trajectory superpositions and four-event pre/post belief histories. |
| `src/perception_env/families.py` | Nested complexity levels, membership, and reproducible world sampling. |
| `src/perception_env/records.py` | Versioned saved-record schemas, validation, and array/metadata persistence. |
| `src/perception_env/shattering.py` | Label matrices, growth counts, probe-set search, and repeated estimates. |
| `src/perception_env/identifiability.py` | Perceptual-equivalence grouping and parameter-recovery eligibility. |
| `src/perception_env/visualize.py` | World, rollout, belief, reachability, shattering, and equivalence figures. |
| `src/perception_env/cli.py` | Reproducible generation, inspection, and benchmark commands. |
| `configs/world_families/v1.yaml` | Versioned operator supports, ladder rules, and sampling distributions. |
| `tests/` | Unit, contract, integration, snapshot-metadata, and CLI tests mirroring source modules. |

### Task 1: Package foundation and versioned world code

**Files:**
- Create: `pyproject.toml`
- Create: `src/perception_env/__init__.py`
- Create: `src/perception_env/config.py`
- Create: `src/perception_env/world.py`
- Create: `src/perception_env/visualize.py`
- Create: `tests/test_world.py`
- Create: `tests/test_world_visuals.py`

**Interfaces:**
- Consumes: none.
- Produces: `OperatorSlot`, `WorldCode`, `WorldSchema`, `SimulationConfig`, `RenderConfig`, `WorldCode.to_flat() -> np.ndarray`, `WorldSchema.from_flat(values: np.ndarray) -> WorldCode`, and `render_world_card(world: WorldCode, path: Path) -> Path`.

- [ ] **Step 1: Write failing world-code tests**

  Test that schema version `v1` has fixed field order, flatten/unflatten round-trips, inactive slots canonicalize parameters, active values normalize correctly, malformed widths fail, and active values inside each configured neutral interval fail validation.

- [ ] **Step 2: Run the focused tests and confirm failure**

  Run: `pytest tests/test_world.py -v`

  Expected: FAIL because `perception_env.world` does not exist.

- [ ] **Step 3: Implement the typed configuration and world-code contracts**

  Implement immutable dataclasses and explicit validation. Use operator slots `vertical_acceleration`, `horizontal_acceleration`, `damping`, `vertical_wall`, and `horizontal_wall`; keep constant motion implicit and spatial force zones out of schema `v1`.

- [ ] **Step 4: Run world-code tests**

  Run: `pytest tests/test_world.py -v`

  Expected: PASS.

- [ ] **Step 5: Write and run the world-card visual test**

  Assert that `render_world_card` creates a nonempty PNG from a saved `WorldCode`, includes all operator names in returned figure metadata, and reproduces identical metadata for the same code.

  Run: `pytest tests/test_world_visuals.py -v`

  Expected: PASS and a temporary 64-DPI diagnostic image.

- [ ] **Step 6: Commit**

  ```bash
  git add pyproject.toml src/perception_env tests/test_world.py tests/test_world_visuals.py
  git commit -m "feat: define versioned world codes"
  ```

### Task 2: Deterministic operator execution and latent rollout

**Files:**
- Create: `src/perception_env/operators.py`
- Create: `src/perception_env/simulate.py`
- Create: `tests/test_operators.py`
- Create: `tests/test_simulate.py`
- Create: `tests/test_rollout_visuals.py`
- Modify: `src/perception_env/visualize.py`

**Interfaces:**
- Consumes: `WorldCode`, `SimulationConfig` from Task 1.
- Produces: `BallState`, `Trajectory`, `TerminationReason`, `step(state: BallState, world: WorldCode, config: SimulationConfig) -> BallState`, `rollout(initial: BallState, world: WorldCode, config: SimulationConfig) -> Trajectory`, and `render_trajectory_strip(trajectory: Trajectory, world: WorldCode, path: Path) -> Path`.

- [ ] **Step 1: Write analytic and operator-order tests**

  Cover constant velocity, one-axis constant acceleration, damping, each boundary reflection, deterministic replay, off-domain termination, and a compound acceleration-plus-wall event whose expected state pins the documented update order.

- [ ] **Step 2: Run tests and confirm failure**

  Run: `pytest tests/test_operators.py tests/test_simulate.py -v`

  Expected: FAIL because rollout interfaces are absent.

- [ ] **Step 3: Implement pure operator functions and rollout**

  Keep operator functions side-effect free. Store every latent state, collision event, validity flag, seed, and termination reason in `Trajectory`.

- [ ] **Step 4: Run numerical tests**

  Run: `pytest tests/test_operators.py tests/test_simulate.py -v`

  Expected: PASS with analytic position and velocity errors below the asserted tolerance.

- [ ] **Step 5: Add the latent-trajectory visual proof**

  Render a strip containing the 2D path, time-colored points, boundary geometry, and a text summary of active operators. Test stable panel titles, axes, seed metadata, and output creation rather than raw PNG bytes.

  Run: `pytest tests/test_rollout_visuals.py -v`

  Expected: PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add src/perception_env/operators.py src/perception_env/simulate.py src/perception_env/visualize.py tests
  git commit -m "feat: simulate deterministic operator worlds"
  ```

### Task 3: Rendering, occlusion, and observable events

**Files:**
- Create: `src/perception_env/render.py`
- Create: `src/perception_env/events.py`
- Create: `tests/test_render.py`
- Create: `tests/test_events.py`
- Create: `tests/test_observation_visuals.py`
- Modify: `src/perception_env/visualize.py`

**Interfaces:**
- Consumes: `Trajectory`, `RenderConfig`.
- Produces: `Occluder`, `RenderedSequence`, `ReappearanceEvent`, `render_sequence(trajectory: Trajectory, occluders: tuple[Occluder, ...], config: RenderConfig) -> RenderedSequence`, `extract_reappearance(sequence: RenderedSequence) -> ReappearanceEvent | None`, and `render_observation_strip(sequence: RenderedSequence, path: Path) -> Path`.

- [ ] **Step 1: Write rendering and invariance tests**

  Assert image shape and dtype, ball visibility outside occluders, complete hiding inside an occluder, exact visibility masks, minimum hidden-run enforcement, reappearance extraction, and bitwise equality of latent states before and after rendering.

- [ ] **Step 2: Run tests and confirm failure**

  Run: `pytest tests/test_render.py tests/test_events.py -v`

  Expected: FAIL because rendering and event modules are absent.

- [ ] **Step 3: Implement rasterization and event extraction**

  Use nearest-pixel ball centers with configurable radius and axis-aligned rectangular occluders. Record rejection reasons for no occlusion, partial-only occlusion, insufficient hidden duration, and no visible reappearance.

- [ ] **Step 4: Run numerical tests**

  Run: `pytest tests/test_render.py tests/test_events.py -v`

  Expected: PASS.

- [ ] **Step 5: Add observation and one-operator comparison visuals**

  `render_observation_strip` must show latent path, visible frames, visibility timeline, hidden interval, and reappearance. Add `render_operator_comparison` that holds seed and initial state fixed while varying one operator value.

  Run: `pytest tests/test_observation_visuals.py -v`

  Expected: PASS with required panels and parameter labels present in figure metadata.

- [ ] **Step 6: Commit**

  ```bash
  git add src/perception_env/render.py src/perception_env/events.py src/perception_env/visualize.py tests
  git commit -m "feat: render occluded ball observations"
  ```

### Task 4: Exact reachability and belief-space fixtures

**Files:**
- Create: `src/perception_env/reachability.py`
- Create: `src/perception_env/beliefs.py`
- Create: `tests/test_reachability.py`
- Create: `tests/test_beliefs.py`
- Create: `tests/test_belief_visuals.py`
- Modify: `src/perception_env/visualize.py`

**Interfaces:**
- Consumes: `BallState`, `WorldCode`, `SimulationConfig`, `ReappearanceEvent`.
- Produces: `EntryStateGrid`, `Probe`, `ReachabilityResult`, `BeliefEvent`, `BeliefHistory`, `enumerate_reachable(entry_states: EntryStateGrid, world: WorldCode, query_steps: tuple[int, ...], config: SimulationConfig) -> ReachabilityResult`, `rasterize_endpoints(result: ReachabilityResult, render: RenderConfig) -> np.ndarray`, `superpose(trajectories: Sequence[Trajectory], weights: np.ndarray, render: RenderConfig) -> np.ndarray`, and `build_belief_history(events: Sequence[BeliefEvent], length: int = 4) -> BeliefHistory`.

- [ ] **Step 1: Write oracle and belief tests**

  Use hand-computable entry-state grids to test reachable endpoints, tolerance-based probe labels, binary occupancy, normalized weighted superposition, empty and saturated sets, chronological four-event padding, and exact `post - pre` channels.

- [ ] **Step 2: Run tests and confirm failure**

  Run: `pytest tests/test_reachability.py tests/test_beliefs.py -v`

  Expected: FAIL because oracle and belief interfaces are absent.

- [ ] **Step 3: Implement finite enumeration and belief construction**

  Make uncertainty bounds and grid resolution explicit fields of `EntryStateGrid`; never infer them from a rendered frame. Keep binary oracle maps separate from derived weighted densities.

- [ ] **Step 4: Run numerical tests**

  Run: `pytest tests/test_reachability.py tests/test_beliefs.py -v`

  Expected: PASS.

- [ ] **Step 5: Add reachability and belief visuals**

  Render sampled paths, endpoint scatter, binary occupancy, density, pre-belief, post-belief, difference belief, and the true endpoint only in a separate first-order diagnostic panel that is excluded from the Model 2 tensor.

  Run: `pytest tests/test_belief_visuals.py -v`

  Expected: PASS, including empty-set and saturated-set figures without warnings.

- [ ] **Step 6: Commit**

  ```bash
  git add src/perception_env/reachability.py src/perception_env/beliefs.py src/perception_env/visualize.py tests
  git commit -m "feat: add reachability and belief superpositions"
  ```

### Task 5: Nested world families and durable benchmark records

**Files:**
- Create: `configs/world_families/v1.yaml`
- Create: `src/perception_env/families.py`
- Create: `src/perception_env/records.py`
- Create: `tests/test_families.py`
- Create: `tests/test_records.py`
- Create: `tests/test_dataset_visuals.py`
- Modify: `src/perception_env/visualize.py`

**Interfaces:**
- Consumes: schemas and records from Tasks 1–4.
- Produces: `WorldFamily`, `BenchmarkRecord`, `Model2InputRecord`, `load_family(path: Path, level: str) -> WorldFamily`, `sample_worlds(family: WorldFamily, count: int, seed: int) -> list[WorldCode]`, `save_record(record: BenchmarkRecord, root: Path) -> Path`, and `load_record(path: Path) -> BenchmarkRecord`.

- [ ] **Step 1: Write family and persistence tests**

  Assert nested membership from `W0` through `W5`, stable seeded sampling, exclusion of neutral active parameters, schema-version retention, record round-trip, rejection-reason persistence, and reconstruction of every visualization input without simulation.

- [ ] **Step 2: Add the first-order leakage test**

  Construct a `Model2InputRecord` containing each forbidden field in turn—raw frames, positions, visibility masks, detector features, and direct event markers—and assert validation failure. Assert acceptance of only pre-belief, post-belief, difference, derived entropy, history mask, and target world code.

- [ ] **Step 3: Run tests and confirm failure**

  Run: `pytest tests/test_families.py tests/test_records.py -v`

  Expected: FAIL because family and record modules are absent.

- [ ] **Step 4: Implement versioned families and records**

  Encode exact parameter supports, neutral intervals, activity patterns, sampling distributions, entry-state uncertainty, and invalid-rollout policy in YAML. Store metadata as JSON-compatible data and arrays in compressed NPZ files.

- [ ] **Step 5: Run contract tests**

  Run: `pytest tests/test_families.py tests/test_records.py -v`

  Expected: PASS.

- [ ] **Step 6: Add dataset contact-sheet visual**

  Render a seeded grid spanning ladder levels with world card, observation strip, and belief panel per example. Generate it from reloaded records.

  Run: `pytest tests/test_dataset_visuals.py -v`

  Expected: PASS.

- [ ] **Step 7: Commit**

  ```bash
  git add configs/world_families/v1.yaml src/perception_env/families.py src/perception_env/records.py src/perception_env/visualize.py tests
  git commit -m "feat: add nested world families and records"
  ```

### Task 6: Empirical growth and shattering search

**Files:**
- Create: `src/perception_env/shattering.py`
- Create: `tests/test_shattering.py`
- Create: `tests/test_shattering_visuals.py`
- Modify: `src/perception_env/visualize.py`

**Interfaces:**
- Consumes: `WorldCode`, `Probe`, `ReachabilityResult`, `WorldFamily`.
- Produces: `LabelMatrix`, `ShatteringEstimate`, `build_label_matrix(worlds: Sequence[WorldCode], probes: Sequence[Probe]) -> LabelMatrix`, `count_labelings(matrix: LabelMatrix) -> int`, `search_shattered_set(matrix: LabelMatrix, max_n: int, budget: int, seed: int, same_context: bool = True) -> ShatteringEstimate`, and `render_shattering_matrix(estimate: ShatteringEstimate, path: Path) -> Path`.

- [ ] **Step 1: Write calibration fixtures**

  Test constant classifiers, thresholds over ordered points, and a finite Boolean fixture with known shattered-set size. Assert distinct-row counts, exhaustive results for small fixtures, deterministic budgeted search, repeated-seed summaries, and explicit failure to label a mixed-context result as `context_conditioned`.

- [ ] **Step 2: Run tests and confirm failure**

  Run: `pytest tests/test_shattering.py -v`

  Expected: FAIL because `perception_env.shattering` is absent.

- [ ] **Step 3: Implement label matrices and search**

  Make the single-context estimate primary. Implement the multi-context global estimate as a separately named diagnostic. Persist operator support, world count, probe count, uncertainty grid, tolerance, algorithm, budget, and seed list with every estimate.

- [ ] **Step 4: Run numerical tests**

  Run: `pytest tests/test_shattering.py -v`

  Expected: PASS and exact recovery on exhaustive synthetic fixtures.

- [ ] **Step 5: Add shattering-matrix visual proof**

  Plot worlds by probes, group duplicate labeling rows, highlight a discovered shattered subset, and annotate realized versus possible label counts and search budget.

  Run: `pytest tests/test_shattering_visuals.py -v`

  Expected: PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add src/perception_env/shattering.py src/perception_env/visualize.py tests/test_shattering.py tests/test_shattering_visuals.py
  git commit -m "feat: estimate empirical shattering complexity"
  ```

### Task 7: Identifiability and recovered-complexity interfaces

**Files:**
- Create: `src/perception_env/identifiability.py`
- Create: `tests/test_identifiability.py`
- Create: `tests/test_identifiability_visuals.py`
- Modify: `src/perception_env/visualize.py`

**Interfaces:**
- Consumes: `LabelMatrix`, world codes, possibility rasters.
- Produces: `EquivalenceClass`, `IdentifiabilityReport`, `group_equivalent_worlds(worlds: Sequence[WorldCode], labels: LabelMatrix, tolerance: float) -> IdentifiabilityReport`, `parameter_scoring_mask(report: IdentifiabilityReport) -> np.ndarray`, and `render_equivalence_gallery(report: IdentifiabilityReport, path: Path) -> Path`.

- [ ] **Step 1: Write equivalence and scoring tests**

  Test identical codes, distinct codes with identical labels, distinct codes with different labels, unresolved budget status, parameter-scoring masks, and separation between exact-code recovery and possibility-equivalent recovery.

- [ ] **Step 2: Run tests and confirm failure**

  Run: `pytest tests/test_identifiability.py -v`

  Expected: FAIL because identifiability interfaces are absent.

- [ ] **Step 3: Implement equivalence grouping and report schema**

  Use held-out probe label agreement as the canonical equivalence test for `v1`; retain raster overlap as a secondary continuous diagnostic.

- [ ] **Step 4: Run numerical tests**

  Run: `pytest tests/test_identifiability.py -v`

  Expected: PASS.

- [ ] **Step 5: Add equivalence-gallery visual**

  Show each equivalent world's card and reachability map side by side, with parameter differences and label agreement visible.

  Run: `pytest tests/test_identifiability_visuals.py -v`

  Expected: PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add src/perception_env/identifiability.py src/perception_env/visualize.py tests
  git commit -m "feat: audit perceptual identifiability"
  ```

### Task 8: Reproducible CLI and end-to-end visual benchmark

**Files:**
- Create: `src/perception_env/cli.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_end_to_end.py`
- Create: `README.md`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: all prior task interfaces.
- Produces: console command `perception-env` with `sample`, `inspect`, `estimate-shattering`, and `demo` subcommands.

- [ ] **Step 1: Write CLI contract tests**

  Assert help text, invalid configuration errors, deterministic output manifests, refusal to overwrite an existing run without `--force`, and successful recreation of visuals from a saved run.

- [ ] **Step 2: Write the end-to-end benchmark test**

  For a tiny seeded `W0`–`W2` configuration, generate valid records, render all required panels, build a label matrix, estimate empirical shattering, run identifiability grouping, and validate that Model 2 input records contain no forbidden fields.

- [ ] **Step 3: Run tests and confirm failure**

  Run: `pytest tests/test_cli.py tests/test_end_to_end.py -v`

  Expected: FAIL because the CLI is absent.

- [ ] **Step 4: Implement the CLI and run manifest**

  Every command writes a manifest containing command arguments, configuration hashes, schema versions, seeds, package versions, output paths, rejection counts, and wall-clock duration.

- [ ] **Step 5: Add the end-to-end demonstration**

  `perception-env demo --config configs/world_families/v1.yaml --seed 7 --output <dir>` must create a world-family contact sheet, trajectory and observation strips, belief panels, reachability overlays, a shattering matrix, an equivalence gallery, and an HTML index linking them.

- [ ] **Step 6: Run the complete test and quality suite**

  Run: `pytest -q`

  Expected: PASS.

  Run: `ruff check .`

  Expected: no violations.

- [ ] **Step 7: Run and inspect the demonstration**

  Run: `perception-env demo --config configs/world_families/v1.yaml --seed 7 --output artifacts/environment-demo`

  Expected: successful manifest with all visual artifacts present and no first-order fields in the exported Model 2 input sample.

- [ ] **Step 8: Document reproduction and interpretation**

  In `README.md`, document installation, the demo command, world-code fields, visual panels, the distinction between oracle possibility and weighted belief, the meaning and limitations of the empirical shattering estimate, and the first-order-input prohibition for Model 2.

- [ ] **Step 9: Commit**

  ```bash
  git add pyproject.toml src/perception_env/cli.py tests/test_cli.py tests/test_end_to_end.py README.md
  git commit -m "feat: ship reproducible environment demo"
  ```

## Final verification

- [ ] Run `pytest -q` and record the passing test count.
- [ ] Run `ruff check .` and record the clean result.
- [ ] Run the seeded environment demo into `artifacts/environment-demo`.
- [ ] Open the HTML index and visually inspect every required panel for clipping, illegible labels, misleading normalization, and seed/config annotations.
- [ ] Inspect one exported `Model2InputRecord` and confirm it contains belief-space tensors only.
- [ ] Re-run the demo into a second directory and compare manifest hashes for deterministic numerical outputs.
- [ ] Confirm the shattering report includes `context_conditioned`, world/probe counts, uncertainty grid, event tolerance, search method, budget, and seeds.

