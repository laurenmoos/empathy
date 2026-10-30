# Model 2 Inverse-World CNN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a real PyTorch CNN that infers the fixed-width physical world code from four-event belief histories alone, with reproducible possibility-based evaluation and saved visual evidence.

**Architecture:** A separate `perception_env.model2` package owns validated belief-only inputs, a compact CNN, masked supervision, training, and evaluation. An offline fixture builder may access environment truth, but writes input arrays separately from targets and evaluator-only context; the production forward path accepts only belief tensors and their validity mask. Oracle/perturbed fixtures make this subplan independently runnable before Model 1 exists; a strict producer adapter later accepts Model 1 belief outputs without changing the network interface.

**Tech Stack:** Python 3.11+, PyTorch, existing NumPy/PyYAML/Matplotlib stack, pytest, Ruff; CPU is the deterministic reference backend.

**Spec:** `docs/superpowers/specs/2026-10-30-parameterized-occlusion-environment-design.md`; Subplan 3 and its gate in `docs/superpowers/plans/2026-10-30-perceptual-coherence-research-roadmap.md`.

## Global Constraints

- “Model 2 never receives raw frames, ball coordinates, detector features, segmentation masks, or a direct reappearance marker.”
- “Model 2's canonical input is the last four belief events, with channels for” pre-belief, post-belief, and post minus pre.
- “Optional entropy or mass summaries must also be derived solely from belief tensors.” V1 CNN omits these optional inputs.
- “The serializer must preserve both the structured representation and a flat tensor representation with a versioned field order.”
- “Parameter normalization ranges are part of the dataset specification, not learned from a dataset instance.”
- “Parameter recovery is scored only where the generated context is sufficiently informative.”
- “Visuals must be generated from saved records rather than transient training state, include seeds and configuration identifiers, and have deterministic snapshot tests where practical.”
- “64-by-64 grayscale canonical frames, with resolution configurable.” Use 64-by-64 belief rasters in the research configuration and 16-by-16 in the smoke configuration.
- One history has one persistent true world; event order is chronological. IDs, absolute times, contexts, source labels, targets, audit masks, and evaluator metadata are never network inputs.
- Keep the environment usable without torch. Add an `ml` optional dependency and lazy CLI imports; do not import torch in `perception_env.__init__`.
- Supervised offline learning only: no online weight updates, replay learner, Model 1 training, or streaming convergence loop in this subplan.
- Do not reinterpret family `v1`: its current W2 permits at most one acceleration slot. Record the actual family configuration and its hash; a ladder change belongs in a separately versioned environment change.

## Review Focus

- Histories sharing a world, overlapping event windows, or duplicated world codes must not cross train/validation/test boundaries; Task 2 tests grouped splits and duplicate rejection.
- All-inactive targets, all-unidentifiable targets, and padded histories must produce finite losses and zero gradients for masked parameters; Tasks 1, 3, and 4 cover them.
- An active prediction in a neutral parameter interval must decode deterministically to a valid world without consulting truth or the requested family label; Task 3 tests endpoints, ties, and projection counts.
- Targets, first-order sidecars, or filenames must not affect forward predictions when beliefs stay fixed; Tasks 1, 5, and 8 test this through the actual training/inference path.
- Empty reachable sets, one-class probe labels, missing classes, and insufficient audit budgets must produce explicit denominators/statuses rather than misleading perfect scores; Task 6 covers them.

---

## Existing contracts and implementation decisions

`WorldSchema.v1()` has five slots and 12 flat fields. Activity indices are `(0,2,4,6,9)`; parameter indices are `(1,3,5,7,8,10,11)`. Parameters belong to slots `(0,1,2,3,3,4,4)`. Derive these mappings from `schema.specs`, then assert these v1 constants in tests. Preserve physical normalization and inactive zeros from `WorldCode.to_flat(schema)`.

`BeliefHistory.tensor` is NumPy `[4,3,H,W]`, `mask` is boolean `[4]`, and `steps` is diagnostic only. `Model2InputRecord` currently bundles target fields with inputs; validate legacy records, then copy only the three belief channels and history mask into a new input-only type. Never forward a `Model2InputRecord`, `BenchmarkRecord`, arbitrary dictionary, or sample ID to the CNN.

The environment APIs reused here are `build_belief_history`, `superpose`, `rollout`, `render_sequence`, `extract_reappearance`, `enumerate_reachable`, `rasterize_endpoints`, `sample_worlds`, `group_equivalent_worlds`, and `search_shattered_set`. `parameter_scoring_mask` currently returns one boolean per world, not per slot. Expand that conservative result over active parameter fields; do not claim fine-grained slot identifiability that the existing audit has not established.

### Artifact contract

Dataset root contains `manifest.json`, `inputs.npz`, `targets.npz`, `evaluation.json`, `evaluation.npz`, and optional `diagnostics/`. When privileged ablations are requested, save `diagnostics/first_order_control.npz` with `features` (`float32[N,4,9]`: entry four-state, reappearance four-state, elapsed time) and `history_mask` (`bool[N,4]`); production loaders never open it. Input NPZ keys are exactly `beliefs` (`float32[N,4,3,H,W]`) and `history_mask` (`bool[N,4]`). Targets contain `world_flat` (`float32[N,12]`), `parameter_identifiable` (`bool[N,7]`), and integer `split` (`0/1/2`). Row identity, persistent-world identity, source mode, event ordinals, producer version, configuration, and content hashes live in the manifest. Evaluator files contain entry grids, simulation settings, separately named audit/held-out probes, true labels, and saved truth rasters. The input loader must not open evaluator or diagnostics files.

Canonical content hashes include array names, dtype, shape, and contiguous bytes plus canonical JSON; they exclude output paths and timing. Every checkpoint/evaluation verifies the dataset hash and world/input schema versions. Use JSON plus NPZ with `allow_pickle=False`; model checkpoints contain a torch state dict and primitive metadata, loaded with explicit `weights_only=True` and `map_location`.

### Experiment defaults

Provide `configs/model2/oracle_v1.yaml` and `configs/model2/smoke_v1.yaml`. Research defaults: registered training seeds `[7,19,31]` on one fixed fixture dataset (fixture seed 7), 120 accepted unique worlds across W1–W5, four episodes per world, four events per episode, split fractions `[0.7,0.15,0.15]`, 64-square beliefs, batch size 32, 50 epochs, AdamW learning rate `0.001`, weight decay `0.0001`, activity/parameter loss weights `1.0/1.0`, CPU reference, `num_workers=0`. Smoke: seed 7, 12 unique W1/W2 worlds, two episodes per world, four events, 16-square beliefs, batch size 4, two epochs. W0 is an explicitly reported diagnostic fixture; its one canonical code must never be duplicated across primary world-disjoint splits. Declare accepted/rejected counts and observed operator support; do not promise that finite rejection sampling covers every combination.

Fixture simulation defaults: `dt=.05`, `steps=80`, occluder `(x_min=.35,x_max=.65,y_min=.05,y_max=.95)`, radius 2 at 64 pixels and 1 at 16, minimum hidden run 2. Sample initial x uniformly `[.10,.25]`, y `[.25,.75]`, vx `[.15,.35]`, vy `[-.10,.10]`; retain only in-domain trajectories with visible pre-occlusion evidence, a fully hidden interval, and visible reappearance. Use at most 200 context attempts per requested world and at most 20 times the requested world count in sampled-world attempts; an unmet requested count is an explicit incomplete-generation error, never an infinite retry or silent smaller dataset.

## File map

| File | Responsibility |
|---|---|
| `src/perception_env/model2/__init__.py` | Lightweight package/version exports |
| `src/perception_env/model2/contracts.py` | Input/target/prediction types, schema mapping, legacy adapter |
| `src/perception_env/model2/config.py` | Strict typed fixture/training/evaluation configs and YAML loading |
| `src/perception_env/model2/data.py` | Separate input/target loaders, grouped splits, collation |
| `src/perception_env/model2/fixtures.py` | Oracle/perturbed producer and saved evaluator context |
| `src/perception_env/model2/network.py` | Belief-only CNN |
| `src/perception_env/model2/decode.py` | Valid world-code projection and diagnostics |
| `src/perception_env/model2/losses.py` | Activity BCE and masked parameter loss |
| `src/perception_env/model2/train.py` | Deterministic training, checkpoint selection and inference |
| `src/perception_env/model2/evaluate.py` | Saved predictions, oracle regeneration, recovery metrics |
| `src/perception_env/model2/transforms.py` | Pure belief-only ablation transforms shared by training/inference |
| `src/perception_env/model2/controls.py` | Constant, single-belief and shuffled-history experiment runners |
| `src/perception_env/model2/first_order_control.py` | Explicitly isolated privileged ablation; never production import |
| `src/perception_env/model2/visualize.py` | Figures from saved dataset/evaluation records only |
| `src/perception_env/model2/adapters.py` | Later Model 1 belief producer adapter |
| `src/perception_env/model2/cli.py` | Model 2 subcommand handlers |
| `configs/model2/oracle_v1.yaml`, `configs/model2/smoke_v1.yaml` | Frozen experiment defaults |
| `tests/model2/` | Contract, learning, metric, leakage, CLI, and visual tests |
| `pyproject.toml`, `uv.lock`, `README.md`, `src/perception_env/cli.py` | Optional dependency, documented commands, lazy dispatch |

### Task 1: Typed input-only boundary and supervised dataset contract

**Files:** Create `model2/__init__.py`, `contracts.py`, `config.py`, `data.py` under `src/perception_env/`; create `tests/model2/test_contracts.py`, `tests/model2/test_data.py`; modify `pyproject.toml`, `uv.lock`.

**Interfaces:**

- `SchemaLayout.from_schema(schema: WorldSchema) -> SchemaLayout` exposes activity indices, parameter indices, and parameter-to-slot indices.
- Frozen `BeliefInputs(beliefs: np.ndarray, history_mask: np.ndarray)` validates on construction; `from_mapping(values: Mapping[str, object]) -> BeliefInputs` rejects every key except its two field names.
- `WorldTargets(world_flat: np.ndarray, parameter_identifiable: np.ndarray)` validates canonical worlds and `[N,7]` masks.
- `from_legacy(record: Model2InputRecord) -> tuple[BeliefInputs, WorldTargets]` creates one batched row, with identifiability initially false until an audit supplies it.
- `Model2Batch(beliefs: torch.Tensor, history_mask: torch.Tensor, world_flat: torch.Tensor, parameter_identifiable: torch.Tensor)` is a supervised container, not a forward argument.
- `BeliefDataset(root: Path, split: Literal['train','validation','test'])`, `__getitem__(index: int) -> tuple[BeliefInputs, WorldTargets]`, `collate_examples(examples: Sequence[tuple[BeliefInputs, WorldTargets]]) -> Model2Batch`.
- `FixtureConfig`, `TrainingConfig`, `EvaluationConfig` are frozen dataclasses matching the defaults above; `load_config(path: Path) -> tuple[FixtureConfig, TrainingConfig, EvaluationConfig]` reads exactly the `fixtures`, `training`, and `evaluation` YAML sections and rejects unknown keys and invalid positive counts/fractions. `FixtureConfig.seed` is independent of `TrainingConfig.seed` (both default 7); `EvaluationConfig.search_seeds=(7,19,31)`. `TrainingConfig.input_mode` is `full`, `single_belief`, or `shuffled_history`, default `full`; it controls the transform applied consistently at fitting and inference, and is stored in the checkpoint. All simulation, raster, uncertainty, support, attempt-limit, probe, audit-bank, and split settings described in this plan are explicit fixture config fields rather than hidden module globals.

- [ ] **Step 1: Write failing contract/data tests.** Assert v1 mappings; `[N,4,3,H,W]` shape; finite pre/post in `[0,1]`; exact difference within `atol=1e-6, rtol=0`; boolean suffix masks; at least one valid event; zero padding; spatial dimensions at least 8; and common batch resolution. For a valid fixture assert `inputs.beliefs.shape == (2, 4, 3, 16, 16)`, `inputs.history_mask.dtype == np.bool_`, and `np.allclose(inputs.beliefs[:,:,2], inputs.beliefs[:,:,1] - inputs.beliefs[:,:,0], atol=1e-6, rtol=0)`. Parameterize forbidden keys including frames, positions, visibility masks, features, event markers, target, entropy, and arbitrary metadata. Assert copying isolates caller mutation, malformed targets fail, all-zero valid beliefs succeed, all-masked rows fail, and a guarded file opener proves the dataset never opens evaluator/diagnostic files.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/model2/test_contracts.py tests/model2/test_data.py -q`; expect missing-module failures.
- [ ] **Step 3: Implement the interfaces and optional `ml` dependency.** Add `torch>=2.6,<3` to the extra, resolve the current compatible release into the lockfile using `uv sync --extra dev --extra ml`, and record the resolved version at runtime. Keep all tensor creation out of the base package import path. Validate direct construction, not just mapping helpers. No input tensor includes targets or audit masks.
- [ ] **Step 4: Run** the two test files plus existing `tests/test_records.py`; expect all pass. In a subprocess that blocks torch imports, `import perception_env` and base CLI help must still succeed.
- [ ] **Step 5: Commit** these files: `feat: define strict belief-only model2 data contracts`.

### Task 2: Reproducible oracle and perturbed belief histories

**Files:** Create `fixtures.py`, the two YAML configs, `tests/model2/test_fixtures.py`; modify `data.py`.

**Interfaces:**

- `FixtureManifest` contains artifact paths, hashes, row/world/episode IDs, split counts, source modes, generation attempts, rejection reasons, seeds, schema versions, and producer configuration.
- `generate_fixtures(config: FixtureConfig, output: Path) -> FixtureManifest` writes the artifact contract above and refuses existing output.
- `split_world_groups(world_ids: Sequence[str], fractions: tuple[float,float,float], seed: int) -> np.ndarray` assigns a persistent world to exactly one split before expanding episodes/windows.
- `build_fixture_event(true_world: WorldCode, prior_world: WorldCode, initial: BallState, config: FixtureConfig, ordinal: int, seed: int) -> tuple[BeliefEvent, dict[str, object]]` returns an event plus offline evaluator provenance; it raises a named rejection for inadmissible events.

- [ ] **Step 1: Write failing fixture tests.** `test_history_has_one_world_and_four_ordered_events`, `test_split_groups_duplicate_codes_and_overlapping_windows`, `test_reappearance_precedes_assimilation`, `test_same_seed_same_array_hashes`, `test_target_not_in_input_archive`, `test_prior_perturbation_is_valid_and_distinct`, and `test_bounded_rejection_reports_incomplete_run`. Assert the true hidden-entry state is explicitly represented in its uncertainty grid and audit probes are disjoint from held-out probes.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/model2/test_fixtures.py -q`; expect absent fixture API failures.
- [ ] **Step 3: Implement offline generation.** Sample unique canonical worlds with `sample_worlds`, hash rounded canonical world codes for grouping, and use independent named RNG streams for worlds, contexts, perturbations, splits, audit probes, and held-out probes. One event uses an actual `extract_reappearance` outcome; hidden-entry grids are centered on the recorded hidden-entry state with x/y half-width `.005`, vx/vy half-width `.02`, resolution `(3,3,3,3)`, clipped position bounds, and center retained. Query time is elapsed steps from hidden entry to observed reappearance, not the absolute clip index.

  Build pre-belief from equally weighted trajectories under a distinct valid perturbed prior world: choose one slot uniformly, toggle it, or resample its active parameters from the configured support, rejecting unchanged codes. Build oracle post-belief from true-world candidate trajectories whose visible outcome lies within tolerance `.025` of the observed reappearance. Normalize surviving weights and rasterize paths only through the query time; reject empty post ensembles. This is an explicitly privileged **oracle bootstrap**, not evidence of trained Model 1 performance. Produce 25% oracle pre/post pairs (true-world pre ensemble, outcome-conditioned post) and 75% perturbed-prior pairs; source mode stays out of inputs. Never paint a ground-truth point or Gaussian directly into a belief image. Assign ordinal event indices for histories; persist physical event times only in sidecars.

  Save non-overlapping episode histories and optional prefixes padded on the left. Audit each target against a recorded bank of 64 independently sampled comparison worlds from the declared family support plus that target, using dedicated audit probes across its history; comparison-bank generation must not inspect validation/test target lists when preparing training audits; expand the conservative world-level identifiability boolean over active parameters. Unfinished audits mark all parameters unidentifiable. Save separately sampled held-out probe contexts that never generate input beliefs. Stratify split assignment by available activity patterns when possible without breaking world grouping; report absent strata. Do not silently change supports to meet quotas.
- [ ] **Step 4: Run** fixture/data tests; expect deterministic hashes, disjoint world groups, bounded failures, and complete smoke artifacts. Reload input/target/evaluation arrays and compare to their declared hashes.
- [ ] **Step 5: Commit**: `feat: generate world-disjoint oracle belief histories`.

### Task 3: CNN and deterministic decoding to executable worlds

**Files:** Create `network.py`, `decode.py`, `tests/model2/test_network.py`, `tests/model2/test_decode.py`.

**Interfaces:**

- `WorldPrediction(activity_logits: torch.Tensor, normalized_parameters: torch.Tensor)` has shapes `[N,5]` and `[N,7]`.
- `InverseWorldCNN(nn.Module)`, `forward(self, beliefs: torch.Tensor, history_mask: torch.Tensor) -> WorldPrediction`.
- `DecodedWorlds(worlds: tuple[WorldCode,...], flat: np.ndarray, projection_counts: np.ndarray)` records per-row number of active parameter projections.
- `decode_prediction(prediction: WorldPrediction, schema: WorldSchema, *, activity_threshold: float = .5) -> DecodedWorlds` operates on detached CPU values and needs no target/family/context.

- [ ] **Step 1: Write failing network/decoder tests.** Assert output widths for 16/64-square inputs, real `nn.Conv2d` layers and trainable gradients, batch size one, padded-value masking before convolution, deterministic eval predictions, and rejection of unknown keyword arguments. Assert `prediction.activity_logits.shape == (2, 5)` and `prediction.normalized_parameters.shape == (2, 7)` and `any(isinstance(layer, torch.nn.Conv2d) for layer in model.modules())`. Test inactive canonical zeros, extreme logits, all parameter bounds, forbidden active neutral values, exact neutral midpoint tie behavior, finite-value failures, and world schema round-trip.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/model2/test_network.py tests/model2/test_decode.py -q`; expect missing implementation failures.
- [ ] **Step 3: Implement.** Zero masked events, reshape `[N,4,3,H,W]` to chronological 12 channels; three Conv2d blocks use `(12→32→64→64)`, kernel 3, stride 2, padding 1, GroupNorm with 8 groups, and ReLU. Adaptive average pool to `4×4`, flatten to 1024, Linear to 128 plus ReLU, then independent activity and parameter heads. Apply sigmoid only to the seven parameter outputs; keep activity logits raw. No dropout, batch norm, coordinate channels, hidden caches, sidecars, or recurrent external state.

  Decoder thresholds sigmoid activity at `.5`, denormalizes using the schema, and projects active parameters falling in a neutral interval to the nearest valid physical boundary plus/minus `1e-6` (tie chooses the lower valid side; one-sided intervals choose the available side). Clamp only to schema support, count projections, re-normalize and call schema validation; never force membership using the known true family. Preserve raw outputs for parameter-error metrics so projection cannot conceal model error.
- [ ] **Step 4: Run** network/decoder/contract tests; expect finite gradients and every decoded finite prediction to satisfy `WorldSchema.v1().validate`.
- [ ] **Step 5: Commit**: `feat: add belief-history inverse-world cnn`.

### Task 4: Masked activity and parameter supervision

**Files:** Create `losses.py`, `tests/model2/test_losses.py`.

**Interfaces:**

- `LossResult(total: torch.Tensor, activity: torch.Tensor, parameters: torch.Tensor, parameter_count: int)`.
- `world_code_loss(prediction: WorldPrediction, targets: torch.Tensor, parameter_identifiable: torch.Tensor, *, activity_weight: float = 1., parameter_weight: float = 1.) -> LossResult`.

- [ ] **Step 1: Write failing tests.** Compare activity loss against hand-computed BCE; assert parameter MSE equals its masked numerator divided by number of selected scalar fields. Change inactive/unidentifiable targets and verify unchanged parameter loss and zero corresponding parameter-head gradients. All-inactive and all-unidentifiable batches must satisfy `result.parameter_count == 0`, `result.parameters.item() == 0.0`, and `torch.isfinite(result.total)`; after backward, assert `torch.count_nonzero(prediction.normalized_parameters.grad) == 0`. Wrong activity predictions must not disable parameter supervision for true active slots.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/model2/test_losses.py -q`; expect missing loss API.
- [ ] **Step 3: Implement** mean `binary_cross_entropy_with_logits` across all five true activity flags. Parameter mask is `true_activity[:,parameter_to_slot] & parameter_identifiable`; average squared normalized error over selected fields, with denominator clamped to one. Return explicit counts, preserve differentiability at zero mask, and reject invalid shapes/nonfinite targets. There is no loss term consuming frames, coordinates, or a differentiable simulator.
- [ ] **Step 4: Run** loss/network tests; expect analytical values and masked-gradient assertions to pass.
- [ ] **Step 5: Commit**: `feat: train world heads with identifiable active-parameter masks`.

### Task 5: Deterministic training, checkpointing, and belief-only inference

**Files:** Create `train.py`, `tests/model2/test_train.py`, `tests/model2/test_leakage.py`; modify `data.py` only if collation integration requires it.

**Interfaces:**

- `TrainingRun(root: Path, best_checkpoint: Path, metrics_path: Path, manifest_path: Path)`.
- `train_model(dataset_root: Path, config: TrainingConfig, output: Path) -> TrainingRun`.
- `load_model(checkpoint: Path, *, device: str = 'cpu') -> InverseWorldCNN`.
- `predict(model: InverseWorldCNN, inputs: BeliefInputs, *, batch_size: int = 32) -> WorldPrediction` returns CPU tensors in original row order.

- [ ] **Step 1: Write failing tests.** A fixed 8-example learnable synthetic belief fixture must reduce total loss below 20% of its initial value within 200 AdamW updates at `.01` and reproduce predictions after reload. Two same-seed short CPU runs must have identical weights/predictions/history metrics excluding timing. Assert validation, not test, selects the best epoch; source hash/schema mismatch rejects loading into an experiment. Hook the actual forward call and assert its arguments contain exactly beliefs and history mask. Poison evaluator frames/coordinates and targets after input loading: inference must remain unchanged; changing a real belief must be able to change predictions.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/model2/test_train.py tests/model2/test_leakage.py -q`; expect missing trainer APIs.
- [ ] **Step 3: Implement** seeded Python/NumPy/torch RNGs, deterministic algorithms, CPU thread count 1 for reference tests, seeded DataLoader generator, no multiprocessing by default, AdamW, and gradient clipping norm 1. Select minimum mean validation total loss with earliest epoch breaking ties; do not read test rows during fitting. Log per-epoch summed loss numerators/denominators, counts, learning rate and duration. Save best/final state dicts, primitive optimizer/config metadata, train/validation row IDs, seed, library/device versions, dataset hashes, and schema field order. Refuse overwrites and validate all input paths/configs before creating output. Preserve safe CPU loading and explicit device selection; unavailable requested accelerators fail clearly, never silently change backend.
- [ ] **Step 4: Run** training/leakage/loss tests; expect successful overfit, exact same-runtime CPU repeat, identical reload inference, and no first-order files opened by the model/data loader. Document that bitwise parity across devices/releases is not guaranteed.
- [ ] **Step 5: Commit**: `feat: train and checkpoint belief-only world inference`.

### Task 6: Held-out possibility recovery and isolated controls

**Files:** Create `evaluate.py`, `transforms.py`, `controls.py`, `first_order_control.py`, `tests/model2/test_evaluate.py`, `tests/model2/test_controls.py`.

**Interfaces:**

- `EvaluationResult(root: Path, metrics: dict[str, object], prediction_path: Path, manifest_path: Path)`.
- `evaluate_model(checkpoint: Path, dataset_root: Path, config: EvaluationConfig, output: Path) -> EvaluationResult`.
- `score_predictions(prediction: WorldPrediction, targets: WorldTargets, evaluator: Mapping[str, object]) -> dict[str, object]` receives evaluator context only after forward inference.
- `fit_constant_baseline(train_targets: WorldTargets) -> WorldPrediction` returns a single reusable prediction row; use empirical activity probabilities and active identifiable parameter means, with `.5` for missing parameter support.
- `transform_control(inputs: BeliefInputs, mode: Literal['single_belief','shuffled_history'], seed: int) -> BeliefInputs` keeps only the latest post belief repeated as pre/post with zero difference and one valid event for the single-belief control; shuffles valid event triples jointly for the shuffled control.
- `run_controls(dataset_root: Path, config: TrainingConfig, output: Path) -> dict[str, Path]` trains identical CNNs separately with each `TrainingConfig.input_mode`, using identical splits/seed/epoch budget and validation selection. Define `transform_control` in `transforms.py` so training does not import the control experiment runner; inference reads the checkpoint mode automatically. Do not apply the transform twice.
- `train_first_order_control(dataset_root: Path, config: TrainingConfig, output: Path) -> TrainingRun` lives in the isolated ablation module, consumes a separately saved privileged feature matrix, and is never called by production training/evaluation. Features are four events of entry/reappearance `(x,y,vx,vy)` and elapsed time, with padding mask; a two-layer 128-unit MLP predicts the same heads. Its results are marked `privileged_first_order_ablation` everywhere.

- [ ] **Step 1: Write failing metric/control tests.** Exact predictions give perfect label agreement and IoU; distinct codes with the same oracle labels count as possibility-equivalent but not exact; active parameter MAE excludes inactive/unidentified values. Test both-empty rasters (IoU 1 with explicit empty-pair count), empty-versus-nonempty (0), zero positive/negative probe classes (undefined class metric is null with denominator 0), unresolved audit masks, invalid predicted trajectories (empty reachability, never dropped), and duplicate predicted worlds. Assert held-out probes were not audit probes and test rows were not fit/selection rows. Constant baseline must depend only on train targets; shuffled transforms preserve channel differences and masks. A static/import test ensures production modules do not import the privileged ablation.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/model2/test_evaluate.py tests/model2/test_controls.py -q`; expect missing evaluator/control APIs.
- [ ] **Step 3: Implement** inference first, then detached decoding and oracle regeneration for each saved held-out context. Report per-operator/micro activity precision/recall, exact activity match, normalized and physical parameter MAE on true-active audited-identifiable fields, code match at `atol=1e-6`, projection rates, raster IoU, held-out possibility agreement, positive/negative recall, and macro averages by persistent world and actual family. Report target ambiguity status independently of predicted code accuracy. No metric silently omits failed/empty predictions.

  Regenerate labels with the same entry uncertainty, relative query steps, tolerance and probes for true/predicted worlds. Use primary single-context shattering searches for true/recovered banks on each of the first three held-out contexts in sorted context-ID order (all bank worlds are evaluated on each identical context/probe set), `max_n=3`, budget 100 and seeds `[7,19,31]`; save distinct row counts, individual dimensions, and context IDs. Do not call this the integrated complexity frontier of Subplan 5. Save raw logits/parameters, decoded worlds, masks, labels, rasters, per-world metrics and all search metadata. Bootstrap whole persistent worlds 1,000 times at seed 7 for 95% metric/difference intervals; never treat repeated episodes as independent worlds. Report search-seed min/max separately from statistical confidence intervals.

  Generate 64 audit and 128 held-out probes per context using distinct RNG streams: half uniform locations and half jittered true reachable endpoints (uniform disk radius `.05`), tolerance `.025`. If no endpoint exists, use uniform sampling and record that fallback; do not guarantee balance. Record world/probe sampling methods, actual supports/ranges, accepted counts, uncertainty, tolerance, budgets, seeds, and whether searches exhausted candidates. Privileged controls write to their own directories with an incompatible architecture tag so their checkpoints cannot load as `InverseWorldCNN`.
- [ ] **Step 4: Run** evaluator/control tests plus environment reachability/identifiability/shattering tests; expect known fixture values and exact partition isolation.
- [ ] **Step 5: Commit**: `feat: evaluate held-out possibility recovery and model2 controls`.

### Task 7: Saved visual proof and Model 1 producer adapter

**Files:** Create `visualize.py`, `adapters.py`, `tests/model2/test_visuals.py`, `tests/model2/test_adapters.py`.

**Interfaces:**

- `render_evaluation(dataset_root: Path, evaluation_root: Path, output: Path) -> tuple[VisualArtifact,...]` reads saved arrays and metadata only.
- `Model1BeliefPacket(episode_id: str, event_ordinal: int, pre_belief: np.ndarray, post_belief: np.ndarray)` is producer-side metadata, never a network argument.
- Existing `model1.bridge.belief_event_from_predictions` outputs can be wrapped as a packet using only `event.pre`, `event.post`, and caller-supplied episode/ordinal metadata; no Model 1 network dependency is imported.
- `adapt_model1_events(events: Sequence[Model1BeliefPacket]) -> BeliefInputs` requires one episode, strictly increasing unique ordinals, common resolution, and at least one event; retains the last four, derives difference and padding. No dependency on the unimplemented Model 1 network.
- `parse_model1_packet(values: Mapping[str, object]) -> Model1BeliefPacket` rejects unexpected fields rather than discarding them.

- [ ] **Step 1: Write failing tests.** Require history panels, predicted/true normalized and physical world cards, activity recovery plot, raw parameter error plot with denominators, predicted/true/difference possibility maps, equivalent-code gallery, training curves and control comparison table. Assert seed/config/dataset/checkpoint hashes and oracle-bootstrap labeling in sidecars. Monkeypatch model inference, rollout and oracle calls to raise: rendering saved results must still succeed. Adapter tests reject mixed episodes, decreasing/duplicate times, raw frames/coordinates/markers/targets and malformed beliefs; valid packets must yield the same tensor as the existing history builder.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/model2/test_visuals.py tests/model2/test_adapters.py -q`; expect missing APIs.
- [ ] **Step 3: Implement** deterministic exemplar selection by saved world ID, including success, failure, empty reachable set, and equivalent distinct codes when present. Use fixed `[0,1]` scales for beliefs/occupancy and symmetric `[-1,1]` differences, annotate undefined metrics/absent equivalence examples, and wrap parameter labels. Every figure and HTML index links its saved source arrays. The adapter computes differences internally and passes only the typed belief input; no special producer path reaches the CNN.
- [ ] **Step 4: Run** visual/adapter tests, open all smoke figures, and check clipping, labels, padding display, and truthful empty/ambiguous cases. Rendering must not rerun simulation or inference.
- [ ] **Step 5: Commit**: `feat: visualize saved inverse-world results and adapt belief producers`.

### Task 8: CLI, end-to-end demonstration, and research gate

**Files:** Create `model2/cli.py`, `tests/model2/test_cli.py`, `tests/model2/test_end_to_end.py`; modify `src/perception_env/cli.py`, `README.md`.

**Interfaces:** Add lazy `perception-env model2` subcommands:

- `fixtures --config PATH --output DIR` → `generate_fixtures`.
- `train --data DIR --config PATH --output DIR` → `train_model`.
- `evaluate --data DIR --checkpoint PATH --config PATH --output DIR` → `evaluate_model`.
- `controls --data DIR --config PATH --output DIR` → `run_controls`; privileged control requires explicit `--include-first-order-ablation`.
- `inspect --data DIR --evaluation DIR --output DIR` → `render_evaluation`.
- `demo --config PATH --output DIR` → fixtures, production train/evaluate, unprivileged controls, saved figures and index under separate child directories.
- `main(argv: Sequence[str] | None = None) -> int` returns 2 for input/config errors and 0 for a successfully executed experiment, even if the scientific gate is unmet; record that gate independently.

- [ ] **Step 1: Write failing CLI/end-to-end tests.** Smoke demo must train actual weights, write a reloadable checkpoint, regenerate held-out labels, and render from saved arrays. Assert production checkpoint architecture, exactly two input-array keys, three channels/four events, disjoint world splits, separated privileged control, and all evidence paths/hashes. Reject existing output, source/output overlap, incompatible checkpoint schema, missing torch with actionable `uv sync --extra ml` text, bad configs and empty validation splits. No force/delete flag in these new commands. Guard opens and forward hooks through the end-to-end path to prove diagnostics and evaluator truth never reach the CNN.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/model2/test_cli.py tests/model2/test_end_to_end.py -q`; expect unknown command/API failures.
- [ ] **Step 3: Implement** CLI dispatch, manifests and README commands. Record command/config, source hashes, split IDs, seed, schema versions, package/device versions, model parameter count, selected epoch, training duration, rejection counts, control mode and artifact hashes. Separate `software_checks_passed` from `research_gate_passed`. Research gate requires paired 95% world-bootstrap lower confidence bound above zero for held-out possibility agreement improvement against both constant and single-belief controls, across the registered three training seeds, plus a passed leakage suite. Average per-world effects over seeds before bootstrapping; save individual-seed scores. If unsupported by data, report `not_established`, keep the evidence, and do not tune on test results or claim success. Software acceptance requires functional real learning and accurate reporting, not a fabricated scientific win.
- [ ] **Step 4: Run** `.venv/bin/pytest -q` and `.venv/bin/ruff check .`; expect all tests pass and no violations. Run `.venv/bin/perception-env model2 demo --config configs/model2/smoke_v1.yaml --output artifacts/model2-smoke`, then a second output directory; compare canonical array/state-dict content hashes, not container bytes/timings. Run `inspect` with simulator/inference unavailable and view every figure. Execute the research configuration for seeds 7, 19 and 31 with separate run directories (CLI permits `TrainingConfig.seed` override through `--seed` on `train`, `controls`, and `demo`, leaving `FixtureConfig.seed` fixed), collect the registered comparison, and report the actual gate outcome. Any training resource limit remains an explicit outstanding research run, not a completed gate.
- [ ] **Step 5: Commit**: `feat: ship reproducible model2 training and evaluation commands`.

## Final acceptance and handoff

- [ ] Input-only archive/schema, forward signature, loader isolation, and direct construction leakage tests all pass.
- [ ] CPU smoke training updates real Conv2d weights and lowers a fixed learnable-fixture loss; checkpoints reload identically in the same runtime.
- [ ] Inactive/unidentified parameters have zero masked loss gradients; raw parameter errors and decoder projections are separately visible.
- [ ] Histories have a persistent world, observed assimilation events and grouped world-disjoint splits; no held-out probe enters fitting or identifiability selection.
- [ ] Every saved prediction can regenerate possibility labels with explicit uncertainty/tolerance; empty/undefined cases and ambiguity remain visible.
- [ ] Saved visuals reproduce without model/simulator execution and all arrays/configurations/checkpoints have verifiable content identities.
- [ ] Constant, single-belief, shuffled-history and explicitly isolated privileged controls are distinguishable in files and reports.
- [ ] Model 1 adapter is tested with synthetic producer packets; real Model 1 integration and online streaming remain Subplans 2/4.
- [ ] Report test count, lint result, reproduction hashes, measured metrics and research gate status. Never substitute passing software tests for evidence that the scientific hypothesis succeeded.

## Reference notes for implementation

PyTorch documents deterministic execution controls and the limitations of cross-platform reproducibility in its [reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html). Use explicit state-dict persistence and `weights_only=True` as described in [saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models) and [torch.load](https://docs.pytorch.org/docs/stable/generated/torch.load.html). Freeze the resolved torch version in `uv.lock`; the CPU reference guarantee applies to that runtime, not arbitrary future releases.
