# Model 1 Belief Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, real PyTorch Model 1 that turns first-order occlusion clips into calibrated weighted trajectory hypotheses and exports only their pre/post belief superpositions for Model 2.

**Architecture:** A compact convolutional frame encoder and GRU clip encoder feed an auxiliary visible-state head plus a best-of-eight trajectory-mixture head. The same network runs once before reappearance and again after assimilation; a differentiable bilinear rasterizer converts each weighted mixture into time-resolved and collapsed belief tensors. Training examples, oracle pre-beliefs, and actual-path post targets come from the existing deterministic environment, while the only downstream bridge is the existing `BeliefEvent`/`Model2InputRecord` belief-only contract.

**Tech Stack:** Python 3.11, PyTorch 2.5+, NumPy, PyYAML, Matplotlib, Pillow, pytest, existing `perception_env` simulator and records.

**Spec:** `docs/superpowers/plans/2026-10-30-perceptual-coherence-research-roadmap.md`; `docs/superpowers/specs/2026-10-30-parameterized-occlusion-environment-design.md`

## Global Constraints

- Model 1 may consume rendered frames, frame times, padding masks, and first-order supervision during training; no Model 1 first-order field may cross the Model 2 bridge.
- Model 2's production input remains a four-event history containing only `pre_belief`, `post_belief`, `belief_difference`, belief-derived entropy, and a history mask.
- Do not provide `world_flat`, family level, latent state, visibility, frame features, or outcome coordinates as Model 1 inputs; family/world fields may exist only in saved diagnostic metadata and dataset stratification.
- Use one ball, static occluders, 64-by-64 grayscale canonical frames, normalized coordinates in `[0, 1]`, and the existing versioned `WorldSchema.v1()`.
- The canonical Model 1 trajectory mixture has eight modes and twelve predicted steps; both values are configurable but fixed within a checkpoint.
- The pre view ends at the last fully hidden frame before reappearance; the post view contains the same event through its first visible reappearance. Both predict the trajectory beginning at the first hidden step.
- State supervision is auxiliary and ablatable by setting its loss weight to zero; the model must never receive ground-truth state as an input.
- Every random operation uses an explicit registered seed; dataset splits, initialization, batching, and saved evaluation outputs must replay deterministically on CPU.
- Every visual must render from a saved prediction record without rerunning simulation or training.
- Keep this subplan independent of Model 2 training, the streaming loop, and the final coherence/shattering experiment.

## Review Focus

- A padded clip must produce the same prediction when values in masked timesteps change; pin this in Task 2's encoder test.
- A trajectory that reaches the image boundary or falls just outside `[0, 1]` must rasterize to finite bounded beliefs without invalid indexing; pin this in Task 3's rasterizer test.
- A sample whose available target is shorter than twelve steps must contribute loss only on valid target steps; pin this in Task 4's masked-loss test.
- A degenerate mixture with one dominant or non-finite logit must either remain finite after softmax or fail with an explicit validation error; pin this in Tasks 2 and 4.
- The Model 2 export path must reject frames, states, family labels, mode coordinates, and weights, leaving only belief-derived arrays; pin this in Task 3's bridge test and Task 8's end-to-end test.

---

## File map

- `configs/model1/v1.yaml`: frozen dataset, model, loss, training, and evaluation defaults.
- `src/perception_env/model1/config.py`: typed YAML parsing and validation.
- `src/perception_env/model1/data.py`: durable Model 1 example records, seeded environment generation, split manifests, datasets, and collation.
- `src/perception_env/model1/types.py`: tensor-only batch, trajectory-mixture, belief-prediction, and evaluation record types.
- `src/perception_env/model1/model.py`: frame CNN, temporal encoder, state head, trajectory-mixture head, and `Model1`.
- `src/perception_env/model1/rasterize.py`: differentiable trajectory splatting, temporal collapse, actual-path targets, and the belief-only bridge.
- `src/perception_env/model1/losses.py`: masked best-of-mixture, mode, state, and belief losses.
- `src/perception_env/model1/metrics.py`: displacement, likelihood, calibration, coverage, and belief metrics.
- `src/perception_env/model1/train.py`: deterministic loaders, training/validation loop, and checkpoint IO.
- `src/perception_env/model1/evaluate.py`: prediction-record generation and aggregate/by-family reports.
- `src/perception_env/model1/visualize.py`: example panels, calibration plots, and family-grouped failure galleries from saved records.
- `src/perception_env/model1/cli.py`: `generate`, `train`, `evaluate`, `inspect`, and `demo` commands.
- `tests/model1/`: focused tests matching the modules above.

### Task 1: Freeze Model 1 configuration and generate environment-backed records

**Files:**
- Create: `src/perception_env/model1/__init__.py`
- Create: `src/perception_env/model1/config.py`
- Create: `src/perception_env/model1/data.py`
- Create: `src/perception_env/model1/types.py`
- Create: `configs/model1/v1.yaml`
- Modify: `pyproject.toml`
- Test: `tests/model1/test_config.py`
- Test: `tests/model1/test_data.py`

**Interfaces:**
- Consumes: `load_family(path: Path, level: str) -> WorldFamily`, `sample_worlds(family, count, seed) -> list[WorldCode]`, `rollout`, `render_sequence`, `extract_reappearance`, `enumerate_reachable`, and `rasterize_endpoints` from Subplan 1.
- Produces: `Model1Config`, `Model1ExampleRecord`, `Model1Dataset`, `Model1Batch`, `generate_model1_dataset(...) -> DatasetManifest`, `save_example(...) -> Path`, `load_example(...) -> Model1ExampleRecord`, and `collate_model1(...) -> Model1Batch`.

- [ ] **Step 1: Write failing configuration tests**

  In `tests/model1/test_config.py`, assert that `load_model1_config(Path("configs/model1/v1.yaml"))` fixes: 64-by-64 frames, clip length 16, horizon 12, 8 modes, hidden width 64, W0-W4 training levels, CPU seed 7, and nonnegative loss weights. Add parameterized failures for zero modes, horizon, batch size, or calibration radius.

- [ ] **Step 2: Run the configuration tests and verify import failure**

  Run: `pytest tests/model1/test_config.py -v`

  Expected: FAIL because `perception_env.model1.config` does not exist.

- [ ] **Step 3: Add the dependency and typed configuration**

  Add `torch>=2.5` to project dependencies. Implement frozen dataclasses `DatasetConfig`, `ModelConfig`, `LossConfig`, `TrainingConfig`, `EvaluationConfig`, and `Model1Config`, plus:

  ```python
  def load_model1_config(path: Path) -> Model1Config
  ```

  The YAML defaults are: `clip_steps=16`, `prediction_steps=12`, `modes=8`, `frame_size=64`, `hidden_size=64`, `batch_size=32`, `epochs=20`, `learning_rate=1e-3`, `weight_decay=1e-4`, `calibration_radius=0.05`, `calibration_bins=10`, split counts `256/64/64`, and quick counts `16/8/8`. Define initial-state ranges, normalized-state velocity support `[-4.0, 4.0]`, and one vertical occluder in the file, not in Python. Assign requested examples across W0-W4 round-robin so each split's family counts differ by at most one.

- [ ] **Step 4: Run configuration tests**

  Run: `pytest tests/model1/test_config.py -v`

  Expected: PASS.

- [ ] **Step 5: Write failing durable-record and dataset tests**

  In `tests/model1/test_data.py`, cover:

  - save/load round-trip for a `Model1ExampleRecord` containing frames, frame indices, hidden/reappearance indices, target states, target mask, oracle pre-belief, family level, seed, and world identifier;
  - `generate_model1_dataset(..., quick=True)` produces deterministic train/validation/test IDs on two runs with seed 7;
  - all accepted records have at least two fully hidden steps and a visible reappearance;
  - a collated batch has `pre_frames` and `post_frames` shaped `[B,16,1,64,64]`, corresponding `[B,16]` time/mask tensors, `target_positions` shaped `[B,12,2]`, `target_mask` shaped `[B,12]`, normalized `entry_state` shaped `[B,4]`, and oracle/path beliefs shaped `[B,64,64]`;
  - changing `world_flat` or family metadata does not alter model inputs; changing a copied latent target changes supervision but not `pre_frames`, `post_frames`, frame times, or frame masks;
  - generation stops with an explicit `RuntimeError` and rejection counts when `max_attempts` is exhausted.

- [ ] **Step 6: Run data tests and verify failure**

  Run: `pytest tests/model1/test_data.py -v`

  Expected: FAIL because the record and generator are missing.

- [ ] **Step 7: Implement the tensor and durable-record contracts**

  In `types.py`, define validated frozen dataclasses:

  ```python
  @dataclass(frozen=True)
  class Model1Batch:
      pre_frames: torch.Tensor       # [B,T,1,H,W]
      pre_times: torch.Tensor        # [B,T]
      pre_mask: torch.Tensor         # bool [B,T]
      post_frames: torch.Tensor
      post_times: torch.Tensor
      post_mask: torch.Tensor
      entry_state: torch.Tensor      # normalized [B,4], target only
      target_positions: torch.Tensor # [B,P,2]
      target_mask: torch.Tensor      # bool [B,P]
      oracle_pre_belief: torch.Tensor # [B,H,W]
      actual_path_belief: torch.Tensor # [B,H,W]
      example_ids: tuple[str, ...]
      family_levels: tuple[str, ...] # diagnostics only; never passed to Model1.forward
  ```

  In `data.py`, define `Model1ExampleRecord`, `DatasetManifest`, `save_example`, `load_example`, and deterministic record IDs derived from schema version, split, seed, and accepted-example index.

- [ ] **Step 8: Implement environment-backed generation and batching**

  Implement:

  ```python
  def generate_model1_dataset(
      config: Model1Config,
      family_config: Path,
      output: Path,
      *,
      seed: int,
      quick: bool = False,
  ) -> DatasetManifest

  class Model1Dataset(torch.utils.data.Dataset[Model1ExampleRecord]): ...

  def collate_model1(
      records: Sequence[Model1ExampleRecord], config: Model1Config
  ) -> Model1Batch
  ```

  Generate worlds and initial states from explicit config ranges; reject invalid rollouts, entry states outside the configured `[-4.0, 4.0]` velocity normalization support, or events without the required hidden run/reappearance. The pre view ends at `hidden_end`; the post view ends at `reappearance_step`; clips are chronologically cropped to the last 16 frames then right-padded. Normalize pixels to `[0,1]`, time relative to `hidden_start` by the twelve-step horizon, and entry velocity using the fixed normalization support. Targets begin at `hidden_start`, use a mask for unavailable tail steps, use the oracle endpoint raster as `oracle_pre_belief`, and rasterize the one actual latent path as `actual_path_belief`.

- [ ] **Step 9: Run Task 1 tests and the existing environment suite**

  Run: `pytest tests/model1/test_config.py tests/model1/test_data.py tests/test_records.py tests/test_end_to_end.py -v`

  Expected: PASS.

- [ ] **Step 10: Commit Task 1**

  ```bash
  git add pyproject.toml configs/model1/v1.yaml src/perception_env/model1 tests/model1/test_config.py tests/model1/test_data.py
  git commit -m "feat: add Model 1 dataset contract"
  ```

### Task 2: Implement the clip/state encoder and multimodal trajectory model

**Files:**
- Create: `src/perception_env/model1/model.py`
- Modify: `src/perception_env/model1/types.py`
- Test: `tests/model1/test_model.py`

**Interfaces:**
- Consumes: `Model1Batch` clip tensors and `ModelConfig` from Task 1.
- Produces: `TrajectoryMixture` and `Model1.forward(frames, times, mask) -> TrajectoryMixture`.

- [ ] **Step 1: Write failing model contract tests**

  In `tests/model1/test_model.py`, assert that a batch of two clips yields:

  ```python
  prediction.positions.shape == (2, 8, 12, 2)
  prediction.logits.shape == (2, 8)
  prediction.weights.shape == (2, 8)
  prediction.entry_state.shape == (2, 4)
  ```

  Assert positions and normalized state lie in `[0,1]`, weights are finite/nonnegative and sum to one, backward propagation reaches the frame encoder, and CPU initialization/forward are identical under a reset seed. Add `test_masked_timesteps_do_not_change_prediction` by replacing every padded pixel/time with large random values and comparing outputs. Add a validation test that rejects non-finite logits before belief export.

- [ ] **Step 2: Run the model tests and verify failure**

  Run: `pytest tests/model1/test_model.py -v`

  Expected: FAIL because `model.py` and `TrajectoryMixture` are missing.

- [ ] **Step 3: Implement validated prediction types**

  Add to `types.py`:

  ```python
  @dataclass(frozen=True)
  class TrajectoryMixture:
      positions: torch.Tensor # [B,M,P,2]
      logits: torch.Tensor    # [B,M]
      entry_state: torch.Tensor # normalized [B,4]

      @property
      def weights(self) -> torch.Tensor: ...

      def validate(self) -> None: ...
  ```

  Validation fixes the rank relationships and rejects non-finite values; softmax is computed in float32.

- [ ] **Step 4: Implement the small PyTorch model**

  In `model.py`, implement:

  ```python
  class FrameEncoder(nn.Module):
      def forward(self, frames: Tensor) -> Tensor: ...

  class ClipStateEncoder(nn.Module):
      def forward(self, frames: Tensor, times: Tensor, mask: Tensor) -> Tensor: ...

  class TrajectoryMixtureHead(nn.Module):
      def forward(self, encoding: Tensor) -> tuple[Tensor, Tensor]: ...

  class Model1(nn.Module):
      def forward(self, frames: Tensor, times: Tensor, mask: Tensor) -> TrajectoryMixture: ...
  ```

  Use three stride-two `Conv2d` blocks with channels 8/16/32, global average pooling, concatenated scalar frame time, and a one-layer GRU of width 64 using packed valid lengths so right padding is ignored. A sigmoid state head emits four normalized values. The mixture head emits eight logits and absolute normalized coordinates `[M,12,2]` via sigmoid. Do not pass `entry_state`, world code, family level, or visibility into `forward`.

- [ ] **Step 5: Run model tests**

  Run: `pytest tests/model1/test_model.py -v`

  Expected: PASS.

- [ ] **Step 6: Commit Task 2**

  ```bash
  git add src/perception_env/model1/model.py src/perception_env/model1/types.py tests/model1/test_model.py
  git commit -m "feat: predict weighted hidden trajectories"
  ```

### Task 3: Rasterize trajectory mixtures and enforce the belief-only bridge

**Files:**
- Create: `src/perception_env/model1/rasterize.py`
- Modify: `src/perception_env/model1/types.py`
- Test: `tests/model1/test_rasterize.py`
- Test: `tests/model1/test_bridge.py`

**Interfaces:**
- Consumes: `TrajectoryMixture`; existing `BeliefEvent`, `build_belief_history`, and `Model2InputRecord`.
- Produces: `BeliefPrediction`, `rasterize_mixture(...) -> BeliefPrediction`, `rasterize_actual_path(...) -> Tensor`, and `belief_event_from_predictions(...) -> BeliefEvent`.

- [ ] **Step 1: Write failing rasterization tests**

  In `tests/model1/test_rasterize.py`, use hand-authored coordinates and weights to assert:

  - `time_belief` shape `[B,12,H,W]` and `collapsed` shape `[B,H,W]`;
  - an interior coordinate uses bilinear weights whose per-time mass is one;
  - mode weights 0.25/0.75 produce the corresponding mixture mass;
  - temporal collapse uses `1 - product(1 - p_t)` per mode before the weighted mode sum, so every pixel stays in `[0,1]`;
  - exact edges, slightly out-of-range values, and repeated points remain finite, bounded, and differentiable;
  - gradients reach predicted positions and logits.

- [ ] **Step 2: Run rasterization tests and verify failure**

  Run: `pytest tests/model1/test_rasterize.py -v`

  Expected: FAIL because the rasterizer is missing.

- [ ] **Step 3: Implement differentiable belief rasterization**

  Add:

  ```python
  @dataclass(frozen=True)
  class BeliefPrediction:
      time_belief: torch.Tensor # [B,P,H,W]
      collapsed: torch.Tensor   # [B,H,W]

  def rasterize_mixture(
      prediction: TrajectoryMixture, *, height: int, width: int
  ) -> BeliefPrediction

  def rasterize_actual_path(
      positions: Tensor, mask: Tensor, *, height: int, width: int
  ) -> Tensor
  ```

  Clamp only the splat coordinates used for indexing, while zeroing contributions for points outside a one-pixel margin. Use `scatter_add` for four-neighbor bilinear splats; do not round coordinates or detach tensors.

- [ ] **Step 4: Run rasterization tests**

  Run: `pytest tests/model1/test_rasterize.py -v`

  Expected: PASS.

- [ ] **Step 5: Write failing bridge-isolation tests**

  In `tests/model1/test_bridge.py`, create pre/post `BeliefPrediction`s, call:

  ```python
  event = belief_event_from_predictions(pre, post, event_step=9)
  history = build_belief_history([event] * 4)
  record = Model2InputRecord.from_mapping(...)
  ```

  Assert the exported mapping's keys are exactly the six fields accepted by `Model2InputRecord`. Parameterize forbidden attempts using frames, entry state, latent positions, family level, modes, logits, or weights and assert rejection. Verify `belief_difference == post_belief - pre_belief` exactly.

- [ ] **Step 6: Implement the only Model 1-to-Model 2 bridge**

  Implement:

  ```python
  def belief_event_from_predictions(
      pre: BeliefPrediction,
      post: BeliefPrediction,
      *,
      event_step: int,
      batch_index: int = 0,
  ) -> BeliefEvent
  ```

  The function detaches only `collapsed[batch_index]`, converts it to CPU `float32` NumPy, and constructs `BeliefEvent`. It accepts no frames, coordinates, metadata, or model features.

- [ ] **Step 7: Run bridge and existing belief-contract tests**

  Run: `pytest tests/model1/test_bridge.py tests/model1/test_rasterize.py tests/test_beliefs.py tests/test_records.py -v`

  Expected: PASS.

- [ ] **Step 8: Commit Task 3**

  ```bash
  git add src/perception_env/model1/rasterize.py src/perception_env/model1/types.py tests/model1/test_rasterize.py tests/model1/test_bridge.py
  git commit -m "feat: rasterize Model 1 beliefs"
  ```

### Task 4: Add masked multimodal and belief objectives

**Files:**
- Create: `src/perception_env/model1/losses.py`
- Test: `tests/model1/test_losses.py`

**Interfaces:**
- Consumes: pre/post `TrajectoryMixture`, their `BeliefPrediction`s, `Model1Batch`, and `LossConfig`.
- Produces: `Model1LossTerms` and `model1_loss(...) -> Model1LossTerms`.

- [ ] **Step 1: Write failing objective tests**

  In `tests/model1/test_losses.py`, assert:

  - the best-of-eight trajectory loss selects the lowest masked ADE mode;
  - the mode cross-entropy target is that best mode and gradients reach every logit;
  - state MSE uses only the auxiliary entry-state target and becomes exactly zero-weighted when `state_weight=0`;
  - pre-belief BCE targets `oracle_pre_belief`, while post-belief BCE targets `actual_path_belief`;
  - changing padded target positions cannot change any loss;
  - a row with no valid target steps raises `ValueError`;
  - NaN positions/logits/beliefs raise `ValueError` before reduction.

- [ ] **Step 2: Run objective tests and verify failure**

  Run: `pytest tests/model1/test_losses.py -v`

  Expected: FAIL because `losses.py` is missing.

- [ ] **Step 3: Implement loss terms and composite loss**

  Implement:

  ```python
  @dataclass(frozen=True)
  class Model1LossTerms:
      total: Tensor
      trajectory: Tensor
      mode: Tensor
      state: Tensor
      pre_belief: Tensor
      post_belief: Tensor

  def model1_loss(
      pre_prediction: TrajectoryMixture,
      post_prediction: TrajectoryMixture,
      pre_belief: BeliefPrediction,
      post_belief: BeliefPrediction,
      batch: Model1Batch,
      config: LossConfig,
  ) -> Model1LossTerms
  ```

  Compute masked best-of-mixture ADE separately for pre and post, cross-entropy from logits to the detached best-mode index, normalized-state MSE for both views, and clamped binary cross-entropy for the two belief targets. Sum only the five explicit YAML weights; do not add an unconfigured smoothness or diversity term.

- [ ] **Step 4: Run objective tests**

  Run: `pytest tests/model1/test_losses.py -v`

  Expected: PASS.

- [ ] **Step 5: Commit Task 4**

  ```bash
  git add src/perception_env/model1/losses.py tests/model1/test_losses.py
  git commit -m "feat: train calibrated trajectory mixtures"
  ```

### Task 5: Train deterministically and save portable checkpoints

**Files:**
- Create: `src/perception_env/model1/train.py`
- Test: `tests/model1/test_train.py`

**Interfaces:**
- Consumes: config, datasets, `Model1`, rasterizer, and loss from Tasks 1-4.
- Produces: `TrainingResult`, `train_model1(...) -> TrainingResult`, `save_checkpoint(...) -> Path`, and `load_checkpoint(...) -> tuple[Model1, CheckpointMetadata]`.

- [ ] **Step 1: Write failing training/checkpoint tests**

  In `tests/model1/test_train.py`, create a deterministic four-example in-memory dataset and assert:

  - two CPU training runs with the same seed have identical loss histories and parameters;
  - 30 optimization steps overfit the four examples enough that final total loss is below 70% of initial loss;
  - validation runs under `torch.inference_mode()` and does not alter parameters;
  - a save/load round-trip preserves predictions bit-for-bit on CPU;
  - the checkpoint contains schema version, complete frozen config, epoch, seed, model state, optimizer state, best validation loss, and package versions;
  - loading a checkpoint with mismatched mode count, horizon, or schema version fails explicitly.

- [ ] **Step 2: Run training tests and verify failure**

  Run: `pytest tests/model1/test_train.py -v`

  Expected: FAIL because the training module is missing.

- [ ] **Step 3: Implement seeded loaders and the train/validation loop**

  Implement:

  ```python
  def seed_model1(seed: int) -> torch.Generator

  def train_model1(
      config: Model1Config,
      train_dataset: Model1Dataset,
      validation_dataset: Model1Dataset,
      output: Path,
      *,
      device: torch.device,
      seed: int,
  ) -> TrainingResult
  ```

  Use AdamW, configured batch size/epochs, deterministic CPU algorithms in tests, gradient-norm clipping at 1.0, and best-validation checkpoint selection. Record per-epoch component losses as JSON Lines. No scheduler, mixed precision, distributed training, or experiment tracker in this subplan.

- [ ] **Step 4: Implement checkpoint IO**

  Implement `save_checkpoint` and `load_checkpoint` with `torch.save`/`torch.load(weights_only=False)` to an explicit path. Validate schema/model dimensions before loading weights and restore optimizer state only when resuming training.

- [ ] **Step 5: Run training tests**

  Run: `pytest tests/model1/test_train.py -v`

  Expected: PASS.

- [ ] **Step 6: Commit Task 5**

  ```bash
  git add src/perception_env/model1/train.py tests/model1/test_train.py
  git commit -m "feat: train and checkpoint Model 1"
  ```

### Task 6: Measure trajectory calibration and assimilation quality

**Files:**
- Create: `src/perception_env/model1/metrics.py`
- Create: `src/perception_env/model1/evaluate.py`
- Test: `tests/model1/test_metrics.py`
- Test: `tests/model1/test_evaluate.py`

**Interfaces:**
- Consumes: saved checkpoint, test `Model1Dataset`, predictions, and beliefs.
- Produces: `TrajectoryMetrics`, `CalibrationReport`, `EvaluationSummary`, and `evaluate_model1(...) -> EvaluationSummary` plus durable prediction records.

- [ ] **Step 1: Write failing metric tests with analytic mixtures**

  In `tests/model1/test_metrics.py`, hand-construct mixtures whose answers are exact and assert:

  - masked `minADE`, `minFDE`, and top-one ADE;
  - fixed-kernel mixture NLL using isotropic sigma `0.03`;
  - top-mode correctness means masked ADE at most radius `0.05`;
  - ten-bin ECE is zero for perfectly calibrated fixtures and positive for overconfident wrong fixtures;
  - 50%, 80%, and 90% credible-mode sets are formed by descending weight until cumulative mass reaches the requested level, and coverage is whether any included mode is within the radius;
  - belief soft-IoU and BCE have exact values for identical, disjoint, and empty pairs;
  - post-minus-pre changes are reported with the sign “positive means post is better.”

- [ ] **Step 2: Run metric tests and verify failure**

  Run: `pytest tests/model1/test_metrics.py -v`

  Expected: FAIL because `metrics.py` is missing.

- [ ] **Step 3: Implement metrics without training-state dependencies**

  Implement pure tensor/NumPy functions:

  ```python
  def displacement_metrics(prediction, targets, mask) -> TrajectoryMetrics
  def mixture_nll(prediction, targets, mask, *, sigma: float = 0.03) -> Tensor
  def calibration_report(prediction, targets, mask, *, radius: float, bins: int) -> CalibrationReport
  def belief_metrics(predicted: Tensor, target: Tensor) -> dict[str, Tensor]
  ```

  Keep pre and post metrics separate and include deltas; never calibrate against padded steps.

- [ ] **Step 4: Run metric tests**

  Run: `pytest tests/model1/test_metrics.py -v`

  Expected: PASS.

- [ ] **Step 5: Write failing evaluation-record tests**

  In `tests/model1/test_evaluate.py`, run inference on two families and assert that evaluation writes one immutable prediction record per example plus `summary.json`. The summary must contain overall and per-family counts, all pre/post metrics, deltas, calibration bins, credible coverage, config/checkpoint hashes, seed, and no training batches. Loading a record must reconstruct its figures' inputs without a model or simulator.

- [ ] **Step 6: Implement evaluation and durable prediction records**

  Define `Model1PredictionRecord` with diagnostic arrays: input pre/post clips, masks/times, true positions/mask, pre/post mode positions and weights, pre/post time beliefs and collapsed beliefs, oracle/path belief targets, family level, example ID, and seed. Implement:

  ```python
  def evaluate_model1(
      checkpoint: Path,
      dataset: Model1Dataset,
      output: Path,
      *,
      device: torch.device,
      seed: int,
  ) -> EvaluationSummary
  ```

  The diagnostic record is not a Model 2 record. Export Model 2 inputs only through Task 3's belief bridge.

- [ ] **Step 7: Run evaluation tests**

  Run: `pytest tests/model1/test_metrics.py tests/model1/test_evaluate.py -v`

  Expected: PASS.

- [ ] **Step 8: Commit Task 6**

  ```bash
  git add src/perception_env/model1/metrics.py src/perception_env/model1/evaluate.py tests/model1/test_metrics.py tests/model1/test_evaluate.py
  git commit -m "feat: evaluate Model 1 calibration"
  ```

### Task 7: Render saved visual evidence and failure galleries

**Files:**
- Create: `src/perception_env/model1/visualize.py`
- Test: `tests/model1/test_visualize.py`

**Interfaces:**
- Consumes: saved `Model1PredictionRecord` files and `EvaluationSummary` from Task 6 only.
- Produces: `render_prediction_panel`, `render_calibration_plot`, and `render_failure_gallery`, each returning the existing `VisualArtifact` shape/path convention.

- [ ] **Step 1: Write failing saved-record visual tests**

  In `tests/model1/test_visualize.py`, save a deterministic prediction record, delete the model and source dataset references, and assert:

  - `render_prediction_panel(record, path)` creates a nonempty PNG showing input clip, true path, all eight pre/post modes with opacity proportional to weight, and pre/post/difference belief images;
  - `render_calibration_plot(summary, path)` includes top-mode reliability and 50/80/90 credible coverage;
  - `render_failure_gallery(records, path, top_n=8)` ranks by pre minADE, groups or labels by family, and handles fewer than eight records;
  - repeated renders have identical dimensions and pixel arrays under the Agg backend;
  - no visual function imports simulation, training, or model modules.

- [ ] **Step 2: Run visual tests and verify failure**

  Run: `pytest tests/model1/test_visualize.py -v`

  Expected: FAIL because `visualize.py` is missing.

- [ ] **Step 3: Implement record-only visuals**

  Implement:

  ```python
  def render_prediction_panel(record: Model1PredictionRecord, path: Path) -> VisualArtifact
  def render_calibration_plot(summary: EvaluationSummary, path: Path) -> VisualArtifact
  def render_failure_gallery(
      records: Sequence[Model1PredictionRecord], path: Path, *, top_n: int = 8
  ) -> VisualArtifact
  ```

  Use fixed colors/scales and include example ID, family, seed, minADE, confidence, and belief IoU in titles. Keep actual first-order frames confined to the diagnostic input panel; label belief-only panels explicitly.

- [ ] **Step 4: Run visual tests**

  Run: `pytest tests/model1/test_visualize.py tests/test_belief_visuals.py -v`

  Expected: PASS.

- [ ] **Step 5: Commit Task 7**

  ```bash
  git add src/perception_env/model1/visualize.py tests/model1/test_visualize.py
  git commit -m "feat: visualize Model 1 beliefs"
  ```

### Task 8: Add the reproducible CLI and end-to-end belief-only acceptance test

**Files:**
- Create: `src/perception_env/model1/cli.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Test: `tests/model1/test_cli.py`
- Test: `tests/model1/test_end_to_end.py`

**Interfaces:**
- Consumes: all Model 1 modules and existing `Model2InputRecord` validation.
- Produces: `perception-model1` CLI, manifests, checkpoints, evaluation records, visual index, and belief-only Model 2 fixture.

- [ ] **Step 1: Write failing CLI tests**

  In `tests/model1/test_cli.py`, call `main([...])` directly and assert:

  - `generate` creates the configured splits and a deterministic manifest;
  - `train` creates best/last checkpoints and loss history;
  - `evaluate` creates summary, saved prediction records, example panels, calibration plot, failure gallery, and `index.html`;
  - `inspect` regenerates all figures from saved records after the source dataset is moved away;
  - every command refuses to overwrite an existing output unless `--force` is set, and refuses broad/overlapping source-output paths;
  - manifests include command arguments, config/checkpoint/content hashes, schema versions, seed, package versions, rejection counts, output paths, and wall-clock duration outside the deterministic hash.

- [ ] **Step 2: Run CLI tests and verify failure**

  Run: `pytest tests/model1/test_cli.py -v`

  Expected: FAIL because the CLI is missing.

- [ ] **Step 3: Implement the CLI and package entry point**

  Add `perception-model1 = "perception_env.model1.cli:main"` and implement:

  ```text
  perception-model1 generate --config ... --family-config ... --seed 7 --output ... [--quick]
  perception-model1 train --config ... --data ... --seed 7 --output ... [--device cpu]
  perception-model1 evaluate --config ... --data ... --checkpoint ... --seed 7 --output ...
  perception-model1 inspect --run ... --output ...
  perception-model1 demo --config ... --family-config ... --seed 7 --output ... --quick
  ```

  `demo --quick` runs tiny generation, two training epochs, test evaluation, and inspection. Reuse the safe-output, hash, manifest, version, and HTML conventions from `perception_env.cli` without importing its private functions; extract shared helpers to a public `src/perception_env/artifacts.py` only if duplication would otherwise diverge behavior, and add focused regression tests if extraction is needed.

- [ ] **Step 4: Write the failing end-to-end invariant test**

  In `tests/model1/test_end_to_end.py`, run `demo --quick`, load four saved belief events, build a `Model2InputRecord`, and assert its archive keys are exactly:

  ```python
  {
      "pre_belief",
      "post_belief",
      "belief_difference",
      "entropy",
      "history_mask",
      "target_world",
  }
  ```

  Assert forbidden first-order and Model 1 latent fields are absent, the pre/post beliefs differ for at least one event, all arrays are finite, repeated seed-7 runs have the same deterministic hash, and all visual artifacts are nonempty.

- [ ] **Step 5: Implement the demo export and documentation**

  Export one `model2-input.npz` from four belief events using the existing `build_belief_history`, `belief_entropy`, and `Model2InputRecord`. `target_world` comes from dataset metadata only at fixture-construction time and is never passed into Model 1. Update `README.md` with install, quick-demo, output-layout, metric definitions, state-supervision ablation, and the explicit first-order/belief-only boundary.

- [ ] **Step 6: Run the complete acceptance suite**

  Run: `pytest tests/model1 tests -q`

  Expected: PASS with no Model 2 input leakage and no existing environment regression.

  Run: `ruff check src tests`

  Expected: PASS.

- [ ] **Step 7: Run the deterministic visual demonstration**

  Run:

  ```bash
  .venv/bin/perception-model1 demo \
    --config configs/model1/v1.yaml \
    --family-config configs/world_families/v1.yaml \
    --seed 7 \
    --output artifacts/model1-demo \
    --quick
  ```

  Expected: exit 0 and produce `index.html`, checkpoints, summary, prediction records, calibration plot, family failure gallery, example panels, `model2-input.npz`, and a manifest with their content hashes.

- [ ] **Step 8: Commit Task 8**

  ```bash
  git add pyproject.toml README.md src/perception_env/model1/cli.py tests/model1/test_cli.py tests/model1/test_end_to_end.py
  git commit -m "feat: ship Model 1 belief demonstration"
  ```

## Self-review outcome

- **Spec coverage:** The plan covers environment-generated training data, a real clip/state encoder, multimodal weighted paths, differentiable belief superpositions, pre/post assimilation, state-supervision ablation, calibration, saved visual proof, CLI reproducibility, and the belief-only Model 2 boundary. Streaming history integration beyond fixture export remains correctly deferred to Subplan 4.
- **Step scan:** Each task has a failing test, one bounded implementation responsibility, verification, and a commit. Algorithms are specified only where their choice changes the research object: clipping/window alignment, mixture loss, rasterization, and calibration.
- **Type consistency:** `Model1Batch -> Model1 -> TrajectoryMixture -> BeliefPrediction -> BeliefEvent -> Model2InputRecord` is used consistently throughout. Shapes are fixed by config and checked at every boundary.
- **Review Focus coverage:** Masked frames are pinned in Task 2; boundaries in Task 3; short targets and non-finite loss inputs in Task 4; non-finite logits in Task 2; first-order leakage in Tasks 3 and 8.
- **Proportion:** The plan deliberately excludes Model 2, streaming parameter updates, and final shattering/coherence analysis; each is already a separate roadmap subplan.

