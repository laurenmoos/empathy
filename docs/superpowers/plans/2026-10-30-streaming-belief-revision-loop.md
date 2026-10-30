# Streaming Belief-Revision Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic continual-learning prototype that carries one latent-world estimate across repeated occlusion events, feeds only Model 1 belief revisions to Model 2, and visibly improves its world and possibility estimates over time.

**Architecture:** A seeded event generator produces several occlusion episodes from one persistent ground-truth world. `StreamingSession` owns a four-event belief buffer and a current predicted world; it asks a Model 1 runtime for pre/post belief images, passes a belief-only `Model2Features` object to a Model 2 runtime, and feeds the predicted world back to Model 1 for a bounded correction loop. A separate replay learner may update network weights between completed events, but its calls and checkpoints are logged independently from per-event latent-state updates.

**Tech Stack:** Python 3.11, NumPy, PyTorch (from Subplans 2–3), Pillow/Matplotlib, dataclasses, pytest, existing `perception_env` simulator and record formats.

**Spec:** `docs/superpowers/specs/2026-10-30-parameterized-occlusion-environment-design.md`

## Global Constraints

- One persistent `WorldCode` produces every event in a stream; changing initial ball state or occluder geometry does not change the world.
- Model 2 receives only four-event pre/post/difference belief histories, derived entropy, and the history mask; raw frames, positions, masks, reappearance markers, targets, and ground-truth metadata are forbidden.
- Model 1 may consume first-order frames and a predicted-world prior, but the ground-truth world is never supplied to its inference path.
- Online latent-world updates and network-weight updates use different APIs, logs, counters, and checkpoint fields.
- Every run is deterministic from configuration, seed, and input checkpoints; evaluation renders from saved records without rerunning inference.
- The correction loop stops when the observed reappearance is covered or after `max_feedback_iterations`; it never loops without a fixed bound.
- Canonical images are 64×64, world codes use `WorldSchema.v1()`, belief history length is four, and inactive world parameters remain canonical zeros.

## Review Focus

- A partially filled history must be right-aligned and masked; padded zero planes must not influence Model 2.
- A reappearance outside every current belief must cause bounded correction, then preserve an explicit unresolved status rather than silently reporting success.
- Missing or rejected events must not enter the replay buffer or advance the four-event history.
- Resuming from a saved session must reproduce the next event, world prediction, and update counters bit-for-bit.
- A weight update must not mutate the current latent world prediction, and a latent update must not change model parameters.

---

### Task 1: Define the stream and model-runtime contracts

**Files:**
- Create: `src/perception_env/streaming.py`
- Test: `tests/test_streaming.py`

**Interfaces:**
- Consumes: `WorldCode`, `WorldSchema`, `BeliefEvent`, `build_belief_history`, `Model2InputRecord`-compatible belief arrays, and saved Model 1/Model 2 checkpoints.
- Produces: `OcclusionEpisode`, `Model2Features`, `WorldPrediction`, `BeliefRuntime`, `WorldRuntime`, `StreamUpdate`, and `StreamingSessionState`.

- [ ] **Step 1: Write failing contract tests**

  Add `test_model2_features_rejects_first_order_fields`, `test_partial_history_is_right_aligned_and_masked`, and `test_session_state_round_trips_without_model_objects`. Assert `Model2Features.from_mapping(...)` accepts exactly `pre_belief`, `post_belief`, `belief_difference`, `entropy`, and `history_mask`, requires `[4,H,W]` belief arrays, and rejects `frames`, `position`, `visible`, `outcome`, and `target_world`.

- [ ] **Step 2: Run the tests and verify RED**

  Run: `.venv/bin/pytest tests/test_streaming.py -q`

  Expected: FAIL because `perception_env.streaming` does not exist.

- [ ] **Step 3: Implement immutable data contracts and protocols**

  In `streaming.py`, add:

  - `OcclusionEpisode(event_id: str, context_id: str, frames: np.ndarray, visible: np.ndarray, reappearance_step: int, reappearance_xy: tuple[float, float], evaluation: dict[str, Any])`;
  - `Model2Features.from_events(events: Sequence[BeliefEvent], length: int = 4) -> Model2Features` and strict `from_mapping` validation with no target field;
  - `WorldPrediction(flat_world: np.ndarray, activity_probabilities: np.ndarray, parameter_values: np.ndarray, confidence: float)`;
  - `BeliefRuntime.propose(episode: OcclusionEpisode, world_prior: np.ndarray | None) -> np.ndarray` and `BeliefRuntime.assimilate(episode: OcclusionEpisode, pre_belief: np.ndarray) -> np.ndarray` protocols;
  - `WorldRuntime.predict(features: Model2Features) -> WorldPrediction` protocol;
  - JSON-safe `StreamUpdate` and `StreamingSessionState` dataclasses whose state contains arrays, counters, event IDs, and belief events but no live model objects.

- [ ] **Step 4: Run focused and full tests**

  Run: `.venv/bin/pytest tests/test_streaming.py -q && .venv/bin/pytest -q`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/perception_env/streaming.py tests/test_streaming.py
  git commit -m "feat: define streaming belief-only contracts"
  ```

### Task 2: Generate deterministic persistent-world event streams

**Files:**
- Modify: `src/perception_env/streaming.py`
- Create: `src/perception_env/stream_records.py`
- Test: `tests/test_stream_records.py`

**Interfaces:**
- Consumes: `SimulationConfig`, `RenderConfig`, `Occluder`, `rollout`, `render_sequence`, `extract_reappearance`, a single `WorldCode`, and a sequence of initial states.
- Produces: `generate_stream(world: WorldCode, initial_states: Sequence[BallState], occluders: Sequence[Occluder], simulation: SimulationConfig, render: RenderConfig, stream_id: str) -> tuple[OcclusionEpisode, ...]`, plus `save_stream` and `load_stream`.

- [ ] **Step 1: Write failing persistent-world and rejection tests**

  Add `test_every_episode_uses_one_persistent_world`, `test_event_ids_and_outputs_are_seed_reproducible`, `test_episode_requires_visible_reappearance`, and `test_rejected_episode_does_not_advance_sequence`. Assert saved evaluation metadata contains the true world only under an `evaluation` namespace.

- [ ] **Step 2: Run the tests and verify RED**

  Run: `.venv/bin/pytest tests/test_stream_records.py -q`

  Expected: FAIL because stream generation and persistence are missing.

- [ ] **Step 3: Implement stream generation and durable records**

  Reject episodes without the configured minimum fully hidden interval and later visible reappearance. Save `metadata.json` and `arrays.npz` per stream, including episode frames and evaluation-only true state/world fields. `load_stream` must reconstruct episodes without simulation.

- [ ] **Step 4: Run focused and full tests**

  Run: `.venv/bin/pytest tests/test_stream_records.py -q && .venv/bin/pytest -q`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/perception_env/streaming.py src/perception_env/stream_records.py tests/test_stream_records.py
  git commit -m "feat: add persistent-world event streams"
  ```

### Task 3: Implement bounded belief revision and world feedback

**Files:**
- Modify: `src/perception_env/streaming.py`
- Test: `tests/test_streaming.py`

**Interfaces:**
- Consumes: the Task 1 runtimes, Task 2 episodes, `WorldSchema.v1().from_flat`, and a possibility-raster callback `possibility(world: WorldCode, episode: OcclusionEpisode) -> np.ndarray`.
- Produces: `StreamingSession.process(episode: OcclusionEpisode) -> StreamUpdate`, `StreamingSession.snapshot() -> StreamingSessionState`, and `StreamingSession.restore(...) -> StreamingSession`.

- [ ] **Step 1: Write failing feedback-loop tests**

  Add `test_uncovered_outcome_triggers_world_feedback`, `test_loop_stops_when_outcome_is_covered`, `test_unresolved_loop_stops_at_iteration_limit`, `test_history_keeps_last_four_completed_events`, and `test_checkpoint_resume_reproduces_next_update`. Use deterministic fake runtimes and assert exact call order: propose → assimilate → world predict → possibility check → optional re-propose.

- [ ] **Step 2: Run the tests and verify RED**

  Run: `.venv/bin/pytest tests/test_streaming.py -q`

  Expected: FAIL because `StreamingSession.process` is missing.

- [ ] **Step 3: Implement `StreamingSession`**

  Add constructor parameters `belief_runtime`, `world_runtime`, `possibility`, `initial_world_flat`, `max_feedback_iterations`, and `outcome_tolerance_pixels`. Each completed event appends exactly one final `BeliefEvent`; intermediate corrections live in `StreamUpdate.iterations` but do not consume history slots. Set status to `covered`, `unresolved`, or `rejected`.

- [ ] **Step 4: Run focused and full tests**

  Run: `.venv/bin/pytest tests/test_streaming.py -q && .venv/bin/pytest -q`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/perception_env/streaming.py tests/test_streaming.py
  git commit -m "feat: add bounded streaming world feedback"
  ```

### Task 4: Separate replay learning from latent updates

**Files:**
- Create: `src/perception_env/continual.py`
- Test: `tests/test_continual.py`

**Interfaces:**
- Consumes: completed `StreamUpdate`, Model 2 training examples from Subplan 3, and a trainer callback `train_step(batch: Sequence[ReplayExample]) -> TrainingStepResult`.
- Produces: `ReplayExample`, `ReplayBuffer`, `ContinualLearner.observe(update)`, `ContinualLearner.maybe_train()`, and `ContinualLog`.

- [ ] **Step 1: Write failing replay and separation tests**

  Add tests for deterministic reservoir sampling, rejected-event exclusion, configured warmup/batch cadence, fixed capacity, and checkpoint restoration. Add `test_latent_update_does_not_change_weights` and `test_weight_update_does_not_change_session_world` using parameter and world snapshots before and after each API call.

- [ ] **Step 2: Run the tests and verify RED**

  Run: `.venv/bin/pytest tests/test_continual.py -q`

  Expected: FAIL because the continual module does not exist.

- [ ] **Step 3: Implement the replay learner**

  `ReplayBuffer(capacity: int, seed: int)` stores only completed belief-only examples and uses seeded reservoir replacement. `ContinualLearner` exposes separate `observe` and `maybe_train` calls, monotonic `latent_update_count` and `weight_update_count`, distinct log records, and serialization of buffer RNG state.

- [ ] **Step 4: Run focused and full tests**

  Run: `.venv/bin/pytest tests/test_continual.py -q && .venv/bin/pytest -q`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/perception_env/continual.py tests/test_continual.py
  git commit -m "feat: separate continual replay learning"
  ```

### Task 5: Save and render the streaming visual proof

**Files:**
- Modify: `src/perception_env/stream_records.py`
- Modify: `src/perception_env/visualize.py`
- Create: `tests/test_streaming_visuals.py`

**Interfaces:**
- Consumes: saved episodes, `StreamUpdate` records, world schema, true world from evaluation metadata, and continual logs.
- Produces: `render_streaming_timeline(run: LoadedStreamingRun, output: Path) -> VisualArtifact` and a saved `streaming-summary.json`.

- [ ] **Step 1: Write failing saved-record visual tests**

  Assert a reloaded run renders a nonblank timeline containing, for each event, the first-order clip thumbnail, Model 1 pre/post/difference beliefs, Model 2 predicted operator activity/parameters, ground truth in a visually separate evaluation row, coverage status, feedback iteration count, and separate latent/weight update markers.

- [ ] **Step 2: Run the tests and verify RED**

  Run: `.venv/bin/pytest tests/test_streaming_visuals.py -q`

  Expected: FAIL because the renderer is missing.

- [ ] **Step 3: Implement record-complete rendering**

  Persist every array required by the timeline before rendering. The renderer must accept only the loaded run, never model or simulator objects. Include stream ID, seed, schema version, checkpoint hashes, and event indices in the figure footer.

- [ ] **Step 4: Run focused and full tests**

  Run: `.venv/bin/pytest tests/test_streaming_visuals.py -q && .venv/bin/pytest -q`

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add src/perception_env/stream_records.py src/perception_env/visualize.py tests/test_streaming_visuals.py
  git commit -m "feat: render streaming belief revisions"
  ```

### Task 6: Add a reproducible streaming CLI and real-model smoke test

**Files:**
- Modify: `src/perception_env/cli.py`
- Modify: `README.md`
- Create: `configs/streaming/v1.yaml`
- Create: `tests/test_streaming_cli.py`

**Interfaces:**
- Consumes: Subplan 2 and 3 checkpoints, stream configuration, fixed seed, and Tasks 1–5.
- Produces: `perception-env stream --model1 ... --model2 ... --config ... --seed ... --output ...`, a resumable run directory, and `index.html` linking the timeline.

- [ ] **Step 1: Write failing CLI and integration tests**

  Test `--quick` with tiny trained checkpoints or deterministic fixtures, then test one CPU training/inference smoke run with real Model 1 and Model 2 classes. Assert the manifest records input checkpoint hashes, accepted/rejected counts, coverage by event, latent and weight update counts, and output content hashes.

- [ ] **Step 2: Run the tests and verify RED**

  Run: `.venv/bin/pytest tests/test_streaming_cli.py -q`

  Expected: FAIL because the command is absent.

- [ ] **Step 3: Implement the command, config, and documentation**

  Default `configs/streaming/v1.yaml` to four history events, eight stream events, three feedback iterations, replay capacity 64, warmup four, batch size four, and weight updates disabled unless explicitly enabled. `--resume` must verify config and checkpoint hashes before continuing.

- [ ] **Step 4: Verify the complete stage**

  Run: `.venv/bin/pytest -q && .venv/bin/ruff check .`

  Run a quick stream twice with the same seed and compare saved numerical arrays and deterministic hashes.

  Expected: all tests pass and outputs match.

- [ ] **Step 5: Commit**

  ```bash
  git add src/perception_env/cli.py README.md configs/streaming/v1.yaml tests/test_streaming_cli.py
  git commit -m "feat: expose continual belief revision demo"
  ```

## Completion Gate

- A real four-event run shows Model 1 beliefs, Model 2 world predictions, and bounded corrections while keeping first-order data out of Model 2.
- The same saved stream can be resumed deterministically and rendered without rerunning inference.
- Coverage is reported before and after feedback for every event.
- Latent update count and weight update count are independently auditable.
- The full test suite and Ruff pass.
