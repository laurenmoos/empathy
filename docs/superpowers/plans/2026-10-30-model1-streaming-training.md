# Model 1 Streaming Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the README's predict–record–reveal–replay training procedure for Model 1 as a deterministic, runnable command.

**Architecture:** Extract environment-controlled Model 1 record generation from the existing offline dataset builder, then add a focused streaming trainer that owns a single Model 1, a bounded replay buffer of fully supervised episodes, and frozen pre-update belief recording. Each environment contributes a batch of varied initial conditions; all beliefs in that batch are recorded before any optimizer step, then revealed records enter replay and a small configured number of updates run without resetting either weights or replay.

**Tech Stack:** Python 3.11, NumPy, PyTorch, PyYAML, pytest.

**Spec:** `README.md` — “Model 1 Seeing,” especially the streaming predict–record–reveal–update paragraph.

## Global Constraints

- Model 1 inputs remain video frames and frame timing only; no operator labels, world code, or physical parameters enter its forward pass or loss.
- Both early and completed clips use the existing trajectory, mode, entry-state, pre-belief, and post-belief supervision.
- All episodes in an environment batch are inferred and recorded with frozen weights before any update from that batch.
- Model weights and replay carry forward when the environment changes.
- Replay is bounded and seeded; generated environments and initial-state batches are deterministic under the run seed.
- Existing uncommitted `README.md`, `explorer.html`, `tests/test_explorer.py`, and `docs/images/` changes are user-owned and must not be overwritten or included in implementation commits.

## Review Focus

- A batch must not partially train before its remaining beliefs are recorded; test identical `weight_update_count` across every event in one environment.
- Replay must retain records from earlier environments after a transition; test the provenance of sampled update batches.
- Rejected trajectories must consume the configured attempt budget rather than shrinking the environment batch silently.
- Same seed and configuration must reproduce environment IDs, event arrays, update provenance, and final weights.
- Stream configuration values of zero, oversized replay batches, or unknown family levels must fail before training creates partial output.

---

### Task 1: Environment-controlled Model 1 records

**Files:**
- Modify: `src/perception_env/model1/data.py`
- Test: `tests/model1/test_data.py`

**Interfaces:**
- Produces: `make_model1_record(config: Model1Config, world: WorldCode, initial: BallState, *, example_id: str, split: str, family_level: str, seed: int) -> tuple[Model1ExampleRecord | None, str | None]`.
- Preserves: `generate_model1_dataset(...)` output and determinism by routing `_make_record(...)` through the new factory.

- [ ] **Step 1: Write failing tests** proving two supplied initial states under one supplied world produce different records with the same `world_id`, and an invalid supplied trajectory returns a rejection reason.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/model1/test_data.py -q` and confirm failure because `make_model1_record` is absent.
- [ ] **Step 3: Implement** the public factory by extracting the simulation/render/reappearance/targets/reachability logic from `_make_record`; keep random world and initial-state sampling only in `_make_record`.
- [ ] **Step 4: Run** `.venv/bin/pytest tests/model1/test_data.py -q` and confirm pass.
- [ ] **Step 5: Commit** `feat: generate model1 episodes within fixed worlds`.

### Task 2: Streaming configuration, environment batches, and replay

**Files:**
- Create: `src/perception_env/model1/stream_train.py`
- Create: `configs/model1/stream_v1.yaml`
- Create: `tests/model1/test_stream_train.py`

**Interfaces:**
- Produces: `StreamTrainingConfig`, `load_stream_training_config(path: Path)`, `EnvironmentBatch`, `generate_environment_batches(...)`, and `Model1ReplayBuffer`.
- Consumes: Task 1 `make_model1_record(...)`.

- [ ] **Step 1: Write failing tests** for strict positive configuration validation, deterministic same-world batches with varied initial states, exact batch size or attempt-budget failure, bounded seeded replay, and preservation of earlier-environment provenance.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/model1/test_stream_train.py -q` and confirm import failure.
- [ ] **Step 3: Implement** configuration and deterministic environment generation. Schedule `levels` cyclically, sample one world per environment, vary initial states from Model 1 dataset bounds, and reject until `episodes_per_environment` records are accepted or `max_attempts_per_environment` is exhausted.
- [ ] **Step 4: Implement** reservoir replay over complete `Model1ExampleRecord` objects, including `environment_index` provenance and seeded sampling without replacement.
- [ ] **Step 5: Run** `.venv/bin/pytest tests/model1/test_stream_train.py -q` and confirm pass.
- [ ] **Step 6: Commit** `feat: add deterministic model1 environment streams`.

### Task 3: Frozen recording and replay updates

**Files:**
- Modify: `src/perception_env/model1/stream_train.py`
- Modify: `tests/model1/test_stream_train.py`

**Interfaces:**
- Produces: `RecordedBeliefEvent`, `StreamTrainingResult`, and `train_model1_stream(config, stream_config, family_config, output, *, device, seed) -> StreamTrainingResult`.
- Saves: `events/*.npz`, `history.jsonl`, `stream-last.pt`, and `stream-summary.json`.

- [ ] **Step 1: Write failing tests** using a tiny real Model 1 stream to prove every event in an environment records before that environment's first update, pre/post/difference arrays are saved, update batches include replay provenance, weights change, and the next environment begins from the carried update count.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/model1/test_stream_train.py -q` and confirm failure because the trainer is absent.
- [ ] **Step 3: Implement** frozen `model.eval()`/`torch.inference_mode()` belief recording with the existing two clips and rasterizer; write events before adding revealed records to replay.
- [ ] **Step 4: Implement** configured replay update steps with the existing `_run_batch` loss, one persistent optimizer/model, gradient clipping, JSONL provenance, and final portable checkpoint.
- [ ] **Step 5: Add deterministic replay test** comparing same-seed event arrays, update provenance, and final model tensors across two runs.
- [ ] **Step 6: Run** `.venv/bin/pytest tests/model1/test_stream_train.py -q` and confirm pass.
- [ ] **Step 7: Commit** `feat: train model1 through streaming replay`.

### Task 4: Runnable command and end-to-end artifact contract

**Files:**
- Modify: `src/perception_env/model1/cli.py`
- Modify: `tests/model1/test_cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `perception-env model1 stream-train --config ... --stream-config ... --family-config ... --output ...`.
- Consumes: Task 3 `train_model1_stream(...)`.

- [ ] **Step 1: Write failing CLI tests** that run a two-environment tiny stream and assert the summary, event files, history, checkpoint, frozen-batch update counts, and overwrite refusal.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/model1/test_cli.py tests/test_cli.py -q` and confirm the command is unknown.
- [ ] **Step 3: Add** the `stream-train` parser and runner, reuse existing output safety helpers, support `--device`, `--seed`, `--force`, and `--quick`, and write a deterministic manifest referencing all core artifacts.
- [ ] **Step 4: Run** `.venv/bin/pytest tests/model1/test_cli.py tests/test_cli.py -q` and confirm pass.
- [ ] **Step 5: Run** `.venv/bin/pytest -q`, `.venv/bin/ruff check src tests`, and `git diff --check`; all must pass.
- [ ] **Step 6: Commit** `feat: expose model1 streaming training command`.
