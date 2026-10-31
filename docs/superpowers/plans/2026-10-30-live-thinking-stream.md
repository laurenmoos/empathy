# Live Thinking Stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the single-rollout demo into a continuous sequence of offset-varied episodes with visibly separate world, belief, and second-order layers, progressive belief revisions, and one auditable Model 2 weight update after each completed guess.

**Architecture:** A `LiveThinkingSession` holds fixed physical parameters, the last four completed belief events, a deterministic offset schedule, and update counts. For each episode, Model 1 produces a trajectory mixture; progressively larger candidate subsets become visible thought steps, Model 2 reads each belief revision and guesses before training, and the final completed event performs one small Model 2 update. The local server owns sessions and the browser requests the next episode as playback finishes.

**Tech Stack:** Python 3.11, PyTorch, NumPy, standard-library `http.server`, static HTML/CSS/JavaScript, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-parameterized-occlusion-environment-design.md`

## Global Constraints

- Model 2 receives only four-event belief histories containing pre-belief, post-belief, difference, and a validity mask; it never receives frames, coordinates, initial state, visibility, or ground truth as inference input.
- The displayed Model 2 guess is captured before its training update for that episode.
- Gravity, wind, and damping remain fixed for a session; accepted episodes vary only deterministic initial height and velocity offsets.
- Online latent-state updates and network-weight updates are counted and displayed separately.
- Ground truth, Model 1 belief space, and Model 2 second-order perception remain visually disjoint.
- The implementation is a hackathon-scale continual adaptation demonstration, not evidence of general continual-learning performance.

## Review Focus

- Offset candidates that leave bounds or fail to reappear must be skipped deterministically and return an actionable 422 only when the bounded candidate bank is exhausted.
- Starting a new session or changing parameters must invalidate late responses and clear the previous episode queue.
- Model 2 training targets may enter only the loss function after the pre-update guess has been saved.
- Concurrent next-episode requests for one session must not duplicate episode indices or race model prediction and training.
- A session with one fixed world must be labeled adaptation, not generalization, and update counts must never imply that Model 1 was trained.

---

### Task 1: Implement real live thinking sessions

**Files:**
- Create: `src/perception_env/live_thinking.py`
- Modify: `src/perception_env/live_inference.py`
- Create: `tests/test_live_thinking.py`
- Modify: `tests/test_live_inference.py`

**Interfaces:**
- Consumes: loaded Model 1 and Model 2 checkpoints, `LiveParameters`, `BeliefInputs`, and `world_code_loss`.
- Produces: `OnlineModel2Runtime.predict(...)`, `OnlineModel2Runtime.train_step(...)`, `LiveThoughtStep`, `LiveThinkingEpisode`, and `LiveThinkingSession.next_episode()`.

- [ ] Write failing tests proving that one episode uses a non-ground-truth initial offset, exposes four ordered belief revisions, supplies only right-aligned `BeliefInputs` to Model 2, records the guess before training, and performs exactly one weight update after the guess.
- [ ] Write failing tests proving successive accepted episodes use distinct deterministic offsets, completed final events persist in a four-event suffix, and invalid offset candidates are skipped within a fixed attempt bound.
- [ ] Implement progressive Model 1 candidate rasterization at 1, 2, all modes, then the post-reappearance belief; each thought step uses the prior belief as `pre`, the new belief as `post`, and decodes one Model 2 guess.
- [ ] Implement `OnlineModel2Runtime` with AdamW at `1e-4`, gradient clipping at `1.0`, and one `world_code_loss` step after the final pre-update prediction; return loss and weight-update count without saving over the source checkpoint.
- [ ] Keep `LiveInferenceEngine.infer(...)` backward-compatible by projecting the first thinking episode into the existing `LiveInferenceResult`.
- [ ] Run `pytest tests/test_live_inference.py tests/test_live_thinking.py -q`, run Ruff on changed files, and commit.

### Task 2: Add persistent streaming session endpoints

**Files:**
- Modify: `src/perception_env/live_server.py`
- Modify: `tests/test_live_server.py`

**Interfaces:**
- Consumes: `LiveThinkingSession` and `LiveThinkingEpisode.to_payload()`.
- Produces: `POST /api/session/start` with physical parameters and `POST /api/session/next` with a server-issued session ID.

- [ ] Write socket-free failing tests for session creation, distinct sequential episode indices, unknown session IDs, exhausted valid offsets, malformed bodies, and exact parameter locking.
- [ ] Implement a thread-safe `LiveSessionRegistry` that creates UUID session IDs, serializes `next_episode()` per session, and never reloads checkpoints.
- [ ] Preserve `POST /api/infer` for compatibility and route the two new endpoints through pure request processors testable without binding a port.
- [ ] Run `pytest tests/test_live_server.py -q`, run Ruff, and commit.

### Task 3: Build the disjoint continuous thinking interface

**Files:**
- Modify: `explorer.html`
- Modify: `tests/test_explorer.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: session start/next JSON with ground-truth frames, four belief thought steps, pre-update guesses, initial offset, and update counts.
- Produces: one continuous Play interaction with separate World, Belief Space, and Second-Order Perception panels.

- [ ] Write failing UI contract tests for three canvases/panels, session endpoints, episode queueing, thought-step labels, separate latent/weight counts, stale-response invalidation, and removal of belief compositing from the world canvas.
- [ ] Replace the single overlaid canvas with a responsive three-panel layout: ground truth plays continuously, belief candidates appear only in the belief panel during occlusion, and Model 2 guesses update only in the second-order panel.
- [ ] Make **Play thinking stream** start a session and automatically request the next offset episode at clip completion; parameter edits stop the stream, clear all three panels, and require a new session.
- [ ] Display episode number, initial offset, thought-step progress, pre-update parameter guess, true parameters, latent update count, weight update count, and latest online loss; label the behavior **adaptation within one fixed world**.
- [ ] Document `.venv/bin/perception-env live-demo`, the belief-only boundary, and the limitation that same-world adaptation can learn a constant answer.
- [ ] Start the real server and visually verify at least three successive offsets, belief/world separation, one parameter reset, and a rejected parameter combination.
- [ ] Run `pytest -q`, `ruff check .`, and `git diff --check`, then commit.

