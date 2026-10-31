# Activation Channel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make detached late Model 1 activations the production evidence channel into Model 2 while retaining belief images as reconstruction targets, diagnostics, and an ablation.

**Architecture:** Model 1 exposes its final GRU state alongside its existing trajectory prediction. Streaming events record pre/post activations with their rendered beliefs and checkpoint identity. A small shared decoder reconstructs pre/post belief maps from activations; the existing belief discrepancy and SMC search then operate unchanged on those reconstructed maps.

**Tech Stack:** Python 3.14, PyTorch, NumPy, pytest

**Spec:** `README.md`

## Global Constraints

- Model 1's existing `forward` result remains backward compatible.
- Activation tensors are detached before crossing the model boundary.
- Model 2 receives no raw frames, world parameters, or gradients into Model 1.
- Direct belief-image input remains available only as an explicit ablation.
- Existing SMC candidate generation and discrepancy semantics remain unchanged.

## Review Focus

- Masked frames must not alter the recorded activation.
- Packet validation must reject nonfinite, mismatched, or mixed-checkpoint traces.
- Adapter outputs must obey the existing belief contract, including exact post-minus-pre differences.
- Adapter training must never create gradients on source activations or Model 1.
- Activation-derived SMC input must be deterministic in evaluation mode.

---

### Task 1: Traced Model 1 forward pass

**Files:**
- Modify: `src/perception_env/model1/types.py`
- Modify: `src/perception_env/model1/model.py`
- Modify: `tests/model1/test_model.py`

**Interfaces:**
- Produces: `Model1Trace(prediction: TrajectoryMixture, activation: torch.Tensor)` and `Model1.forward_with_trace(...) -> Model1Trace`.

- [ ] Add tests for shape, equality with ordinary predictions, masking, and gradient detachment behavior at the boundary.
- [ ] Run the focused tests and confirm they fail because the traced API is absent.
- [ ] Implement a shared encoding path and the traced forward API without changing `forward`.
- [ ] Run the focused tests and the Model 1 suite.

### Task 2: Streaming activation records

**Files:**
- Modify: `src/perception_env/model1/stream_train.py`
- Modify: `tests/model1/test_stream_train.py`

**Interfaces:**
- Consumes: `Model1.forward_with_trace`.
- Produces: event NPZ fields `pre_activation`, `post_activation`, `activation_difference`, and `checkpoint_id` alongside existing belief fields.

- [ ] Extend the stream-record test with activation and checkpoint assertions.
- [ ] Run it and confirm the new fields are absent.
- [ ] Record detached float32 activations before each update and identify their shared Model 1 snapshot.
- [ ] Run the stream-training tests.

### Task 3: Learnable activation adapter

**Files:**
- Create: `src/perception_env/model2/activation_adapter.py`
- Create: `tests/model2/test_activation_adapter.py`

**Interfaces:**
- Produces: `Model1ActivationPacket`, `ActivationBeliefAdapter`, `activation_reconstruction_loss`, checkpoint save/load, and deterministic fitting from recorded events.

- [ ] Test packet validation, decoder shape/range/difference invariants, detached inputs, loss reduction, and checkpoint replay.
- [ ] Run the tests and confirm imports fail.
- [ ] Implement the smallest shared MLP decoder and training/checkpoint helpers that satisfy the tests.
- [ ] Run the adapter tests.

### Task 4: Activation evidence into SMC

**Files:**
- Modify: `src/perception_env/model2/adapters.py`
- Modify: `src/perception_env/model2/smc.py`
- Modify: `tests/model2/test_adapters.py`
- Modify: `tests/model2/test_smc.py`

**Interfaces:**
- Consumes: activation packets and a fitted `ActivationBeliefAdapter`.
- Produces: `adapt_model1_activations(...) -> BeliefInputs`; `BeliefWorldSMC.infer_activations(...) -> SMCResult`; existing `infer(BeliefInputs)` remains the belief-map ablation.

- [ ] Test activation history padding, mixed-checkpoint rejection, deterministic adapter output, and equivalence between explicit adapted input and `infer_activations`.
- [ ] Run the focused tests and confirm the activation bridge is absent.
- [ ] Implement the bridge and thin SMC entry point while reusing the existing search.
- [ ] Run focused tests, lint, and the full suite.

### Task 5: Ranking objective and controls

**Files:**
- Modify: `src/perception_env/model2/activation_adapter.py`
- Modify: `src/perception_env/model2/adapters.py`
- Modify: `tests/model2/test_activation_adapter.py`
- Modify: `tests/model2/test_adapters.py`

**Interfaces:**
- Produces: a differentiable candidate-ranking loss and deterministic shuffled-activation control; the existing first-order ablation remains the video control.

- [ ] Test that matching candidate worlds outrank mismatches and propagate gradients into the adapter.
- [ ] Test that activation shuffling preserves the evidence contract while changing its content.
- [ ] Implement the loss and control mode.
- [ ] Run focused tests, lint, and the full suite.
