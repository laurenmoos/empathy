# Perceptual Coherence Research Roadmap

## Purpose

This is the plan of plans for a research demonstration of second-order perception over time. The system studies whether a model that sees only superpositions of another model's beliefs can infer a latent physical world, revise those beliefs continually, and recover the possible/impossible distinctions of an increasingly complex operator class.

The roadmap is intentionally hierarchical. Each subplan must yield independently runnable software, quantitative checks, and a saved visual demonstration before the next dependent subplan begins.

## Architectural invariant

Model 1 may consume first-order frames. Model 2 may not. Model 2 consumes a four-event history of belief superpositions and belief revisions derived from Model 1. No raw frame, position, segmentation, detector feature, or direct outcome marker may enter Model 2.

## Visual invariant

Every subplan must provide a deterministic command that renders its main objects and claims from saved records. Visual output is an acceptance criterion, not optional polish.

## Plan hierarchy

### Subplan 1: Parameterized environment and operator class

**Question:** Can we construct nested, reproducible families of toy worlds whose possible/impossible judgments change in measurable ways?

**Outputs:** One-ball simulator, operator bank, occlusion renderer, world-code schema, exact finite reachability oracle, belief fixtures, nested world families, empirical shattering search, and dataset records.

**Visual proof:** World cards, trajectory strips, one-operator comparison grids, reachability overlays, belief panels, and shattering matrices.

**Gate:** Analytic trajectories pass, observation never changes latent state, small Boolean fixtures calibrate shattering search, and all core records render without rerunning simulation.

**Executable plan:** `docs/superpowers/plans/2026-10-30-parameterized-occlusion-environment.md`

### Subplan 2: Model 1 belief generator

**Question:** Can first-order perception generate a calibrated distribution of hidden trajectories rather than a single point prediction?

**Outputs:** Frame encoder, visible-state estimator, conditional trajectory sampler, trajectory weights, rasterized pre/post belief tensors, and calibration metrics. Start with environment state supervision, then ablate it.

**Visual proof:** Input clip beside sampled paths; weighted superposition through time; calibration plots; failure galleries grouped by world family.

**Gate:** Held-out trajectories have calibrated coverage, belief tensor schema matches Subplan 1, and assimilation produces a post-belief without exposing first-order fields downstream.

**Executable plan:** `docs/superpowers/plans/2026-10-30-model-1-belief-generator.md`

### Subplan 3: Model 2 inverse-world CNN

**Question:** How much latent operator complexity can be recovered from belief histories alone?

**Outputs:** Belief-history dataset, fixed-width activity-and-parameter head, world-code loss with inactive-slot masking, possibility-equivalence metrics, and exact controls that receive shuffled or first-order inputs only as ablations.

**Visual proof:** Belief history beside predicted and true world cards; operator-wise recovery plots; perceptually equivalent world pairs; recovered-versus-true possibility maps.

**Gate:** The production model contains no first-order input path, beats constant and single-belief baselines, and regenerates held-out possibility judgments from its predicted world code.

**Executable plan:** `docs/superpowers/plans/2026-10-30-model-2-inverse-world-cnn.md`

### Subplan 4: Streaming belief-revision loop

**Question:** Does repeated second-order inference improve a persistent world estimate across occlusion events?

**Outputs:** Stateful event stream, four-event belief buffer, world-code update rule, convergence/iteration policy, replay-buffer learner, and strict separation between online latent updates and weight updates.

**Visual proof:** An animated or tiled timeline of frames visible only to Model 1, belief revisions visible to Model 2, world-code estimates, and error after each event.

**Gate:** The loop replays deterministically, improves possibility coverage over multiple events, and does not silently route first-order data to Model 2.

**Executable plan:** `docs/superpowers/plans/2026-10-30-streaming-belief-revision-loop.md`

### Subplan 5: Coherence and complexity recovery

**Question:** At what empirical operator-class complexity does the perceptual loop stop maintaining coherent possible/impossible judgments over time?

**Outputs:** Temporal-coherence definition, empirical shattering estimates for true and recovered world banks, complexity ladder experiments, uncertainty intervals, and capacity-breakpoint analysis.

**Visual proof:** True-versus-recovered shattering curves, temporal coherence traces, labeling matrices, capacity frontier, and representative belief failures near the breakpoint.

**Gate:** Metrics recover known synthetic controls, report search budgets, separate parameter error from perceptual equivalence, and reproduce across registered seeds.

**Executable plan:** `docs/superpowers/plans/2026-10-30-coherence-complexity-recovery.md`

### Subplan 6: Research demonstration and reproducibility package

**Question:** Can another researcher inspect, rerun, and understand the central claim?

**Outputs:** Frozen configurations, compact benchmark dataset, training/evaluation commands, ablation table, seeded result bundle, paper-ready figures, and a narrated demonstration.

**Visual proof:** One end-to-end dashboard or video showing a world, Model 1 beliefs, Model 2 world inference, iterative correction, and the complexity comparison.

**Gate:** A clean environment reproduces the headline result and every reported number links to a configuration and saved artifact.

**Executable plan:** `docs/superpowers/plans/2026-10-30-research-demo-reproducibility.md`

## Dependency flow

Subplan 1 defines the contracts used by every later plan. Subplans 2 and 3 can then proceed in parallel against oracle-generated fixtures. Subplan 4 integrates them. Subplan 5 evaluates the integrated system. Subplan 6 freezes and communicates the result.

## Research checkpoints

1. **Environment checkpoint:** Operator families produce visibly and measurably different reachable sets.
2. **Belief checkpoint:** Model 1 represents multimodal uncertainty during occlusion.
3. **Second-order checkpoint:** Model 2 recovers world information without first-order leakage.
4. **Continual checkpoint:** Repeated belief revisions improve the latent world estimate.
5. **Complexity checkpoint:** Recovery degrades as empirical operator-class complexity increases.
6. **Demonstration checkpoint:** The full claim is visible in one reproducible artifact.

## Explicit deferrals

The first research demonstration excludes multiple moving objects, high-fidelity physics, photorealism, unrestricted operator programs, formal VC-dimension proofs, and deployment-scale continual learning.

