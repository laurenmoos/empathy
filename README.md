# Empathy

Empathy asks whether one system can partially recover another **as a perceiver**, through the intentional structure preserved in its experience. **Model 1 sees; Model 2 thinks by imagining the world that could have made Model 1 see as it did.**

## Run

```bash
uv run empathy
```

Open <http://127.0.0.1:8765/explorer.html>.

To watch Seeing learn without the browser:

```bash
uv run empathy train-seeing --episodes 50 --log-dir runs/seeing
```

This prints one compact JSON line per throw and writes a checkpoint, `metrics.jsonl`, and TensorBoard-compatible events. `prediction_loss` scores the frozen pre-exit belief against the later visible exit and departure. One episode is one online update and one TensorBoard step. Optionally inspect the run with `uv run tensorboard --logdir runs/seeing`.

To train in the source world and then reset the fast world state while carrying the slow network into an unseen environment:

```bash
uv run empathy train-seeing --episodes 100 --heldout-episodes 100 --log-dir runs/heldout
```

The held-out phase keeps the same single loss and continues updating both fast and slow parameters. Its log includes their distance from the start of the new environment.

## Environment

A ball moves behind a fixed occluder. Each environment is defined by gravity, wind, and damping. These parameters compose at every time step to transform one ball state into the next, so the hidden trajectory carries their joint constraints without directly revealing their values.

## Model 1: Seeing

Model 1 uses a small 3D video CNN to encode the visible approach. That encoding initializes a distribution over recurrent states. A GRU then predicts acceleration and integrates position and velocity one physical time step at a time, producing a superposition of possible hidden rollouts. The visible exit position, velocity, and first departure frames reweight those rollouts; they do not disclose a unique hidden trajectory.

Within one environment, repeated throws vary the initial conditions while gravity, wind, and damping remain fixed. A fast per-world latent updates after each throw, shared weights move more slowly, and recent observations are replayed. The hidden simulator path is available only for evaluation and the ground-truth line in the visualization; it is not a training target.

## Model 2: Thinking

Model 2 receives the detached pre-belief, post-belief, and belief-change activations. It therefore encounters the internal form of Model 1's experience.

The pre-belief, post-belief, and difference panels are readable projections reconstructed from the underlying activation traces `z_pre`, `z_post`, and `z_post - z_pre`.

Sequential Monte Carlo maintains candidate settings of gravity, wind, and damping. A modular imagination loop simulates each candidate environment and privately calls the frozen Model 1 on its counterfactual clip. Only the resulting detached activations cross into Model 2, which gives greater weight to worlds whose activations resemble Model 1's actual activation history. Raw frames never enter Model 2's inference boundary.

This is a practical approximation to second-order perception: Model 1 sees an event, and Model 2 tries to understand the conditions under which that seeing took the form it did. The simulator search remains reflective thinking rather than immediate perception.

Model 2 recovers the **intentional structure** of Model 1's experience only as an intermediate representation in its task of reconstructing the environment reflected in those activations.

## Coherence

Model 2 is coherent when it can imagine a world whose conditions could have produced Model 1's experience. Measure coherence through parameter error and posterior concentration as episodes accumulate, plus stability across initial conditions and unseen environments.

Because gravity, wind, and damping are compositional operators, successful recovery asks whether Model 1's activations preserve reusable physical constraints rather than individual trajectory templates. The VC-dimension connection remains a capacity analogy: recoverable dimension describes how many independent distinctions among possible worlds survive in Model 1's perceptual state.

Model 1 remains independently trained, and its activations are detached before reaching Model 2. Model 2 learns to read the experience without shaping the system that produced it.
