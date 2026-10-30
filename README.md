# Perception Environment

A visually inspectable toy environment for experiments on second-order perceptual coherence. One ball moves under deliberately simple forces, becomes occluded, and produces exact reachable sets and belief superpositions for downstream models.

## Interactive parameter explorer

Open [`explorer.html`](explorer.html) directly in a browser. It is a self-contained interface with presets, sliders, play/pause, and reset controls for gravity, wind, damping, walls, the initial ball state, and occluder geometry. The observed world and the superposition of possible hidden paths update together; no server or command line is required.

## Install and run

```bash
uv sync --extra dev --extra ml
.venv/bin/perception-env demo \
  --config configs/world_families/v1.yaml \
  --seed 7 \
  --output artifacts/environment-demo
```

Open `artifacts/environment-demo/index.html` to inspect the world cards, latent trajectories, occluded observations, belief revisions, shattering matrix, and perceptual-equivalence gallery. Add `--quick` for a smaller smoke-test run.

The other commands are:

- `sample`: save a seeded bank of flat world codes from one complexity level;
- `inspect`: regenerate a contact sheet from saved records without rerunning simulation;
- `estimate-shattering`: rerun the search from a demo's saved label matrix;
- `demo`: produce the complete benchmark and visual index.

## Learned models

The repository now contains two real PyTorch models rather than only visual stand-ins.
Model 1 encodes an occlusion clip with a small CNN/GRU and predicts eight weighted hidden
trajectories, which are differentiably rasterized into a belief superposition. Model 2 is a
separate CNN whose production input is limited to four pre/post/difference belief events and a
history mask; it predicts operator activity and normalized parameters.

Run small CPU demonstrations with:

```bash
.venv/bin/perception-env model1 demo \
  --config configs/model1/v1.yaml \
  --family-config configs/world_families/v1.yaml \
  --seed 7 --device cpu --quick \
  --output artifacts/model1-learned-demo

.venv/bin/perception-env model2 demo \
  --config configs/model2/smoke_v1.yaml \
  --seed 7 \
  --output artifacts/model2-learned-demo

.venv/bin/perception-env stream-demo \
  --seed 7 --events 4 \
  --output artifacts/streaming-demo
```

The first two commands train and evaluate the actual learned models. The quick streaming command
exercises the production belief-only feedback and persistence contracts with deterministic fixture
runtimes; adapters and checkpoints for the learned models are available, but a trained-checkpoint
streaming command remains the next integration step. The smoke Model 2 run is a systems check, not
a successful research result: its possibility agreement does not yet beat the constant baseline.

## World code

Schema `v1` has five optional operator slots: vertical acceleration, horizontal acceleration, damping, a vertical wall, and a horizontal wall. Every slot has an activity flag and normalized parameter block. Constant motion is implicit. Active force values cannot lie in a configured neutral interval, so “active but zero” is not silently confused with “inactive.”

The simulator uses fixed-step semi-implicit Euler updates in this order: acceleration, damping, velocity update, position update, collision resolution. It is a controlled rule generator, not a realistic physics engine.

## Possibility versus belief

The reachability oracle enumerates an explicit grid of admissible hidden-entry states. Its binary occupancy image answers whether a location is reachable under a world. A belief image is different: it is a weighted superposition of sampled trajectory masks and can express concentration or uncertainty.

Model 2 receives only four-event histories of:

1. pre-assimilation belief;
2. post-assimilation belief;
3. their difference.

It never receives raw frames, positions, visibility masks, detector features, or a direct reappearance marker. `Model2InputRecord` rejects those fields, and the demo exports a belief-only `model2-input.npz`.

## Empirical shattering estimate

Each world acts as a classifier over candidate reappearance probes. The search selects probe subsets and counts whether the sampled world bank realizes all binary labelings. The result is a context-conditioned, empirical shattering dimension—not a formal VC-dimension proof.

Every report records operator support, entry-state uncertainty, event tolerance, world and probe counts, search algorithm, budget, and seeds. Multi-context searches are labeled separately and cannot be presented as context-conditioned estimates.

## Visual panels

- **World cards** show operator activity and physical parameters.
- **Trajectory strips** expose latent paths, collisions, and seeds.
- **Observation strips** separate latent motion from visible frames and occlusion timing.
- **Belief panels** show paths, endpoint reachability, pre/post beliefs, and differences; the true endpoint stays in a separate first-order diagnostic panel.
- **Shattering matrices** group duplicate world labelings and highlight a shattered probe subset.
- **Equivalence galleries** show when different world codes have the same possible/impossible consequences.

All canonical arrays and metadata are stored independently of the figures, so visual inspection can be repeated from saved records.
