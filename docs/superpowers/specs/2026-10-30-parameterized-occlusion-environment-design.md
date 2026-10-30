# Parameterized Occlusion Environment Design

## Intent

Build a deliberately simple 2D environment for studying how much latent world complexity a perceptual system can recover over time. A single ball moves behind an occluder under a small, compositional family of hand-written motion operators. The system exposes complete latent trajectories, rendered observations, possible reappearance events, and the true world description needed to evaluate an inverse model.

The environment is a research instrument, not a realistic physics engine. Its value comes from transparent rules, controlled composition, deterministic replay, and a measurable family of possible/impossible judgments.

## Research role

The larger system contains two learned components:

1. A trajectory model generates several candidate hidden paths from the visible clip.
2. An inverse-world CNN receives only a short history of aggregate belief images and belief-space revisions, then predicts the latent world that could have produced the observation.

The predicted world is fed back to trajectory generation. New paths and belief images are produced until the observation is accommodated or inference reaches a fixed iteration limit. The environment supplies the controlled worlds, observations, exact labels, and evaluation oracles needed to study this loop.

Model 2 never receives raw frames, ball coordinates, detector features, segmentation masks, or a direct reappearance marker. Its architectural purpose is second-order perception over Model 1's beliefs, not a second route for first-order perception.

## Core definitions

### State and time

Use a fixed-step, continuous 2D state:

\[
z_t=(x_t,y_t,v^x_t,v^y_t), \qquad t\in\{0,\ldots,T\}.
\]

All coordinates are normalized to \([0,1]\). A rollout uses a fixed \(\Delta t\), an explicit seed, and an ordered operator list. Identical configuration and seed must produce identical latent states and frames.

### Operator bank

The first version contains a fixed bank of operator types:

| Operator | Activity | Parameters | Effect |
|---|---:|---|---|
| Constant motion | always on | none | Integrates position from velocity. |
| Vertical acceleration | optional | \(a_y\) | Adds a constant vertical force; includes gravity. |
| Horizontal acceleration | optional | \(a_x\) | Adds a constant horizontal force; includes wind. |
| Linear damping | optional | \(\lambda\) | Multiplies velocity by a fixed decay per step. |
| Horizontal wall | optional | \(x_w, r\) | Reflects horizontal velocity at a vertical boundary with restitution \(r\). |
| Floor/ceiling | optional | \(y_w, r\) | Reflects vertical velocity at a horizontal boundary with restitution \(r\). |
| Spatial force zone | deferred extension | bounds, \(a_x,a_y\) | Applies a constant force only inside a rectangle. |

Operators are intentionally elementary. No rotation, deformation, multi-body contact, fluid effects, or general rigid-body solver is included.

### World code

Let \(K\) be the number of optional operator slots. A world is represented by a fixed-width structured code

\[
w=(m_1,\theta_1,\ldots,m_K,\theta_K),
\]

where \(m_i\in\{0,1\}\) is an activity flag and \(\theta_i\) is a fixed-width, normalized parameter block. Inactive parameter blocks take a canonical neutral value and are ignored by the simulator. This same schema is the inverse-world CNN's target.

The serializer must preserve both the structured representation and a flat tensor representation with a versioned field order. Parameter normalization ranges are part of the dataset specification, not learned from a dataset instance.

### Observation model

Occlusion is not a physical operator. It is an observation transform applied after latent simulation. The base observation contains:

- an RGB or grayscale frame sequence with the ball hidden where it intersects a rectangular occluder;
- an exact visibility mask per frame;
- the complete latent ball state for supervision and evaluation;
- the first visible post-occlusion ball location and time, when reappearance occurs.

Keeping observation transforms separate prevents occluder geometry from being mistaken for physical complexity.

### Candidate reappearance event

A probe is a candidate event

\[
q=(c, x, y, t),
\]

where \(c\) identifies an observed pre-occlusion context and \((x,y,t)\) specifies a candidate post-occlusion location and time. A tolerance radius converts the continuous location into an event region.

For a fixed world \(w\), define the possibility classifier

\[
h_w(q)=1
\]

when at least one latent trajectory consistent with the visible context, its uncertainty specification, and world \(w\) reaches the event region at time \(t\). Otherwise \(h_w(q)=0\).

In the deterministic baseline, uncertainty comes from an explicit finite set or bounded grid of admissible hidden-entry states. The oracle enumerates those states; it does not rely on a learned model.

### Belief and possibility representation

For a chosen query time, rasterize all reachable candidate locations into a binary occupancy image. Optionally retain a count-normalized density image for visualization and learned inputs. The benchmark's possible/impossible labels always come from the binary oracle.

Model 1 produces a pre-assimilation belief superposition

\[
B_t^- = \sum_j \alpha_j R(\tau_j),
\]

where \(R\) rasterizes a sampled trajectory. After Model 1 processes new first-order evidence, it produces a revised belief \(B_t^+\). Model 2's canonical input is the last four belief events, with channels for \(B_t^-\), \(B_t^+\), and \(B_t^+-B_t^-\). Optional entropy or mass summaries must also be derived solely from belief tensors.

The environment provides oracle and deliberately perturbed belief pairs to bootstrap this interface before Model 1 is trained. Later integration may mix them with Model 1-generated beliefs, but it may not add first-order inputs to Model 2.

## Empirical operator-class complexity

The physical hypothesis class is the family \(\mathcal H=\{h_w:w\in\mathcal W\}\). Occlusion creates the inference problem but does not itself enlarge the physical operator class.

For \(n\) sampled probes \(Q=(q_1,\ldots,q_n)\), each world produces a binary labeling vector

\[
L(w,Q)=(h_w(q_1),\ldots,h_w(q_n)).
\]

Estimate the sampled growth function by counting distinct vectors over a finite world bank. Estimate an empirical shattering dimension as the largest \(n\) for which the search finds a probe set realizing all \(2^n\) labelings. The result must always be reported with:

- operator bank and parameter ranges;
- world sampling method and count;
- probe sampling method and count;
- search algorithm and budget;
- state uncertainty and event tolerance;
- random seeds and confidence intervals across repeated searches.

This quantity is deliberately described as empirical and search-relative, not as a proof of the VC dimension of a continuous dynamical family.

## Nested complexity ladder

Define versioned, nested world families so complexity can be increased one controlled step at a time:

1. \(\mathcal W_0\): constant motion only;
2. \(\mathcal W_1\): add one constant acceleration axis;
3. \(\mathcal W_2\): allow both acceleration axes;
4. \(\mathcal W_3\): add damping;
5. \(\mathcal W_4\): add one collision boundary;
6. \(\mathcal W_5\): allow multiple active optional operators;
7. \(\mathcal W_6\): optionally add a spatial force zone.

Each level defines exact activity constraints, parameter ranges, and sampling distributions in configuration. Family membership must be testable from a world code. A level may be revised only by creating a new version, because changing its parameter support changes the measured hypothesis class.

## Dataset record

Each generated example contains:

- schema and complexity-ladder versions;
- seed and full generation configuration;
- structured and flat world code;
- initial state and all latent states;
- rendered frames and visibility masks;
- occluder geometry;
- admissible hidden-entry state set or bounds;
- true reappearance event;
- oracle possibility image at configured query times;
- pre-assimilation, post-assimilation, and difference belief tensors for up to four belief events;
- validity flags and rejection reason, if generation failed constraints.

Store metadata and numeric arrays separately from optional video exports. The canonical dataset is array-based; videos are diagnostic artifacts.

## Inverse-model contract

The environment does not implement the final CNN, but it fixes its input and target contract. Given a four-event history of pre-assimilation, post-assimilation, and difference belief images, the model predicts:

- activity logits for all \(K\) optional operator slots;
- normalized parameters for every slot;
- an optional possible/impossible logit for the observed outcome.

Evaluation distinguishes exact parameter recovery from perceptual equivalence. Two world codes may be different yet induce the same reachable set for a short clip. Report:

- activity precision, recall, and exact-match rate;
- parameter error on active and identifiable slots;
- regenerated possibility-image overlap;
- possible/impossible agreement on held-out probes;
- empirical growth/shattering complexity of the recovered world bank versus the true bank.

## Streaming and continual protocol

The first streaming benchmark keeps a latent world estimate across a sequence of occlusion events in one persistent world. Model 1 updates first-order perception and emits belief tensors; Model 2 updates only from belief histories. After each completed event, the system may update Model 2 using a small replay buffer, but online world-code updates and network-weight updates must be logged separately.

Bootstrap training uses oracle beliefs generated under deliberately perturbed sampled worlds. Each input contains the wrong-world pre-belief and a post-belief formed after the first model assimilates the outcome. The target is the true world code. Later stages mix these controlled examples with Model 1-generated belief histories.

## Visual demonstrability

Every executable stage must produce a human-inspectable visual artifact. At minimum:

- world cards show active operators, normalized and physical parameters, geometry, and initial state;
- rollout strips show latent trajectory, visible trajectory, occluded interval, and reappearance;
- operator comparison grids hold initial conditions fixed and vary one operator or parameter;
- belief panels show sampled paths, their superposition, pre-belief, post-belief, and difference belief;
- reachability panels overlay oracle reachable cells and sampled endpoints;
- shattering matrices show worlds by probes with distinct labeling rows highlighted;
- streaming timelines show successive belief revisions, predicted world codes, and true world code;
- identifiability views place perceptually equivalent worlds side by side.

Visuals must be generated from saved records rather than transient training state, include seeds and configuration identifiers, and have deterministic snapshot tests where practical. A task is not complete when its numerical API works but its corresponding visual proof is absent.

## Identifiability protocol

Parameter recovery is scored only where the generated context is sufficiently informative. The environment must support an identifiability audit that searches the sampled world bank for distinct world codes producing equivalent possibility labels within tolerance. Results are tagged as:

- identifiable within the sampled bank;
- perceptually equivalent under the current probe set;
- unresolved because the search budget was insufficient.

The primary scientific claim should use possibility and complexity recovery; exact parameter recovery is a secondary diagnostic when identifiability holds.

## Validation requirements

- Analytic constant-motion and constant-acceleration trajectories match the simulator within numerical tolerance.
- Collision and damping behavior remain bounded and deterministic.
- Occluder changes visibility but never latent state.
- Serialization round-trips without changing a world.
- Possibility rasters contain every enumerated reachable endpoint and no endpoint outside the configured tolerance.
- Complexity levels are nested by construction and produce reproducible sampled banks.
- Empirical shattering search recovers known answers on small synthetic Boolean classifier fixtures before being applied to worlds.
- Model 2 input records contain no first-order observation fields.
- Every benchmark record can be rendered into its required diagnostic visual without rerunning simulation.

## Non-goals

- realistic or high-fidelity physics;
- multiple moving objects;
- learned simulation;
- formal proof of VC or shattering dimension;
- unrestricted operator programs;
- photorealistic rendering;
- treating occlusion geometry as a force operator.

## Initial implementation calls

These are builder decisions made to keep the first research demonstration small:

- Python 3.11 and PyTorch tensors for model-facing data;
- NumPy for deterministic simulation and oracle enumeration;
- Pillow or NumPy rasterization rather than a game engine;
- Hydra/OmegaConf is avoided initially; typed dataclasses plus YAML are sufficient;
- pytest with property-style parameterized tests, adding Hypothesis only if hand-written cases prove inadequate;
- 64-by-64 grayscale canonical frames, with resolution configurable;
- fixed-step semi-implicit Euler for force updates, documented as part of the environment definition.

## Delivery sequence

1. Define schemas, operator ordering, integration semantics, and analytic tests.
2. Implement latent rollout and deterministic replay.
3. Add rendering, occlusion, event extraction, and rollout visualizations.
4. Implement exact finite-state reachability, belief rasterization, and pre/post belief-history fixtures.
5. Add nested world-family sampling and dataset records.
6. Implement empirical growth/shattering search with synthetic calibration fixtures.
7. Add identifiability and recovered-complexity evaluation interfaces.
8. Produce a small reproducible benchmark release and the complete visual diagnostic suite.

