# Empathy

Empathy asks whether one system can partially recover another **as a perceiver**, through the intentional structure preserved in its experience. **Model 1 sees; Model 2 thinks by imagining the world that could have made Model 1 see as it did.**

![A ball approaches an occluder, becomes a superposition of possible hidden trajectories, exits, and continues visibly.](assets/experiment.svg)

_One fixed world, many throws: Model 1 predicts through the occlusion; Model 2 searches for the world that would produce those beliefs._

## The thought experiment

Imagine that one system learns to anticipate a moving object without ever being told the laws of its world. A second system cannot inspect that world, the object's true hidden path, or the first system's training targets. It can only observe the *form of the first system's experience*: detached activations before and after the object disappears.

The second system is allowed to think. It proposes a world, runs a private counterfactual simulation, lets a frozen copy of the first system perceive it, and asks whether the resulting internal response resembles the response that actually occurred. Repeating this turns gravity, wind, and damping into competing hypotheses rather than supervised labels.

The toy asks: **can a system recover the constraints organizing another system's perception by searching for worlds under which that perception would make sense?** If it can, coherence is not merely agreement with an observed trajectory. It is agreement between a hypothesized world and the structure of another perceiver's possible experience. “Empathy” is the deliberately ambitious name for that second-order inference.

## Run

```bash
uv run empathy
```

Open <http://127.0.0.1:8765/explorer.html>.

To watch Seeing learn without the browser:

```bash
uv run empathy train-seeing --episodes 50 --log-dir runs/seeing
```

Each throw is one online update. The command writes a checkpoint, JSON metrics, and TensorBoard events; `prediction_loss` measures how well the belief formed before occlusion agrees with the later visible exit and departure. Inspect it with `uv run tensorboard --logdir runs/seeing`.

To train in the source world and then reset the fast world state while carrying the slow network into an unseen environment:

```bash
uv run empathy train-seeing --episodes 100 --heldout-episodes 100 --log-dir runs/heldout
```

The held-out phase resets the fast world state, carries the shared video model forward, and continues the same observable-only learning procedure in a new environment.

## Environment

A ball moves behind a fixed occluder. Gravity, wind, and damping define one world and remain fixed while repeated throws vary the ball's initial height. The simulator supplies visible pixels before and after the gap; its hidden path is retained only as ground truth for the demonstration.

## Model 1: Seeing

Seeing is a small continual video world model. A shared 3D CNN encodes the visible approach together with the measured entry state and a fast latent for the current world. It produces the mean and scale of a distribution over GRU states. Samples from that distribution predict acceleration, update velocity, and integrate position one physical time step at a time, forming the visible fan of possible hidden trajectories.

When the ball reappears, its visible exit position, exit velocity, and first departure points score the frozen pre-exit samples. A single evidence loss updates the fast world latent and, more slowly, the shared CNN and GRU; a short replay buffer revisits recent throws. The hidden simulator trajectory never enters this loss.

## Model 2: Thinking

Thinking is currently an explicit inference process, not a second trained CNN. After each throw it receives a new frozen Seeing checkpoint and a detached activation packet: the means and scales encoded before and after occlusion, plus their change. It never receives the actual frames, hidden path, or true world parameters.

Sequential Monte Carlo maintains a persistent population of candidate values for gravity, wind, and damping. For every candidate, a private imagination runner executes the complete toy simulator, renders a counterfactual clip, and passes that clip through the same frozen Seeing checkpoint. Candidate videos are batched, but only their detached activation packets cross back into Thinking.

Worlds are weighted by normalized distance between their imagined and observed pre-activation, post-activation, and activation change. The posterior persists across throws, resampling and gently revising hypotheses when its effective population becomes too small. The interface displays the posterior mean, marginal ranges, and a few joint possible worlds rather than pretending that one trajectory uniquely identifies the rules.

Seeing and Thinking run asynchronously. Seeing continues learning from the real stream; Thinking periodically receives increasingly recent frozen checkpoints and searches against each without blocking the animation. Seeing never learns which candidates Thinking considered and is never optimized to make their recovery easy.

This separation is the point of the toy: Seeing predicts what could happen; Thinking asks what kind of world would make those predictions and revisions take that form.

## Coherence

The system is coherent when candidate worlds that reproduce Seeing's internal response also approach the hidden world's constraints. The demo makes that test visible by placing the true parameter markers behind Thinking's posterior. Useful measurements are parameter error, posterior concentration, and stability across varied entry heights and held-out worlds.

The experiment is therefore about identifiability as much as accuracy. Different operator combinations can explain one exit; varied throws should eliminate some combinations while preserving genuine alternatives. Success means the activation space retains reusable information about forces rather than memorizing a trajectory template.

The shattering connection remains a capacity question rather than a claim made by the live demo: how many independent distinctions among possible worlds can survive in Seeing's states and be recovered by Thinking before coherence breaks down?
