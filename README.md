# Empathy

Empathy asks whether one system can partially recover another **as a perceiver**, through the intentional structure preserved in its experience. **Model 1 sees; Model 2 thinks by imagining the world that could have made Model 1 see as it did.**

## Environment

A ball moves behind a fixed occluder. Gravity, wind, and damping define one world and remain fixed while repeated throws vary the ball's initial height. The simulator supplies visible pixels before and after the gap; its hidden path provides ground truth for the demonstration.

![A ball approaches an occluder, becomes a superposition of possible hidden trajectories, exits, and continues visibly.](assets/experiment.svg)

_One fixed world, many throws: Model 1 predicts through the occlusion; Model 2 searches for the world that would produce those beliefs._

## The thought experiment

Model 1 learns to predict the motion of an occluded object through experience with pixels and visible reappearances. Its learned dynamics carry the regularities of the world. Model 2's perceptual channel consists of one detached three-part packet from Model 1: the visible-before activation, Model 1's superposition of predicted hidden trajectories, and the visible-after activation. Event and checkpoint identifiers align comparisons.

Model 2 maintains candidate values for gravity, wind, and damping. For each candidate, it runs a counterfactual simulation and passes the resulting clip through a frozen copy of the current Model 1. Candidates whose activations resemble Model 1's observed response receive more posterior weight.

The experiment asks whether a perceptual model's internal response contains enough structure to recover the constraints organizing its experience.

Recovering gravity, wind, and damping is a **surrogate task** with a known answer. The deeper target is Model 1's learned distribution of prediction errors: the structured pattern of fit and mismatch its beliefs produce across counterfactual worlds. Recovering that error distribution amounts to recovering Model 1's perceptual frame—the constraints that shape what it expects, where it remains uncertain, and how visible evidence revises its beliefs.

## Model 1: Seeing

Seeing is a small continual video world model. A shared 3D CNN encodes the visible approach together with the measured entry state and a fast latent for the current world. It produces the mean and scale of a distribution over GRU states. Samples from that distribution predict acceleration, update velocity, and integrate position one physical time step at a time, forming the visible fan of possible hidden trajectories.

When the ball reappears, its visible exit position, exit velocity, and first departure points score the frozen pre-exit samples. A single evidence loss uses this visible reappearance to update the fast world latent and, more slowly, the shared CNN and GRU; a short replay buffer revisits recent throws.

## Model 2: Thinking

Thinking currently takes the form of an explicit inference process. After each throw it receives a new frozen Seeing checkpoint and the three-part packet described above. The middle term contains all of Seeing's sampled hidden rollouts, interpolated onto one fixed time grid. The before and after activations provide context and normalization; the predicted hidden region is the primary inference signal. This packet forms Thinking's complete perceptual interface.

Sequential Monte Carlo maintains a persistent population of candidate values for gravity, wind, and damping. For every candidate, a private imagination runner executes the complete toy simulator, renders a counterfactual clip, and passes that clip through the same frozen Seeing checkpoint. Candidate videos are batched inside the imagination runner, which returns detached activation packets to Thinking.

Worlds are weighted by component-normalized distance between the observed packet and the packet produced by the counterfactual Seeing run. The hidden-rollout term receives the primary weight; the two visible endpoint terms act as lower-weight anchors. The posterior persists across throws, resampling and gently revising hypotheses when its effective population becomes too small. The interface displays the posterior mean, marginal ranges, and several joint possible worlds, preserving the ambiguity carried by each trajectory.

Seeing and Thinking run asynchronously. Seeing continues learning from the real stream; Thinking periodically receives increasingly recent frozen checkpoints and searches against each while the animation continues. Seeing follows its local prediction objective, and Thinking performs a separate search over candidate worlds.

This separation is the point of the toy: Seeing predicts what could happen; Thinking asks what kind of world would make those predictions and revisions take that form.

## Coherence

The system is coherent when candidate worlds reproduce Seeing's internal response and approach the hidden world's constraints. The demo makes this surrogate test visible by placing the true parameter markers behind Thinking's posterior. Parameter error, posterior concentration, and stability across varied entry heights and held-out worlds measure how well the search recovers the learned error geometry that frames Seeing's experience.

The experiment is therefore about identifiability as much as accuracy. Different operator combinations can explain one exit; varied throws should narrow some combinations while preserving genuine alternatives. Success means the activation space retains reusable information about forces and expresses a stable distribution of errors across trajectories.

The shattering connection supplies a capacity question for future experiments: how many independent distinctions among possible worlds can survive in Seeing's states and remain recoverable through Thinking's search?

## Run

```bash
uv run empathy
```

Open <http://127.0.0.1:8765/explorer.html>.
