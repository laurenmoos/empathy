# Seeing Architecture

## Goal

Seeing forms a belief about the ball's hidden trajectory, then corrects that belief when the ball reappears.

```text
observed approach → entry → hidden belief → exit → observed departure
```

The environment parameters are never given to the model.

## Video encoder

A small 3D CNN processes the visible approach:

```text
z_pre = VideoEncoder(frames before occlusion)
```

Motion is therefore encoded across time rather than frame by frame.

## Streaming world model

The approach representation, observed entry state, and a fast per-world latent initialize a distribution over recurrent states:

```text
h_0 ~ InitialState(z_pre, entry_state, world_latent)
```

A GRU transition then predicts acceleration and integrates position and velocity one physical time step at a time:

```text
h_(t+1) = GRU(h_t, state_t, world_latent)
acceleration_t = Head(h_(t+1))
state_(t+1) = Integrate(state_t, acceleration_t, dt)
```

Sampling several initial recurrent states produces the visible superposition of rollout lines. The number of lines is only a sampling choice, not a fixed number of modes.

## Post correction

The exit does not reveal the hidden trajectory. It only rules rollouts in or out. Every sampled rollout implies an exit position, velocity, and early departure. Reappearance supplies those same observable quantities:

```text
p_post(rollout) ∝ p_pre(rollout) × compatibility(rollout, observed exit and departure)
```

Incompatible rollouts lose probability; multiple compatible rollouts remain possible. Hidden simulator trajectories are never passed into the online update.

## Single streaming objective

- the frozen pre-belief is scored against the later visible exit position, velocity, and departure;
- the score is a negative log-evidence loss over all sampled rollouts, not a best-path distance;
- a fast world latent updates after every throw while shared encoder weights move slowly;
- recent throws are replayed so one observation does not erase earlier constraints.

There is deliberately no contrastive or auxiliary loss in the current experiment. One throw is one online update and one TensorBoard step—the streaming equivalent of an epoch counter.

Repeated throws keep gravity, wind, and damping fixed while varying initial conditions. The compatible belief sets progressively intersect.

## Outputs

```text
initial point
→ observed approach
→ entry marker
→ pre-belief samples
→ corrected samples + hidden ground truth for evaluation only
→ exit marker
→ observed departure
```

The same runtime powers the browser and the headless trainer. The trainer emits one `prediction_loss`, plus non-loss diagnostics for belief spread and update count.
