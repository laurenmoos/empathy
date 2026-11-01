# Seeing Architecture

## Goal

Seeing forms a belief about the ball's hidden trajectory, then corrects that belief when the ball reappears.

```text
observed approach → entry → hidden belief → exit → observed departure
```

The environment parameters are never given to the model.

## Shared video encoder

A small 3D CNN processes both observations with shared weights:

```text
z_pre  = VideoEncoder(frames before occlusion)
z_post = VideoEncoder(frames after reappearance)
```

Motion is therefore encoded across time rather than frame by frame.

## Pre-belief

The pre-occlusion representation, observed entry state, and a fast per-world latent define a continuous distribution over smooth cubic Bézier paths:

```text
p_pre(path) = Prior(z_pre, entry_state, world_latent)
```

The number of lines in the interface is only a sampling choice, not a fixed number of trajectory modes.

## Post correction

The exit does not reveal the hidden trajectory. It only rules paths in or out. Every sampled path implies an exit position, velocity, and early departure. Reappearance supplies those same observable quantities:

```text
p_post(path) ∝ p_pre(path) × compatibility(path, observed exit and departure)
```

Incompatible paths lose probability; multiple compatible paths remain possible. Hidden simulator trajectories are never passed into the online update.

## Streaming objective

- endpoint position, velocity, and departure supply geometric evidence;
- matched pre/post clips should agree more than clips swapped between throws;
- a fast world latent updates after every throw while shared encoder weights move slowly;
- recent throws are replayed so one observation does not erase earlier constraints.

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

The same runtime powers the browser and the headless trainer. The trainer emits JSON lines and TensorBoard-compatible events for loss, exit error, belief spread, effective sample count, and update count.
