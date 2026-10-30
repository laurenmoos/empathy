import numpy as np

from perception_env.beliefs import BeliefEvent
from perception_env.continual import ContinualLearner, ReplayBuffer, ReplayExample
from perception_env.streaming import StreamUpdate


def _update(index: int, status: str = "covered") -> StreamUpdate:
    pre = np.zeros((8, 8), np.float32)
    post = pre.copy()
    post[4, index % 8] = 1
    return StreamUpdate(f"event-{index}", status, BeliefEvent(index, pre, post), ())


def test_replay_buffer_is_bounded_and_seed_deterministic() -> None:
    left = ReplayBuffer(3, seed=5)
    right = ReplayBuffer(3, seed=5)
    for index in range(10):
        example = ReplayExample.from_update(_update(index))
        left.add(example)
        right.add(example)
    assert [item.event_id for item in left.examples] == [item.event_id for item in right.examples]
    assert len(left.examples) == 3


def test_continual_learner_separates_observation_from_weight_updates() -> None:
    batches: list[list[str]] = []

    def train_step(examples):
        batches.append([item.event_id for item in examples])
        return {"loss": 0.25}

    learner = ContinualLearner(
        ReplayBuffer(8, seed=2), train_step, warmup=2, batch_size=2, update_every=2
    )
    learner.observe(_update(0, "unresolved"))
    assert learner.weight_update_count == 0
    assert learner.maybe_train() is None
    learner.observe(_update(1))
    learner.observe(_update(2))
    result = learner.maybe_train()
    assert result == {"loss": 0.25}
    assert learner.latent_update_count == 3
    assert learner.weight_update_count == 1
    assert len(batches[0]) == 2

