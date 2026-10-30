import json

import numpy as np
from PIL import Image

from perception_env.beliefs import BeliefEvent
from perception_env.continual import ContinualLog
from perception_env.stream_records import load_streaming_run, save_streaming_run
from perception_env.streaming import (
    FeedbackIteration,
    OcclusionEpisode,
    StreamingSessionState,
    StreamUpdate,
    StreamWorldPrediction,
)
from perception_env.streaming_demo import render_streaming_timeline
from perception_env.world import WorldSchema


def _run_parts():
    schema = WorldSchema.v1()
    true_world = schema.world(horizontal_acceleration={"a_x": 0.2})
    predicted = StreamWorldPrediction(
        true_world.to_flat(schema),
        np.asarray([0.1, 0.9, 0.1, 0.1, 0.1]),
        np.linspace(0.1, 0.7, 7),
        0.85,
    )
    frames = np.zeros((4, 8, 8), dtype=np.float32)
    frames[0, 4, 1] = 1.0
    frames[3, 4, 6] = 1.0
    episode = OcclusionEpisode(
        "stream-000",
        "stream-context-000",
        frames,
        np.asarray([True, False, False, True]),
        3,
        (0.75, 0.5),
        {
            "true_world_flat": true_world.to_flat(schema).tolist(),
            "initial_state": [0.1, 0.5, 0.2, 0.0],
            "seed": 13,
        },
    )
    pre = np.zeros((8, 8), dtype=np.float32)
    pre[4, 2:6] = 0.4
    post = pre.copy()
    post[4, 6] = 1.0
    event = BeliefEvent(3, pre, post)
    update = StreamUpdate(
        episode.event_id,
        "covered",
        event,
        (FeedbackIteration(0, predicted, True),),
    )
    state = StreamingSessionState(
        predicted.flat_world,
        (event,),
        (episode.event_id,),
        1,
        0,
    )
    logs = (
        ContinualLog(1, 1, 0, None),
        ContinualLog(1, 1, 1, {"loss": np.float32(0.5)}),
    )
    metadata = {
        "stream_id": "stream",
        "seed": 13,
        "schema_version": "v1",
        "checkpoint_hashes": {"model1": "fixture-one", "model2": "fixture-two"},
    }
    return (episode,), (update,), state, logs, metadata


def test_complete_streaming_run_round_trips_all_visual_inputs(tmp_path) -> None:
    episodes, updates, state, logs, metadata = _run_parts()
    path = save_streaming_run(
        tmp_path / "run", episodes, updates, state, logs, metadata
    )

    loaded = load_streaming_run(path)

    assert loaded.episodes[0].event_id == "stream-000"
    assert loaded.updates[0].status == "covered"
    assert np.array_equal(loaded.updates[0].event.pre, updates[0].event.pre)
    assert np.array_equal(
        loaded.updates[0].iterations[0].prediction.flat_world,
        updates[0].iterations[0].prediction.flat_world,
    )
    assert loaded.updates[0].iterations[0].covered is True
    assert loaded.session_state.processed_event_ids == ("stream-000",)
    assert loaded.continual_logs == logs
    assert loaded.metadata == metadata
    summary = json.loads((path / "streaming-summary.json").read_text())
    assert summary["coverage_by_event"] == {"stream-000": True}
    assert summary["feedback_iterations"] == {"stream-000": 1}
    assert summary["latent_update_count"] == 1
    assert summary["weight_update_count"] == 1


def test_timeline_renders_only_from_reloaded_run(tmp_path) -> None:
    episodes, updates, state, logs, metadata = _run_parts()
    path = save_streaming_run(
        tmp_path / "run", episodes, updates, state, logs, metadata
    )
    loaded = load_streaming_run(path)

    artifact = render_streaming_timeline(loaded, tmp_path / "timeline.png")

    image = np.asarray(Image.open(artifact.path).convert("RGB"))
    assert image.std() > 10
    assert artifact.metadata["source"] == "reloaded_streaming_records"
    assert artifact.metadata["panels"] == [
        "first_order_clip",
        "pre_belief",
        "post_belief",
        "belief_difference",
        "model2_world_prediction",
        "ground_truth_evaluation",
    ]
    assert artifact.metadata["statuses"] == ["covered"]
    assert artifact.metadata["feedback_iterations"] == [1]
    assert artifact.metadata["latent_update_count"] == 1
    assert artifact.metadata["weight_update_count"] == 1
    assert artifact.metadata["checkpoint_hashes"] == metadata["checkpoint_hashes"]

