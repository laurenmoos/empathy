import inspect

import numpy as np
from PIL import Image

from perception_env.model1.evaluate import (
    EvaluationSummary,
    Model1PredictionRecord,
    load_prediction_record,
    save_prediction_record,
)
from perception_env.model1.visualize import (
    render_calibration_plot,
    render_failure_gallery,
    render_prediction_panel,
)


def prediction_record(index: int = 0) -> Model1PredictionRecord:
    modes = np.stack(
        [
            np.column_stack(
                (np.linspace(0.2, 0.8, 12), np.full(12, 0.2 + i * 0.08))
            )
            for i in range(8)
        ]
    ).astype(np.float32)
    arrays = {
        "pre_frames": np.zeros((16, 1, 64, 64), np.float32),
        "pre_times": np.zeros(16, np.float32),
        "pre_mask": np.ones(16, bool),
        "post_frames": np.zeros((16, 1, 64, 64), np.float32),
        "post_times": np.zeros(16, np.float32),
        "post_mask": np.ones(16, bool),
        "target_positions": modes[0],
        "target_mask": np.ones(12, bool),
        "pre_positions": modes,
        "pre_weights": np.full(8, 0.125, np.float32),
        "post_positions": modes[::-1].copy(),
        "post_weights": np.linspace(0.05, 0.2, 8).astype(np.float32),
        "pre_time_belief": np.zeros((12, 64, 64), np.float32),
        "post_time_belief": np.zeros((12, 64, 64), np.float32),
        "pre_belief": np.eye(64, dtype=np.float32),
        "post_belief": np.fliplr(np.eye(64, dtype=np.float32)).copy(),
        "belief_difference": np.fliplr(np.eye(64, dtype=np.float32))
        - np.eye(64, dtype=np.float32),
        "oracle_pre_belief": np.eye(64, dtype=np.float32),
        "actual_path_belief": np.eye(64, dtype=np.float32),
    }
    metrics = {
        "pre_min_ade": 0.1 + index,
        "pre_confidence": 0.4,
        "pre_belief_iou": 0.7,
    }
    return Model1PredictionRecord(f"example-{index}", f"W{index}", index, metrics, arrays)


def test_saved_record_visuals_are_deterministic_and_self_contained(tmp_path) -> None:
    path = save_prediction_record(prediction_record(), tmp_path / "records")
    record = load_prediction_record(path)
    first = render_prediction_panel(record, tmp_path / "first.png")
    second = render_prediction_panel(record, tmp_path / "second.png")

    assert first.path.stat().st_size > 0
    assert first.metadata_path.is_file()
    assert np.array_equal(np.asarray(Image.open(first.path)), np.asarray(Image.open(second.path)))


def test_calibration_and_failure_gallery_render_from_saved_values(tmp_path) -> None:
    summary = EvaluationSummary(
        count=2,
        overall={},
        per_family={},
        calibration={
            "pre": {
                "ece": 0.2,
                "bin_confidence": [None] * 9 + [0.9],
                "bin_accuracy": [None] * 9 + [1.0],
                "bin_count": [0] * 9 + [2],
                "credible_coverage": {"0.5": 0.5, "0.8": 0.75, "0.9": 1.0},
            },
            "post": {
                "ece": 0.1,
                "bin_confidence": [None] * 9 + [0.9],
                "bin_accuracy": [None] * 9 + [1.0],
                "bin_count": [0] * 9 + [2],
                "credible_coverage": {"0.5": 0.75, "0.8": 1.0, "0.9": 1.0},
            },
        },
        config_hash="config",
        checkpoint_hash="checkpoint",
        seed=7,
        record_paths=(),
    )

    calibration = render_calibration_plot(summary, tmp_path / "calibration.png")
    gallery = render_failure_gallery(
        [prediction_record(0), prediction_record(1)], tmp_path / "failures.png", top_n=8
    )

    assert calibration.path.stat().st_size > 0
    assert gallery.path.stat().st_size > 0
    source = inspect.getsource(__import__("perception_env.model1.visualize", fromlist=["*"]))
    assert "model1.model" not in source
    assert "model1.train" not in source
    assert "perception_env.physics.simulate" not in source
