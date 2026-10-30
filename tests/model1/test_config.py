from pathlib import Path

import pytest

from perception_env.model1.config import load_model1_config


def test_v1_config_fixes_mvp_dimensions() -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))

    assert config.dataset.frame_size == 64
    assert config.dataset.clip_steps == 16
    assert config.dataset.prediction_steps == 12
    assert config.dataset.levels == ("W0", "W1", "W2", "W3", "W4")
    assert config.model.modes == 8
    assert config.model.hidden_size == 64
    assert config.training.seed == 7
    assert config.training.batch_size == 32
    assert all(value >= 0 for value in config.loss.weights())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model", "modes"), 0),
        (("dataset", "prediction_steps"), 0),
        (("training", "batch_size"), 0),
        (("evaluation", "calibration_radius"), 0.0),
    ],
)
def test_config_rejects_nonpositive_dimensions(tmp_path, path, value) -> None:
    source = Path("configs/model1/v1.yaml").read_text()
    import yaml

    payload = yaml.safe_load(source)
    payload[path[0]][path[1]] = value
    target = tmp_path / "bad.yaml"
    target.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError):
        load_model1_config(target)
