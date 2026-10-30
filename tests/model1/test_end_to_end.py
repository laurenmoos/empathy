import json
from pathlib import Path

import numpy as np
import yaml

from perception_env.model1.cli import main
from perception_env.records import Model2InputRecord


def test_quick_demo_is_reproducible_and_exports_belief_only_fixture(tmp_path) -> None:
    payload = yaml.safe_load(Path("configs/model1/v1.yaml").read_text())
    payload["dataset"]["quick_counts"] = [4, 2, 4]
    payload["training"]["batch_size"] = 2
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(payload))
    hashes = []
    for name in ("one", "two"):
        output = tmp_path / name
        assert main(
            [
                "demo",
                "--config",
                str(config),
                "--family-config",
                "configs/world_families/v1.yaml",
                "--seed",
                "7",
                "--output",
                str(output),
                "--quick",
            ]
        ) == 0
        manifest = json.loads((output / "manifest.json").read_text())
        hashes.append(manifest["deterministic_hash"])
        with np.load(output / "model2-input.npz", allow_pickle=False) as archive:
            assert set(archive.files) == {
                "pre_belief",
                "post_belief",
                "belief_difference",
                "entropy",
                "history_mask",
                "target_world",
            }
            record = Model2InputRecord.from_mapping(
                {key: archive[key].copy() for key in archive.files}
            )
            assert np.isfinite(record.pre_belief).all()
            assert np.any(record.pre_belief != record.post_belief)
        assert (output / "evaluation" / "index.html").is_file()
        assert (output / "evaluation" / "calibration.png").stat().st_size > 0
    assert hashes[0] == hashes[1]
