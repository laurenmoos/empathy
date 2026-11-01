from __future__ import annotations

import json

from empathy.model1.online_cli import run_online_training


def test_headless_training_writes_json_events_and_checkpoint(tmp_path, capsys) -> None:
    output = tmp_path / "seeing"
    metrics = run_online_training(episodes=2, log_dir=output, seed=11)
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert len(lines) == 2
    assert lines == metrics
    assert all(
        set(row) == {"episode", "prediction_loss", "belief_spread", "updates"}
        for row in lines
    )
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 2
    assert list(output.glob("events.out.tfevents.*"))
    assert (output / "online-seeing.pt").is_file()


def test_headless_training_is_seeded(tmp_path, capsys) -> None:
    first = run_online_training(episodes=2, log_dir=tmp_path / "one", seed=5)
    capsys.readouterr()
    second = run_online_training(episodes=2, log_dir=tmp_path / "two", seed=5)
    capsys.readouterr()
    assert first == second
