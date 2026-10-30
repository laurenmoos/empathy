import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).parents[1]
EXPLORER = ROOT / "explorer.html"
LEARNED_EXAMPLE = ROOT / "learned-example.json"


class ExplorerParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.canvases: set[str] = set()
        self.external_urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        element_id = values.get("id")
        if element_id:
            self.ids.add(element_id)
            if tag == "canvas":
                self.canvases.add(element_id)
        for name in ("src", "href"):
            value = values.get(name)
            if value and value.startswith(("http://", "https://")):
                self.external_urls.append(value)


def parsed_explorer() -> tuple[str, ExplorerParser]:
    source = EXPLORER.read_text()
    parser = ExplorerParser()
    parser.feed(source)
    return source, parser


def test_explorer_is_one_self_contained_learned_rollout_interface() -> None:
    source, parser = parsed_explorer()

    assert {"playback", "reset", "rolloutSelect", "worldCanvas"} <= parser.ids
    assert parser.canvases == {"worldCanvas"}
    assert not parser.external_urls
    assert "requestAnimationFrame" in source
    assert "Learned rollout" in source
    assert 'fetch("learned-example.json")' in source
    assert "Observed rollout + Model 1 beliefs" in source
    assert "Model 2 · Second-order perception" in source
    for removed in ("simulationMode", "runLearned", "modelOnePanel", "beliefCanvas", "view-switch"):
        assert removed not in source


def test_learned_example_connects_ground_truth_model1_and_model2() -> None:
    payload = json.loads(LEARNED_EXAMPLE.read_text())

    assert payload["schema_version"] == "learned-demo-v2"
    assert len(payload["rollouts"]) >= 2
    worlds = {tuple(rollout["ground_truth"]["world_flat"]) for rollout in payload["rollouts"]}
    assert len(worlds) >= 2
    for rollout in payload["rollouts"]:
        truth = rollout["ground_truth"]
        assert truth["frame_count"] > 1
        assert 0 <= truth["reappearance_step"] < truth["frame_count"]
        assert rollout["ground_truth"]["frames_base64"]
        assert len(rollout["model1"]["pre_belief_u8"]) > 100
        assert len(rollout["model1"]["post_belief_u8"]) > 100
        assert len(rollout["model2"]["predicted_world_flat"]) == 12
        assert len(rollout["ground_truth"]["world_flat"]) == 12


def test_learned_square_frames_preserve_their_aspect_ratio() -> None:
    source, _ = parsed_explorer()

    assert "const side = Math.min(worldCanvas.width, worldCanvas.height);" in source
    assert "drawImage(buffer, 0, 0, worldCanvas.width, worldCanvas.height)" not in source
    assert 'worldCtx.imageSmoothingQuality = "high";' in source


def test_beliefs_are_composited_into_the_observed_rollout() -> None:
    source, parser = parsed_explorer()

    assert {
        "truthPanel",
        "modelTwoPanel",
        "rolloutSelect",
        "truthPhase",
        "beliefLegend",
        "comparisonRows",
    } <= parser.ids
    assert "drawBeliefOverlay" in source
    assert "worldCtx.globalCompositeOperation = \"screen\";" in source
    assert "worldCtx.drawImage(beliefBuffer" in source
    assert "learnedFrame >= truth.hidden_start && learnedFrame <= truth.hidden_end" in source
    assert "purple = Model 1’s possible hidden paths" in source
    assert "learnedFrame >= truth.reappearance_step" in source
    assert "payload.rollouts" in source
    assert "validateLearnedPayload" in source
    assert "validateLearnedRollout" in source
    assert "Couldn’t load learned rollouts" in source
    assert "selectLearnedRollout" in source
    assert '$("rolloutSelect").addEventListener("change"' in source
    assert "Ground-truth parameters" in source
    assert "Guessed parameters" in source
    assert "normalized parameters" in source
    assert "beliefPhase" not in parser.ids
    assert "Belief state" not in source


def test_belief_overlay_renders_byte_values_without_clipping() -> None:
    source, _ = parsed_explorer()

    assert "const normalized = value / 255;" in source
    assert "Math.sqrt(normalized)" in source
    assert 'worldCtx.imageSmoothingQuality = "high";' in source

