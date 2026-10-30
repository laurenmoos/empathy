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


def test_explorer_is_a_self_contained_parameter_interface() -> None:
    source, parser = parsed_explorer()

    required_controls = {
        "runSimulation",
        "runLearned",
        "reset",
        "wind",
        "gravity",
        "damping",
        "initialY",
        "initialVx",
        "occluderLeft",
        "occluderRight",
    }
    assert required_controls <= parser.ids
    assert parser.canvases == {"worldCanvas", "beliefCanvas"}
    assert not parser.external_urls
    assert "requestAnimationFrame" in source
    assert "Play simulation" in source
    assert "Learned rollout" in source
    assert 'fetch("learned-example.json")' in source
    assert "Start with an example" not in source
    assert 'id="preset"' not in source
    assert "const presets" not in source
    assert "playing = false" in source
    assert 'id="horizontalWallEnabled"' not in source
    assert "p.horizontalWallEnabled = true" in source
    assert 'id="verticalWallEnabled"' not in source
    assert "p.verticalWallEnabled = true" in source
    assert "· off" not in source
    assert "always on" not in source
    assert "<h2>Walls</h2>" not in source
    for control in (
        "verticalWallPosition",
        "verticalRestitution",
        "horizontalWallPosition",
        "horizontalRestitution",
    ):
        assert f'id="{control}"' not in source
    assert "p.verticalWallPosition = 0.9" in source
    assert "p.horizontalWallPosition = 0.1" in source
    assert "Initial horizontal speed" in source
    assert "world parameter vector" not in source
    assert "active world operators" not in source
    assert "Ball position" in source
    assert "Forces on the ball" in source
    assert "right wall" in source
    assert "floor" in source
    assert "Ground truth" in source
    assert "Model 1 · Belief superposition" in source
    assert "Model 2 · Second-order perception" in source
    assert "World guess" in source
    assert 'class="model-map"' not in source
    assert "simulator-generated stand-ins" not in source
    assert "Adjust the controls below" not in source
    assert "Set the physical parameters, then run" not in source
    assert '<br><span id="visibilityState">' not in source
    assert 'class="status"' not in source
    assert 'class="section-title"' not in source


def test_learned_example_connects_ground_truth_model1_and_model2() -> None:
    payload = json.loads(LEARNED_EXAMPLE.read_text())

    assert payload["schema_version"] == "learned-demo-v2"
    assert len(payload["rollouts"]) >= 2
    worlds = {tuple(rollout["ground_truth"]["world_flat"]) for rollout in payload["rollouts"]}
    assert len(worlds) >= 2
    for rollout in payload["rollouts"]:
        assert rollout["ground_truth"]["frame_count"] > 1
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


def test_view_selection_is_separate_from_playback() -> None:
    source, parser = parsed_explorer()

    assert {"simulationMode", "runLearned", "runSimulation", "reset", "modeContext"} <= parser.ids
    assert "Simulation view" in source
    assert "change forces and inspect the world" in source
    assert "Learned rollout" in source
    assert "watch truth, beliefs, and the world guess together" in source
    assert "Play simulation" in source


def test_learned_view_shows_truth_beliefs_and_guess_on_one_timeline() -> None:
    source, parser = parsed_explorer()

    assert {
        "truthPanel",
        "modelOnePanel",
        "modelTwoPanel",
        "rolloutPicker",
        "rolloutSelect",
        "truthPhase",
        "truthLegend",
        "comparisonRows",
    } <= parser.ids
    assert '$("truthPanel").hidden = false;' in source
    assert '$("modelOnePanel").hidden = !learnedMode;' in source
    assert '$("modelTwoPanel").hidden = !learnedMode;' in source
    assert '$("truthLegend").hidden = learnedMode;' in source
    assert "drawLearnedWorld(); drawLearnedBelief();" in source
    assert "if (learnedMode && learnedData && learnedPlaying) {" in source
    assert "payload.rollouts" in source
    assert "selectLearnedRollout" in source
    assert '$("rolloutSelect").addEventListener("change"' in source
    assert "Actual world" in source
    assert "Model 2 guess" in source
    assert "normalized parameters" in source


def test_model_view_renders_byte_beliefs_without_clipping() -> None:
    source, _ = parsed_explorer()

    assert "const normalized = value / 255;" in source
    assert "Math.sqrt(normalized)" in source
    assert 'beliefCtx.imageSmoothingQuality = "high";' in source
    assert "brighter = more rollouts agree" in source
    assert "current true state" not in source


def test_model_view_removes_irrelevant_simulation_controls_and_stretched_card() -> None:
    source, _ = parsed_explorer()

    assert source.count('class="card simulation-settings"') == 3
    assert 'document.body.classList.toggle("model-view", learnedMode);' in source
    assert ".model-view .simulation-settings { display: none; }" in source
    assert ".model-two { align-self: start;" in source
    assert "main, .model-view main { grid-template-columns: 1fr; }" in source


def test_explorer_exposes_the_same_operator_update_order_as_the_simulator() -> None:
    source, _ = parsed_explorer()

    order = [
        source.index("// 1. acceleration"),
        source.index("// 2. damping"),
        source.index("// 3. position"),
        source.index("// 4. collision resolution"),
    ]
    assert order == sorted(order)

