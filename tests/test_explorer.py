import subprocess
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).parents[1]
EXPLORER = ROOT / "explorer.html"
EXPLORER_STATE = ROOT / "explorer_state.js"


class ExplorerParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.canvases: set[str] = set()
        self.input_types: list[str] = []
        self.external_urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        element_id = values.get("id")
        if element_id:
            self.ids.add(element_id)
            if tag == "canvas":
                self.canvases.add(element_id)
        if tag == "input" and values.get("type"):
            self.input_types.append(values["type"])
        for name in ("src", "href"):
            value = values.get(name)
            if value and value.startswith(("http://", "https://")):
                self.external_urls.append(value)


def parsed_explorer() -> tuple[str, ExplorerParser]:
    source = EXPLORER.read_text()
    parser = ExplorerParser()
    parser.feed(source)
    return source, parser


def test_explorer_uses_minimal_numeric_controls_and_plain_layer_names() -> None:
    source, parser = parsed_explorer()

    assert parser.input_types == ["number", "number", "number"]
    assert "<h2>Ground truth</h2>" in source
    assert "<h2>Beliefs</h2>" in source
    assert "<h2>Second-order perception</h2>" in source
    for unwanted in (
        "Choose one world",
        "down ↔ up",
        "left ↔ right",
        "Set the world",
        "adaptation within one fixed world",
        "Model 1",
        "Model 2",
        "Waiting for a belief",
        "Waiting for an episode",
        "Actual rollout.",
        "Possible hidden paths accumulate",
        "No belief yet",
        "<small>Episode</small>",
        "Initial offset",
        "Latent updates",
        "Weight updates",
        "Latest online loss",
        "Thought step",
        ">Ready.</p>",
    ):
        assert unwanted not in source
    state_source = EXPLORER_STATE.read_text()
    assert "Model 1" not in state_source
    assert "Model 2" not in state_source


def test_explorer_has_three_visually_disjoint_perceptual_layers() -> None:
    source, parser = parsed_explorer()

    assert {
        "worldPanel",
        "worldCanvas",
        "beliefPanel",
        "beliefCanvas",
        "secondOrderPanel",
        "comparisonRows",
    } <= parser.ids
    assert parser.canvases == {"worldCanvas", "beliefCanvas"}
    assert not parser.external_urls
    assert "Ground truth" in source
    assert "Beliefs" in source
    assert "Second-order perception" in source
    assert "drawBeliefOverlay" not in source
    assert 'globalCompositeOperation = "screen"' not in source


def test_stream_controls_start_sessions_and_queue_offset_episodes() -> None:
    source, parser = parsed_explorer()

    assert {
        "gravity",
        "wind",
        "damping",
        "streamToggle",
        "reset",
    } <= parser.ids
    assert 'postJson("/api/session/start"' in source
    assert 'postJson("/api/session/next"' in source
    assert "async function startSession" in source
    assert "async function requestNextEpisode" in source
    assert "episodeQueue" in source
    assert "requestNextEpisode();" in source
    assert "Play thinking stream" in source


def test_thought_steps_and_world_guess_have_visible_state() -> None:
    source, parser = parsed_explorer()

    assert {"thoughtReadout", "operatorReadout"} <= parser.ids
    assert {
        "latentUpdates",
        "weightUpdates",
        "onlineLoss",
        "thoughtStepMetric",
    }.isdisjoint(parser.ids)
    for label in (
        "first possibility",
        "more possibilities",
        "full belief",
        "reappearance revision",
    ):
        assert label in source
    assert "Ground-truth parameters" in source
    assert "Pre-update guess" in source
    assert "function updateSecondOrder" in source


def test_parameter_edits_invalidate_the_stream_and_clear_all_layers() -> None:
    source, _ = parsed_explorer()

    assert "requestSerial += 1;" in source
    assert "sessionId = null;" in source
    assert "episodeQueue.length = 0;" in source
    assert "clearWorldCanvas();" in source
    assert "clearBeliefCanvas();" in source
    assert "clearSecondOrder();" in source
    assert "Parameters changed. Start a new thinking stream." in source


def test_live_episode_validation_checks_frames_beliefs_and_four_thoughts() -> None:
    source, _ = parsed_explorer()

    assert 'payload.schema_version !== "live-thinking-v1"' in source
    assert "payload.thought_steps.length !== 4" in source
    assert "frame byte length" in source
    assert "belief byte length" in source
    assert "predicted_world_flat" in source
    assert "initial_state" in source
    assert "requestAnimationFrame" in source


def test_frame_narration_preserves_stream_failures() -> None:
    script = """
const {frameNarration} = require('./explorer_state.js');
const failure = 'Stream stopped: Network unavailable';
if (frameNarration(false, false, failure) !== failure) process.exit(1);
if (frameNarration(true, false, null) !== 'Possible hidden paths are expanding.') {
  process.exit(2);
}
"""

    result = subprocess.run(
        ["node", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
