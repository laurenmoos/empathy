import subprocess
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).parents[1]
EXPLORER = ROOT / "explorer.html"


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
    assert "Ground-truth world" in source
    assert "Model 1 · Belief space" in source
    assert "Model 2 · Second-order perception" in source
    assert "Model 2 never sees the world panel" in source
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
        "episodeReadout",
        "offsetReadout",
    } <= parser.ids
    assert 'postJson("/api/session/start"' in source
    assert 'postJson("/api/session/next"' in source
    assert "async function startSession" in source
    assert "async function requestNextEpisode" in source
    assert "episodeQueue" in source
    assert "requestNextEpisode();" in source
    assert "Play thinking stream" in source
    assert "adaptation within one fixed world" in source


def test_thought_steps_and_online_updates_have_visible_state() -> None:
    source, parser = parsed_explorer()

    assert {
        "thoughtReadout",
        "operatorReadout",
        "latentUpdates",
        "weightUpdates",
        "onlineLoss",
    } <= parser.ids
    for label in (
        "first possibility",
        "more possibilities",
        "full belief",
        "reappearance revision",
    ):
        assert label in source
    assert "Ground-truth parameters" in source
    assert "Pre-update guess" in source
    assert "latent_update_count" in source
    assert "weight_update_count" in source
    assert "training.loss" in source
    assert "function updateLearningReadout" in source
    assert "index === 3" in source


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
if (frameNarration(true, false, null) !== 'Model 1 is expanding possible hidden paths.') {
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
