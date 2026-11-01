from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).parents[1]
EXPLORER = ROOT / "explorer.html"


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
    assert "<title>Empathy</title>" in source
    assert "<header><h1>Empathy</h1></header>" in source
    assert "text-align: center" in source
    for decorative_rule in (
        "border-bottom: 1px solid #565d56",
        "border-bottom: 1px solid #737a73",
        "border-top: 1px solid var(--line)",
        "border-bottom: 1px solid var(--line)",
    ):
        assert decorative_rule not in source
    assert "reset" not in parser.ids
    assert "Reset stream" not in source
    assert "<h2>Seeing</h2>" in source
    assert "<h2>Thinking</h2>" in source
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
def test_explorer_combines_ground_truth_path_and_belief_fuzz() -> None:
    source, parser = parsed_explorer()

    assert {
        "perceptionPanel",
        "perceptionCanvas",
        "secondOrderPanel",
        "comparisonRows",
    } <= parser.ids
    assert {"worldPanel", "worldCanvas", "beliefPanel", "beliefCanvas"}.isdisjoint(
        parser.ids
    )
    assert parser.canvases == {"perceptionCanvas"}
    assert not parser.external_urls
    assert "<h2>Seeing</h2>" in source
    assert "<h2>Thinking</h2>" in source
    assert "function drawGroundTruthPath" in source
    assert "function drawBeliefSuperposition" in source
    assert "drawBeliefFuzz" not in source
    assert "step.path_weights" in source
    assert "lineWidth" in source
    assert "globalAlpha" in source


def test_explorer_uses_vector_trajectories_without_ball_pixels() -> None:
    source, _ = parsed_explorer()

    assert "function drawOccluder" in source
    assert "function clipToOccluder" in source
    assert "function drawInitialPoint" in source
    assert "drawInitialPoint(currentEpisode.initial_state);" in source
    assert "function drawVisibleTrajectory" in source
    assert "drawVisibleTrajectory(truth);" in source
    assert "function drawEntryExitPoints" in source
    assert "drawEntryExitPoints(truth);" in source
    assert "boundaryPoint(truth.path, truth.occluder[0])" in source
    assert "boundaryPoint(truth.path, truth.occluder[1])" in source
    assert "clipToOccluder(truth.occluder, () =>" in source
    assert "drawGroundTruthPath(truth.path)" in source
    assert "progressiveGroundTruthPath" not in source
    for unwanted in (
        "function drawInitialCondition",
        "function drawExitPoint",
        "function drawBallMarker",
        "function imageBuffer",
        "frames_base64",
        "frameBytes",
        "drawImage",
        "createRadialGradient",
    ):
        assert unwanted not in source


def test_stream_controls_start_sessions_and_queue_offset_episodes() -> None:
    source, parser = parsed_explorer()

    assert {
        "gravity",
        "wind",
        "damping",
        "streamToggle",
    } <= parser.ids
    assert 'postJson("/api/session/start"' in source
    assert 'postJson("/api/session/next"' in source
    assert "async function startSession" in source
    assert "async function requestNextEpisode" in source
    assert "episodeQueue" in source
    assert "requestNextEpisode();" in source
    assert "Throw the ball" in source
    assert "Play thinking stream" not in source


def test_world_guess_shows_all_three_current_numeric_parameters() -> None:
    source, parser = parsed_explorer()

    assert "operatorReadout" not in parser.ids
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
    assert "function updateSecondOrder" in source
    assert "function formatParameter" in source
    assert "function posteriorElements" in source
    assert "step.parameter_posterior" in source
    assert "posterior-particle" in source
    assert "border-radius: 50%; background: #758073" in source
    assert "posterior-band" in source
    assert "posterior-mean" in source
    assert "posterior-truth" in source


def test_second_order_panel_shows_weighted_ranges_without_summary_block() -> None:
    source, parser = parsed_explorer()

    assert {"coherenceScore", "coherenceFill", "coherenceDetail"}.isdisjoint(parser.ids)
    assert parser.canvases == {"perceptionCanvas"}
    assert "function scalePercent" in source
    assert "posterior.intervals[slot.posteriorIndex]" in source
    assert "posterior.mean[slot.posteriorIndex]" in source
    assert "posterior.particles.forEach" in source
    assert 'axis.setAttribute("role", "img")' in source
    assert "hidden world" in source
    for noise in (
        "constraint-axis",
        "constraint-target",
        "constraint-guess",
        "constraint-previous",
        "constraint-legend",
        "constraint-values",
    ):
        assert noise not in source
    assert "mean closeness to hidden parameters" not in source


def test_parameter_edits_invalidate_the_stream_and_clear_all_layers() -> None:
    source, _ = parsed_explorer()

    assert "requestSerial += 1;" in source
    assert "sessionId = null;" in source
    assert "episodeQueue.length = 0;" in source
    assert "clearPerceptionCanvas();" in source
    assert "clearSecondOrder();" in source
    assert 'stopAndClear();' in source


def test_normal_playback_is_visual_without_narration_feeds() -> None:
    source, parser = parsed_explorer()

    assert "worldPhase" not in parser.ids
    assert "frameNarration" not in source
    for narration in (
        "The ball is approaching the occluder.",
        "Generating another offset…",
        "Ball hidden · the world continues",
        "Ball reappeared · evidence arrives",
        "Ball visible",
        "Stream reset. Start again when ready.",
        "Parameters changed. Start a new thinking stream.",
    ):
        assert narration not in source
    assert 'statusOverride = `Stream stopped: ${error.message}`' not in source


def test_playback_runs_at_fast_visual_cadence() -> None:
    source, _ = parsed_explorer()

    assert "const FRAME_INTERVAL_SECONDS = .004;" in source
    assert "while (accumulator >= FRAME_INTERVAL_SECONDS" in source
    assert "accumulator -= FRAME_INTERVAL_SECONDS;" in source


def test_live_episode_validation_checks_trajectory_beliefs_and_four_thoughts() -> None:
    source, _ = parsed_explorer()

    assert 'payload.schema_version !== "live-thinking-v1"' in source
    assert "payload.thought_steps.length !== 4" in source
    assert "const occluder = truth.occluder ?? DEFAULT_OCCLUDER;" in source
    assert "ground_truth: {...truth, occluder}" in source
    assert "ground-truth occluder must contain four normalized values" in source
    assert "belief byte length" in source
    assert "predicted_world_flat" in source
    assert "parameter_posterior" in source
    assert "posterior particles and weights are invalid" in source
    assert "posterior intervals or mean are invalid" in source
    assert "constraint recovery is invalid" in source
    assert "initial_state" in source
    assert "model1_checkpoint_version" in source
    assert "model2_checkpoint_version" in source
    assert "checkpoint versions are invalid" in source
    assert "requestAnimationFrame" in source
