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


def test_explorer_is_one_live_parameter_inference_interface() -> None:
    source, parser = parsed_explorer()

    assert {
        "gravity",
        "gravityValue",
        "wind",
        "windValue",
        "damping",
        "dampingValue",
        "runInference",
        "playback",
        "reset",
        "worldCanvas",
    } <= parser.ids
    assert parser.canvases == {"worldCanvas"}
    assert not parser.external_urls
    assert "requestAnimationFrame" in source
    assert "Set the world" in source
    assert 'fetch("/api/infer"' in source
    assert 'gravity: Number($("gravity").value)' in source
    assert 'wind: Number($("wind").value)' in source
    assert 'damping: Number($("damping").value)' in source
    assert "Observed rollout + Model 1 beliefs" in source
    assert "Model 2 · Second-order perception" in source
    for removed in (
        "simulationMode",
        "runLearned",
        "modelOnePanel",
        "beliefCanvas",
        "view-switch",
        "rolloutSelect",
        "learned-example.json",
    ):
        assert removed not in source


def test_live_request_has_loading_error_and_comparison_states() -> None:
    source, _ = parsed_explorer()

    assert '"Content-Type": "application/json"' in source
    assert "JSON.stringify(parameters)" in source
    assert "response.ok" in source
    assert "payload.error" in source
    assert '$("runInference").disabled = true' in source
    assert '$("runInference").disabled = false' in source
    assert "Ground-truth parameters" in source
    assert "Guessed parameters" in source
    assert 'return "inactive"' in source


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
    assert 'payload.schema_version !== "live-inference-v1"' in source
    assert "validateLiveResult" in source
    assert "Couldn’t run inference" in source
    assert "runInference" in source
    assert "Ground-truth parameters" in source
    assert "Guessed parameters" in source
    assert "normalized parameters" not in source
    assert "beliefPhase" not in parser.ids
    assert "Belief state" not in source


def test_belief_overlay_renders_byte_values_without_clipping() -> None:
    source, _ = parsed_explorer()

    assert "const normalized = value / 255;" in source
    assert "Math.sqrt(normalized)" in source
    assert 'worldCtx.imageSmoothingQuality = "high";' in source

