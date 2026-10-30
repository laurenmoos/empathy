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


def test_explorer_is_a_self_contained_parameter_interface() -> None:
    source, parser = parsed_explorer()

    required_controls = {
        "runSimulation",
        "reset",
        "wind",
        "gravity",
        "damping",
        "verticalWallPosition",
        "verticalRestitution",
        "horizontalWallPosition",
        "horizontalRestitution",
        "initialY",
        "initialVx",
        "occluderLeft",
        "occluderRight",
    }
    assert required_controls <= parser.ids
    assert parser.canvases == {"worldCanvas", "beliefCanvas"}
    assert not parser.external_urls
    assert "requestAnimationFrame" in source
    assert "Run simulation" in source
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
    assert source.count("<h2>Walls</h2>") == 1
    assert "Adjust the controls below" not in source
    assert "Set the physical parameters, then run" not in source
    assert '<br><span id="visibilityState">' not in source
    assert 'class="status"' not in source
    assert 'class="section-title"' not in source


def test_explorer_exposes_the_same_operator_update_order_as_the_simulator() -> None:
    source, _ = parsed_explorer()

    order = [
        source.index("// 1. acceleration"),
        source.index("// 2. damping"),
        source.index("// 3. position"),
        source.index("// 4. collision resolution"),
    ]
    assert order == sorted(order)

