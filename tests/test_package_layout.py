import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "src" / "empathy"

GROUPED_MODULES = {
    "physics": (
        "config",
        "world",
        "operators",
        "simulate",
        "render",
        "events",
        "families",
    ),
    "reasoning": (
        "beliefs",
        "reachability",
        "recovery",
        "identifiability",
        "shattering",
        "coherence",
        "records",
        "visualize",
    ),
    "live": ("inference", "thinking", "server"),
}


def test_source_modules_are_grouped_by_responsibility() -> None:
    assert PACKAGE.is_dir()
    from empathy.cli import _parser

    for package, modules in GROUPED_MODULES.items():
        assert (PACKAGE / package / "__init__.py").is_file()
        for module in modules:
            assert (PACKAGE / package / f"{module}.py").is_file()
            importlib.import_module(f"empathy.{package}.{module}")

    old_names = {
        *GROUPED_MODULES["physics"],
        *GROUPED_MODULES["reasoning"],
        "live_inference",
        "live_thinking",
        "live_server",
    }
    assert not [name for name in old_names if (PACKAGE / f"{name}.py").exists()]
    assert "live-demo" in _parser().format_help()


def test_live_cycle_imports_in_either_order() -> None:
    for first, second in (
        ("empathy.live.inference", "empathy.live.thinking"),
        ("empathy.live.thinking", "empathy.live.inference"),
    ):
        result = subprocess.run(
            [sys.executable, "-c", f"import {first}; import {second}"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_distribution_and_console_command_are_named_empathy() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

    assert project["name"] == "empathy"
    assert project["scripts"] == {"empathy": "empathy.cli:main"}
    assert "torch>=2.5" in project["dependencies"]
    assert "ml" not in project["optional-dependencies"]
    assert {path.name for path in (ROOT / "src").iterdir() if path.is_dir()} == {"empathy"}
