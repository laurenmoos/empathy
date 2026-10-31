import importlib
import subprocess
import sys
from pathlib import Path

from perception_env.cli import _parser

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "src" / "perception_env"

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
    for package, modules in GROUPED_MODULES.items():
        assert (PACKAGE / package / "__init__.py").is_file()
        for module in modules:
            assert (PACKAGE / package / f"{module}.py").is_file()
            importlib.import_module(f"perception_env.{package}.{module}")

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
        ("perception_env.live.inference", "perception_env.live.thinking"),
        ("perception_env.live.thinking", "perception_env.live.inference"),
    ):
        result = subprocess.run(
            [sys.executable, "-c", f"import {first}; import {second}"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
