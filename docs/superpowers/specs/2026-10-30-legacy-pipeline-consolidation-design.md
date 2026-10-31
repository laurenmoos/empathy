# Legacy Pipeline Consolidation Design

## Intent

Make the repository easier to understand without flattening focused research modules into larger files. Remove the superseded prototype lane that predates the live Pinball demo and the current Model 1 streaming trainer.

## Audit finding

The package contains about 11,000 lines across environment, Model 1, Model 2, evaluation, and visualization modules. Most similarly named files are scoped to different models and should remain separate. One isolated legacy lane is genuinely redundant:

- `continual.py`
- `streaming.py`
- `stream_records.py`
- `streaming_demo.py`
- `export_learned_example.py`
- `configs/streaming/v1.yaml`
- `explorer_state.js`

The five Python modules total 1,337 lines. They are referenced only by their own tests and the root CLI's old `stream-demo` command. The current interactive path uses `live_inference.py`, `live_thinking.py`, and `live_server.py`; current Model 1 continual training uses `model1/stream_train.py`. `explorer_state.js` is no longer loaded by the browser and contains narration that the visible interface deliberately removed.

## Goals

- Remove the isolated legacy streaming/export lane and its dedicated tests.
- Remove the `stream-demo` CLI command while preserving `live-demo`.
- Remove the unused browser narration helper and its stale test.
- Keep the Pinball interface and live API behavior unchanged.
- Keep Model 1 and Model 2 training, evaluation, visualization, and their package-local CLIs unchanged.
- Keep core simulation, reachability, shattering, identifiability, recovery, and coherence research code unchanged.
- Leave historical plan documents intact as records of earlier work.

## Non-goals

- Do not combine Model 1 and Model 2 modules merely because they share filenames such as `config.py`, `train.py`, or `visualize.py`; their responsibilities and dependency boundaries differ.
- Do not merge `live_inference.py`, `live_thinking.py`, and `live_server.py` into one large file.
- Do not alter model architecture, training behavior, physics, API schemas, or the Pinball visual design.
- Do not delete artifacts, checkpoints, or user-authored README changes.

## Safety criteria

Before deleting a file, confirm it has no importer outside the retirement set. Replace positive tests for retired commands with a root-CLI test that confirms `live-demo` remains and `stream-demo` is absent. Run the focused CLI, live-thinking, live-server, and explorer tests, then the full suite with the moved repository's source path supplied explicitly. Ruff must pass without cache writes.

## Result

The root package loses one obsolete execution path and roughly 1,350 lines of source/helpers, while the current live demo and both learned-model pipelines retain their existing boundaries. The cleanup reduces conceptual surface area without creating oversized replacement modules.
