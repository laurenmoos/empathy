# Package Structure Cleanup Design

## Intent

Make the repository feel small and legible without changing the Pinball demo, model behavior, training paths, or command-line interface. Remove generated and obsolete bulk, then group the remaining source by responsibility instead of leaving a large flat package.

## Approved structure

```text
src/perception_env/
  cli.py
  physics/
  reasoning/
  live/
  model1/
  model2/
```

- `physics/` owns world schemas, simulation configuration, forces, rollout, rendering, occlusion events, and world-family sampling.
- `reasoning/` owns beliefs, reachability, recovery, identifiability, shattering/coherence measures, records, and research visualizations.
- `live/` owns the browser-facing inference engine, continual thinking session, and HTTP server.
- `model1/` and `model2/` remain intact because they already form clear subsystems.
- `cli.py` remains at package root as the single command entry point.

The change updates imports and tests to the new package paths. It does not add compatibility wrapper modules: this is a private hackathon codebase, and wrappers would preserve the flat-file clutter the cleanup is intended to remove.

## Generated and obsolete material

Before deleting generated artifacts, move the one runtime dependency
`artifacts/model1-learned-demo/training/best.pt` to the tracked path
`checkpoints/model1.pt` and make that the live demo default. The current
`--model2-checkpoint` option is removed: the live engine explicitly discards
that argument and uses the online SMC runtime instead, so the option and its
artifact path are misleading dead interface.

Delete:

- ignored `artifacts/` outputs;
- ignored `.superpowers/` execution scratch data;
- Python, pytest, and Ruff caches;
- the empty `configs/streaming/` directory;
- tracked `learned-example.json`, which the live browser no longer reads.

Keep all six tracked YAML configs. They are small, active experiment presets referenced by current source or tests. Keep historical plans/specs and the user-owned `README.md` and `docs/images/` changes.

## Constraints

- Preserve the `perception-env` entry point and every current CLI command; remove only the unused `--model2-checkpoint` option.
- Preserve the Pinball UI and live API behavior.
- Preserve Model 1 and Model 2 behavior, checkpoints, and training interfaces.
- Move code without opportunistic behavioral edits.
- Do not combine semantically distinct modules merely to reduce the file count.
- Update all internal imports and tests; no stale imports from old flat module paths may survive.
- Verify with Ruff, the complete pytest suite, CLI help, and a live-server smoke test.

## Success

The checkout loses generated bulk and the unused embedded rollout fixture; the source tree reads as five coherent subsystems; all existing tests and commands still work; user-owned uncommitted files remain untouched.
