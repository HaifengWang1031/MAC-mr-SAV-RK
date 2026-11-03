# Working in this project
Read CONTEXT.md before changing numerical semantics. For the confirmed scope and test seams, read docs/spec.md. Before changing module ownership, dependencies, controller configuration, persistence or analysis, read [docs/architecture.md](docs/architecture.md): it maps current owners, extension points, compatibility rules and tests. Update that document in the same change when those contracts change.

Keep model, numerical stepping, persistence and plotting separate. Use type annotations at public Python interfaces. Preserve the documented MAC array layout and use full divergence checks after linear solves. Meaningful tests exercise operators, stage/integration results and public run/load/analyze workflows; use independently known solutions as references. Keep one representative test per mechanism: `tests/` carries the mechanisms that still exist, and each new feature lands with its own focused tests.

Run a focused test file after each numerical change and mypy regularly. Run the full suite for release. Keep runs and reports separate; analysis consumes explicit run IDs and never launches computation. Record validation limits in docs/validation.md. Source material in the original desktop projects is read-only.

## Agent skills

### Issue tracker

Issues live in local markdown under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five default triage roles as local issue status values. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` and `docs/adr/`. See `docs/agents/domain.md`.
