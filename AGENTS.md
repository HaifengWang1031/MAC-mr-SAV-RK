# Working in this project
Read CONTEXT.md before changing numerical semantics. For the confirmed scope and test seams, read docs/spec.md. For storage and run identities, read docs/architecture.md.

Keep model, numerical stepping, persistence and plotting separate. Use type annotations at public Python interfaces. Preserve the documented MAC array layout and use full divergence checks after linear solves. Meaningful tests exercise operators, stage/integration results and public run/load/analyze workflows; use independently known solutions as references.

Run a focused test file after each numerical change and mypy regularly. Run the full suite for release. Keep runs and reports separate; analysis consumes explicit run IDs and never launches computation. Record validation limits in docs/validation.md. Source material in the original desktop projects is read-only.
