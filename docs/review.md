# Code review and resolution

Review base: empty repository baseline `367ec0d`; initial implementation `b5ffa2f`. Two independent review axes were run as required by the implement/code-review skills. The following fixes were reviewed again within their original scope.

## Standards

- Analysis could be marked complete before temporal-analysis outputs finished. Fixed with running/failed/complete finalization around all additional operations; an injected plotting-failure test verifies the failed status.
- Analysis read unchecked configuration metadata. Fixed through shared `load_record`, validating configuration and HDF5 checksums for every reader.
- Possible divergent responsibilities in workflow.py. Plotting and ordinary analysis now live in experiments/analysis.py, separate from computation and persistence.

Reviewer confirmed all three findings closed. The last item was a maintainability judgement, not a hard standard violation.

## Spec

- P1: numerical splitting of a repeated real root could discard the minimum-|r| branch. Fixed degree-dependent roundoff neighborhoods followed by real-axis scaled-residual validation. The regression `(r−1)^2(r+3)` retains three numerical candidates and selects r≈1.
- P2: edited config metadata could distort temporal convergence labels/orders. Closed by the shared integrity gate; complete records are required for temporal comparisons.

Reviewer confirmed both findings closed. Stage coefficients, cubic algebra, pressure scaling, MAC constraints and step-sequence semantics inspected were consistent with the specification.

Original findings: Standards 3 (one judgement call); Spec 2. Remaining findings from these reviews: 0 on each axis. Numerical root screening remains a floating-point policy, not certified algebraic isolation.
