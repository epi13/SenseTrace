# ADR-004: Phase 0 controls, uncertainty, and CPU-only CI

## Decision

Phase 0 uses materialized observations for the shuffled-label control. The
shuffled condition reuses the injected observation traces and changes only the
label association, recording the parent dataset fingerprint, both label-stream
fingerprints, and the permutation reference. The permutation is constrained to
declared exchangeability strata.

Every enabled model is evaluated for null and shuffled controls. A single
`run phase0` is explicitly `UNCALIBRATED` and cannot open Phase 1. The
`calibrate phase0` command materializes independent datasets, builds an
empirical maximum-statistic null distribution across models and metrics, and
then evaluates a fresh gate-validation ensemble. Point estimates and marginal
p-values remain visible, but the family-wise adjusted decision is the gate.
Confidence intervals resample the configured experimental unit, which is
session-level by default in the legacy single-run report.

Construction audits are visible in each condition but are never inference
features or valid physical claims. The repository CI runs format, lint, mypy,
and pytest on CPU-only GitHub-hosted runners. Hardware, SSH, Fabric, and
privileged checks remain separate acceptance evidence.

## Consequence

The previously observed boosted-tree null elevation is not suppressed by a
threshold. It remains an explicit investigation result until independent null
resampling and group-balance evidence resolves finite-sample and split
alternatives.
