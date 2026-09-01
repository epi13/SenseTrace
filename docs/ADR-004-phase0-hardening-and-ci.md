# ADR-004: Phase 0 controls, uncertainty, and CPU-only CI

## Decision

Phase 0 uses materialized observations for the shuffled-label control. The
shuffled condition reuses the injected observation traces and changes only the
label association, recording the parent dataset fingerprint, both label-stream
fingerprints, and the permutation reference.

Every enabled model is evaluated for null and shuffled controls. Control
acceptance is model-aware: `PASS` means the reported intervals are consistent
with chance, `WARN` means a numerical elevation remains uncertain, and `FAIL /
INVESTIGATE` means an interval excludes chance. WARN and FAIL both close the
Phase 1 gate. Confidence intervals resample the configured experimental unit,
which is session-level by default in Phase 0.

Construction audits are visible in each condition but are never inference
features or valid physical claims. The repository CI runs format, lint, mypy,
and pytest on CPU-only GitHub-hosted runners. Hardware, SSH, Fabric, and
privileged checks remain separate acceptance evidence.

## Consequence

The previously observed boosted-tree null elevation is not suppressed by a
threshold. It remains an explicit investigation result until independent null
resampling and group-balance evidence resolves finite-sample and split
alternatives.
