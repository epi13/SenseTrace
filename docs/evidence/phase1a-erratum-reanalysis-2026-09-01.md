# Phase 1A split erratum and reanalysis — 2026-09-01

This is an erratum to the retained worker-03 Phase 1A evidence record. The
original report and raw run are preserved; the historical artifact is not
rewritten.

## Defect

The original implementation named the strict level
`E_unseen_boot_session` and grouped it by `(boot_id,
acquisition_session_id)`. All samples in the campaign had the same recorded
OS boot ID, while acquisition-session IDs differed. That combination therefore
created enough groups to report E as available even though no unseen OS boot
was held out.

## Corrected interpretation

The original E result is not unseen-boot evidence and is downgraded to
unavailable under the corrected implementation. A corrected E split groups
only by `boot_id` and requires at least three genuine independent boot groups.
D remains a valid acquisition-session holdout because the campaign had four
independently started sessions. The B primary result is unchanged: logistic
AUROC was approximately 0.4959, boosted-tree AUROC approximately 0.4912, and
the paired latency-delta sign-flip p-value was 0.4118 with an interval spanning
zero.

Nothing in this correction creates evidence for physical DRAM-state inference.
The retained campaign remains a negative exploratory result for safe
commodity-memory host observables under its tested measurement regime.

## Reanalysis status

The retained raw dataset is not present in the versioned repository; its
controller-fetched summary remains under the ignored local `runs/` location
and the worker raw run remains on worker-03. When the retained raw shards are
available, the corrected analyzer can regenerate D and mark E unavailable
without changing trace, label, or model inputs. New runs use the corrected
`E_unseen_boot` name and record split invariants.
