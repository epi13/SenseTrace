# Self-forecasting confirmation protocol — 2026-09-12

Status: frozen independent confirmation protocol. The initial worker run is
development/discovery data; this document records the unchanged analysis
before the fresh confirmation acquisition.

The first real worker run selected `future_timing_delta_sign` at horizon `1`
as the primary exploratory signal. This selection is explicitly post-discovery
and therefore is not presented as a preregistered discovery claim. The
confirmation run must execute the complete frozen configuration and report all
targets, horizons, models, and controls regardless of its outcome.

Frozen implementation and configuration:

- source commit: `4b9b2de592b88a098c16efcd21a1efa66bf431a6`
- configuration: `configs/self-forecasting-trace-worker03.example.yaml`
- normalized configuration fingerprint: `2644dc8fb517ec701c9ae25409aac5da9ddf926ab6e95dfb38d7b23543560ed2`
- acquisition: one fresh `CommodityDramBackend` session, `random_word`, 64 complete samples, 32 native measurement repetitions per sample
- targets: `future_timing_level` and `future_timing_delta_sign`
- horizons: 1, 2, 4, 8, and 16 `native_measurement_repetition` units
- models and controls: exactly the declared `horizon.models` list
- training seeds: 11, 23, and 37
- bootstrap/permutation repetitions: 400/400
- grouping: complete sample trajectory IDs; no cross-sample windows
- primary exploratory criterion: balanced accuracy ≥ 0.55 and max-statistic corrected p ≤ 0.05 at the selected target/horizon

The confirmation output is a new worker run directory with a new acquisition
session ID. It is not to be pooled with the development run for model or
target selection. A fresh boot is not required for this confirmation because
the acquisition session, allocation, and sample trajectories are new; the
boot ID must still be retained and any cross-boot replication remains a later
validation question.

Claim boundary: real SenseTrace commodity measurement-trajectory analysis
only. The protocol does not test hidden physical state, DRAM-origin
information, precognition, or model-inference foresight.
