# worker-03 Experiment Host

## Purpose

`worker-03` is the intended dedicated SenseTrace acquisition and training host for early experiments. It can run long-duration experiments without interfering with the primary workstation and can be treated as disposable/recoverable research infrastructure when more aggressive DRAM tests are introduced.

Inventory observed from the controller on 2026-08-31:

- Fedora Linux 43 KDE Desktop, kernel `6.17.1-300.fc43.x86_64`;
- Dell Precision Tower 3431, board `01TN68`, BIOS `1.24.0`;
- Intel Core i7-9700, 8 cores, 32 GiB installed RAM, 8 GiB zram swap;
- NVIDIA Quadro P620 plus Intel UHD Graphics 630;
- WDC WD2500BEKT 232.9G disk, Btrfs root/home with approximately 215G free at inventory time;
- `eno1` at `192.168.1.113`, SSH active and enabled;
- DIMM/SPD, memory frequency/timings, voltage sensors, and Linux `sensors` unavailable from the unprivileged session;
- `/dev/watchdog`, `/dev/watchdog0`, and `/dev/watchdog1` exist, but the provider was not identified and systemd watchdog use is not enabled;
- `kernel.panic=0`, `kernel.panic_on_oops=0`, `graphical.target`, and `powersave` CPU governors;
- `sudo -n` requires a password, so the live deployment is currently user-scoped.

The monitor is not part of the tested workflow. The SSH alias in the controller's `~/.ssh/config` is the normal management path.

The experiment should not depend on the GPU. The initial model ladder is deliberately small enough to run on CPU, while the GPU may be used opportunistically where supported.

## Terminology

The **runner should operate unattended**, but the initial hidden-bit inference task is supervised: SenseTrace deliberately writes known random `0`/`1` target states and uses those states as labels. Unsupervised or self-supervised methods may later be useful for discovering latent DRAM behaviors, clustering cells, or learning representations of physical traces.

## Design goal

Treat `worker-03` as a dedicated SenseTrace appliance rather than as an interactive workstation.

The target lifecycle is:

```text
boot
  -> start SenseTrace runner
  -> recover experiment journal
  -> validate last completed shard
  -> resume acquisition
  -> periodically train/evaluate configured baselines
  -> record results and environment metadata
  -> continue until experiment completion or explicit stop
```

A reboot, kernel panic, power interruption, or intentionally destabilizing DRAM experiment should not invalidate already completed acquisition work.

## Runner responsibilities

The implemented `sensetrace-runner` coordinates the safe synthetic path through:

- experiment configuration loading;
- cryptographically random balanced target generation;
- controlled memory allocation and target writes;
- acquisition of enabled telemetry channels;
- in-memory buffering;
- chunked/sharded dataset writes;
- atomic shard finalization and checksums;
- experiment journaling and resume state;
- environment/configuration snapshots;
- dataset validation and fingerprinting;
- optional periodic model training;
- evaluation and negative controls;
- orderly stop and recovery behavior.

The runner must separate **acquisition metadata** from **model inputs** so host identity, addresses, run ordering, and similar nuisance variables cannot silently leak into the classifier.

## Storage strategy

The HDD is expected to be adequate for early experiments if acquisition is designed around sequential writes.

Do not create one file per trace. Use buffered append or chunked/sharded storage instead:

```text
acquisition loop
    -> memory buffer
    -> shard writer
    -> run-XXXX/shard-000123.tmp
    -> flush + checksum + metadata
    -> atomic rename/finalize
    -> run-XXXX/shard-000123.<format>
```

Initial shard targets may be on the order of 256 MB to 1 GB and should be tuned empirically.

Raw waveform volume can become large quickly. For example, one million traces with 2,048 32-bit samples is about 8 GB **per channel** before metadata. Multi-channel acquisition should therefore preserve raw data while making storage volume an explicit experiment parameter.

## Crash-safe acquisition

SenseTrace should assume that future experiments can crash `worker-03`.

Requirements:

1. Never mark a shard complete before all payload, metadata, and checksum data have been flushed.
2. Write incomplete shards to a temporary name.
3. Finalize with an atomic rename where the storage format permits it.
4. Maintain an append-only or otherwise crash-resilient experiment journal.
5. On startup, discard or quarantine incomplete temporary shards.
6. Verify the last finalized shard before resuming.
7. Resume from a deterministic experiment checkpoint without duplicating samples silently.

A crash is an operational event, not automatically an experiment failure. Any crash must still be logged because instability can itself correlate with DRAM settings and must not contaminate labels or splits.

## Environment and reproducibility ledger

Every acquisition run should capture enough information to reproduce or audit the result later. At minimum record:

- `run_id` and `session_id`;
- timestamp range;
- host identifier;
- DIMM identifiers and available SPD information;
- CPU and motherboard identifiers;
- BIOS/UEFI version and relevant memory settings where obtainable;
- operating system and kernel;
- memory frequency and exposed timings;
- CPU governor/frequency configuration;
- temperature telemetry and sensor provenance;
- relevant supply-voltage telemetry where obtainable;
- experiment configuration hash;
- code commit SHA;
- random-seed references used for reproducible software behavior;
- dataset/shard hashes;
- model/config hashes for any training run;
- crash/reboot/interruption events.

Do not use host, DIMM, row, bank, address, session, or ordering identifiers as model features by default. They are primarily grouping and audit fields.

## Host stabilization

For controlled measurements, reduce avoidable environmental noise where practical and record whatever cannot be controlled.

Potential controls include:

- minimize unnecessary background services;
- avoid unrelated workloads during acquisition;
- use a stable CPU frequency/governor configuration when appropriate;
- record thermal state rather than assuming it is constant;
- record memory configuration and boot-time changes;
- avoid automatic package/kernel updates during a long acquisition campaign unless intentionally scheduled between sessions.

The goal is not to create an unrealistically sterile machine. The goal is to know which variables changed.

## Memory-region isolation

Early experiments should use a deliberately controlled portion of installed RAM rather than treating all 32 GB as experimental space.

Where the platform and operating system allow, reserve or pin the acquisition region and minimize overlap with memory used by the OS and runner. More aggressive timing, refresh, or disturbance experiments should be introduced only after the runner can recover automatically and the affected region is well bounded.

The exact mechanism will depend on what commodity-controller access the host exposes. SenseTrace must document the actual isolation guarantee rather than assuming physical row or bank placement from virtual addresses.

## Compute expectations

The planned model sizes are small:

- logistic regression and boosted trees: CPU-first;
- tiny MLP/CNN baselines: CPU-capable;
- residual 1D CNN / small TCN: expected roughly 10K-100K parameters initially.

A 2 GB GPU is not expected to constrain the scientific objective. Training throughput should be secondary to acquisition integrity, split correctness, and negative controls.

## Cross-machine validation opportunity

Because `worker-03` and the primary workstation are closely matched, they provide a useful future generalization test.

A small validation campaign on the primary workstation could test whether a model trained on `worker-03` learned:

- a reusable physical relationship;
- a particular DIMM/device fingerprint;
- a motherboard/controller-specific signal;
- or a host-specific nuisance correlation.

The strongest version uses physically different DIMMs of the same model/revision, if available, because that preserves architecture while varying silicon instance.

The primary workstation should not be used for prolonged or intentionally destabilizing acquisition merely to obtain this validation. A small controlled holdout dataset is sufficient for the scientific question.

## Future hardware path

Commodity-host measurements are the starting point. If Phase 1 produces evidence worth pursuing, the experiment may progress to tighter control with research hardware such as an FPGA-based memory controller and, later, richer electrical instrumentation.

The worker-03 harness should therefore keep acquisition interfaces modular so the same dataset, split, validation, and model code can survive changes in measurement hardware.

## Initial implementation milestones

1. Host inventory records the reproducibility ledger with explicit `unavailable` values.
2. The synthetic backend exercises buffering, shard finalization, journaling, and deterministic resume.
3. Tests cover temporary and invalid finalized shards, checksums, split fingerprints, and identity rejection.
4. Phase 0 implements null, injected-signal, shuffled-label, grouped-split, repeated-seed, and baseline model controls.
5. `CommodityDramBackend` is an explicit safe placeholder; no disturbance, refresh disabling, voltage, or firmware manipulation is exposed.
6. The live worker user service and remote process-restart acceptance have been exercised. Kernel panic, watchdog, firmware power-loss, and reboot persistence remain privileged-host work.

## Evidence boundary

The worker campaign is synthetic. Its result can support the claim that a known synthetic perturbation is recovered under the recorded grouped split. It cannot support a claim about physical DRAM-state inference, commodity DRAM topology, or cross-machine transfer.
