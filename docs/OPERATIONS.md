# SenseTrace Operations

SenseTrace has one controller-side remote interface. The CLI uses the `worker-03` SSH alias from `~/.ssh/config`, resolves that alias, and performs file transfer and commands through Fabric. Direct SSH remains appropriate for diagnostics and emergency recovery.

The equivalent Fabric task namespace is `fab worker03.<task>` from this checkout; for example, `fab worker03.doctor` and `fab worker03.status`.

## Controller setup

From the repository checkout:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

The sklearn baselines are the default CPU-only path. Install the optional `ml` extra only when a suitable local PyTorch wheel is available; the tiny MLP and CNN require PyTorch, but CUDA is not required by the models. Remote deployment intentionally installs only the core dependencies.

## Discover and deploy the host

```bash
sensetrace host doctor worker-03
sensetrace host inventory worker-03
sensetrace host bootstrap worker-03
sensetrace host deploy worker-03
# only when an operator deliberately wants the example to replace live config:
sensetrace host deploy worker-03 --reset-config
sensetrace host status worker-03
sensetrace host logs worker-03 --lines 100
sensetrace host boot-profile worker-03
sensetrace host services worker-03
sensetrace host noise-baseline worker-03
sensetrace host verify-boot worker-03
```

`deploy` is repeatable. It initializes a missing live configuration from the example, then preserves it on normal deployments. `--reset-config` is the deliberate replacement operation. The result records configuration hashes before and after deployment. With noninteractive sudo it first stops and disables the user fallback, installs `/opt/sensetrace`, `/etc/sensetrace`, `/var/lib/sensetrace`, the system unit, the candidate `sensetrace.target`, tmpfiles policy, and the documented kernel recovery sysctls. Without it, deployment uses `~/.local/share/sensetrace`, a user systemd unit, and records that reboot persistence is not proven until user-manager linger or a system unit is configured.

The controller reports exactly one management mode: `system`, `user-fallback`, `invalid-dual-service`, or `not-installed`. Service actions refuse `invalid-dual-service` and operate only on the authoritative scope.

The privileged script intentionally does not change the graphical boot target or disable SDDM. If headless boot is later desired, validate a fresh SSH connection first, make the change separately, and retain a rollback path.

The root installer is:

```bash
sudo bash /opt/sensetrace/source/infra/worker03/install-root.sh /opt/sensetrace/source
```

It sets `kernel.panic=10` and `kernel.panic_on_oops=1`. This addresses kernel panic/oops reboot behavior only; it does not prove recovery from a hard hang, power interruption, or firmware power-loss policy.

## Run and inspect Phase 0

Run a compact campaign remotely while retaining raw data on `worker-03`:

```bash
sensetrace run phase0 --host worker-03 \
  --config configs/phase0.example.yaml \
  --samples 12000 --trace-length 256 \
  --models logistic_regression boosted_trees \
  --conditions null injected shuffled
sensetrace results latest --host worker-03
sensetrace results fetch latest --host worker-03 --destination evidence/worker-03
```

Use `--curve` with `--conditions injected` to materialize the configured injection-strength curve. Result directories contain `run.json`, `host.json`, `config.json`, `environment.json`, `events.jsonl`, per-condition dataset manifests and split records, and `metrics.json`.

The runner service is intentionally idle until a campaign is explicitly launched. It can be inspected with:

```bash
sensetrace status --host worker-03
sensetrace host logs worker-03
```

## Recovery checks

The local and remote recovery check exercises a bounded acquisition, an interruption, a finalized-shard validation, a deliberately incomplete `.tmp` shard, deterministic resume, and duplicate-range detection:

```bash
sensetrace selftest recovery
sensetrace host verify-recovery worker-03
```

The remote command also kills only the user-service runner process with `SIGKILL` and verifies that systemd starts a new PID. It does not claim a kernel crash or power-loss test.

## Stop and storage

Stop the user fallback service directly over SSH if needed:

```bash
ssh worker-03 'systemctl --user stop sensetrace.service'
```

The configured acquisition guard records a `storage_guard_stop` event and stops before the configured minimum free GB and percentage thresholds. Raw traces remain on the worker by default; `results fetch` retrieves small evidence files only.

## Reboot limitations

`sensetrace host reboot worker-03` requires noninteractive sudo and is intentionally refused when that authorization is unavailable. A controlled reboot acceptance test must verify a new SSH connection, a changed boot ID, automatic service start, journal recovery, and resumed samples. `sensetrace host headless` changes the default to `multi-user.target` only after that authorization is available; `sensetrace host isolate-target` and `sensetrace host set-target` are separate explicit operations for the candidate target. The current worker has `graphical.target` as its default and `user manager linger=no`; therefore the user fallback is not an automatic-after-reboot appliance.

The worker exposes `/dev/watchdog`, `/dev/watchdog0`, and `/dev/watchdog1`. The current unprivileged inventory identifies `intel_oc_wdt` and `iTCO_wdt` with inactive 60-second and 30-second sysfs timeouts, respectively; systemd watchdog use remains disabled. SenseTrace does not enable a watchdog speculatively. The inventory also records RAPL domain names without pretending that unavailable energy reads are measurements.

## Phase 1A gate and safe observables

Phase 1A is gated by a Phase 0 report. The local command requires a report whose `acceptance.phase1_gate` is true:

```bash
sensetrace run phase1a --config configs/worker03.example.yaml \
  --phase0-report evidence/phase0/metrics.json --output runs
sensetrace host run-phase1a worker-03 \
  --config configs/worker03.example.yaml \
  --phase0-report evidence/phase0/metrics.json
```

The safe backend uses an anonymous page-aligned buffer, records whether `mlock` actually succeeded, performs ordinary user-space writes and reads, and records `perf_counter_ns` timing traces. The ordinary digital read is audit-only and is not in the feature matrix. `eviction_buffer` is explicitly best-effort cache eviction and does not prove a DRAM access. No physical address, row/bank identity, refresh disabling, voltage change, or disturbance loop is exposed.
