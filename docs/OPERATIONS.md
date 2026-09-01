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
sensetrace host status worker-03
sensetrace host logs worker-03 --lines 100
```

`deploy` is repeatable. With noninteractive sudo it installs `/opt/sensetrace`, `/etc/sensetrace`, `/var/lib/sensetrace`, the system unit, tmpfiles policy, and the documented kernel recovery sysctls. Without it, deployment uses `~/.local/share/sensetrace`, a user systemd unit, and records that reboot persistence is not proven until user-manager linger or a system unit is configured.

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

`sensetrace host reboot worker-03` requires noninteractive sudo and is intentionally refused when that authorization is unavailable. A controlled reboot acceptance test must verify a new SSH connection, a changed boot ID, automatic service start, journal recovery, and resumed samples. The current worker has `graphical.target` as its default and `user manager linger=no`; therefore the user fallback is not an automatic-after-reboot appliance.

The worker exposes `/dev/watchdog`, `/dev/watchdog0`, and `/dev/watchdog1`, but no provider was safely identified from the unprivileged session. SenseTrace does not enable a watchdog speculatively.
