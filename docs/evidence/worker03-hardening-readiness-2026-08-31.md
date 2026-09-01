# worker-03 hardening readiness evidence

This is a pre-transition evidence record. It deliberately does not claim that
privileged host changes or reboot acceptance have passed.

## Controller and live host

- SSH/Fabric doctor: passed on 2026-08-31.
- Host: `worker-03`, Fedora 43 KDE, kernel `6.17.1-300.fc43.x86_64`.
- Boot ID observed: `4114c2dc-89fd-45ee-8f68-2ff4ad58b00d`.
- Current default target: `graphical.target`.
- Current running service count: 40.
- `sddm.service`: active.
- SSH: active and enabled.
- SenseTrace system unit: not installed/inactive.
- SenseTrace user fallback: active and enabled.
- Authoritative controller mode: `user-fallback`.
- Runner process count: 1 (PID 176356 during this observation).
- Root filesystem: approximately 214 GiB available at observation time.
- Watchdog inventory: `intel_oc_wdt`/60 s and `iTCO_wdt`/30 s, both inactive; systemd watchdog use disabled.
- Energy inventory: RAPL package/core/uncore/DRAM domain names visible, energy reads unavailable to `worker-03`.
- PMU inventory: `perf` unavailable to the unprivileged environment.

## Privileged boundary

`sudo -n true` failed. Therefore the following remain unapplied and unclaimed:

- system service migration;
- `multi-user.target` default;
- `sensetrace.target` installation/isolation/default;
- display-manager disablement;
- `kernel.panic=10` and `kernel.panic_on_oops=1`;
- watchdog enablement or reset-behavior testing;
- firmware AC-power recovery inspection/configuration;
- remote reboot acceptance.

The repository now contains explicit, reversible controller operations for the
authorized transition. No password, broad sudo rule, or undocumented firmware
mechanism was introduced.

## Scientific gate

The Phase 0 implementation now repairs trailing journal tails, materializes
true shuffled-label controls from the same injected observations, evaluates all
enabled models, reports group-unit confidence intervals, and emits visible
leakage audits. The boosted-tree null elevation remains `FAIL / INVESTIGATE`
when its uncertainty excludes chance in the recorded grouped assessment. The
Phase 1 gate therefore remains closed pending independent null-resampling and
group-balance investigation.

## Claim boundary

This artifact establishes repository/controller readiness evidence and a clear
privilege boundary. It does not establish headless boot, crash/panic/watchdog
recovery, direct DRAM access, physical row/bank topology, or a physical
measurement result.
