# worker-03 appliance readiness — 2026-09-01

## Observed fact

The controller re-inspected the live host before making changes.

- SSH and Fabric connectivity: working.
- Host: `worker-03`; one runner PID (`186479`) under the user fallback.
- Authoritative management mode: `user-fallback`.
- System service: not installed/enabled/active.
- User service: enabled and active.
- Default target: `graphical.target`.
- Active display manager: SDDM.
- Running services: 39.
- CPU governor: `powersave` on all eight CPUs; `intel_pstate`; turbo active;
  `turbostat` available.
- Watchdogs: `intel_oc_wdt` (60 s) and `iTCO_wdt` (30 s) both loaded and
  inactive; both report `nowayout=0`. Systemd runtime watchdog is `0`; reboot
  watchdog is `10min`.
- `sudo -n true`: not authorized; an interactive sudo password was not
  available to the controller session.
- Final unprivileged deployment source marker: commit
  `0aa130fd761d92781125028b5158363840fbdd90`.
- The live operator config remained unchanged across deployment:
  `fd04b5ed629f96c44e5bb7049e1466a9494b9b8584e0ba3568c26f7d1114e569`.
- The deployment attempted the native build, but worker-03 has no `make`; the
  build therefore did not install the native timing library. The user service
  nevertheless restarted with exactly one runner (current observed PID
  `192955`).
- A remote Phase 0 smoke run materialized two replicates per condition across
  both balance modes and wrote run
  `phase0-calibration-20260901T050827Z-c30294d1`. Its frozen protocol hash was
  `3e10e9a1e9b4cf8b98dd9e52855b430e41ed1fd415bf22ff9b1f310d80e2c0db`;
  fresh null/shuffled rates were `0/2`, while the one-replicate-per-mode weak
  injection smoke did not meet the configured detection threshold.

## Interpretation

The live state is still the safe pre-transition state recorded after PR #4.
No headless, system-service, dedicated-target, reboot, or watchdog mutation was
performed. The controller now exposes an explicit terminal-only `authorize-sudo`
operation; it never stores or logs credentials. Once authorized, the staged
workflow is system service, fresh-SSH validation, multi-user baseline, target
isolation, and only then default-target/reboot acceptance.

The native deployment path builds the timing library when a compiler is
available, and Phase 1A refuses to run when its configured native requirement is
not met.

## Unresolved confounds

No multi-user baseline, fresh post-reboot SSH acceptance, system-service boot,
dedicated-target isolation, or hardware watchdog reset test has yet been
performed. The remote smoke is engineering validation only, not a Phase 0
gate ensemble. Display-manager removal, service-count changes, CPU-profile
changes, and watchdog selection remain operator-authorized actions. The worker
also requires a local native-toolchain installation or a separately approved
native-library provisioning path before Phase 1A can run.

## Claim boundary

worker-03 is not yet a proven dedicated headless appliance. The current evidence
supports only the user-service fallback and live read-only inventory.
