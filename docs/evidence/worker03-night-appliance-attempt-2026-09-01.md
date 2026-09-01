# worker-03 appliance transition attempt — 2026-09-01

## Observed facts before the dedicated-target reboot

- Controller commit deployed to the system installation: `b64cd3753333ed969335e7255fdbc63959e03431`.
- Worker identity: `worker-03` (`uid=1000`, member of `wheel`), Fedora 43,
  kernel `6.17.1-300.fc43.x86_64`, Dell Precision Tower 3431 / Core i7-9700.
- The controller's key-based SSH alias, Fabric, and `sudo -n true` all worked.
- The old user fallback was stopped and disabled.  The authoritative
  `sensetrace.service` was enabled and active with exactly one runner process.
- The pre-migration user configuration SHA-256 was
  `fd04b5ed629f96c44e5bb7049e1466a9494b9b8584e0ba3568c26f7d1114e569`.
  The system configuration at `/etc/sensetrace/worker03.yaml` had the same
  hash after every deployment; no reset was requested.
- `kernel.panic=10` and `kernel.panic_on_oops=1` were installed.  This proves
  only the configured panic/oops reboot policy, not hard-hang recovery.
- Fedora packages `make` and `gcc` were installed after the initial cached RPM
  signature failure was resolved by clearing the package cache; signature
  checking was not bypassed.
- On worker-03, `make -C /opt/sensetrace/source/native test` passed.  The
  native library SHA-256 was
  `d4219ec48e00f16078f9b5b678f71a35bbb9ab9c4ba4bb95d9213fef090839b7`;
  its provenance reports `sensetrace-native-kernel-v1`, RDTSC/RDTSCP timing,
  and CLFLUSH support.
- A 2,048-sample native-control check measured cached median 22 TSC cycles and
  CLFLUSH-control median 189 cycles.  This is differentiated cache-control
  evidence only; it is not a DRAM-row or physical-address measurement.
- The recovery self-test passed, including journal-tail repair, temporary-shard
  quarantine, deterministic resume, and a deliberate system-runner process
  kill/restart (`203015` -> `204722`).

## Boot-profile transition

- `multi-user.target` was set as the headless baseline after a fresh SSH check.
  Fresh SSH and Fabric worked, SDDM/display-manager was inactive, and the
  system runner remained active.
- Five observed multi-user reboot returns occurred with new boot IDs and with
  SSH, `sshd`, the system runner, and the preserved configuration available.
  This exceeds the requested three-cycle multi-user acceptance count, although
  controller transport retries prevented one compact single-command report.
- `sensetrace.target` was isolated successfully before it became default.
  During isolation, fresh SSH and Fabric worked and `sensetrace.target`,
  `sshd.service`, and `sensetrace.service` were all active.

## Dedicated-target failure and current boundary

`sensetrace.target` was then made the default and a controlled reboot was
requested.  The controller observed the expected SSH disappearance, but the
worker did not return to `192.168.1.113`: ARP, ICMP, and SSH remained absent;
an ARP/host scan found no replacement address with an SSH service.  A standard
Wake-on-LAN packet was sent to the previously observed wired MAC
`e4:54:e8:68:93:39` without recovery.

Therefore the dedicated-target reboot acceptance is **failed/unproven**.  No
claim is made that worker-03 is currently remotely recoverable or that the
dedicated target is safe as a default boot target.  No watchdog was enabled:
`intel_oc_wdt` (60 s) and `iTCO_wdt` (30 s) were only inventoried, and no
hard-hang recovery was tested.

No Phase 0 or Phase 1A campaign was started after this failure.  The required
measurement-stability baseline and a fresh Phase 0 gate cannot be established
while the worker is unreachable.
