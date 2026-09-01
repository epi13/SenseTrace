# Native timing calibration — 2026-09-01

## Observed fact

The native library was built with `make -C native` and exercised for 200
repetitions on the controller's x86 host.

- Kernel: `sensetrace-native-kernel-v1`.
- Timer: serialized LFENCE/RDTSC start and RDTSCP/LFENCE end.
- Flush control: CLFLUSH followed by MFENCE.
- CLFLUSH support: present on the test CPU.
- Timer-only median: 24 cycles; mean 23.805; standard deviation 0.761.
- Cached-load median: 22 cycles; mean 22.865; standard deviation 1.146.
- CLFLUSH-load median: 188 cycles; mean 596.085; standard deviation 5470.550;
  95th percentile 264 cycles and 99th percentile 1211.85 cycles.
- Idle-control median: 24 cycles; mean 23.280; standard deviation 1.085.

## Interpretation

The native path has materially finer resolution and a visibly separated cached
versus flushed distribution on this host. The flushed path has a heavy tail and
must be reported with percentiles rather than only a mean. The Python timing
path remains a regression/control backend.

## Unresolved confounds

These measurements do not identify the cache level or prove that a flushed load
reached DRAM. CPU frequency, interrupts, thermal state, and operating-system
scheduling can still affect the cycle distribution. A worker-local calibration
under its selected acquisition profile is required before a physical campaign.

## Claim boundary

This is a timing/cache-path separation result only, not a DRAM-state or DRAM-row
measurement result.
