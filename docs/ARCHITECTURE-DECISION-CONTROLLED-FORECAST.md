# Architecture decision: controlled predictive-state campaign

Status: implemented on `worker-03`; protocol `controlled-predictive-state-v1`.

The campaign reuses the repository's validated configuration loader, Fabric
`RemoteHost` discovery/deployment path, `ControlledMemoryBuffer`, existing
PRBS/Walsh/active-quiet schedule contracts, raw artifact hashing, and grouped
session reporting. The historical passive `real_horizon` and construction-null
pipelines remain unchanged and retain their original claim boundaries.

The versioned extension is deliberately separate. `native/measurement_kernel.c`
keeps the historical entry points and adds the v5 trajectory ABI: a native
bounded loop for cached/preloaded, CLFLUSH, eviction, and timer-only controls;
retained raw endpoint TSC/TSC_AUX values; counterbalanced paired acquisition
order; and a native bounded memory-pressure exciter. Python allocates the
buffers and records metadata outside the loop. `controlled_forecast.py` owns
the acquisition journal, requested-versus-actual excitation records, causal
feature firewall, held-out model ladder, exact streaming replay, and synthetic
null/positive controls.

The model ladder is intentionally small: training mean, persistence, current
observation plus position, workload-only, current observation plus workload
history, ARX observation/workload history, and a rank-bounded delay-embedded
PCA/state regression. Development sessions select the strongest eligible
baseline and history candidate; confirmation is fitted only from development
data. No trajectory identifiers, seeds, future schedule, future witness state,
or whole-trajectory normalization enters model features.

The measured durations are raw TSC ticks. v5 includes the RDTSCP endpoint
overhead and records endpoint AUX values; it does not claim that TSC ticks are
core cycles. A separate monotonic/TSC calibration converts elapsed-time lead
for reporting. Paired probes are sequential and may perturb one another; the
reference spacing establishes distinct observed cache lines only, not DRAM
banks, rows, channels, or physical addresses.
