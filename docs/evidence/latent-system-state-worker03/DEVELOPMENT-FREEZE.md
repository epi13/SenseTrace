# Latent system-state confirmation freeze

Protocol: `latent-system-state-v1`  
Authoritative development: `development-r2`  
Configuration hash: `0ed6802e9c276ef51502edf98b574e98cb808a4c6a820895c40ea048b35882b9`  
Acquisition code: `46c8378e7b17e7d5b0dc6cdd5f8bb537d28d27a8`  
Boot: `92cdb521-ac36-4649-a35c-bfb55c6ac870`

The confirmation protocol was frozen after development-r2 and before
confirmation acquisition:

- primary: `read_pressure / cached_preloaded / future_block_mean / horizon 8`
- sampling interval: 100 µs; target block length: 4 repetitions
- 20 independent sessions per family/condition cell; one trajectory per session
- families: `read_pressure`, `active_quiet`, `sham`, `passive`
- target conditions: `cached_preloaded`, `timer_only`
- origin bundle: `cached_preloaded`, `timer_only`, `clflush`, `eviction`
- Tier 1 local witness subset: `tsc_aux`
- PMU witnesses: calling-thread `cache-references` and `cache-misses`, origin-only
- witness scope: origin-only for the low-overhead acquisition
- model ladder: training mean, persistence, current primary, current multichannel,
  current plus workload, primary history, multichannel history, ridge state, and
  rank-bounded PCA delay state
- match tolerance: 8 TSC ticks
- practical observer thresholds: 2× wall latency, 2× timing variance, 20 ticks
  median shift, and 25% thread CPU utilization

No witness subset, tolerance, sampling interval, model class, primary cell, or
confirmation analysis rule was changed after confirmation data were acquired.
The timescale sweep used two sessions per cell and is descriptive development
evidence only; it did not select a confirmation timescale.
