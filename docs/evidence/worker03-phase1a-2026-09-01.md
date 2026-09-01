# worker-03 Phase 1A campaign — 2026-09-01

This record captures the explicitly gated safe commodity-memory campaign after
the separate Phase 0 v2 final gate passed. The raw run remains on worker-03 at
`/home/worker-03/.local/share/sensetrace/runs/phase1a-20260901T170018Z-fe4463bc`.
The controller fetched the report and ledgers into the ignored local
`runs/worker03-phase1a-20260901T170018Z-fe4463bc/` directory.

## Campaign and provenance

- Run: `phase1a-20260901T170018Z-fe4463bc`
- Campaign: `campaign-phase1a-20260901T170018Z-fe4463bc`
- Status: `completed` (`2026-09-01T17:00:18Z` to `17:50:21Z` UTC)
- Deployed code commit: `0801ffb9d7214a4a344807b1537eedbe47580a8a`
- Run configuration hash: `2758a638b2aa2f8d201d6a5117a093732c97e57cb3810fdf47de7fbf1e961613`
- Host: worker-03, boot ID
  `697e8ad6-ef00-4d2f-a9ae-c944867c9d46`, kernel
  `6.17.1-300.fc43.x86_64`
- Sessions: four independently started source sessions per acquired
  condition; each session recorded its UUID, start time, host snapshot, boot
  ID, fresh anonymous buffer allocation, label fingerprint, journal, cache and
  timing-kernel provenance, and hashes.
- Acquired conditions: `paired_single_bit`, `cache_hit_control`,
  `idle_no_memory_operation`, `all_zero_vs_all_one`, and `random_word_null`.
  Each combined dataset contains 32,768 balanced rows and preserves its four
  source session manifests. `paired_single_bit_shuffled` is the derived label
  permutation control and also contains 32,768 rows.
- Recovery integrity: campaign completion was recorded; zero temporary shards
  remained; the remote system service stayed singular and active.

## Split and control coverage

Every condition, including the shuffled control, had all five declared split
levels available and independently evaluated:

`A_repeated_trial_holdout`, `B_unseen_location`,
`C_unseen_acquisition_block`, `D_unseen_acquisition_session`, and
`E_unseen_boot_session`.

Pair-order auditing was exact: 16,384 `label_0_first` and 16,384
`label_1_first` rows globally, with 8,192 of each label at each pair
position. The audit was true for every condition. Acquisition-order, CPU,
frequency/governor, thermal, cache, session, boot, and block diagnostics were
reported as audit-only channels.

On the primary B split of the paired condition, the exploratory model summary
was:

| Model | AUROC mean | Balanced accuracy mean |
| --- | ---: | ---: |
| logistic regression | 0.4959 | 0.4996 |
| boosted trees | 0.4912 | 0.4924 |

The preregistered paired diagnostic on that test partition had 2,464 complete
pairs. The sample-median latency delta (label 1 minus label 0) had mean
`0.3155`, median `0.0000`, two-sided pair-level sign-flip `p=0.4118`, and a
95% percentile interval `[-0.1624, 0.8629]` resampled over complete
`acquisition_session_id × acquisition_block` clusters. Secondary diagnostics
remain exploratory and are not used as claims.

The shuffled-label and random-word controls stayed near chance on B: shuffled
AUROC means were 0.4950 (boosted trees) and 0.4926 (logistic regression), and
random-word AUROC means were 0.4916 and 0.5022, respectively. These values are
descriptive campaign results, not confirmatory inference.

## Cache-control and claim boundary

The flushed path was explicitly recorded as `_mm_clflush(address)` followed by
`_mm_mfence()` before the timed load, with LFENCE/RDTSC start and
RDTSCP/LFENCE end timing fences. This requests cache-line invalidation when
native CPU support is reported; it does not prove a DRAM access or expose a
physical address, row, bank, subarray, chip, or DIMM identity.

This campaign supports only exploratory characterization of safe commodity
host observables under the recorded virtual-buffer/session protocol. It does
not establish physical DRAM-state inference, physical topology, or
device-independent generalization.
