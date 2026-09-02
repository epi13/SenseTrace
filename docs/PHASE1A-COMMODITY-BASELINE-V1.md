# Phase 1A commodity baseline v1

`phase1a-commodity-baseline-v1` is the frozen comparison identity for the
commodity CLFLUSH/timed-load experiment. A run records both this identity and
the SHA-256 `protocol_hash` of its serialized contract in `run.json`,
`campaign.json`, `protocol.json`, and the final report.

The contract freezes the decisions that give the historical observable its
meaning:

- each independently started acquisition session gets a fresh page-aligned
  anonymous allocation, a new allocation/session identity, a host snapshot,
  genuine OS `boot_id`, and an append-only journal;
- labels are balanced in matched pairs; each pair shares a random base word and
  differs only in the declared target bit; pair order is exactly counterbalanced
  per virtual location;
- the operation is an ordinary user-space write/read followed by the selected
  cache control and a native LFENCE/RDTSC/RDTSCP timed observation;
- raw traces are retained; the ordinary digital read, addresses, allocation,
  session, boot, order, and oracle metadata are audit-only;
- A/B/C/D/E holdouts are evaluated independently, with E grouped only by real
  OS boot IDs and unavailable below three boot groups;
- models remain the small configured baseline set, and paired median timing
  statistics, label shuffles, cache controls, and recovery invariants remain
  part of the protocol;
- CLFLUSH means that invalidation was requested before the timed load. It does
  not establish DRAM access or physical row/bank/channel/topology identity.

Changing one of those decisions changes the protocol hash and requires a new
protocol identity or an explicit superseding protocol. Increasing `N` while
leaving the uncertain observable unchanged is not, by itself, scientific
progress. Historical runs and evidence are retained; new code does not rewrite
their records.

