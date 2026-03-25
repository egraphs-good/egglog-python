# `srtree` Witness Notes

This folder records the commands and conclusions from the `srtree` A/C witness
scan and the matching Egglog container translation work.

The source helpers live in the sibling repo at:

- `/Users/saul/p/srtree/examples/find_eqsat_cap_witness.hs`
- `/Users/saul/p/srtree/examples/egglog_ac_multiset_from_srtree.py`

## Saved Commands

- `./run_witness_scan.sh`
  Scans the saved `srtree` result corpora with the Haskell helper and ranks
  candidate witnesses by:
  1. `full_hit_cap=true` first.
  2. Otherwise largest A/C growth fallback.

- `./run_row18_translation.sh`
  Runs the Egglog translation on the strongest fallback witness from the scan:
  row 18 of `results/ablation/1199_pareto_random`. This row is heavy, so the
  saved command uses `--iters 5`.

- `./run_row7_translation.sh`
  Runs the Egglog translation on a smaller real witness from the same corpus:
  row 7 of `results/ablation/1199_pareto_random`. This is the stable default
  smoke test and works with `--iters 50`.

## Environment

The runs were validated with:

- `ghc 9.10.1`
- `nlopt 2.10.1`
- `pkg-config` available
- `cabal build lib:srtree` succeeding with Homebrew `nlopt`

## Conclusions

- Replaying the saved corpus expressions one-at-a-time in fresh `srtree`
  e-graphs did **not** produce a `full_hit_cap=true` example at `--max-iters 50`.
- The strongest fallback witness at that budget is row 18 of
  `/Users/saul/p/srtree/results/ablation/1199_pareto_random`.
  Its Haskell replay metrics were:
  - `full` nodes `28 -> 459`
  - `full` e-classes `28 -> 89`
  - `A/C` nodes `28 -> 633`
  - `A/C` e-classes `28 -> 95`
- Even after increasing that row to `--max-iters 200`, it still stayed below the
  repo’s hard `1500` e-class stop.
- On the Egglog side, the same row 18 witness clearly shows the container win:
  - binary A/C total size `27 -> 4016`
  - multiset total size `27 -> 68`
  - this was verified with `--iters 5`
- For a practical default run, row 7 from the same corpus is much lighter and
  still demonstrates the same effect:
  - binary A/C total size `18 -> 200`
  - multiset total size `18 -> 34`
  - this was verified with `--iters 50`

## Notes

- The Haskell helper is measuring saved single-expression replays, not replaying
  the original GP population state.
- The Egglog script currently containerizes additive structure only. That is
  enough to show the A/C blow-up reduction on these witnesses.
