# Param-Eq Replication Status

This file is the hypothesis and experiment ledger for the retained `param_eq`
baseline.

It is intentionally not the place for corpus-level plots or current summary
numbers. For those, use the comparison notebook:

- [replication.py](replication.py)
- [replication.ipynb](replication.ipynb)

In particular, any statement here about whether Egglog is or is not meeting the
archived Haskell baseline should be read together with the notebook sections:

- `## 4. Haskell vs Egglog Comparison`
- `## 5. Current Limitations and Likely A/C Effects`

## Source Of Truth

- Haskell implementation being mirrored:
  - `../param-eq-haskell/src/FixTree.hs`
  - `../param-eq-haskell/src/Main.hs`
- Egglog translation:
  - [pipeline.py](pipeline.py)
- Haskell-backed reduced canaries:
  - [haskell_golden.json](haskell_golden.json)
  - [generate_haskell_golden.py](generate_haskell_golden.py)
- Corpus-level artifact generation:
  - [normalize_archives.py](normalize_archives.py)
  - [run_egglog_corpus.py](run_egglog_corpus.py)

## Current Overall Read

The current baseline is a partial reproduction, not full parity.

That claim is supported by the corpus-level notebook results in
[replication.py](replication.py):

- Egglog preserves some of the archived Haskell behavior on many rows and often
  lands near the same parameter-count target.
- Egglog does not yet preserve all of the qualitative separation between the
  paper pipeline stages that the archived Haskell plots show.
- The remaining misses are concentrated in a narrower hard tail, not spread
  uniformly across the whole corpus.

So the right current claim is:

- Egglog is close enough to act as a serious pre-multiset baseline.
- Egglog is not yet faithful enough to claim the Haskell baseline is fully
  reproduced.

## Accepted Baseline

The accepted Python baseline in [pipeline.py](pipeline.py) is:

- four bounded rounds of `analysis_schedule.saturate() + rewrite_schedule`
- one final `analysis_schedule.saturate()`

This is an accepted approximation, not a literal copy of Haskell's internal
loop structure.

Why it stays accepted:

- it keeps the reduced Haskell canaries green
- it avoids the longer-running behavior reopened by the more literal
  alternatives
- the corpus-level consequences of that choice are already visible in the
  notebook results in [replication.py](replication.py)

## Mismatch Frontier

| Family | Status | Smallest repro | Current read |
| --- | --- | --- | --- |
| Negative-base integer powers | Fixed | `(-2) ** 2`, `(-2) ** 3` | The Python translation now matches the sampled Haskell cases by only folding negative bases when the exponent round-trips through `to_i64` / `from_i64`. |
| Small constant pruning | Aligned on canaries | `2 - 2`, `2 / 2`, `0 / x0` | The current `delete(...)` approximation is good enough on the confirmed small cases; it is not the next blocker. |
| Reduced schedule mismatch | Fixed in the retained baseline | `sbp_zero_times_quadratic`, `x0_sq_plus_x1_sq`, `pagie_sbp_1` | The bounded schedule recovers the reduced Haskell canaries without reopening the longer one-pass `saturate(...)` behavior. |
| Haskell-shaped inner fixpoint | Rejected | `x0_sq_plus_x1_sq` | A more literal rewrite-no-update inner fixpoint still timed out on the reduced quadratic canary and is no longer in the codebase. |
| Remaining exact-form mismatch | Open | `(1 / ((x0 * x0) + 1)) + (4 * (x1 * x1))` | The current baseline disables `add_comm`, which blocks some left-biased factorization paths that Haskell can take. |
| Corpus mismatch | Open | `pagie_operon_15` | The remaining full-row mismatch is currently best explained by the no-`add_comm` compensation, not by a generic engine-level scheduler bug. |
| Extraction tie-break | Allowed / separate | `sub_add_left_assoc` | Equivalent term, unstable representative choice; not the main semantic blocker. |

## Closed Findings

### Negative-base power folding

This is no longer an open issue.

Relevant source sites:

- Haskell: `FixTree.hs`, power constant evaluation
- Egglog: [pipeline.py](pipeline.py), negative-base exponent handling

The retained behavior is:

- integral constant exponent: fold
- non-integral or non-constant exponent: do not fold

That is enforced by the reduced Haskell-backed tests in
[test_pipeline.py](test_pipeline.py).

### Residual scheduler-match replay

This is no longer the live explanation for the remaining `param_eq` mismatch.

Historical conclusion:

- the old residual-match explanation was real at the engine level
- the supporting engine fix landed in `egg-smol` / `egglog-experimental`
- after that, the remaining `param_eq` mismatch still persisted

So this should not be treated as the current blocker for parity.

## Rejected Experiments

### Direct Haskell-shaped inner fixpoint

Tried:

- initial analysis
- then repeat `(rewrite once; analysis saturate)` until the rewrite step makes
  no database updates

Why it was rejected:

- it still timed out on `x0_sq_plus_x1_sq`
- the bounded schedule already reaches the useful extracted form for the reduced
  canaries much earlier

Do not retry this exact experiment unless an engine change materially changes
the distance between the useful extracted fixpoint and the rewrite-no-update
fixpoint.

### Restoring `add_comm` globally

Tried:

- restore `add_comm` so the blocked factorization paths become reachable again

Why it was rejected:

- it does recover the smaller factorization toy
- it also reopens longer-running behavior on the broader baseline
- so it is not acceptable as the retained default schedule

## Current Live Hypothesis

The remaining exact-form gap is mainly caused by the accepted baseline
compensation itself:

- disabling `add_comm` avoids larger blowups
- but it also prevents some left-biased factorization paths that the archived
  Haskell implementation can take

This is why the current baseline can still be:

- close on many corpus rows
- clearly below full parity on exact-form reproduction

That qualitative conclusion is visible in the notebook-backed comparison
results in [replication.py](replication.py), especially the exact-match scatter
and the Egglog-vs-Haskell reduction-ratio sections.

## Next Probe

If exact parity becomes the priority again, the next step is:

1. reduce `pagie_operon_15` to the first failing toy beyond the already-working
   right-biased mirror-rule cases
2. decide whether that miss should be addressed by:
   - a narrower factorization-path recovery
   - a different `add_comm` compensation
   - or a deliberate decision to keep the current approximation and move on to
     multiset work

That decision should be made after consulting the notebook results, not from
this ledger alone.
