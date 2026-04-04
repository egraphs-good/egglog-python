# `param_eq_hegg` Semantic Review

Source of truth:
- Haskell: [FixTree.hs](/Users/saul/p/param-eq-haskell/src/FixTree.hs)
- Python: [param_eq_hegg.py](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py)
- Golden cases: [param_eq_hegg_haskell_golden.json](/Users/saul/p/egg-smol-python/python/tests/param_eq_hegg_haskell_golden.json)
- Golden generator: [generate_haskell_golden.py](/Users/saul/p/egg-smol-python/python/exp/param_eq_paper/generate_haskell_golden.py)

This review is limited to differences that can change final extracted expressions, final e-graph/class contents, or reported parameter counts in a way that masks semantic differences.

## Stabilized Mental Model

After another Haskell-backed loop, the semantic picture is narrower and more concrete:

1. Local rule translation is mostly aligned on the sampled small cases.
   - `log` / `exp` / `sqrt` guard families match.
   - constant-only canaries such as `2 - 2` and `2 / 2` prune correctly.
   - mixed-class canaries such as `x - x`, `x / x`, and `0 / x0` behave like Haskell.
   - negative-base powers with integral exponents now match Haskell.

2. The strongest remaining open issue is still schedule-sensitive, but the
   narrowed cause changed again after the fresh-rematch engine change.
   - The engine no longer retains concrete residual matches across scheduler iterations.
   - That engine change is real and tested directly in `egg-smol` and `egglog-experimental`.
   - A direct Haskell-shaped inner loop was implemented experimentally as:
     initial analysis, then a rewrite-no-update fixpoint with post-analysis.
   - On the reduced canary `x0_sq_plus_x1_sq`, that direct fixpoint still timed
     out under the real baseline path and was rolled back.
   - By contrast, explicit bounded rounds already reach the Haskell extracted
     form after one round and remain stable through at least 40 rounds; 100
     explicit rounds still finish in about 5.9s, and 200 explicit rounds finally
     time out.
   - So for this explicit-analysis translation, the rewrite-no-update fixpoint
     is much farther away than the useful extracted fixpoint.
   - The Python baseline therefore stays on four bounded
     `analysis_schedule.saturate() + rewrite` rounds plus one final analysis
     saturation.
   - The new smallest Haskell-backed schedule discriminator is
     `sbp_zero_times_quadratic`, and it now matches Haskell under the bounded
     schedule.

3. The remaining open gap is now narrower and no longer points at the engine.
   - `x0_sq_plus_x1_sq`, `sbp_zero_times_quadratic`, and `pagie_sbp_1` now all
     match Haskell under the bounded baseline.
   - `pagie_operon_15` still differs in exact e-class membership from Haskell.
   - The smallest confirmed reason is now the current baseline compensation:
     `add_comm` is disabled, and some left-biased factor-out rewrites become
     unreachable without it.
   - So the remaining mismatch is no longer "Egglog cannot behave like hegg";
     it is "the current no-`add_comm` baseline intentionally gives up some
     Haskell factorization paths to avoid larger blowups."

4. Extraction-shape differences are separate from the main semantic loop.
   - `sub_add_left_assoc` stays in the expected-mismatch bucket as an unstable tie-break case
   - it is not the current blocker for semantic parity

## Mismatch Frontier

| Family | Status | Smallest repro | Current read |
| --- | --- | --- | --- |
| Negative-base integer powers | Fixed | `(-2) ** 2`, `(-2) ** 3` | Python now matches Haskell by allowing negative bases only when the exponent round-trips through `to_i64` / `from_i64`. |
| Small constant pruning | Aligned on canaries | `2 - 2`, `2 / 2`, `0 / x0` | The current `delete(...)` approximation is good enough on the confirmed small cases; it is not the next blocker. |
| Reduced saturate mismatch | Fixed in Python baseline | `sbp_zero_times_quadratic`, `x0_sq_plus_x1_sq` | Four bounded saturated rounds plus one final analysis saturation recover the reduced Haskell schedule canaries without the long one-pass `saturate(...)` behavior. |
| Haskell-shaped inner fixpoint | Rejected and rolled back | `x0_sq_plus_x1_sq` | The direct rewrite-no-update fixpoint still times out on the reduced quadratic canary even though the extracted term stabilizes after one bounded round, so the experiment was removed from the codebase. |
| Remaining exact-form mismatch | Open, now explained by compensation | `(1 / ((x0 * x0) + 1)) + (4 * (x1 * x1))` | Without `add_comm`, the left-biased factor-out rewrites cannot reach factorized forms that Haskell can derive. |
| Corpus mismatch | Open, explained by smaller factorization toy | `pagie_operon_15` | The remaining full-row mismatch is now consistent with the `add_comm` compensation rather than an engine-level schedule bug. |
| Extraction tie-break | Allowed / separate | `sub_add_left_assoc` | Equivalent term, unstable representative choice; do not treat as the main semantic loop. |

## Loop Ledger

### Reduced schedule mismatch

- Status: fixed in the current Python baseline
- Smallest repro: `sbp_zero_times_quadratic`
- Haskell observation:
  - `0.004376 - (0.0 * (x1 * x1))` simplifies to `x1`
  - `x0_sq_plus_x1_sq` normalizes to
    `((-0.00978823600529464 * (x0 * x0)) + (-0.009929236885765901 * (x1 * x1)))`
- Egglog observation:
  - one long
    `scheduler.scope(analysis_rewrite_round.saturate(stop_when_no_updates=True))`
    ran long on these reduced cases after the fresh-rematch engine change
  - the newer direct inner-loop translation
    `initial analysis; then repeat (rewrite once; analysis saturate) until the rewrite step makes no database updates`
    also timed out on `x0_sq_plus_x1_sq` under the real baseline path and has
    now been rolled back from the experimental/Python schedule surface
  - explicit bounded rounds on that same case reach the Haskell extracted form
    after one round, stay unchanged through 40 rounds, still finish in about
    5.9s at 100 rounds, and only time out around 200 rounds
  - four bounded `analysis_schedule.saturate() + rewrite` rounds recover the
    Haskell extracted forms quickly
  - a final `analysis_schedule.saturate()` is needed so `const_value(root)`
    reaches `none` instead of remaining as a nested `join_const_value(...)`
- Current hypothesis:
  - accepted: the remaining reduced mismatch was caused by the long
    `analysis_rewrite_round.saturate(...)` structure itself, not by residual
    matches
  - rejected: a direct Haskell-shaped rewrite-no-update fixpoint would be a
    practical replacement for the bounded Python baseline
- Last attempted fix:
  - replace the bounded baseline with a direct rewrite-no-update inner fixpoint
    plus post-analysis, mirroring the Haskell inner loop more literally
- Observed result:
  - `sbp_zero_times_quadratic` and `pagie_sbp_1` still matched under the direct
    fixpoint, but `x0_sq_plus_x1_sq` timed out
  - the bounded four-round baseline remains the smallest accepted schedule that
    keeps all reduced Haskell canaries green
- Next probe:
  - none; this family is closed
  - do not retry this exact experiment unless a future engine change moves the
    no-update fixpoint materially closer to the useful extracted fixpoint

### Corpus mismatch

- Status: open, but no longer points at the engine
- Smallest repro: `(1 / ((x0 * x0) + 1)) + (4 * (x1 * x1))`
- Haskell observation:
  - with `add_comm`, the left-biased factor-out rules can derive
    `4 * ((x1 * x1) + (0.25 / ((x0 * x0) + 1)))`
- Egglog observation:
  - under the current no-`add_comm` baseline, the toy stays in its unfactored
    form and the factorized form is not in the same e-class
  - under the same bounded schedule with `add_comm` restored, the factorized
    form does enter the e-class
  - a narrower experiment with explicit right-biased mirror rules for
    `b + (a*x)`, `(b*y) + (a*x)`, and `(b/y) + (a*x)` also recovers the
    factorized e-class on smaller toys without restoring global `add_comm`
  - those mirror rules also recover nested toys with two reciprocal terms and
    an outer constant, so the remaining `pagie_operon_15` mismatch is narrower
    than a generic "missing right-biased factorization" explanation
  - the full `pagie_operon_15` row still shows the same tradeoff: restoring
    `add_comm` reopens longer-running behavior, while the no-`add_comm`
    baseline settles on a numerically equivalent but not Haskell-identical form
- Current hypothesis:
  - the remaining exact-form mismatch is caused by the current baseline
    compensation itself: disabling `add_comm` blocks some left-biased
    factorization paths that Haskell can take
- Last attempted fix:
  - restore `add_comm` under bounded schedules
- Observed result:
  - it does make the smaller factorization toy reachable again
  - it does not restore `pagie_operon_15` cleanly without reintroducing longer
    runs, so it is not accepted as the default baseline
- Next probe:
  - if exact Haskell-form parity on `pagie_operon_15` becomes a priority, the
    next step is to reduce the remaining miss to the first failing toy beyond
    the already-working right-biased mirror-rule cases

### Extraction tie-break

- Status: allowed / separate
- Smallest repro: `sub_add_left_assoc`
- Haskell observation:
  - keeps `x0 - (x0 + x1)`
- Egglog observation:
  - may choose an equivalent subtraction reassociation depending on surrounding state
- Current hypothesis:
  - this is extraction tie-breaking rather than an analysis/class-state mismatch
- Last attempted fix:
  - none in the semantic loop
- Next probe:
  - ignore unless a future class-state inspection shows a deeper semantic difference

## Findings

### 1. Negative-base power folding is fixed now and should stay fixed

- Haskell: [FixTree.hs:232](/Users/saul/p/param-eq-haskell/src/FixTree.hs#L232)
- Python: [param_eq_hegg.py:372](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py#L372)
- Repro:
  - `(-2) ** 2`
  - `(-2) ** 3`
  - `(-2) ** x0`
- Effect:
  - final extraction
  - final class contents

Haskell uses `PowerF e1 e2 -> liftA2 (**) e1 e2`. The Python translation now matches the sampled real-valued cases by splitting negative bases into:

- integral constant exponent:
  - fold to `some(a**b)`
- non-integral or non-constant exponent:
  - keep `none`

This is now enforced by the golden fixture and focused regression tests, so it is no longer an open semantic mismatch.

### 2. The current open gap is a raw-product schedule/rebuild mismatch, not a local algebra mismatch

- Haskell:
  - [FixTree.hs:204](/Users/saul/p/param-eq-haskell/src/FixTree.hs#L204)
  - [FixTree.hs:213](/Users/saul/p/param-eq-haskell/src/FixTree.hs#L213)
  - [FixTree.hs:368](/Users/saul/p/param-eq-haskell/src/FixTree.hs#L368)
- Python:
  - [param_eq_hegg.py:319](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py#L319)
  - [param_eq_hegg.py:444](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py#L444)
  - [param_eq_hegg.py:1180](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py#L1180)
- Repro:
  - `pagie_operon_15` in [param_eq_hegg_haskell_golden.json](/Users/saul/p/egg-smol-python/python/tests/param_eq_hegg_haskell_golden.json)
  - reduced full-structure toy:
    `((((2.0 * x0) * (3.0 * x0)) + (1.0 / (((4.0 * x0) * (5.0 * x0)) + 1.0))) + ((6.0 * x1) * (7.0 * x1)))`
  - negative controls:
    - `(((-46.3591499328613281) / (((13.0968494415283203 * x0) * (2.1884925365447998 * x0)) + 34.0089225769042969)) + ((-1351.7908935546875000) / (((17.2840843200683594 * x1) * (47.4518127441406250 * x1)) + 990.7814331054687500)))`
    - `(((2.0 * (x0 * x0)) + (1.0 / ((x0 * x0) + 1.0))) + (3.0 * (x1 * x1)))`
- Effect:
  - final extraction
  - final class contents

The important update now is:

- the residual-match explanation below was real, and the engine now fixes it
- after that engine fix, the remaining reduced mismatch moved back to the long
  `saturate` path itself
- `x0_sq_plus_x1_sq` and the raw-product toy both finish quickly under bounded
  explicit rounds, but one long scheduler-scoped `saturate` still runs much
  longer

That makes the current hypothesis more specific than before:

- the remaining mismatch is not explained by local algebra alone
- it is no longer explained by stale residual tuples either
- it now points at the semantics of one long
  `analysis_rewrite_round.saturate(...)` path relative to bounded explicit
  rounds

### 2a. Engine fix landed: custom schedulers now rematch fresh like `egg`

- Haskell:
  - [Saturation.hs:121-159](/Users/saul/p/param-eq-haskell/dist-newstyle/src/hegg-3b7fc7710d63b98862dea8a6a0207f14aac529a277de5c84edc77b4e17374465/src/Data/Equality/Saturation.hs#L121)
  - [Scheduler.hs:23-83](/Users/saul/p/param-eq-haskell/dist-newstyle/src/hegg-3b7fc7710d63b98862dea8a6a0207f14aac529a277de5c84edc77b4e17374465/src/Data/Equality/Saturation/Scheduler.hs#L23)
- Egglog:
  - [scheduler.rs:20-31](/Users/saul/p/egg-smol/src/scheduler.rs#L20)
  - [scheduler.rs:98-134](/Users/saul/p/egg-smol/src/scheduler.rs#L98)
  - [scheduler.rs:210-253](/Users/saul/p/egg-smol/src/scheduler.rs#L210)
  - [scheduling.rs:357-403](/Users/saul/p/egglog-experimental/src/scheduling.rs#L357)
- Repro:
  - direct engine regressions:
    - [scheduler.rs](/Users/saul/p/egg-smol/src/scheduler.rs) `test_scheduler_rematches_fresh_after_skipped_iteration`
    - [integration_test.rs](/Users/saul/p/egglog-experimental/tests/integration_test.rs) `test_backoff_rematches_fresh_after_ban_expires`
- Effect:
  - final e-graph state
  - runtime / exploration behavior

The most specific implementation difference previously supported by both code
and runs was:

- hegg recomputes the current match set from the rebuilt e-graph every iteration
- egglog custom schedulers retained a backlog of concrete matches across iterations

In hegg, one iteration builds a fresh matching database from the current graph,
runs `ematch`, updates per-rule ban stats, applies the chosen matches, rebuilds,
and then starts over from the rebuilt graph. The scheduler state in
[Scheduler.hs](/Users/saul/p/param-eq-haskell/dist-newstyle/src/hegg-3b7fc7710d63b98862dea8a6a0207f14aac529a277de5c84edc77b4e17374465/src/Data/Equality/Saturation/Scheduler.hs)
stores only per-rule statistics.

That engine difference is now fixed:

- `Scheduler::filter_matches` now only chooses within the current iteration's
  fresh matches
- custom schedulers can skip future searches via `should_search`, but unchosen
  matches are discarded after each iteration
- scheduler query rules are now non-seminaive, so each iteration really
  rematches the rebuilt graph instead of only replaying deltas

The engine-level evidence is direct:

- the core scheduler regression now records copy-rule match sizes `[1, 2]`,
  proving the second scheduled step sees the rebuilt graph rather than only the
  prior skipped delta
- the experimental backoff regression now bans a rule, adds a new row while the
  ban is active, and then copies all three rows after unbanning, proving it
  rematches fresh rather than replaying the old two-row tuple set

So the residual-match explanation should no longer be treated as the current
open blocker. The next open blocker is narrower:

- **one long `analysis_rewrite_round.saturate(...)` still behaves differently
  from bounded explicit rounds, even after fresh rematching is fixed**

The smallest current evidence for that is:

- `x0_sq_plus_x1_sq`:
  - long under one long `saturate`
  - fast and correct under two explicit rounds
- `raw_product_toy`:
  - same pattern
- `pagie_sbp_1`:
  - still rejects the naive “just use two explicit rounds” replacement by
    regressing to `0.004376`

### 3. A naive schedule replacement fixes the reduced case but regresses another canary

- Python schedule site: [param_eq_hegg.py:1180](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py#L1180)
- Repro:
  - fixed by the speculative schedule: `x0_sq_plus_x1_sq`
  - regressed by the same schedule: `pagie_sbp_1`
- Effect:
  - final extraction

A tested but rejected hypothesis was:

- replace the current saturated `analysis + rewrite` pair with two explicit `analysis_saturate + rewrite_schedule` rounds inside one `_run_single_pass`

That schedule did normalize the reduced quadratic-sum case quickly to the Haskell form. But it also regressed `pagie_sbp_1` from the correct current-Haskell result `x1` back to `0.004376`.

So the next schedule change must be more selective than “use two explicit rounds everywhere”.

### 3a. Historical: directly folding the regrouping RHS was not the missing piece

- Python: [param_eq_hegg.py](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py)
- Repro:
  - attempted fix target: `x0_sq_plus_x1_sq`
  - regression canary: `pagie_sbp_1`
- Effect:
  - final extraction
  - runtime / exploration behavior

A narrower hypothesis was that the regrouping rule

- `((a * x) * (b * y)) -> ((a * b) * (x * y))`

was too dependent on a later analysis pass to fold `a * b`, and that emitting
`Num(ca * cb)` directly would approximate Haskell rebuild better.

That hypothesis is now rejected. The isolated product term already normalizes
under the real baseline schedule, and changing the regrouping rule to emit a
folded RHS constant made both the reduced mismatch and `pagie_sbp_1` run away
much harder. So the missing behavior is not “just fold regrouped constants
earlier”.

### 3b. Historical: removing `add_comm` was the first accepted schedule fix

- Python: [param_eq_hegg.py](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py)
- Repro:
  - fixed reduced canary: `x0_sq_plus_x1_sq`
  - preserved canary: `pagie_sbp_1`
  - improved larger canary: `pagie_operon_15` no longer times out, though it still does not exactly match Haskell
- Effect:
  - final extraction
  - runtime / exploration behavior

Ablation against the reduced quadratic-sum case produced:

- baseline: timeout
- `single_pair`: too weak
- `two_rounds`: fixes the case but regresses `pagie_sbp_1`
- `baseline_without_add_comm`: fixes the case
- `baseline_without_mul_comm`: also fixes the case
- `baseline_without_mul_assoc`: also fixes the case
- `baseline_without_add_assoc`: still times out
- `baseline_without_product_regroup`: still times out

At that stage, both `baseline_without_add_comm` and
`baseline_without_mul_comm` preserved the current comparable golden set. That
historical conclusion was overtaken by the later fresh-rematch engine change,
which made the long `saturate` path the active blocker again.

Fresh rerun evidence for the same reduced case makes the search blowup visible:

- current baseline (`baseline_without_add_comm`) finished in about `0.066s`
  with `367` nodes, `103` e-classes, and the expected normalized quadratic sum
- the old `baseline_with_add_comm` variant was still running after `14s` at
  roughly `99%` CPU and `233 MB` RSS, so it was killed rather than left
  wandering further

The same reduced-case runs also show that the problem is not “product regrouping
alone fails”. The isolated product term already normalizes, and a single
`analysis + rewrite` pair is too weak while two explicit rounds are too strong.
That leaves the sum-level search space around the raw quadratic sum as the best
supported explanation for why `add_comm` is the compensating fix in this
translation.

### 3c. Historical: `pagie_operon_15` temporarily looked like an extraction-form difference

- Python: [param_eq_hegg.py](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py)
- Repro:
  - `pagie_operon_15`
- Effect:
  - extracted form

Under the earlier baseline, the full Operon row no longer timed out. The
extracted Egglog form was still not textually equal to the Haskell form, but
sampled evaluation agreed to floating-point precision on multiple points. That
historical read was overtaken by the later fresh-rematch engine change, which
made the long `saturate` path the active blocker again.

- semantic/e-graph parity is good enough on this canary
- extraction/cost tie-breaking still differs

### 4. Constant-class pruning is not the strongest active mismatch hypothesis anymore

- Haskell: [FixTree.hs:213](/Users/saul/p/param-eq-haskell/src/FixTree.hs#L213)
- Python: [param_eq_hegg.py:444](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py#L444)
- Repro:
  - aligned:
    - `2 - 2`
    - `2 / 2`
    - `0 / x0`
- Effect:
  - final class contents

The current `delete(...)`-based `modifyA` approximation is still not a literal port of hegg’s class-level node filtering. But on the small canaries we care about, it now behaves the right way:

- truly constant classes prune to leaf-only representatives
- mixed classes like `0 / x0` keep their non-leaf representative because the class analysis is still `none`

So pruning remains an approximation, but it is not the best current explanation for the remaining open mismatches.

### 5. The Python language is still a paper-subset language, not a full `SRTreeF` port

- Haskell constructors: [FixTree.hs:44](/Users/saul/p/param-eq-haskell/src/FixTree.hs#L44)
- Python language: [param_eq_hegg.py:79](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py#L79)
- Repro:
  - any future `logBase`, trig, hyperbolic, or pre-materialized `Param` case
- Effect:
  - final extraction when those constructors appear

`param_eq_hegg.py` intentionally models only the subset exercised by the current paper notebook. That means the current file is easier to compare against the active corpus, but it is still not a full `FixTree.hs` port:

- no separate `ParamF`
- no `LogBaseF`
- no broader `FunF Function a` family
- no surviving `PowF`

For the current notebook this is acceptable because the archived rows being compared only use the subset already modeled. It is still a semantic gap if the scope expands.

### 6. The current Haskell code and the archived paper CSV are not interchangeable truths

- Current Haskell code: [FixTree.hs](/Users/saul/p/param-eq-haskell/src/FixTree.hs)
- Archived row artifact: [haskell_paper_rows.csv](/Users/saul/p/egg-smol-python/python/exp/param_eq_paper/artifacts/haskell_paper_rows.csv)
- Repro:
  - `pagie_sbp_1` in [param_eq_hegg_haskell_golden.json](/Users/saul/p/egg-smol-python/python/tests/param_eq_hegg_haskell_golden.json)
- Effect:
  - review methodology
  - not a Python semantic bug by itself

The golden fixture is generated from the current `FixTree.hs`, not from the archived paper CSV. `pagie_sbp_1` is a good reminder that “paper row artifact” and “current Haskell source” are separate truth sources and should not be mixed casually.

For this review pass, the source of truth is the current `FixTree.hs`, as requested.

## Likely Non-Issues

### Reporting semantics now line up with the paper harness

- Haskell:
  - [Main.hs:25](/Users/saul/p/param-eq-haskell/src/Main.hs#L25)
  - [FixTree.hs:111](/Users/saul/p/param-eq-haskell/src/FixTree.hs#L111)
- Python:
  - [param_eq_hegg.py:939](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py#L939)

`count_params` now mirrors `recountParams . replaceConstsWithParams`, including the important special case where literal exponents are excluded from parameter counting but non-constant exponent subtrees are still traversed.

### Method costs are intentionally aligned with `cost2`, not `cost`

- Haskell:
  - [FixTree.hs:248](/Users/saul/p/param-eq-haskell/src/FixTree.hs#L248)
  - [FixTree.hs:260](/Users/saul/p/param-eq-haskell/src/FixTree.hs#L260)
- Python:
  - [param_eq_hegg.py:101](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py#L101)

`rewriteTree` uses `cost2`, not `cost`. The current Python constructor costs are therefore closer to the right Haskell extraction model than the more elaborate `cost` function lower in `FixTree.hs`.

### Mixed classes are intentionally preserved

- Haskell:
  - [FixTree.hs:204](/Users/saul/p/param-eq-haskell/src/FixTree.hs#L204)
  - [FixTree.hs:213](/Users/saul/p/param-eq-haskell/src/FixTree.hs#L213)
- Python:
  - [param_eq_hegg.py:291](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py#L291)

The Python file already documents and tests the key Haskell invariant: a class may contain both a constant representative and a non-constant representative while its analysis remains `none`.

## Clarity Refactor Proposal

These are readability changes, not semantic fixes.

1. Reorder [param_eq_hegg.py](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py) to match the Haskell file more literally:
   - language / surface syntax
   - analysis domain and merge
   - analysis algebra
   - Haskell-named guard helpers
   - `_basic_rewrites`
   - `_fun_rewrites`
   - schedule / outer passes
   - reporting helpers

2. Add thin Haskell-named guard helpers and use them directly in the rewrites:
   - `is_const(x, a)`
   - `is_not_const(x)`
   - `is_negative(x, a)`
   - `is_not_zero(x, a)`
   - `is_not_neg_consts(x, y, a, b)`

3. Keep reporting helpers visually separated from EqSat logic:
   - [render_num](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py#L846)
   - [_eval_num](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py#L904)
   - [count_params](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py#L984)
   - [count_nodes](/Users/saul/p/egg-smol-python/python/egglog/exp/param_eq_hegg.py#L1045)

4. Keep the mapping table at the top of the module and extend it if new sections are added:
   - `joinA -> join_const_value`
   - `evalConstant -> const_seed_rules | const_propagation_rules`
   - `modifyA -> const_prune_rules`
   - `rewritesBasic -> _basic_rewrites`
   - `rewritesFun -> _fun_rewrites`
   - `rewriteTree -> _run_single_pass`
   - `simplifyE -> run_paper_pipeline`

## Fix Priority

Must fix for paper-faithful extraction:
- the remaining long-saturation mismatch around
  `analysis_rewrite_round.saturate(...)`
- any broader corpus gaps that remain after that reduced schedule difference is
  explained

Matters mainly for final e-graph structure:
- closer `modifyA` pruning semantics than the current `delete(...)` approximation, if a new concrete counterexample shows up

Matters mainly for extracted surface form:
- subtraction reassociation tie-breaks where Python and Haskell choose different but equivalent representatives

Reporting-only or currently aligned:
- parameter counting projection
- use of `cost2`-style extraction costs
