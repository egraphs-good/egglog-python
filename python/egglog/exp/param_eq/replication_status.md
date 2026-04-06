# Param-Eq Replication Status

This file is the hypothesis and experiment ledger for the retained `param_eq`
baseline.

It is intentionally not the place for corpus-level plots or current summary
numbers. For those, use the comparison notebook:

- [replication.py](replication.py)
- [replication.ipynb](replication.ipynb)

In particular, any statement here about whether Egglog is or is not meeting the
archived Haskell baseline should be read together with the notebook sections:

- `## 5. Current Local Haskell vs Egglog Comparison`
- `## 6. Scheduler Ablation`
- `## 7. Current Limitations`

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

The current worktree baseline is a partial reproduction, not full parity.

That claim is supported by the corpus-level notebook results in
[replication.py](replication.py):

- the current direct Haskell-shaped baseline now saturates all retained rows on
  both original and Sympy inputs
- against the **current local Haskell** artifact, Egglog now matches final
  parameter counts on:
  - `318 / 341` original rows
  - `336 / 341` Sympy rows
- the remaining live Egglog-vs-Haskell gaps are small:
  - largest original-input gap: `2`
  - largest Sympy-input gap: `1`
- the larger apparent Sympy gap in the older notebook view was mostly
  published-archive drift, not a current Egglog-vs-Haskell mismatch

So the right current claim is:

- Egglog is now very close to the current local Haskell baseline on the paper's
  parameter-count metric.
- Egglog is still not at full parity, but the remaining live mismatches are now
  a small tail rather than a broad failure mode.
- The scheduler-ablation results now suggest the retained scheduler stack is
  still necessary as a whole if we want to preserve the current all-row
  saturation and notebook-level conclusions.

One important refinement to that statement:

- some of the remaining notebook mismatches are now clearly **paper-archive
  drift**, not current Egglog-vs-Haskell drift
- for example, `kotanchek:EPLEX#4` is still an archived-paper mismatch in the
  notebook, but the live Haskell `simplifyE` result now reaches the same
  8-parameter form as Egglog
- the next probes should therefore prioritize rows where Egglog still differs
  from the **current local Haskell implementation**, not only from the archived
  CSVs
- the live Haskell artifact itself currently includes `2` archived fallback
  rows because current local Haskell still overflows its stack budget on:
  - `pagie SRjl#18`
  - `kotanchek Bingo#21`

## Accepted Baseline

The accepted Python baseline in [pipeline.py](pipeline.py) is:

- outer pass cap `2`
- inner rewrite cap `30`
- full Haskell rewrite set `rewritesBasic <> rewritesFun`
- one bound fresh-rematch scheduler reused across all inner iterations of one
  pass
- one saturated analysis round after each rewrite round
- stop on whole-pass e-graph size stability, which is the closest current
  analogue to Haskell's rebuild-size check

This is now the retained default, not only an experiment.

Why it stays accepted:

- it is the closest high-level reproduction of Haskell we can currently express
  in Python on top of Egglog
- it fixes the earlier `pagie_operon_15` schedule-boundary miss on the paper
  metric
- the ablation pass did not produce a simpler stack that preserved both the
  live-Haskell medians and the all-row saturation story
- the refreshed notebook results in [replication.py](replication.py) are based
  on this baseline

There is now also a local full-state trace harness:

- [trace_egglog.py](trace_egglog.py)
- [trace_haskell.py](trace_haskell.py)
- [trace_tables.py](trace_tables.py)
- [compare_trace.py](compare_trace.py)

Those tools write ignored per-step JSON snapshots under `trace/<case>/` and
are the source of truth for the state-level trace claims below.

## Latest Probe

Question:

- after switching the notebook to the live-Haskell baseline, can any of the
  scheduler customizations be removed without changing the notebook-level
  conclusions?

Probe commands:

- use `pagie Operon#15` original plus the remaining `pagie GOMEA#17/#18/#28`
  live mismatches as the first targeted screen
- reject an ablation immediately if it times out or clearly regresses those
  rows
- only run broader aggregate checks for the ablations that survive the targeted
  screen
- summarize each ablation against the live-Haskell acceptance bar:
  - exact-match counts
  - saturation totals
  - max gaps
  - overall medians
  - per-algorithm median pattern

Observed result:

- retained baseline:
  - `318 / 341` original exact matches
  - `336 / 341` Sympy exact matches
  - `341 / 341` saturation on both inputs
- `no-haskell-backoff`:
  - rejected immediately
  - timed out on `pagie Operon#15` original while the retained baseline still
    saturated that row in about `11.7s`
- `no-bound-scheduler`:
  - kept the same medians and per-algorithm pattern
  - `315 / 341` original exact matches
  - `336 / 341` Sympy exact matches
  - lost full original-input saturation by timing out on:
    - `pagie EPLEX#25`
    - `pagie SRjl#1`
- `no-fresh-rematch`:
  - kept the same medians and per-algorithm pattern
  - `317 / 341` original exact matches
  - `336 / 341` Sympy exact matches
  - lost full original-input saturation by timing out on:
    - `pagie SRjl#20`
    - `kotanchek Operon#21`
- `no-graph-size-stop`:
  - matched the targeted `pagie Operon#15` and `pagie GOMEA#17/#18/#28`
    parameter counts
  - but slowed `pagie Operon#15` from about `11.7s` to about `57.3s`
  - was not promoted as a simpler retained stack in this pass

Interpretation update:

- the best current read is:
  - no major live-Haskell-vs-Egglog result gap remains
  - archive drift is now a larger interpretive issue than broad Egglog
    divergence
  - the current scheduler customizations still look necessary as a stack:
    - Haskell-style backoff is required for hard cases
    - persistent scheduler state is still required for all-row saturation
    - fresh rematching is still required for all-row saturation
  - the next work should return to the small remaining live mismatch tail and
    the two live-Haskell fallback rows, not reopen broad schedule tuning

Next probe:

- start with the remaining true live gaps rather than more global schedule
  work:
  - `pagie GOMEA#17`
  - `pagie GOMEA#18`
  - `pagie GOMEA#28`
- classify the two live-Haskell fallback rows separately from real Egglog
  mismatches:
  - `pagie SRjl#18`
  - `kotanchek Bingo#21`

### Follow-up probe: retire `kotanchek:EPLEX#4` and move to a live Haskell mismatch

Question:

- is `kotanchek:EPLEX#4` still a meaningful Egglog-vs-Haskell mismatch after
  restoring `add_comm`, or is it only a mismatch against the archived paper
  rows?

Probe commands:

- inspect the archived paper row in
  [artifacts/haskell_paper_rows.csv](artifacts/haskell_paper_rows.csv)
- inspect the live Haskell final expression from
  `trace/kotanchek_eplex_4/haskell/final_simplify_e.json`
- compare it with the Egglog final expression from
  `trace/kotanchek_eplex_4/egglog/final_simplify_e.json`
- recompute the paper parameter count locally with
  [pipeline.py](pipeline.py)'s `count_params`

Observed result:

- the archived paper row still reports `10` final parameters for
  `kotanchek:EPLEX#4`
- but the live Haskell trace now ends at:
  - `((0.002 + ((0.192 * (log(abs((x0 + -0.7770646701367595))) + -2.72876829847128)) ** 2.0)) * (((log(abs((x1 + -7.071787142314785))) + -0.9651127263944101) * (x1 * 0.4)) + 0.096))`
- the current Egglog baseline ends at the same algebraic form up to
  associativity/commutativity:
  - `((0.002 + ((0.192 * (-2.72876829847128 + log(abs((x0 + -0.7770646701367595))))) ** 2.0)) * (((x1 * (-0.9651127263944101 + log(abs((x1 + -7.071787142314785))))) * 0.4) + 0.096))`
- both expressions count as `8` parameters under the current local
  `count_params` implementation

Interpretation update:

- `kotanchek:EPLEX#4` is no longer a good root-cause target for Egglog-vs-live
  Haskell debugging
- it remains useful as a notebook example of drift against the archived paper
  rows
- but it should be retired as a current engine/schedule mismatch

Next probe:

- move to `pagie_operon_15`, where current local Haskell still differs from the
  retained Egglog baseline

### Follow-up probe: pinpoint the live Haskell miss on `pagie_operon_15`

Question:

- after restoring `add_comm` and Haskell-style backoff accounting, is the
  remaining `pagie_operon_15` gap still an engine/state mismatch, or is it now
  specifically the retained four-round schedule stopping too early?

Probe commands:

- run the retained baseline manually for two outer passes and check whether the
  second-pass final e-graph proves the current local Haskell target
- run the literal Haskell-style loop with:
  - outer cap `2`
  - inner cap `30`
  - one reused bound scheduler
  - `haskell_backoff=True`
- record the first literal inner iteration where:
  - the extracted parameter count drops from `8` to `7`
  - the current local Haskell target becomes provable in the e-graph

Observed result:

- current local Haskell `simplifyE` now ends at:
  - `(-0.009788252341175882 * ((x0 * x0) + (((((x1 * x1) + (165.99479114520634 / ((x1 * x1) + 1.2080326581103884))) + (162.89497489770397 / ((x0 * x0) + 1.186536134285651))) * 1.0144051369822908) + -258.7196837451166)))`
- the retained baseline still ends at:
  - `(-2.2516087483e-06 + (-0.009788252341175882 * ((((x0 * x0) + -258.7199137768597) + (165.24149932483223 / ((x0 * x0) + 1.186536134285651))) + (1.0144051369822908 * ((x1 * x1) + (165.99479114520634 / ((x1 * x1) + 1.2080326581103884)))))))`
- retained baseline results:
  - pass 1: `8` params, Haskell target not present
  - pass 2: `8` params, Haskell target still not present
- literal Haskell-style results:
  - iteration 1: `8` params
  - iterations 2-4: still `8` params, but the tiny outer constant has already
    been absorbed into the scaled sum as `+ 0.00023003174313643716`
  - iteration 5: drops to `7` params
  - iteration 6: the current local Haskell target becomes provable in the
    e-graph
  - full two-pass run now completes quickly, about `0.93s`

Interpretation update:

- the current `pagie_operon_15` miss is now a **schedule-boundary mismatch**,
  not the earlier engine blowup story
- with the correct Haskell rewrite set and Haskell-style backoff accounting,
  Egglog reaches the current local Haskell form quickly
- the retained baseline simply stops too early: its fixed four analysis/rewrite
  rounds never reach the iteration-5/6 factorization that absorbs the outer
  constant into the scaled sum
- this is the first current case where the root cause is concrete enough to
  name directly:
  - the accepted baseline schedule is weaker than Haskell's inner loop on this
    row

Next probe:

- inspect whether the literal iteration-5 transition on `pagie_operon_15`
  factorizes through the same reduced `a*x + b -> a * (x + b/a)` family already
  isolated in `reduced_pagie_second_pass_toy`
- if it does, the next fix should be schedule-focused rather than another rule
  or engine change

### Follow-up probe: which rewrite family causes the `Add` / `Mul` explosion?

Question:

- on `pagie_operon_15`, is the literal Egglog blowup caused mainly by:
  - additive A/C closure by itself
  - factorization-style `basic_other` rewrites by themselves
  - or the interaction between those two families?

Probe commands:

- bounded six-iteration literal-mode run on `pagie_operon_15`
- same reused backoff scheduler and same initial analysis as the literal mode
- compare these rewrite-set variants:
  - full literal: `basic_rules | fun_rules`
  - `no_basic_other`
  - `no_add_assoc`
  - `no_add_comm`
  - `no_add_ac`
- record per iteration:
  - rewrite time
  - `Num` class count
  - `Num` member count
  - top `num_matches_per_rule`

Observed result:

- full literal:
  - iteration 5: `584` `Num` classes, `1334` `Num` members, rewrite `0.128s`
  - iteration 6: `1274` `Num` classes, `3341` `Num` members, rewrite `1.966s`
  - top iteration-6 match counts:
    - `add_assoc`: `874`
    - `a*x + b/y -> a * (x + (b/a)/y)`: `786`
    - `add_comm`: `653`
    - `a*x + b*y -> a * (x + (b/a)*y)`: `637`
    - `mul_comm`: `526`
- `no_basic_other`:
  - iteration 6: `79` `Num` classes, `321` `Num` members, rewrite `0.020s`
  - additive A/C rules still match often, but the graph does not explode
- `no_add_assoc`:
  - iteration 6: `235` `Num` classes, `500` `Num` members, rewrite `0.070s`
- `no_add_comm`:
  - iteration 6: `148` `Num` classes, `312` `Num` members, rewrite `0.027s`
- `no_add_ac`:
  - iteration 6: `101` `Num` classes, `235` `Num` members, rewrite `0.037s`

Interpretation update:

- the blowup is **not** “A/C rules alone”
  - with `basic_other` removed, `add_assoc` and `add_comm` still fire, but the
    graph stays small
- the blowup is **not** “factorization alone”
  - with additive A/C removed, `basic_other` still fires, but the graph stays
    small
- the current best explanation is a feedback loop:
  - `basic_other` factorization rewrites create more `Mul(Const, Add(...))` and
    related additive/multiplicative forms
  - `add_assoc` and `add_comm` then enumerate many equivalent binary sum shapes
  - those new sum shapes feed the same factorization rewrites again
- this also explains why the visible size gap is concentrated in `Add` / `Mul`
  members even though the hottest single rule family is not only the generic
  A/C rules

Next probe:

- compare the first `basic_other`-generated factorization shapes at iterations
  5 and 6 between Haskell and Egglog, using a bounded per-rule / per-shape
  probe instead of full-signature trace materialization
- the goal is to tell whether Haskell reaches fewer factorization opportunities
  because of a scheduler/search difference, or whether it creates the same
  opportunities but merges them more aggressively during rebuild

### Follow-up probe: why do more rules match?

Question:

- when the literal Egglog run blows up on `pagie_operon_15`, is that because:
  - analysis marks more classes constant / nonconstant
  - the translated rewrite rules are broader than Haskell's
  - or the same rules drive the two engines to different intermediate states?

Probe commands:

- reuse the bounded literal six-iteration run on `pagie_operon_15`
- compare Haskell outer-pass snapshots vs live Egglog state at:
  - iteration 4
  - iteration 5
- count, for both systems:
  - constant vs nonconstant `Num` classes
  - ordered and unordered `const * nonconst` `Mul` members
  - ordered `const / nonconst` `Div` members
  - guarded factorization opportunities for:
    - `a*x + b*y`
    - `a*x + b/y`
    - `a*x + b`
- compare those opportunity counts to Egglog's actual
  `num_matches_per_rule`

Observed result:

- the two relevant Haskell rules are translated literally in
  `FixTree.hs` / [pipeline.py](pipeline.py):
  - `a * x + b := a * (x + (b / a))`
  - `a * x + b / y := a * (x + (b / a) / y)`
- iteration 4:
  - constant-class counts are essentially the same:
    - Haskell `56` constant classes
    - Egglog `56` constant classes
  - guarded factorization opportunities are also nearly identical:
    - `a*x + b*y`: Haskell `211`, Egglog `209`
    - `a*x + b/y`: Haskell `84`, Egglog `76`
    - `a*x + b`: Haskell `41`, Egglog `35`
- iteration 5:
  - Egglog does **not** have more constant classes:
    - Haskell `106` constant classes
    - Egglog `103` constant classes
  - but Egglog does have more `const * nonconst` multiplication structure:
    - ordered `const * nonconst` `Mul` members:
      - Haskell `345`
      - Egglog `383`
    - unordered one-constant-one-nonconstant `Mul` members:
      - Haskell `456`
      - Egglog `482`
  - the guarded opportunity counts split by rule family:
    - `a*x + b*y`:
      - Haskell `671`
      - Egglog `637`
    - `a*x + b/y`:
      - Haskell `710`
      - Egglog `786`
    - `a*x + b`:
      - Haskell `246`
      - Egglog `296`
- those Egglog opportunity counts exactly predict the next-step rule matches:
  - end of iteration 4 opportunities:
    - `209`, `76`, `35`
    - match iteration-5 rule counts `209`, `76`, `35`
  - end of iteration 5 opportunities:
    - `637`, `786`, `296`
    - match iteration-6 rule counts `637`, `786`, `296`

Interpretation update:

- this is **not** a rule-accounting bug:
  - Egglog's reported match counts line up exactly with the bounded
    opportunity counts in the e-graph state
- this is **not** primarily an analysis bug:
  - Egglog does not have more constant classes than Haskell at the relevant
    boundary
  - the divergence is not “more things are marked constant”
- this is also **not** evidence that the Python rewrite translation is broader
  than Haskell's:
  - the key exploding factorization rules match the Haskell source directly
- the current best explanation is therefore an engine/state-behavior
  difference:
  - by the end of iteration 5, Egglog has accumulated more `const * nonconst`
    multiplicative structure than Haskell
  - that extra structure makes the guarded `a*x + b/y` and `a*x + b` rules
    match more often on the next iteration
  - the earlier A/C result still matters because additive A/C closure feeds
    those opportunities, but the immediate divergence is now localized to the
    state the two engines carry into iteration 6

Next probe:

- find the smallest `pagie_operon_15`-family toy where:
  - iteration 4 guarded opportunities are near parity
  - iteration 5 Egglog has more `const * nonconst` `Mul` structure than
    Haskell
- then inspect which exact iteration-5 rewrites create those extra
  `const * nonconst` `Mul` members in Egglog but not in Haskell

## Mismatch Frontier

| Family | Status | Smallest repro | Current read |
| --- | --- | --- | --- |
| Negative-base integer powers | Fixed | `(-2) ** 2`, `(-2) ** 3` | The Python translation now matches the sampled Haskell cases by only folding negative bases when the exponent round-trips through `to_i64` / `from_i64`. |
| Small constant pruning | Aligned on canaries | `2 - 2`, `2 / 2`, `0 / x0` | The current `delete(...)` approximation is good enough on the confirmed small cases; it is not the next blocker. |
| High-level schedule fidelity | Fixed in the retained baseline | `reduced_pagie_second_pass_toy`, `pagie_operon_15` | The retained baseline now uses the direct Haskell-shaped outer `2` / inner `30` loop with the full rewrite set, so the old four-round schedule mismatch is no longer the main blocker. |
| Scheduler backoff parity | Partially aligned | `pagie_operon_15` | Haskell-style backoff accounting materially reduced the earlier factorization/A-C blowup and helped make the literal loop practical, but it does not by itself explain every remaining paper-row mismatch. |
| Earliest full-state divergence tools | Available | `x0_sq_plus_x1_sq`, `pagie_sbp_1` | The trace harness is still the right tool once a row is confirmed as a true live Egglog-vs-Haskell mismatch, but it is no longer needed for every notebook miss. |
| Remaining live corpus tail | Open | `pagie GOMEA#17`, `pagie GOMEA#18`, `pagie GOMEA#28` | The remaining live Egglog-vs-Haskell gaps are now a small tail with maximum original gap `2` and maximum Sympy gap `1`. |
| Archive drift | Open / separate | `kotanchek:EPLEX#4`, `pagie EPLEX#25` | Many paper-row mismatches are between the archived CSVs and the current local Haskell implementation, not between Egglog and live Haskell. |
| Live-Haskell overflow fallback | Open / separate | `pagie SRjl#18`, `kotanchek Bingo#21` | The current live Haskell corpus artifact still falls back to the archived rows for these two cases because local reruns overflow the Haskell stack budget. |
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

### Earlier stop-on-rewrite-only inner fixpoint

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

### Four-round replacement without persistent scheduler reuse

Historical result:

- simply changing the high-level Python loop shape without reusing one bound
  backoff scheduler did not reproduce Haskell more faithfully
- the useful pass-1 shift only appeared once the same scheduler instance was
  preserved across inner iterations on a single `EGraph`

This is why the current literal-Haskell probe uses low-level scheduler handles
instead of another pure schedule-syntax approximation.

### Restoring `add_comm` globally

Tried:

- restore `add_comm` so the blocked factorization paths become reachable again

Updated result:

- it does recover the smaller factorization toy
- it is part of the Haskell rewrite set and should stay in the retained
  baseline
- with the current scheduler changes, the baseline no longer needs to disable
  it just to finish the corpus
- the meaningful remaining gap is now the schedule/search behavior after using
  the correct ruleset, not whether `add_comm` should be present

## Current Live Hypothesis

The next useful frontier is no longer one dominant scheduling failure.

The strongest current read is:

- the direct Haskell-shaped baseline removed the earlier major schedule
  mismatch and fixed the notebook-level `pagie_operon_15` blocker
- what remains is a mixed tail containing:
  - true live Egglog-vs-Haskell mismatches on some rows
  - archive drift on some rows
  - and a smaller number of extraction/layout-only differences
- the most likely remaining live mismatches are now localized expression-family
  issues rather than one global failure mode

So the next debugging loop should not start from scheduler redesign again. It
should start from one of the largest remaining paper-row gaps and first answer:

- does current local Haskell still differ from Egglog on that row?
- if yes, what is the first meaningful state divergence?
- if no, is the row only archived-paper drift?

## Next Probe

The next probe should target one remaining max-gap row, preferably one where
the archived-paper gap is still `3` parameters:

1. inspect the row in:
   - [artifacts/haskell_paper_rows.csv](artifacts/haskell_paper_rows.csv)
   - [artifacts/egglog_paper_rows.csv](artifacts/egglog_paper_rows.csv)
2. rerun the same source through the current local Haskell implementation and
   current Egglog baseline
3. classify the row as one of:
   - live mismatch
   - archive drift
   - extraction/layout only
4. only for a true live mismatch, use the trace tools to localize the first
   meaningful divergence
