# `param_eq` Container Performance Debug Log

Living ledger for scientific-debugging passes on cases where the container
encoding is slower even when the final e-graph is smaller.

## 2026-05-01: `pagie/31/EPLEX/2/sympy`

- Status: active, root cause narrowed to fixed overheads rather than useful
  rewrite matching.
- Smallest repro: corpus row
  `dataset=pagie`, `raw_index=31`, `algorithm_raw=EPLEX`, `algo_row=2`,
  `input_kind=sympy`.
- Input expression:
  `exp(1.24138165845371 * (log(abs(x0)) - exp(exp(log(abs(x0)) * exp(-0.0146126519824622 * exp(9.0 * log(abs(x1))))) / log(abs(log(abs((x1 ** 6.0 + log(abs(log(abs((log(abs(x0)) + log(abs((log(abs((log(abs(-1.0 * ((x1 ** 9.0 + 0.441) ** 9.0 + 0.24) ** 3.0)) + 0.292) ** 3.0)) + 0.462550286672301) ** 3.0))) ** 3.0)) ** 3.0))) ** 3.0)) ** 3.0)))) ** 3.0 * exp(-1.0 * log(abs(x1)))) + 0.904`
- Exact row-selection probe:
  `uv run python - <<'PY' ... load_original_results()[pagie/31/EPLEX/2/sympy] ... run_paper_pipeline(...) ... run_paper_pipeline_container(...) ... PY`
- Observed source-of-truth behavior from existing CSVs:
  binary `67.47 ms`, container `283.40 ms`, binary total size `87`,
  container total size `56`, both `7` params.
- Observed local behavior on current dirty checkout:
  median binary report time about `0.097s`, median container report time about
  `0.460s`, binary total size `87`, container total size `57`, both `7`
  params.
- Hypothesis H1: container is slower because it creates a larger e-graph.
  Result: falsified for this repro; container total size is smaller.
- Hypothesis H2: container is slower because rewrite matching/applying has many
  useful matches.
  Result: falsified for this repro. Boundary trace shows container rewrite
  rounds with `0` matches, `updated=False`, and sub-millisecond reported
  rule search time, while wall time is `0.10s-0.14s`.
- Hypothesis H3: saved `.egg` logging is the main overhead.
  Result: mostly falsified. With `save_egglog_string=False`, median container
  rewrite time remains about `0.235s`; with saving enabled it is about `0.324s`.
  File saving adds overhead, but it is not the main cause.
- Current hypothesis: the slowdown is dominated by per-fresh-egraph
  materialization/registration of complex container rewrite rules plus
  extraction over large embedded container payloads. This overhead is not
  captured by final `egraph_total_size`.
- Supporting probe:
  splitting the first container rewrite call gave roughly `0.036s` in
  `_add_decls`, `0.089s` in `run_schedule_to_egg`/rule registration, and
  `0.015s` in actual `run_program`; the corresponding binary split was about
  `0.0067s`, `0.014s`, and `0.013s`.
- Supporting probe:
  first container rewrite registration emitted `130` commands and about
  `23.7k` command characters; binary emitted `56` commands and about `10.4k`
  command characters. The slowest container registrations are the larger
  polynomial rewrites and their generated higher-order `primitive _lambda_*`
  callbacks.
- Supporting probe:
  replacing `container_rewrite_ruleset` with the empty `container_extra_rules`
  ruleset for this row preserves the same extracted output and size, and drops
  median report time from about `0.424s` to `0.196s`.
- Supporting extraction probe:
  on the built container e-graph, no-cost extraction takes about `0.034s` to
  `0.048s`; cost extraction takes about `0.095s` to `0.143s`. The wrapped
  `container_cost_model` callback accounts for only about `0.016s` to `0.023s`
  of that, so the Python callback body is not the dominant extraction cost.
- Supporting timing-fraction probe:
  over seven current runs, the median container report-scope time is about
  `0.392s`. Median final extraction time is about `0.138s`, or about `35%` of
  report-scope time. The costed extract alone is about `0.102s`, or about
  `26%`. The comparable binary medians are `0.076s` total, `0.028s` extraction
  (`37%`), and `0.023s` costed extraction (`30%`).
- Supporting full-wall attribution probe:
  an exact instrumented run covered `98.6%` of binary wall time and `99.5%` of
  container wall time with explicit pipeline sections. Binary took `123.1ms`
  full wall / `87.3ms` report-scope; container took `606.2ms` full wall /
  `456.3ms` report-scope. In the container `run main rules` section,
  `256.1ms` total split into `41.3ms` for the final `(run-schedule ...)`
  command, `22.2ms` of reported Egglog rule work, `132.5ms` of ruleset /
  scheduler materialization commands, and `82.3ms` of non-`run_program`
  Python/conversion overhead. So only `8.7%` of that section was reported
  rule execution, while `51.7%` was materialization-command time.
- Supporting payload-size probe:
  the container e-graph reports only `72` total function rows, but the
  `polynomial` function alone has `31` rows whose recursive Python
  representations total about `163k` characters. The largest individual
  polynomial payloads are around `22k` characters. This supports the claim that
  `egraph_total_size` undercounts embedded map/monomial payload complexity for
  containers.
- Secondary finding:
  the current loop initializes `previous_size` before the initial analysis
  round, so an analysis-only size increase forces one extra no-op rewrite
  round. This is not the dominant cost here, but it is measurable and should be
  tested separately as a small control-flow fix.
- Last attempted fix:
  tested moving `previous_size = _graph_size(egraph)` to after the initial
  analysis round. Rejected because the target repro did not improve materially:
  container median report time stayed about `0.426s` versus about `0.424s`
  before the tweak.
- Next probes:
  test a safe way to avoid registering the full container rewrite ruleset on
  rows that are analysis-only without delaying needed rewrites on known
  parameter-win cases; separately add a payload-aware metric so reports do not
  imply that smaller `egraph_total_size` means less extraction work.
- Update after reusable-base `push`/`pop` pipeline:
  with the current timing boundary, which excludes the initial `before_cost`
  extraction but includes reusable-base construction and ruleset prewarm, the
  same `pagie/31/EPLEX/2/sympy` canary measures about `85.8ms` for binary and
  `285.8ms` for containers. The container path still returns `7` params and
  `total_size=72`.
- Current strict overhead accounting:
  counting final `(run-schedule ...)` command time, Rust extraction time, and
  Python cost-model callback time as "egglog-side" work, the median container
  split is about `114.4ms` egglog-side and `171.2ms` non-egglog overhead, so
  non-egglog overhead is still about `59.9%` of the counted run. To reach
  `10%` under this definition at the same total runtime, non-egglog overhead
  would need to fall to about `28.6ms`, a reduction of about `142.7ms`.
- Current broad command accounting:
  if all Rust `run_program` command time is counted as egglog-side work
  including ruleset/materialization commands, the median container split is
  about `223.3ms` command/extraction/cost work and `61.9ms` remaining overhead,
  or `21.7%` non-egglog. Reaching `10%` under this broader definition would
  require reducing remaining overhead by about `33.3ms`.

## 2026-05-01: `kotanchek/160/SBP/11/original`

- Status: active, representative of rows where containers really do grow a
  larger intermediate search space even though the recorded final size is
  smaller than binary.
- Smallest repro: corpus row
  `dataset=kotanchek`, `raw_index=160`, `algorithm_raw=SBP`, `algo_row=11`,
  `input_kind=original`.
- Input expression:
  `0.040413 - 0.000597 * ((6.648 - x0) * ((exp(x1) - x1 * 15.197) * (2.753 + (x0 * 9.675 - x0 * (6.648 - x0) * x0))))`
- Initial observation from existing CSVs:
  binary runtime `~52ms`, container runtime `~166ms`; binary recorded
  `egraph_total_size=137`, container recorded `egraph_total_size=66`.
- Hypothesis H4: some "smaller e-graph but slower" rows are smaller only by
  the last-pass recorded size, while peak per-pass work is actually larger for
  containers.
  Result: supported for this repro.
- Boundary trace:
  binary uses one pass, peaks at `137`, and finishes at `137`. Container pass 1
  registers only `14` rows, then analysis grows to `22`, then rewrite/analysis
  rounds grow through `33`, `48`, `64`, `73`, `86`, `99`, `117`, `138`,
  `145`, `157`, `172`, `181`, and finally `183`. The extracted expression is
  smaller, so pass 2 starts from only `9` registered rows and peaks at `69`.
- Key implication:
  the artifact column currently called `egraph_total_size` is the final size of
  the last outer pass, not the peak or cumulative amount of e-graph work. It
  can therefore say containers are smaller (`~66-69`) even when the actual
  expensive first-pass search peaked above binary (`183` vs `137`).
- Function-growth probe:
  almost all first-pass container growth is in `polynomial`: `4` rows at
  registration, `8` after initial analysis, then `17`, `32`, `45`, `53`, `66`,
  `79`, `97`, `118`, `125`, `137`, `151`, `160`, and `162` rows at the final
  no-change round. Other functions barely grow.
- Rule witness:
  repeated matches are dominated by the singleton polynomial flattening rule in
  `container_basic_rules` and the coefficient-factorization rules. The final
  no-change round still finds `108` matches, mostly `67` singleton-flatten
  matches and `29` coefficient-factor matches, but produces no new rows.
- Timing split:
  first container pass took about `0.79s` with `845` matches and peak `183`;
  second pass took about `0.28s` with `202` matches and peak `69`. The
  reported per-rule search/apply time is much smaller than the wall time, so
  the remaining cost appears to be engine/scheduler overhead over large
  container payloads and duplicate candidate processing rather than Python
  harness time.
- Current hypothesis:
  this family is slower because container polynomial rewrites intentionally
  explore many equivalent polynomial presentations before extraction collapses
  the expression. The last-pass size hides that peak. Backoff does not prevent
  the no-change duplicate-match round because the match counts stay below the
  match-limit threshold and the rules are still eligible.
- Next probes:
  add a peak-size/cumulative-size metric to the instrumentation; test a
  duplicate-presentation guard for singleton flattening and coefficient
  factoring on this repro plus the known parameter-win regressions.

## 2026-05-01: profiler cross-check for `pagie/31/EPLEX/2/sympy`

- Status: profiler evidence confirms the existing hypothesis for this one-shot
  canary. Containers are spending much more time in command/materialization,
  root binding, and extraction/cost payload handling than binary; the profile
  does not point at decode/render or final graph size as primary causes.
- Profiler availability:
  `which py-spy || true` found no `py-spy`; `which sample` returned
  `/usr/bin/sample`, so the primary probe used macOS `sample` with a narrowed
  in-process `cProfile` companion.
- Profiling harness artifacts:
  `/tmp/param_eq_profile_runner.py`, `/tmp/run_param_eq_sample.sh`,
  `/tmp/dump_pstats.py`.
- Native sample commands:
  `/tmp/run_param_eq_sample.sh container 45 10` and
  `/tmp/run_param_eq_sample.sh binary 140 10`.
- Native sample artifacts used:
  `/tmp/param_eq_container_20260501_163846.sample.txt`,
  `/tmp/param_eq_container_20260501_163846.summary.json`,
  `/tmp/param_eq_container_20260501_163846.runner.stdout`,
  `/tmp/param_eq_container_20260501_163846.runner.stderr`,
  `/tmp/param_eq_container_20260501_163846.sample.stdout`,
  `/tmp/param_eq_container_20260501_163846.sample.stderr`,
  `/tmp/param_eq_binary_20260501_163924.sample.txt`,
  `/tmp/param_eq_binary_20260501_163924.summary.json`,
  `/tmp/param_eq_binary_20260501_163924.runner.stdout`,
  `/tmp/param_eq_binary_20260501_163924.runner.stderr`,
  `/tmp/param_eq_binary_20260501_163924.sample.stdout`,
  `/tmp/param_eq_binary_20260501_163924.sample.stderr`.
- Discarded native sample artifacts:
  `/tmp/param_eq_container_20260501_163719.*` and
  `/tmp/param_eq_binary_20260501_163754.*` sampled the `uv` wrapper rather than
  Python and are retained only as raw failed-attempt artifacts.
- Loop-only `cProfile` commands:
  `/Users/saul/p/egg-smol-python/.venv/bin/python /tmp/param_eq_profile_runner.py --variant container --iterations 30 --warmup 1 --json-out /tmp/param_eq_container_20260501_1643.loop_cprofile.summary.json --profile-out /tmp/param_eq_container_20260501_1643.loop.cprofile`
  and
  `/Users/saul/p/egg-smol-python/.venv/bin/python /tmp/param_eq_profile_runner.py --variant binary --iterations 80 --warmup 1 --json-out /tmp/param_eq_binary_20260501_1644.loop_cprofile.summary.json --profile-out /tmp/param_eq_binary_20260501_1644.loop.cprofile`.
- Loop-only `cProfile` pstats commands:
  `/Users/saul/p/egg-smol-python/.venv/bin/python /tmp/dump_pstats.py /tmp/param_eq_container_20260501_1643.loop.cprofile --out /tmp/param_eq_container_20260501_1643.loop.pstats.txt --limit 100`
  and
  `/Users/saul/p/egg-smol-python/.venv/bin/python /tmp/dump_pstats.py /tmp/param_eq_binary_20260501_1644.loop.cprofile --out /tmp/param_eq_binary_20260501_1644.loop.pstats.txt --limit 100`.
- Loop-only `cProfile` artifacts used:
  `/tmp/param_eq_container_20260501_1643.loop.cprofile`,
  `/tmp/param_eq_container_20260501_1643.loop.pstats.txt`,
  `/tmp/param_eq_container_20260501_1643.loop_cprofile.summary.json`,
  `/tmp/param_eq_binary_20260501_1644.loop.cprofile`,
  `/tmp/param_eq_binary_20260501_1644.loop.pstats.txt`,
  `/tmp/param_eq_binary_20260501_1644.loop_cprofile.summary.json`.
- Native sample timing summary:
  container ran `45` iterations with median wall `0.411s` and last reported
  pipeline time `0.277s`; binary ran `140` iterations with median wall
  `0.126s` and last reported pipeline time `0.083s`. Container still returns
  `7` params with `total_size=72`, while binary returns `7` params with
  `total_size=88`.
- Native sample hotspots:
  container samples show heavy time under `EGraph::parse_and_run_program`, with
  visible scheduler/rule execution, but also a prominent materialization path:
  `Assignment::annotate_expr` appears around `913` recursive stack samples in
  the container run versus about `101` in binary. The container sample also
  shows map/container primitive work such as `FoldKv::apply` and
  `eval_compiled_expr`; binary spends proportionally more of its sampled
  extension time in scheduler/rule execution.
- Loop-only `cProfile` hotspots, normalized per measured iteration:
  container `extract` was about `0.320s/iter` versus binary about
  `0.100s/iter`; container `_register_commands` was about `0.225s/iter`
  versus binary about `0.026s/iter`; container `run_schedule_to_egg` /
  ruleset materialization was about `0.117s/iter` versus binary about
  `0.024s/iter`; container `let`/root binding was about `0.125s/iter` versus
  binary about `0.012s/iter`; native `parse_and_run_program` was about
  `0.201s/iter` versus binary about `0.060s/iter`.
- Secondary profiler finding:
  full one-shot container parsing/lowering is measurable but not dominant:
  `parse_expression_container` was about `0.044s/iter`, mostly
  `binary_to_containers`, while binary `parse_expression` was about
  `0.006s/iter`. Container decode back to binary was about `0.011s/iter`, and
  binary render was about `0.005s/iter`, so decode/render remain low priority.
- Interpretation:
  the profiler independently supports the current model that the
  `pagie/31/EPLEX/2/sympy` slowdown is not explained by reported Egglog rule
  execution or final graph size. The biggest profiler deltas are exactly around
  command conversion/materialization, root binding/registration, extraction and
  cost payload handling, plus some container primitive evaluation inside the
  native run.
- Next profiler-driven probe:
  isolate `EGraph.extract(..., cost_model=container_cost_model)` from
  extraction without the Python cost model on the same built container e-graph,
  and separately profile a reusable pre-materialized ruleset/base-egraph path
  with source parsing excluded. These probes would distinguish extraction
  callback/payload cost from materialization/root-binding cost more cleanly.

## 2026-05-01: isolated prewarm / ruleset materialization profile

- Status: confirmed for the current prewarm path only. The probe excludes
  source parse, let-current, inner rewrite rounds, and extraction; each measured
  iteration only constructs `EGraph(Num(0.0), Num(1.0))` and runs
  `analysis_schedule + active_rewrite_ruleset + analysis_schedule`.
- Harness and artifacts:
  `/tmp/param_eq_prewarm_profile.py`,
  `/tmp/run_param_eq_prewarm_sample.sh`,
  `/tmp/param_eq_prewarm_binary_20260501_oneshot.summary.json`,
  `/tmp/param_eq_prewarm_binary_20260501_oneshot.cprofile`,
  `/tmp/param_eq_prewarm_binary_20260501_oneshot.pstats.txt`,
  `/tmp/param_eq_prewarm_container_20260501_oneshot.summary.json`,
  `/tmp/param_eq_prewarm_container_20260501_oneshot.cprofile`,
  `/tmp/param_eq_prewarm_container_20260501_oneshot.pstats.txt`,
  `/tmp/param_eq_prewarm_binary_20260501_190658.sample.txt`,
  `/tmp/param_eq_prewarm_container_20260501_190658.sample.txt`.
- Commands:
  `/Users/saul/p/egg-smol-python/.venv/bin/python /tmp/param_eq_prewarm_profile.py --variant binary --iterations 200 --warmup 5 --collect-text --json-out /tmp/param_eq_prewarm_binary_20260501_oneshot.summary.json --profile-out /tmp/param_eq_prewarm_binary_20260501_oneshot.cprofile`,
  `/Users/saul/p/egg-smol-python/.venv/bin/python /tmp/param_eq_prewarm_profile.py --variant container --iterations 120 --warmup 5 --collect-text --json-out /tmp/param_eq_prewarm_container_20260501_oneshot.summary.json --profile-out /tmp/param_eq_prewarm_container_20260501_oneshot.cprofile`,
  `/Users/saul/p/egg-smol-python/.venv/bin/python /tmp/dump_pstats.py /tmp/param_eq_prewarm_binary_20260501_oneshot.cprofile --out /tmp/param_eq_prewarm_binary_20260501_oneshot.pstats.txt --limit 120`,
  `/Users/saul/p/egg-smol-python/.venv/bin/python /tmp/dump_pstats.py /tmp/param_eq_prewarm_container_20260501_oneshot.cprofile --out /tmp/param_eq_prewarm_container_20260501_oneshot.pstats.txt --limit 120`,
  `/tmp/run_param_eq_prewarm_sample.sh binary 600 8`, and
  `/tmp/run_param_eq_prewarm_sample.sh container 260 12`. `uv run py-spy` was
  available but failed on macOS with "This program requires root on OSX", so
  native sampling used `/usr/bin/sample`.
- Timing result:
  current saved-source prewarm median is binary `26.2ms` versus container
  `110.3ms`; construction median is binary `2.1ms` versus container `2.3ms`.
  A companion sampling run measured binary prewarm `24.5ms` and container
  prewarm `100.9ms`, so the cProfile probe was not a direction-changing
  outlier.
- Command volume:
  construction emits `4` saved-source lines and `86` chars in both variants.
  Binary prewarm emits `85` `run_program` commands, `93` saved-source lines,
  and `13.8k` chars per iteration; container prewarm emits `198`
  `run_program` commands, `217` saved-source lines, and `32.2k` chars.
- Python/native attribution:
  binary cProfile puts prewarm under `run_schedule_to_egg`/`_process_schedule`
  at about `23.0ms/iter`, `ruleset_to_egg` at about `22.9ms/iter`,
  `run_program` at about `21.9ms/iter`, and native
  `parse_and_run_program` at about `20.2ms/iter`. Container puts the same path
  at about `110.7ms`, `110.6ms`, `82.4ms`, and `78.3ms` per iteration,
  respectively. Container also spends about `69.4ms/iter` in
  `command_to_egg`, `57.6ms/iter` in `callable_ref_to_egg`, and
  `40.4ms/iter` in `_primitive_command_to_egg`; binary has no comparable
  `_primitive_command_to_egg` hotspot in the top profile.
- Native sample:
  container samples show `RegisterPrimitive::update` and
  `typecheck_expr_with_bindings_and_output` as the dominant registration stack,
  plus deep recursive `Assignment::annotate_expr` stacks. Binary has no
  comparable `RegisterPrimitive` stack and spends proportionally more in the
  schedule/rule-run path.
- Batching inference:
  batching registration commands should help the saved-source path by reducing
  per-command parse/file/FFI overhead, but it cannot remove Python ruleset
  materialization (`ruleset_to_egg`, `command_to_egg`,
  `callable_ref_to_egg`, `_primitive_command_to_egg`) or native primitive
  registration/typechecking. The upper-bound `save_egglog_string=False` probe
  reduced binary prewarm from `26.2ms` to `21.7ms` and container from
  `110.3ms` to `95.4ms`, which makes batching useful but not sufficient.
- Next implementation seam:
  the narrowest seam is pre-materializing/caching the ruleset registration
  product for the reusable base e-graph, especially container primitive command
  lowering and callable/type registration, before optimizing extraction or
  expression parse/decode. If batching is attempted first, measure it as a
  saved-source/registration transport improvement rather than as a fix for the
  container-specific primitive typechecking cost.

## 2026-05-02: `_add_decls(schedule)` declaration-materialization audit

- Status: root cause narrowed. The `trace.py` phase labeled
  `initial add decls` is primarily first-time expansion of delayed Python rule
  declarations, not steady-state declaration dictionary merging.
- Smallest repro:
  `python/egglog/exp/param_eq/trace.py`, specifically the call:
  `egraph._add_decls(containers_analysis_schedule + container_rewrite_ruleset + containers_analysis_schedule)`.
- Trace context:
  `uv run python python/egglog/exp/param_eq/trace.py` showed
  `initial add decls: 0.05s (10.54%)` in the profiler run and about
  `0.05s (10.03%)` in a main-thread check.
- Profiling artifacts:
  `/tmp/egg_add_decls_profile.py` and
  `/tmp/egg_add_decls_profile_thread2_20260502/summary.json`, with cProfile,
  instrument JSON, hot-reuse results, and trace context in the same directory.
- Profiling commands:
  `uv run python /tmp/egg_add_decls_profile.py cprofile --variant binary --out-dir /tmp/egg_add_decls_profile_thread2_20260502 --limit 160`
  and
  `uv run python /tmp/egg_add_decls_profile.py cprofile --variant container --out-dir /tmp/egg_add_decls_profile_thread2_20260502 --limit 160`;
  also ran `sample`, `instrument`, and `hot-reuse` modes from the same script.
- Fresh-process median `_add_decls` timings:
  binary `11.58ms` (p25-p75 `11.38-11.92ms`) versus container `48.44ms`
  (p25-p75 `46.33-51.96ms`).
- Materialized declaration graph size:
  binary produced about `2,130` objects, `54` dicts / `180` dict entries, and
  `63` rules; container produced about `6,997` objects, `74` dicts / `255`
  dict entries, and `32` rules.
- Hot-reuse finding:
  after first materialization, repeated `_add_decls(schedule)` on the same
  already-materialized schedule collapses to about `0.03-0.04ms`. A main-thread
  probe saw first container `_add_decls` at `43.33ms`, then subsequent calls on
  fresh e-graphs with the same schedule at about `0.03-0.04ms`.
- Direct cProfile attribution:
  nearly all cumulative time is under `DelayedDeclarations.__egg_decls__`,
  `Declarations.create` / `upcast_declarations`, and
  `UnstableCombinedRuleset._create_egg_decls`, but their self time is small.
  The real work is `Ruleset._update_egg_decls` draining deferred rule
  generators. Container-heavy callees include `runtime.__call__`,
  `container_basic_rules`, `resolve_literal`, `_convert_function`, and
  `container_analysis_rules`.
- Secondary costs:
  protocol checks are measurable but not primary. The container cProfile had
  `4,479` `HasDeclarations` protocol checks and about `11.5ms` cumulative
  under `typing.__instancecheck__`. Dict merging/copying was smaller:
  `Declarations.update_other` about `2.19ms` cumulative and
  `Declarations.copy` about `1.63ms` in the profiled container run.
- Storage model:
  `DelayedDeclarations.__egg_decls__` calls the stored thunk every property
  access. `Ruleset._update_egg_decls` drains deferred generators once, then
  returns the same `_current_egg_decls`. `UnstableCombinedRuleset` intentionally
  re-evaluates declarations every request so late child rules are visible.
  Direct probes confirmed `Ruleset.__egg_decls__` returns the same declarations
  object after first access, while `UnstableCombinedRuleset.__egg_decls__`
  returns a new merged `Declarations` object every access.
- Cold copy model:
  for warmup schedules, binary cold `_add_decls` performed `389`
  `Declarations.copy` calls and `1666` `update_other` calls; container cold
  `_add_decls` performed `1164` copies and `4651` updates. Hot access dropped
  to `8` copies and `17` updates for both.
- Push/pop model:
  `EGraph.push()` runs Rust `Push`, stores the old `EGraphState`, and replaces
  current state with `self._state.copy()`. That copy shallow-copies the current
  top-level declaration dictionaries, so it is proportional to declaration
  dict width and aliases nested declaration objects.
- Correctness constraint:
  rulesets are mutable after first declaration access. `Ruleset.append`,
  deferred rule materialization, and default rewrite generation can all mutate
  child rulesets. A blanket cache in `DelayedDeclarations.__egg_decls__` would
  be unsafe unless tied to invalidation/versioning.
- Current ranked fix direction:
  1. Add ruleset versioning first, including deferred-rule materialization and
     explicit rule appends, so cached combined declarations can be invalidated
     safely.
  2. Cache `UnstableCombinedRuleset._create_egg_decls` by combined ident plus
     child identity/version. This targets repeated combined declaration rebuilds
     without hiding late child rules.
  3. Add an `EGraphState` fast path to skip merging an exact already-added
     declaration provider at the same version.
  4. Hoist param-eq inner-loop schedule construction so any per-schedule cache
     has a stable object to hit.
  5. Defer persistent/structurally shared `Declarations`; it is likely a broad
     migration because declaration objects and ruleset rule lists are mutable.
- Expected impact:
  versioned combined-declaration caching should mostly help repeated hot
  schedule accesses and correctness around late updates. It will not remove
  the one-time `~48ms` cold container rule-generation cost unless rules are
  deliberately materialized earlier or cached across program-level setup. That
  cold cost is the specific source of the `trace.py` `initial add decls`
  percentage.

## 2026-05-02: binary canary slowdown after current pipeline edits

- Status:
  profiled one long binary run; no code changes kept.
- Canary:
  `kotanchek/96/GOMEA/7/sympy`
  expression:
  `(-0.007876 * x0 + 0.122148884 * x1 - 0.007876 * exp(x1) - 0.002373 * exp(x0 * (x0 - 2.0)) + 0.052131244) * exp(x0 * (2.0 - x0))`
- Baseline artifact reference:
  existing `artifacts/egglog_results_binary.csv` has this row at about
  `1088.9ms`, `egraph_total_size=398`, `passes=2`, `after_params=4`.
- Current dirty-tree observation:
  direct `run_paper_pipeline(parse_expression(expr))` took about `10.95s` to
  `11.54s`, `passes=2`, `egraph_total_size=8407`, `after_params=4`.
- Falsified hypothesis:
  file-backed `.egg` saving is not the dominant cause. Running the same canary
  with pass e-graphs forced to `save_egglog_string=False` still took about
  `10.38s`, with the same size and params.
- Falsified hypothesis:
  the current global persistent scheduler object and new pre-run-before-let
  shape are not sufficient causes. Controlled variants with/without pre-run
  and with global/fresh-per-pass schedulers all still hit inner counts
  `[30, 13]` and `egraph_total_size=8407`.
- High-level profile:
  current run spends about `10.08s / 11.54s` in `EGraph.run(...)` and
  `run_program`/Rust command execution, about `1.38s` in extraction, and only
  about `0.05s` in schedule lowering. Declaration overhead is not the main
  issue for this binary canary.
- State-growth evidence:
  pass 1 no longer reaches a size fixed point; it runs all `30` inner
  iterations. Sizes grow:
  `50, 76, 126, 264, 491, 962, 1958, 2780, 3279, 4496, 5325, ... 8407`.
  Pass 2 then takes `13` inner iterations and ends at size `775`.
- Top matching rules by cumulative matches in pass 1:
  `18493` matches for
  `(a*x) * (b*y) -> (a*b) * (x*y)`;
  `17207` matches for
  `a*x - b*y -> a * (x - (b/a)*y)`;
  `7772` matches for
  `x - a*y -> x + (-a)*y`;
  `7015` matches for
  `a*x + b -> a * (x + b/a)`;
  `6748` matches for
  `x - a -> x + -a`;
  `6138` matches for
  `(x*y) + (x*z) -> x * (y+z)`.
- Match-limit experiment:
  using the same rules but a smaller backoff match limit gives the same
  extracted param count while avoiding the blowup on this canary:
  `25 -> 0.215s, size=220`;
  `50 -> 0.254s, size=341`;
  `100 -> 0.349s, size=534`;
  `250 -> 0.635s, size=1286`;
  `500 -> 1.643s, size=2508`;
  `1000 -> 9.245s, size=8407`.
- Current hypothesis:
  the regression is search-space pressure from broad binary algebraic rewrites
  under `BACKOFF_MATCH_LIMIT=1000`. This case is slow because rule execution
  keeps producing new equivalent binary presentations until the fixed
  `HASKELL_INNER_ITERATION_LIMIT=30` cap, not because Python declaration
  materialization or `.egg` logging dominates.

## 2026-05-02 follow-up: debug build was the wall-time regression

- Status:
  reran after the current changes were committed; rebuilt the Python extension
  with `uv run maturin develop --release`.
- Initial state:
  the active `python/egglog/bindings.cpython-313-darwin.so` was `48M`, matching
  `target/debug/libegglog.dylib`. The existing release artifacts were `15M`.
- Experiment 1:
  restored only `pipeline.py` from `HEAD~1` and reran
  `kotanchek/96/GOMEA/7/sympy`.
  This restored the old reported `egraph_total_size=398`, but wall time stayed
  around `10.0s` before rebuilding release. Disabling saved `.egg` logging did
  not materially change that.
- Experiment 2:
  after `uv run maturin develop --release`, the same `HEAD~1` pipeline shape
  ran the canary in `0.837s`, `passes=2`, `egraph_total_size=398`,
  `after_params=4`.
- Experiment 3:
  restored committed `HEAD` `pipeline.py` and reran with the release binding.
  The canary ran in `0.844s`, `passes=2`, `after_params=4`, with
  `total_size=8407`.
- Top-five slow artifact rerun with committed `HEAD` plus release binding:
  `kotanchek/96/GOMEA/7/sympy`: old `1088.9ms`, new `831.3ms`;
  `kotanchek/180/SRjl/1/sympy`: old `971.1ms`, new `671.6ms`;
  `pagie/27/Bingo/27/sympy`: old `897.7ms`, new `622.9ms`;
  `kotanchek/181/SRjl/2/sympy`: old `855.8ms`, new `708.1ms`;
  `pagie/163/SBP/13/sympy`: old `835.7ms`, new `710.8ms`.
  All five preserved `after_params`.
- Conclusion:
  no last-commit code revert is needed to recover binary wall-time
  performance; the wall-time regression was caused by running the debug
  extension build. The remaining difference in `egraph_total_size` is a
  reporting semantics change: committed `HEAD` tracks max size across passes,
  while the old pipeline overwrote the value with the final pass size.

## 2026-05-03: container inverse-factor normalization outlier

- Status:
  diagnosis only. Temporary rule edits used for falsification were reverted.
- Canary:
  large expression from the notebook/dashboard request beginning
  `((((((((0.11064466475608078 + -0.010036545250561161 * ...`.
  This is the case the user described as taking about `50x` longer on
  containers.
- Source-of-truth timing, direct production entrypoints:
  `run_paper_pipeline(parse_expression(EXPR))` took about `0.86s` to `0.95s`,
  `passes=2`, `egraph_total_size=10725`, `before_params=49`,
  `after_params=37`.
  `run_paper_pipeline_container(parse_expression_container(EXPR))` took about
  `31s` to `33s`, `passes=2`, `egraph_total_size=4586`,
  `before_params=49`, `after_params=32`.
- Initial observation:
  the container result is semantically better by the param metric and has a
  smaller reported e-graph, so the slowdown is not explained by final graph
  size.
- Phase split, binary:
  rewrite execution was about `0.28s`, extraction about `0.65s`, analysis
  about `0.02s`.
- Phase split, container:
  rewrite execution was about `22.4s`, extraction about `2.3s`, analysis about
  `1.5s`, initial extraction about `0.08s`, and graph-size checks were
  effectively zero.
- First meaningful divergence:
  container pass 1 rewrite rounds dominate. The graph grows from about `160` to
  about `4586`, but rewrite rounds remain expensive near convergence:
  roughly `2.2s` per rewrite round after the graph has almost stopped changing.
  Pass 2 is small.
- Ruleset isolation:
  running only `shared_rewrite_ruleset` or `container_fun_rules` is fast.
  Running `container_basic_rules` alone reproduces the slowdown:
  about `32s`, `16` rounds, `rewrite=28.5s`, `extract=2.3s`,
  `after_params=32`.
  `shared_rewrite_ruleset | container_basic_rules` and the full container
  rewrite ruleset behave the same, so the issue is localized to
  `container_basic_rules`.
- Rule-level aggregation from `RunReport.search_and_apply_time_per_rule` over
  the slow container pass:
  `pipeline.py:647` inverse-factor polynomial normalization:
  about `19.0s`, `328` matches.
  `pipeline.py:609` singleton/sole-monomial flattening:
  about `2.1s`, `2539` matches.
  `pipeline.py:737` greedy Horner factorization:
  about `1.5s`, `1951` matches.
  `pipeline.py:359` integer-ratio coefficient factoring:
  about `1.5s`, `847` matches.
  `pipeline.py:338` representative coefficient factoring:
  about `1.4s`, `7088` matches.
  `pipeline.py:440` repeated scalar coefficient factoring:
  about `0.9s`, `2016` matches.
- Primary hypothesis:
  `pipeline.py:647` is the main regression source. The rule is useful for
  parameter quality, but operationally too broad. For every container
  polynomial candidate it folds all monomials, checks every term, and probes
  `POLYNOMIALS[term]` for reciprocal factors. The outlier combines enough
  reciprocal and singleton-polynomial presentations that this full-polynomial
  fold is expensive even with only hundreds of successful matches.
- Falsification probe 1:
  temporarily disabled only the `pipeline.py:647` rule with an impossible
  guard. Container time dropped from about `26s` to about `12.2s`;
  rewrite time dropped from about `22.4s` to about `8.6s`; `after_params`
  stayed `32`. This confirms the rule is causal and contributes the largest
  share of the slowdown, but it is not the only cost.
- Falsification probe 2:
  temporarily added a cheap eligibility guard so the rule only ran when the
  polynomial had a `-1` exponent term whose base appeared in `POLYNOMIALS`.
  Container time dropped to about `18.3s`; rewrite time dropped to about
  `14.6s`; `after_params` stayed `32`. This validates the guard direction, but
  the expression genuinely has many eligible reciprocal terms, so this guard is
  not sufficient.
- Reduction probe:
  top-level fragments of the expression are individually fast. Several
  fragments run faster in containers than binary. The slowdown appears when the
  fragments are combined, consistent with cross-term accumulation of equivalent
  polynomial and reciprocal presentations rather than a single isolated
  arithmetic subexpression.
- Current conclusion:
  the inverse-factor normalization rule is the primary hotspot. A likely fix
  should not remove it outright; it should narrow it from a whole-polynomial
  fold over all candidates to a one-reciprocal-factor-at-a-time rewrite, or add
  stronger eligibility/canonicality guards so it only runs when it can actually
  change a reciprocal term and not repeatedly rescan stable presentations.

## 2026-05-04: replace inverse-factor rule with scored `SOLE_MONOMIALS`

- Status:
  accepted local fix candidate.
- Question:
  can the expensive inverse-factor normalization rule be removed if
  `SOLE_MONOMIALS` stops keeping a stale first alias?
- Change tested:
  removed the broad reciprocal normalization rewrite from
  `container_basic_rules`. Changed `SOLE_MONOMIALS` from
  `Map[Num, ContainerMonomial]` to `Map[Num, Pair[i64, ContainerMonomial]]`.
  The merge now keeps the lower local score, with old-value tie breaking.
- Witness after only removing the broad rule:
  `0.7096075864465768 / (-5.746100091872222*x0 +
  2.873050045936111*exp(x0) + 1.7208022848793882)` regressed to
  `-0.12349377405560805 / (-0.2994730786735579 + x0 - 0.5*exp(x0))`,
  `after_params=3`.
- Alias evidence:
  the stale `SOLE_MONOMIALS` value for the denominator was
  `-5.746100091872222 * polynomial(-0.2994730786735579 + x0 -
  0.5*exp(x0))`. The better alias was
  `2.873050045936111 * polynomial(0.5989461573471158 - 2*x0 + exp(x0))`.
  The second form has one fewer non-integer coefficient after decoding.
- Failed probe:
  latest-wins merge fixed the witness, but that is still scheduler-order based
  rather than a preference rule.
- Accepted probe:
  a scored merge plus a specific one-level nested-polynomial score picked the
  better alias with score `2`; the stale alias scored worse. The focused
  witness now extracts
  `0.2469875481112161 / (0.5989461573471158 - 2.0*x0 + exp(x0))`,
  `after_params=2`.
- Cycle regression:
  the self-factor cycle canary `-14.792753236262874*x1 + x0` stayed
  extractable through four container rewrite/analysis rounds.
- Sample comparison:
  `make -C python/egglog/exp/param_eq compare-binary-container-sample
  SAMPLE_LIMIT_PER_DATASET=50` compared 100 shared rows. After params were
  `9 better / 0 worse / 91 same` for containers. Param-reduction still showed
  five worse rows because container lowering starts with fewer params on those
  rows; after-param float count did not regress.
- Large outlier timing:
  rerunning `/tmp/param_eq_perf_expr.py` changed the prior container canary
  from about `31s` to about `13.0s` wall/report time, with
  `after_params=32` preserved and e-graph size about `4477`. Binary remained
  about `0.76s` to `0.85s`, now extracting `after_params=36` in this checkout.
- Validation:
  `uv run pytest
  python/egglog/exp/param_eq/test_pipeline.py::test_container_uses_preferred_singleton_polynomial_for_inverse_scale
  python/egglog/exp/param_eq/test_pipeline.py::test_container_preserves_repeated_polynomial_products -q`
  passed.
  `uv run ruff check python/egglog/exp/param_eq/pipeline.py` passed.

## 2026-05-04: remove `POLYNOMIALS` analysis table

- Status:
  accepted local fix candidate.
- Question:
  can the remaining `POLYNOMIALS: Map[Num, ContainerPolynomial]` analysis
  table be removed without losing the parameter-quality parity that the
  repeated-product and outer-scale container rules need?
- Baseline dependency:
  the table was used by two rules in `container_basic_rules`: the
  repeated-polynomial product rule
  `P + c*M*P + R -> P * (1 + c*M) + R`, and the outer-constant scale rule
  `(c*x + d*y + k) * M - c -> c * (M * (...) - 1)`.
  An integer-residual guard also used it only to reject nested polynomial
  terms.
- Failed probe 1:
  replaced the two table lookups with `POLYNOMIAL_MONOMIALS`, keyed by
  `{polynomial(P): 1}`. The focused repeated-product regression with the
  `log(abs(...))` expression regressed from the expected `after_params <= 8`
  to `after_params=9`.
- Failed probe 2:
  changed `POLYNOMIAL_MONOMIALS` to prefer the newer value. The same
  repeated-product regression still extracted with `after_params=9`, so this
  was not just a stale merge choice for that table.
- Falsification probe:
  restored `POLYNOMIALS` only for the outer-scale rule. The focused regression
  still failed with `after_params=9`, proving the repeated-product rule was the
  first missing dependency.
- Accepted probe:
  removed the table lookup entirely and bound the polynomial relation directly:
  after collecting candidate keys, the repeated-product rule now uses
  `counts.count(n) > 0` and `polynomial(poly1) == n`; the outer-scale rule now
  uses `mono[n] == 1` and `polynomial(poly1) == n`. This gives the rule an
  explicit relation dependency without storing a global `Num -> polynomial`
  map.
- Code removed:
  deleted the `POLYNOMIALS` constant, its default analysis initialization, and
  its population rule.
- Focused validation:
  `uv run pytest
  python/egglog/exp/param_eq/test_pipeline.py::test_container_uses_preferred_singleton_polynomial_for_inverse_scale
  python/egglog/exp/param_eq/test_pipeline.py::test_container_preserves_repeated_polynomial_products
  python/egglog/exp/param_eq/test_pipeline.py::test_container_preserves_repeated_numeric_factor_before_like_term_merge
  python/egglog/exp/param_eq/test_pipeline.py::test_container_splits_integer_residual_after_like_term_merge
  python/egglog/exp/param_eq/test_pipeline.py::test_container_matches_binary_coefficient_factoring_choices
  -q` passed with `10 passed`.
  `uv run ruff check python/egglog/exp/param_eq/pipeline.py` passed.
- Broader validation:
  full `test_pipeline.py` still has unrelated existing failures: snapshot
  drift and a test calling the removed `_new_rewrite_scheduler`. The targeted
  POLYNOMIALS regressions are green.
- Sample comparison:
  `make -C python/egglog/exp/param_eq compare-binary-container-sample
  SAMPLE_LIMIT_PER_DATASET=50` compared 100 shared rows. After params were
  `9 better / 0 worse / 91 same` for containers. Extracted cost was
  `46 better / 7 worse / 47 same`; total e-graph size was
  `97 better / 3 worse / 0 same`.
- Direct A/B against the old table:
  temporarily reversed only the `pipeline.py` removal patch and ran
  `run_egglog_corpus --variant container --limit-per-dataset 20` to
  `/tmp/egglog_container_with_polynomials_first20.csv`, then restored the
  patch and reran to `/tmp/egglog_container_no_polynomials_first20.csv`.
  On the 40 shared rows, `after_params` was identical for all rows. Median
  e-graph total size changed from `37` to `36`, with `39 / 1 / 0`
  better / worse / same under the table-free version. Median runtime changed
  from `189.5ms` to `191.8ms`, with `15 / 25 / 0` faster / slower / same,
  so this sample supports a state-size win but not a measured runtime win.
- Current conclusion:
  `POLYNOMIALS` was a workaround for making polynomial-representation
  dependencies visible through higher-order map-derived keys. The direct
  `polynomial(poly1) == n` constraints now make those dependencies explicit
  enough for the relevant rules, so the extra table can be removed.

## 2026-05-04: keep `POLYNOMIAL_MONOMIALS` as a singleton-key index

- Status:
  rejected full removal.
- Question:
  can `POLYNOMIAL_MONOMIALS: Map[ContainerMonomial, ContainerPolynomial]` be
  removed after `POLYNOMIALS` was removed?
- Purpose of the table:
  it maps the concrete singleton monomial key `{polynomial(P): 1}` back to `P`.
  The remaining users are not asking for every polynomial representation of a
  `Num`; they are asking whether an already-present monomial key is exactly a
  nested polynomial term.
- Direct replacement tested:
  replaced lookups such as `polynomial_monomials[candidate_mono]` with
  constraints like `mono == {n: 1}` and `polynomial(poly1) == n`.
- Sample result:
  on a first-20-per-dataset container sample, after params were identical on
  all 40 rows compared with the version that kept `POLYNOMIAL_MONOMIALS`.
  Median runtime looked better in that small sample (`172.5ms` vs `191.8ms`),
  but this did not cover the worst focused canary.
- Falsifying canary:
  `test_container_matches_binary_coefficient_factoring_choices` for
  `0.103875 + 0.01063 * (exp(x0) - exp(x1) + 2.824*x0 + 6.894*x1 +
  7.894*(x0 + x1 - x0*x0))`.
  With direct constraints, the test still passed but the single case took
  about `120.8s`.
- Isolation:
  disabling all three direct singleton-polynomial rules made the same canary
  fast but lost the expected node-count improvement. Re-enabling only the
  exact nested polynomial flattening rule reproduced the slowdown, so the
  expensive pattern is:
  `a*polynomial(P) + R -> a*P + R` with direct
  `mono == {n: 1}; polynomial(P) == n` matching.
- Restored result:
  restoring `POLYNOMIAL_MONOMIALS` reduced that canary to about `9.1s`, and the
  focused 16-test bundle completed in `14.79s`.
- Current conclusion:
  `POLYNOMIAL_MONOMIALS` is still needed as a performance index. The direct
  relation form is semantically valid, but it causes the matcher to enumerate
  many polynomial e-classes before proving that the singleton monomial key is
  present. The table keeps the join keyed by existing monomial keys.

## 2026-05-04: `POLYNOMIALS` removal slowdown root cause

- Status:
  diagnosis only. Temporary guards were reverted.
- Question:
  why did some examples get slower after removing `POLYNOMIALS`, even though
  params stayed equal and e-graph size usually got slightly smaller?
- A/B command:
  temporarily reversed only the current `pipeline.py` patch and ran
  `run_egglog_corpus --variant container --limit-per-dataset 50` to
  `/tmp/egglog_container_with_polynomials_first50.csv`, restored the patch, and
  reran to `/tmp/egglog_container_no_polynomials_first50.csv`.
- A/B result:
  on 100 shared rows, median runtime was similar
  (`169.9ms` with `POLYNOMIALS`, `167.9ms` without), but 51 rows were slower.
  The largest slowdown was `pagie/Bingo/raw=17/algo_row=17/original`, from
  `464ms` to `1077ms`, with after params unchanged at `8` and e-graph total
  size changing from `546` to `545`.
- Canary expression:
  the repeated-product/log expression beginning
  `(-2.0 * -0.16807062259009387 + -0.00015170797954567304*x1) *
  log(abs(...))`.
- Phase split on the canary:
  with `POLYNOMIALS`, total was `0.353s`: analysis `0.063s`, rewrite
  `0.170s`, extraction `0.112s`.
  without `POLYNOMIALS`, total was `0.879s`: analysis `0.057s`, rewrite
  `0.694s`, extraction `0.120s`.
  The graph was not bigger; the slowdown is rewrite matching near convergence.
- Rule attribution:
  old repeated-product rule using `%POLYNOMIALS` took `0.009s` with
  `67` matches.
  new direct repeated-product rule using `counts.count(n) > 0` and
  `polynomial(poly1) == n` took `0.465s` with `93` matches.
  new direct outer-scale rule also took `0.073s` with `0` matches.
- Falsification probe:
  temporarily disabled only the direct repeated-product rule with an impossible
  guard. The canary dropped to `0.191s` total and `0.136s` rewrite time, but
  quality regressed from `after_params=8` to `after_params=10`.
- Current conclusion:
  removing `POLYNOMIALS` changed the join shape of the repeated-product rule.
  The old table made the candidate polynomial keys available through
  `map_restrict_keys(counts, polynomials)`, so the expensive relation join
  happened inside a map primitive over terms already present in the polynomial.
  The direct replacement exposes `polynomial(poly1) == n` as a normal e-matcher
  join for each counted term, which is semantically correct but much more
  expensive on examples with many polynomial presentations. The main slowdown
  is therefore matcher join order/cardinality, not extraction, analysis, final
  e-graph size, or parameter-quality work after extraction.

## 2026-05-04: restore polynomial body cache with explicit index naming

- Status:
  accepted local fix candidate.
- Change:
  restored the old `Num -> ContainerPolynomial` cache as
  `POLYNOMIAL_BODIES`, and documented it as a join-shaping index rather than
  as independent semantic state. `SOLE_MONOMIALS` remains the preferred
  singleton-alias cache, and `POLYNOMIAL_MONOMIALS` remains the singleton
  monomial-key index.
- Concrete comments added:
  `POLYNOMIAL_BODIES` documents the repeated-product example
  `P + c*M*P + R -> P * (1 + c*M) + R`, where candidates should be terms
  already present in the polynomial rather than every `polynomial(body) == n`
  relation in the e-graph.
  `POLYNOMIAL_MONOMIALS` documents the exact nested-flatten example
  `a*polynomial(P) + R -> a*P + R`, where candidates should be existing
  singleton monomial keys rather than constructed `{n: 1}` keys for every
  polynomial relation.
- Canary result:
  `pagie/Bingo/raw=17/algo_row=17/original` is back to the old shape:
  total `0.355s`, rewrite `0.173s`, extraction `0.110s`,
  e-graph total size `546`, `after_params=8`.
  This matches the prior `POLYNOMIALS` behavior and removes the direct-rule
  slowdown (`0.879s`, rewrite `0.694s`).
- Focused validation:
  `uv run pytest ... -q --durations=8` over the 16 container regressions passed
  in `4.44s`; the slowest case was `1.56s`.
- Sample comparison:
  first-50-per-dataset container sample matched the old `POLYNOMIALS` cache on
  all 100 `after_params` values. Median runtime was effectively similar
  (`169.9ms` old, `170.8ms` renamed cache); remaining per-row timing deltas
  look like benchmark noise because e-graph size and params are unchanged.

## 2026-05-04: slow container outlier with small final e-graph

- Status:
  diagnosis only. No rule changes kept.
- Question:
  why does the row
  `0.103875 + 0.01063 * (exp(x0) - (exp(x1) + (x0 + x1 - x0*x0) * -7.894 + (x1 * -7.894 - 2.824*x0 + x1)))`
  take much longer in containers even though the container e-graph is smaller?
- Direct run result:
  binary ran in about `0.56s`, with `passes=2`, e-graph size `9168`,
  `after_nodes=27`, and `after_params=4`.
  containers ran in about `13.2s` to `14.0s`, with `passes=2`, e-graph size
  `4017`, `after_nodes=23`, and `after_params=4`.
- Phase split:
  binary total was `0.570s`: analysis `0.010s`, rewrite `0.191s`,
  extraction `0.359s`.
  container total was `13.282s`: analysis `2.641s`, rewrite `9.357s`,
  extraction `1.260s`.
- Boundary evidence:
  the slowest container phases were late rewrite rounds, not construction or
  extraction: pass 1 rounds 26 through 30 each took about `0.79s` to `0.87s`
  while the total size moved only from about `3633` to `4017`.
- Rule attribution from the container run:
  repeated scalar coefficient-magnitude factoring at `pipeline.py:518` took
  `1.93s` with `8870` matches.
  `SOLE_MONOMIALS` flattening at `pipeline.py:696` took `1.32s` with `6004`
  matches.
  Horner factoring at `pipeline.py:773` took `1.27s` with `4747` matches.
  representative coefficient factoring at `pipeline.py:425` took `1.22s` with
  `8128` matches.
  integer-ratio coefficient factoring at `pipeline.py:442` took `1.17s` with
  `3081` matches.
- Falsified hypotheses:
  the slowdown is not caused by a larger final e-graph: containers end smaller
  (`4017` vs `9168`).
  it is not primarily extraction: container extraction is large (`1.26s`) but
  much smaller than container rewrite time (`9.36s`).
  it is not the restored `POLYNOMIAL_BODIES` lookup: the repeated-product rule
  using that table took only `0.35s` and the outer-scale lookup had `0`
  matches.
- Current hypothesis:
  this row is slow because the container encoding exposes many equivalent
  coefficient-factor presentations inside one polynomial. The broad scalar
  coefficient factoring, sole-monomial flattening, Horner, and representative
  coefficient rules keep finding legal alternate presentations through late
  rounds. The result is smaller and reaches the same parameter count, but the
  schedule pays for repeated map scans and rematching near convergence.

## 2026-05-04: isolate unstaged hunk causing the slow container outlier

- Status:
  diagnosis only. Current `pipeline.py` was not changed by this probe.
- Question:
  which unstaged change made the row above slow?
- Method:
  used a detached `HEAD` worktree at `050e5fd`, copied the local compiled
  `egglog` extension artifacts into it, and selected Python source with
  `PYTHONPATH` so current and `HEAD` used the same runtime.
- Results:
  `HEAD` ran the container case in `1.06s`, with e-graph size `862`,
  `after_nodes=23`, and `after_params=4`.
  Current dirty source ran it in `13.41s`, with e-graph size `4017`,
  `after_nodes=23`, and `after_params=4`.
  Copying the full current `pipeline.py` into the `HEAD` worktree reproduced
  the slow result: `13.35s`, size `4017`.
  Reverting only the post-normalization integer-residual split guard in that
  temp copy restored the fast result: `1.06s`, size `862`.
- Isolated hunk:
  the slow hunk is in the rule
  `a*M + b*N + R -> a*polynomial(M + N) + (b - a)*N + R`.
  The unstaged version changed the suppression guard from the polynomial body
  cache to `SOLE_MONOMIALS`:
  `map_restrict_keys(counts, POLYNOMIALS).length() == 0`
  became
  `map_restrict_keys(counts, SOLE_MONOMIALS).length() == 0`.
- Mechanism:
  the old guard suppresses integer-residual splitting when the current
  polynomial already contains a nested polynomial term. The new guard only
  suppresses when a term has a sole-monomial alias, so it lets the split rule
  run on nested-polynomial presentations. On this row that creates many extra
  coefficient-factor presentations, which then feed repeated scalar coefficient
  factoring, `SOLE_MONOMIALS` flattening, Horner, and representative
  coefficient factoring through late rounds.

## 2026-05-04: coefficient-ratio outlier with large container e-graph

- Status:
  accepted local fix candidate.
- Full expression:
  `0.02889 * x0^3 - 0.18081288 * x0^2 + 0.18081288 * x0 +
  0.14418036 * x1 - 0.00963 * exp(x1) + 0.179028 -
  0.00963 * exp(-15.767 * x0)`.
- Baseline observation:
  binary ran in about `0.20s`, e-graph size `3443`, `after_nodes=29`,
  `after_params=5`.
  containers ran in about `1.35s`, e-graph size `1145`, `after_nodes=29`,
  `after_params=5`.
- Reduced reproducer:
  replacing the exponentials with variables and dropping the unrelated `x1`
  term preserves the container slowdown:
  `0.02889*x0^3 - 0.18081288*x0^2 + 0.18081288*x0 +
  0.179028 - 0.00963*x2 - 0.00963*x3`.
  This ran in about `1.23s`, e-graph size `1035`, `after_nodes=21`,
  `after_params=3`.
- Rule attribution on the reduced reproducer:
  `SOLE_MONOMIALS` flattening took `0.267s` with `4464` matches.
  coefficient-count factoring took `0.172s` with `2914` matches.
  Horner took `0.133s` with `1897` matches.
  representative coefficient factoring took `0.064s` with `2456` matches.
- Failed probe:
  restoring the old `poly.length() <= 5` bound on `SOLE_MONOMIALS` flattening
  did not help. The same rule still fired on smaller equivalent presentations,
  and the full case got slightly slower (`1.35s -> 1.41s`).
- Accepted changes:
  tightened integer-ratio coefficient factoring so a coefficient is counted
  only when it is a smaller exact integer divisor of another coefficient. This
  removes self-counting and equal-magnitude counting from that whole-polynomial
  scale rule; repeated/equal-and-opposite scalar subsets remain handled by the
  dedicated subset rule.
  also bounded the generic representative coefficient rule to polynomials with
  at most four monomials, leaving larger polynomials to the targeted
  coefficient rules.
- Result:
  reduced reproducer improved from `1.23s`, size `1035` to `0.79s`, size
  `703`, with unchanged `after_nodes=21`, `after_params=3`.
  full expression improved from `1.35s`, size `1145` to `1.12s`, size `788`,
  with unchanged `after_nodes=29`, `after_params=5`.
- Full corpus check:
  a fresh 714-row container run saturated every row. Compared with the existing
  container artifact, `after_params` was `0` better, `714` same, `0` worse.
  Compared with the binary artifact, `after_params` was `81` better, `633`
  same, `0` worse. Median runtime moved from `513.5ms` in the existing
  container artifact to `192.1ms`; median `after_params` stayed `5`.

## 2026-05-04: `pagie/144/Operon/25/original` exact-flatten slowdown

- Status:
  rejected local fix candidates and reverted exact-flatten edits. The row is
  still slower than desired, but the only zero-regression variants found either
  increased corpus runtime or depended on a shape-specific singleton rule.
- Full expression:
  `-5.5456266637e-06 + 1.0000027418136597 * ((exp(-4.225025177001953*x1*(1.425803780555725*x1)) + exp(-1.171636939048767*x0*(5.220480918884277*x0))) * 0.2746707499027252 - -1.9792732000350952 - (exp(-1.7990413904190063*x1*(0.5140095949172974*x1)) + exp(1.8088867664337158*x0*(-0.5105684995651245*x0))) * 1.2522673606872559)`.
- Baseline observation:
  current accepted container rules ran the row at about `1.19s` in the corpus
  artifact, e-graph size `699`, `after_nodes=33`, `after_params=7`.
  A direct trace measured about `0.76s` report time, with analysis `0.10s`,
  rewrite `0.48s`, and extraction `0.17s`.
- Reduced reproducer:
  replacing the four exponentials with variables preserved the slowdown:
  `-5.5456266637e-06 + 1.0000027418136597 *
  (0.2746707499027252*(x0 + x1) - -1.9792732000350952 -
  1.2522673606872559*(x2 + x3))`.
  The reduced case ran at about `0.74s`, e-graph size `676`,
  `after_nodes=13`, `after_params=3`.
- Hypothesis:
  the slowdown is caused by exact nested-polynomial flattening expanding
  isolated grouped sums, after which repeated scalar coefficient factoring
  recreates the groups. This creates a flatten/refactor cycle even though the
  extracted expression does not improve.
- Supporting probes:
  disabling the repeated scalar coefficient subset rule made the target fast
  (`~0.12s`, size `122`) but regressed focused parameter tests, so it was
  rejected.
  disabling exact nested flattening made the target fast (`~0.09s`, size `72`)
  but regressed log-polynomial and coefficient canaries, so it was rejected.
  limiting exact flattening to nested polynomials that contain constants made
  the full corpus faster (`153.7s -> 148.2s`, size sum `54051 -> 48683`) but
  introduced two `after_params` regressions:
  `pagie/134/Operon/15/original` and `pagie/163/SBP/13/original`.
- Rejected fix candidates:
  adding a generic single-nested-group helper preserved parameters but made the
  full corpus slower (`153.7s -> 158.1s`).
  adding singleton exact flatten for all constant-bearing bodies preserved
  parameters but was much slower (`153.7s -> 176.8s`) and grew the e-graph
  size sum (`54051 -> 59022`).
  bounding that singleton rule by body length still preserved parameters only
  at the shape needed by `pagie/163` and remained slower (`153.7s -> 171.9s`).
  A final `poly1.length() == 7` singleton probe fixed the known regression but
  was rejected as too example-shaped.
- Current decision:
  keep the accepted baseline rules. Do not add a targeted exact-flatten rule
  for this row. A future viable fix needs a general criterion for when
  flattening a nested polynomial unlocks a downstream simplification, not a
  body-length or row-shape condition.
