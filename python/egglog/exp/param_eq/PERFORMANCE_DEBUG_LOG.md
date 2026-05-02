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
