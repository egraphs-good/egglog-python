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

## 2026-05-04: partial long expression runtime no-repro

- Status:
  no runtime slowdown reproduced on the current tree.
- Input:
  the partial expression ending at
  `(x0 ** 2.0 * (-0.7990490259929379 * x1 * (x1 - 4.734655034994491) +
  0.2566632469880581) - 0.01768299095097286 * exp(2.0 * x0)) *
  exp(-2.0 * x0)`.
- Observation:
  five repeated direct runs were stable. Binary median report time was
  `0.750s`, e-graph size `8352`, `after_nodes=170`, `after_params=25`.
  Container median report time was `0.274s`, e-graph size `226`,
  `after_nodes=164`, `after_params=26`.
  A follow-up 30-pair alternating run and 100-run container-only stress loop
  still did not show intermittent container slowdown. Container-only report
  times were median `0.282s`, p95 `0.293s`, p99 `0.297s`, max `0.304s`,
  with zero runs above `0.5s`.
- Trace:
  binary spent most time in extraction (`0.477s` of `0.754s` traced total)
  after building a much larger graph. Container traced total was `0.271s`,
  with analysis `0.026s`, rewrite `0.129s`, extraction `0.092s`, and let
  `0.019s`. Top container rule family was `SOLE_MONOMIALS` flattening
  (`0.023s`, `286` matches), which is not enough to explain a large slowdown.
- Conclusion:
  the reported container runtime issue is not reproduced by this exact prefix
  under the current accepted rules. If a local notebook shows container much
  slower, it is likely running a different/longer expression, stale rule state,
  or a different checkout. The remaining current mismatch on this exact input is
  quality, not runtime: container extracts one more parameter (`26` vs `25`).

## 2026-05-04: recursive appended expression slowdown

- Status:
  diagnosed; no local rule change accepted.
- Notebook construction detail:
  `add_more_appended_exprs(5)` does not run appended indices `0..4`. Because
  `LAST_ADDED` is incremented inside the loop while the index expression also
  uses `LAST_ADDED + i`, it runs appended indices `[0, 2, 4, 6, 8]`.
- Reproduced observation:
  appended index `8` contains source rows `0..8` and is slow in container mode:
  binary `0.846s`, e-graph size `10725`, `after_nodes=237`,
  `after_params=37`; container `17.91s`, e-graph size `2107`,
  `after_nodes=201`, `after_params=32`.
- First divergence:
  appended index `5` is fast in container mode (`0.301s`, size `226`).
  appended index `6` is still moderate (`0.432s`, size `425`).
  appended index `7` jumps to `6.708s`, size `1663`.
  appended index `8` then jumps to `17.555s`, size `2107`.
- Smallest reproducer found:
  source rows `[6, 7]`, which are the `original` and `sympy` rows for
  `kotanchek/raw_index=4/Bingo/algo_row=4`, already reproduce the rule-family
  behavior: binary `0.636s`, size `6305`, `after_params=6`; container
  `0.739s`, size `817`, `after_params=5`.
  Adding row `0` amplifies the same mechanism: rows `[0, 6, 7]` run binary
  `0.901s`, size `12634`, `after_params=11`; container `6.377s`, size `2551`,
  `after_params=7`.
- Rule attribution:
  for rows `[6, 7]`, container rewrite time was `0.440s`; top families were
  `SOLE_MONOMIALS` flattening (`0.111s`, `1759` matches), Horner (`0.077s`,
  `1049` matches), coefficient-count factoring (`0.061s`, `436` matches),
  representative coefficient factoring (`0.020s`, `1690` matches), and exact
  nested flattening (`0.020s`, `2562` matches).
  for rows `[0, 6, 7]`, the same families dominate at larger scale:
  `SOLE_MONOMIALS` flattening (`1.700s`, `8738` matches), Horner (`1.245s`,
  `6012` matches), coefficient-count factoring (`0.615s`, `2459` matches),
  exact nested flattening (`0.201s`, `12518` matches).
- Hypothesis:
  the container encoding is slow here because the benchmark expression adds an
  original expression and its expanded/sympy equivalent. Exact flattening and
  coefficient factoring expose equivalent expanded and grouped presentations;
  `SOLE_MONOMIALS` then records aliases and Horner repeatedly factors shared
  terms. Row `0` shares terms such as `x0`, `x1`, and `x0^2/exp(x0^2)`, so it
  greatly amplifies the same opportunities.
- Falsified alternatives:
  row `8` alone is not slow (`0.117s`, size `106`), and pairwise `row8` with
  rows `0..7` is not enough to reproduce the large slowdown. Extraction is not
  the root cause: appended index `8` container trace spent `17.451s` in rewrite
  and only `0.745s` in extraction.
- Next useful probe:
  if this benchmark shape matters, optimize a general "original plus expanded
  equivalent" cycle, not row-specific constants. A likely direction is a guard
  that suppresses exact flattening or sole-monomial alias expansion when the
  expanded terms already coexist with the grouped term and no new lower-cost
  representation is introduced.

## 2026-05-04: scale/flatten cycle in recursive appended slowdown

- Status:
  mechanism narrowed; no rule change accepted.
- Smallest synthetic reproducer:
  `0.1 * (1.0 + 2.0*x0 + 3.0*x1) + (0.1 + 0.2*x0 + 0.3*x1)`.
  Binary runs in about `0.37s`, size `5029`, `after_params=1`.
  Containers run in about `4.0s`, size `3507`, `after_params=1`.
  The three-variable version
  `0.1 * (1.0 + 2.0*x0 + 3.0*x1 + 4.0*x2) + (0.1 + 0.2*x0 + 0.3*x1 + 0.4*x2)`
  is much worse: containers previously measured about `32.5s`, size `9521`,
  still with `after_params=1`.
- State inspection:
  after bounded rounds on the two-variable reproducer, `function_values(polynomial)`
  shows many rows with the same normalized value, including `P`,
  `0.1*polynomial(2P)`, `0.2*polynomial(P)`, `2*polynomial(0.5P)`,
  `4*polynomial(0.5P)`, and partially flattened mixtures. A serialized
  round-4 snapshot was written to `/tmp/param_eq_simple2_round4_serialized.json`;
  it had total size `99` with `80` serialized `polynomial` nodes in the
  truncated view.
- One-rule evidence:
  the integer-ratio coefficient factor rule alone transforms the mixed shape
  `0.1*P + 0.1 + 0.2*x0 + 0.3*x1` into
  `0.1*polynomial(P + 1 + 2*x0 + 3*x1)`. Running exact nested flatten next,
  then analysis, normalizes that into `0.1*polynomial(2 + 4*x0 + 6*x1)`.
  This proves a real scale/flatten cycle, but the two-rule chain alone stops
  after about `10` polynomial rows, so it is not the whole blowup.
- Second one-rule evidence:
  representative coefficient factoring alone transforms
  `P = 1 + 2*x0 + 3*x1` into
  `2*polynomial(0.5 + x0 + 1.5*x1)` and similarly creates fractional aliases
  for the mixed outer polynomial. Exact flatten and analysis then make these
  aliases equivalent to the original bodies. This explains the `0.5P` ladder
  seen in the full e-graph.
- Rejected semantic fix:
  guarding representative coefficient factoring against integer `coef` was
  not acceptable. It improved the recursive appended index `8` canary from
  about `17.7s` to `2.5s` with the same `32` params, but it made the minimal
  `simple_2` reproducer much worse: about `4.0s` to `47.9s`, size `3507` to
  `16267`. This falsifies "never factor integer representative coefficients"
  as a global fix; integer representative factoring can be an early compact
  alias even though it is also part of the cycle.
- Redundant-flatten guard probe:
  a local guard that tried to skip exact nested flattening when the expanded
  scaled body already exists did not match using `POLYNOMIAL_MONOMIALS`,
  because that table preserves an older raw body with numeric constants still
  in monomial keys. Using `POLYNOMIAL_BODIES` still could not prove the duplicate
  for the `0.1` example because map equality is exact over `f64` coefficients:
  `0.1 * 3.0` produces `0.30000000000000004`, while the flat input contains
  `0.3`.
- Floating-point evidence:
  f64 exactness is an amplifier, but not the only cause. A quarter-scale exact
  version ran in about `5.3s`, close to the `0.1` version, and a half-scale
  version timed out at `20s` in a subprocess probe. Rationals or coefficient
  rounding may help duplicate detection, but they will not by themselves remove
  the grouped/flattened scale-presentation cycle.
- Scheduler evidence:
  reducing the container backoff match limit is a real lever but not a complete
  semantic fix. For rows `[0, 6, 7]`, match limit `300` preserved `7` params and
  reduced runtime to about `1.7s`; match limit `200` reduced runtime further but
  regressed to `8` params. For appended index `8`, match limit `200` reduced
  runtime to about `2.8s` and preserved `32` params, while match limit `100`
  regressed to `33` params.
- Current hypothesis:
  the underlying cause is a cycle of equivalent presentations:
  `a*P + expanded(a*P)` can be represented as a mixed polynomial, a flat
  polynomial, a scaled nested polynomial, or nested scaled variants such as
  `a*polynomial(kP)`. The parameter extractor eventually picks the desired
  compact form, but saturation spends time materializing many equivalent
  scale/group/flatten choices. Lowering the container scheduler budget can
  avoid some of that search, but the safe budget appears case-dependent.
- Next probes:
  test a tolerance-aware or rounded coefficient subset detector for the
  redundant-flatten case, and separately test a staged schedule that runs
  broad scale factoring for a small fixed budget before the expensive
  flatten/Horner rules. Keep any candidate only if it preserves all container
  vs binary `after_params` outcomes on the artifact comparison.

## 2026-05-04: option comparison for removing the scale/flatten cycle

- Status:
  compared candidate interventions; no code change accepted.
- Option A, lower container backoff match limit:
  this budgets the cycle rather than removing it. Prior measurements showed
  rows `[0, 6, 7]` improved from about `5.96s` at limit `1000` to `1.72s` at
  limit `300` with the same `7` params, but limit `200` regressed to `8`
  params. appended index `8` improved from about `17.7s` to `2.8s` at limit
  `200` with the same `32` params, but limit `100` regressed to `33` params.
  This is plausible as a pragmatic scheduler override, but it does not prevent
  the semantic cycle and needs a full artifact after-param check before use.
- Option B, representative factor only when shallow param score decreases:
  tested by binding `score == shallow_polynomial_param_score(poly)` and
  `score1 == f64_param_score(coef) + shallow_polynomial_param_score(poly1)`.
  This strongly fixed the minimal cycle: `simple2` went to about `0.064s`,
  size `27`, `after_params=1`; `rows[6]+rows[7]` went to about `0.120s`,
  size `118`, `after_params=5`.
  Rejected because it lost real parameter wins: rows `[0, 6, 7]` regressed
  `7 -> 8` params and appended index `8` regressed `32 -> 35` params. This
  option is semantically clean, but only viable with compensating rules for the
  downstream wins that currently require a locally non-improving factor alias.
- Option C, block representative factoring only on all-free integer bodies:
  tested with `shallow_polynomial_param_score(poly) > 0` as a guard. This
  preserves the real appended params (`32`) and rows `[0, 6, 7]` params (`7`),
  but it makes the reduced cycle worse: `simple2` about `17.3s`, size `6686`,
  and `simple3` timed out at `30s`. Rejected. The all-free integer factor path
  is part of the cycle, but it can also create an early compact alias that keeps
  the reduced case from exploring even more.
- Option D, tolerance-aware duplicate expanded-body subsumption:
  tested a local rule for `a*P + expanded(a*P) -> 2a*P` using
  `POLYNOMIAL_BODIES` and `CONST_MERGE_TOLERANCE`, with `subsume=True`.
  The naive local rule was not sufficient: `simple2` worsened to about `9.6s`,
  size `4735`; `simple3` improved versus the current `32s` case but still took
  about `12s`; rows `[0, 6, 7]` improved to about `4.2s` with `7` params; and
  appended index `8` was essentially unchanged at about `17.7s`.
  This remains conceptually attractive, but the rule must be much more targeted
  or integrated into normalization; as an added rewrite it adds matching work
  and does not stop the later aliases soon enough.
- Option E, rational or rounded polynomial coefficients:
  evidence says this would help duplicate detection because exact `f64` map
  equality fails on cases like `0.1 * 3.0` versus `0.3`. It is not a complete
  fix: exact-scale variants such as the quarter and half examples still show
  large slowdowns. Rationals/rounding should be viewed as a support mechanism
  for canonical duplicate detection, not as the primary cycle breaker.
- Current conclusion:
  the cleanest route to preserving parameter quality while preventing blowup is
  probably two-stage:
  first, replace the broad representative coefficient rule with a local
  score-decreasing rule; second, add explicit replacements for the small set of
  downstream parameter wins that depended on locally non-improving aliases.
  The scheduler-limit option is simpler and may be good enough empirically, but
  it cannot guarantee the cycle is avoided. A duplicate-expanded subsumption
  rule should only be revisited if it can be made normalization-like, cheap, and
  proven to fire before exact flatten/Horner create further aliases.

## 2026-05-04: subsumption-first versus explicit-rule replacement trial

- Status:
  tried both requested directions and reverted both. `pipeline.py` is unchanged.
- Subsumption in analysis:
  tested a `subsume=True` analysis normalization for the exact duplicate shape
  `a*P + expanded(aP) + R -> 2a*P + R`, using `POLYNOMIAL_BODIES` plus
  `CONST_MERGE_TOLERANCE`.
  This is much stronger than the earlier naive rewrite form because the
  subsumption runs from analysis and can hide the redundant presentation before
  later rules match on it.
- Subsumption evidence:
  `simple2 = 0.1*(1 + 2*x0 + 3*x1) + (0.1 + 0.2*x0 + 0.3*x1)` improved to
  about `0.066s`, size `23`, `after_params=1`.
  `simple3 = 0.1*(1 + 2*x0 + 3*x1 + 4*x2) + (0.1 + 0.2*x0 + 0.3*x1 + 0.4*x2)`
  improved to about `0.064s`, size `25`, `after_params=1`.
  The real appended rows `[0, 6, 7]` improved from about `5.9s`, size `2551`,
  to about `4.3s`, size `1857`, with the same `7` params.
  The larger appended index `8` stayed essentially unchanged at about `17.3s`,
  size about `2142`, and `32` params.
  The smaller rows `[6, 7]` canary slowed slightly, about `0.74s -> 0.87s`,
  with unchanged `5` params.
- Subsumption conclusion:
  this is a plausible narrow cleanup for the direct grouped/expanded duplicate
  cycle, but it is not a complete fix for the large appended slowdown. The
  remaining slowdown is not only `a*P + expanded(aP)`; it also involves deeper
  global scale/factor aliases that the subsumption does not eliminate.
- Explicit-rule replacement:
  tested replacing broad representative coefficient factoring with local
  score-decreasing variants, then relying on existing explicit rules. Also
  tested combining that with the duplicate subsumption above.
- Explicit-rule evidence:
  the score-decreasing replacement made many runs fast:
  `simple2` about `0.061s`, size `27`, `after_params=1`;
  `rows[6]+rows[7]` about `0.107s`, size `118`, `after_params=5`;
  appended index `8` about `0.44s`, size `368`.
  It is not acceptable because it loses parameter reductions:
  rows `[0, 6, 7]` regressed `7 -> 8` params, and appended index `8`
  regressed `32 -> 35` params. Combining it with duplicate subsumption fixed
  the synthetic duplicate cases but did not recover those real parameter wins.
  A local score-scanning variant had the same issue: faster, but still `8`
  params on rows `[0, 6, 7]` and `35` params on appended index `8`.
- Explicit-rule conclusion:
  the missing behavior is not a small local rule like "pick a better coefficient
  in this polynomial." The accepted extraction for rows `[0, 6, 7]` depends on
  a non-local/global scale factor, for example extracting a shared
  `0.042813... * (...)` across heterogeneous nested terms. Replacing the broad
  representative alias with a maintainable finite set of small explicit rules
  would likely require a larger global beneficial-factor extraction rule, not
  a handful of simple cases.

## 2026-05-04: delayed-win and stop-condition research

- Status:
  investigated whether the current slowdown could be avoided by stopping on
  extraction stability instead of e-graph size stability. No code change
  accepted.
- Reduced duplicate-cycle evidence:
  On `simple2`, the extracted result reaches `after_params=1` by inner
  iteration `3`, but the graph keeps growing:
  iteration `3` size `62`, extract about `12ms`; iteration `10` size `3253`,
  extract about `703ms`.
  On `simple3`, the extracted result reaches `after_params=1` by inner
  iteration `3`, but the graph still grows through iteration `10`:
  size `66 -> 624`, extract about `16ms -> 158ms`.
  This confirms that the synthetic duplicate cycle is mostly wasted closure
  after the useful extraction is already present.
- Naive extraction-stability stop:
  stopping after two identical extracted results made the synthetic cases fast:
  `simple2` about `0.15s` instead of about `4.0s`, and `simple3` about `0.15s`
  instead of about `32s`.
  Rejected as a general stop condition because it misses delayed real wins:
  rows `[0, 6, 7]` stopped at `8` params instead of the accepted `7`;
  appended index `8` stopped at `33` params instead of the accepted `32`.
- Delayed real-win trace:
  rows `[0, 6, 7]` improves from `11 -> 10 -> 9 -> 8` params by iteration `4`,
  remains at `8` params through iteration `11`, and only reaches `7` params at
  iteration `12` when the global scale factor appears. Size at the delayed win
  is about `1306`.
  appended index `8` improves from `39 -> 37 -> 34 -> 33`, remains at `33`
  through iteration `8`, and only reaches `32` params at iteration `9`.
- Current conclusion:
  early extraction stability is useful as a diagnostic but unsafe as the main
  algorithm. The useful delayed wins require keeping some globally factored
  aliases alive long after the local expression has stabilized. A better fix
  should either represent those global factored views without feeding them back
  into generic polynomial rewrites, or compute the global factorization
  deterministically outside broad e-matching.

## 2026-05-04: rule-level best-scale and residual-scale experiments

- Status:
  tried the requested Python-rule implementation direction and reverted it.
  `pipeline.py` is unchanged.
- Best float-reducing scale as a rule:
  implemented a deterministic selector using nested `map-fold-kv`: for each
  coefficient candidate `c`, compute the coefficient paid-float gain from
  `P -> c*(P/c)`, tie-breaking by the number of integer quotients, and emit
  only one scale when the gain is positive.
- Rule-level selector evidence:
  unbounded and `poly.length <= 6` versions were too slow before the first
  canary completed. With `poly.length <= 4`, the selector fixed `simple2`
  (`~0.07s`, size `29`, `after_params=1`) and improved `rows[6]+rows[7]`
  (`~0.14s`, size `156`, `after_params=5`), but it missed the known delayed
  global wins: rows `[0, 6, 7]` stayed at `8` params and appended index `8`
  stayed at `35` params.
- Whole-polynomial residual scale as a rule:
  added a second Python-rule selector that chooses a scale when the divided
  coefficients contain a group whose residuals differ by exact integers, then
  relies on the existing residual-split rule to factor that divided body.
- Residual-scale evidence:
  the rule recovered or improved some parameter counts:
  rows `[0, 6, 7]` reached `6` params and rows `[6, 7]` reached `4` params.
  It was not acceptable: `simple3` regressed to about `34s`, size `5517`, and
  the extracted shape was suspicious (`0.4 * (1 + 2*x0 + 3*x1 + 4*x2)` for the
  reduced duplicate case where the expected scale is `0.2`). Appended index `8`
  still missed the accepted `32` params, reaching only `33`.
- Current conclusion:
  implementing the selector as nested Python-level map folds is the wrong
  mechanism. The bounded float-reducing rule validates the ranking idea on
  `simple2`, but the real delayed wins need a more global decoded-cost view.
  The residual-scale idea is powerful enough to find extra params, but as a
  broad rewrite it creates too many opportunities and needs either a carefully
  audited primitive or an extraction/opaque-view implementation where the
  chosen factored form does not feed back into generic polynomial rewrites.

## 2026-05-04: primitive best-common-float-scale experiment

- Status:
  implemented `map-best-common-float-scale` as a partial Rust primitive over
  `Map[_, f64]` and exposed it to Python as `map_best_common_float_scale`.
  Kept the primitive-backed `param_eq` coefficient factoring rule in the
  working tree for further investigation.
- Primitive behavior:
  the primitive rounds coefficients to a fixed decimal denominator, tries
  pairwise gcd-derived scales, and returns only a non-integer scale that turns
  at least two coefficients into small integer quotients while decreasing the
  approximate paid-float count. Otherwise the primitive returns no value.
- Focused validation:
  `cargo test --test files --features bin map -- --nocapture` passed.
  `uv run pytest python/tests/test_high_level.py::test_map_best_common_float_scale
  python/tests/test_high_level.py::test_map_best_common_float_scale_uses_useful_pair
  python/tests/test_high_level.py::test_map_best_common_float_scale_fails_without_useful_scale -q`
  passed.
- Pipeline failure list:
  `uv run pytest python/egglog/exp/param_eq/test_pipeline.py -q --tb=no`
  reported `26 failed, 122 passed, 15 skipped`. The failures are container-only
  for the changed rule families, plus unrelated/stale snapshot/scheduler tests.
- True primitive-related breakage:
  fourteen `test_rules[containers-...]` cases fail for binary-style arbitrary
  coefficient factoring such as `a*x+b -> a*(x+b/a)`, quotient-denominator
  factoring, and `(b+a*x)/(c+d*y)` style quotient factoring.
  Seven container simplification regressions lose parameter improvements:
  scaled polynomial factors, scaled terms in larger polynomials, nested log
  constant absorption, and exact nested log polynomial flattening.
  Two repeated-polynomial-product tests keep the expected parameter count but
  miss the expected node count.
- Direct primitive probes:
  `map_best_common_float_scale({3.14, 2.71})` fails, so the new rule cannot
  produce the old `3.14*(x+2.71/3.14)` witness. It also fails for the
  coefficient pairs in the scaled product and nested-log canaries:
  `{0.0077679147943854, -0.0477729775011539}`,
  `{0.003918940250258392, 0.7556389413872189}`, and the exact nested-log
  coefficient pairs. It succeeds for the intended duplicate-scale canaries:
  `{0.2, 0.4, 0.6000000000000001} -> 0.2`,
  `{0.1, 0.2, 0.333} -> 0.1`, and
  `{0.009239, -0.042748853, -0.009239} -> 0.009239`.
- Current conclusion:
  the primitive is correctly implementing the narrower "locally reduces paid
  float count" heuristic, but the existing container rule suite relies on a
  broader reachability rule that factors arbitrary representative coefficients
  even when the local float count does not improve. Replacing the old rule
  requires adding separate maintainable reachability rules for binary-style
  factoring, or changing the primitive/heuristic to return non-improving scales
  when those scales are needed for downstream rules.

## 2026-05-04: two-term reachability rule after primitive scale

- Status:
  accepted a narrow two-term representative-coefficient reachability rule in
  `container_basic_rules`, while keeping the profitable `map-best-common-float-scale`
  primitive as the first scale choice.
- Missing-case classification:
  the primitive does not cover locally neutral two-term factoring:
  `a*M + b*N -> a*(M + (b/a)*N)`. This broke binary-shaped container witnesses
  such as `a*x+b`, `a*x+b*y`, two-term denominator factoring, and two-term
  polynomial arguments under `log`. These cases are useful because later
  quotient/log/scaled-polynomial rules can consume the factored two-term form
  even though local float count does not improve.
- Rule shape:
  restored the old representative coefficient factoring only for
  `poly.length() == 2`, with at least one non-constant monomial and no
  non-constant coefficient already equal to `1.0`. Larger length-3/4
  polynomials still use the primitive or targeted coefficient rules.
- Focused test result:
  `uv run pytest test_rules + scaled polynomial/log/repeated product families`
  passed `95 passed`.
- Full pipeline result:
  `uv run pytest python/egglog/exp/param_eq/test_pipeline.py -q --tb=short`
  now has only three non-rule failures left:
  stale snapshot `test_end_to_end`, stale snapshot
  `test_constant_folding_containers`, and
  `test_container_self_factor_cycle_remains_extractable` referencing removed
  helper `_new_rewrite_scheduler`.
- Real-row samples:
  first 20 canonical rows vs previous rule:
  mean wall ratio `0.253`, mean report ratio `0.226`, mean size delta `-1.85`,
  and `0` after-param regressions.
  Spread sample of 16 rows across the corpus:
  mean wall ratio `0.283`, mean report ratio `0.246`, mean size delta `-0.06`,
  and `0` after-param regressions.
- Cumulative canaries:
  `simple2` improved from about `4.45s`, size `3507` to about `0.12s`, size
  `29`, with `after_params=1`.
  `simple3` improved from about `32.8s`, size `9521` to about `0.09s`, size
  `54`, with `after_params=1`.
  `rows_6_7` improved from `5` params to `4` params and from about `0.82s` to
  about `0.68s`.
  `appended8` preserved `32` params and similar runtime (`~18.6s` previous,
  `~19.1s` current) while shrinking size from `2107` to `1950`.
  `rows_0_6_7` remains the known gap: previous reached `7` params in about
  `4.28s`; current reaches `8` params in about `3.24s`.
- Current conclusion:
  the two-term rule is the right first narrow reachability replacement. It
  recovers the broad unit-test families and real-row param quality without
  reopening the synthetic duplicate-scale blowup. The remaining uncovered
  family is not local two-term factoring; it is a delayed global scale across
  an already-composed larger polynomial. That needs a separate global-scale
  rule or extraction-only view, and should not be folded into the two-term
  reachability rule.

## 2026-05-04: notebook appended run `Unextractable root`

- Status:
  investigated a notebook failure from `add_more_appended_exprs(10)` with
  `LAST_ADDED == 16`. No current-checkout reproduction found.
- Observed notebook error:
  `Unextractable root Value(...) with sort EqSort { name: "Num" }` while the
  progress bar was around 60%.
- Exact current-checkout probes:
  running container-only appended indices `16..25` in a fresh process succeeded.
  Running the notebook-shaped sequence `binary_report = run_paper_pipeline(...)`
  followed by `container_report = run_paper_pipeline_container(...)` for
  appended indices `0..25` reproduced one transient failure at appended index
  `18`, but repeated narrower probes did not reproduce it.
  Fresh `run_paper_pipeline(parse_expression(appended18))` succeeds with
  `before_params=94`.
  Fresh `EGraph().extract(parse_expression_container(appended18),
  include_cost=True, cost_model=container_cost_model)` succeeds with
  `ParamCost(90, 482)`.
  A longer history probe running all 714 source rows plus appended `0..17` in
  the same process still left the appended18 container initial extraction
  healthy.
- Falsified hypotheses:
  appended18 is not intrinsically unextractable in the current checkout.
  container-only history does not poison extraction.
  binary-only history does not poison extraction.
  the full original-row notebook history does not poison extraction in a fresh
  process.
- Current hypothesis:
  the saved notebook is using stale in-kernel function objects or scheduler
  state from an earlier `pipeline.py` version. The notebook imports
  `run_paper_pipeline` and `run_paper_pipeline_container` directly, so editing
  `pipeline.py` on disk does not update those names until the cell is rerun or
  the kernel is restarted.
- Recommended recovery:
  restart the notebook kernel and rerun the import/setup cells, or replace the
  direct function imports with a module import plus `importlib.reload(pipeline)`
  during experiments.

## 2026-05-04: appended initial custom-cost extraction root

- Status:
  fixed the repeated appended-expression `Unextractable root` failure at the
  initial `EGraph(...).extract(..., include_cost=True, cost_model=...)` step.
- Smallest reliable repro:
  in a fresh process, build recursive `APPENDED_EXPRS` from
  `load_original_results()` and repeatedly run only:
  `EGraph(save_egglog_string=False).extract(parse_expression(expr),
  include_cost=True, cost_model=param_cost_model)`.
  Before the fix this failed around appended index `13..19` with
  `ValueError: Unextractable root`.
- Falsified hypotheses:
  the expression itself was not intrinsically unextractable:
  `appended19` succeeded alone.
  a cost-model callback rejection was not the cause:
  a logging wrapper saw `0` callback exceptions before the extractor reported
  an unextractable root.
  the e-class was not missing entirely:
  the default extractor could extract the same root after the custom-cost
  extractor failed.
- Confirmed mechanism:
  disabling repeated-subexpression synthetic lets made the repeated initial
  extraction pass through appended39. The failure also disappeared with
  `save_egglog_string=True`, which uses the parse-and-run path rather than the
  direct command API. The failing boundary was therefore the combination of
  direct command registration, synthetic let factoring, and custom object-cost
  extraction. The fix registers the exact structural extraction root for
  custom-cost extraction instead of the synthetic-let optimized command form.
- Validation:
  added
  `test_repeated_initial_custom_cost_extraction_handles_appended_roots`.
  `uv run pytest
  python/egglog/exp/param_eq/test_pipeline.py::test_repeated_initial_custom_cost_extraction_handles_appended_roots
  -q` passed.
  The direct repeated initial extraction probe now succeeds through appended39.
- Appended performance after fix:
  with the then-current `45/30` container scheduler, appended0..79 all ran
  without extraction failures. No case approached the 5-minute timeout; by
  appended79 binary was about `2.8s` report time and containers about `4.4s`.
  Containers stayed much smaller (`~1.5k` total size vs binary `~17k`) and
  usually had fewer params, but report time was often higher after appended11.
- Scheduler budget probe:
  `30/20` was tested as a lower container budget. On the 714 retained corpus
  rows it had `0` failures and `0` after-param regressions, with mean runtime
  about `161ms`. On appended0..79 it was substantially faster than `45/30`;
  the max slowdown over binary was about `0.65s` at appended43, and many later
  cases were close to binary time. Remaining appended param regressions under
  `30/20` were indices `1..5` and `47`; indices `1..5` did not improve with
  larger backoff, so those are missing-rule or cost/extraction differences
  rather than search-budget differences.

## 2026-05-04: larger appended expressions container slowdown

- Status:
  investigated why larger synthetic appended expressions still run slower in
  container mode even though the container e-graph is much smaller and the
  extracted parameter count is better.
- Repro:
  built recursive appended expressions from `load_original_results()` in
  canonical row order and sampled appended indices `79, 100, 120, 140, 160,
  180, 200` with `/tmp/param_eq_large_sample.py`.
- Baseline observation with current `30/20` container scheduler:
  appended79: binary report `3.05s`, container report `3.67s`; binary size
  `16948`, container size `1456`; params `430 -> 419`.
  appended100: binary report `4.22s`, container report `5.79s`; binary size
  `17927`, container size `2347`; params `611 -> 594`.
  appended120: binary report `5.41s`, container report `7.21s`; binary size
  `21261`, container size `2863`; params `735 -> 704`.
  appended140: binary report `7.47s`, container report `9.10s`; binary size
  `21950`, container size `3028`; params `822 -> 774`.
  appended160: binary report `8.33s`, container report `9.81s`; binary size
  `22582`, container size `3195`; params `910 -> 851`.
  appended180 hit Python recursion depth in both variants, and appended200 hit
  Python parser nesting limits in both variants.
- Phase probe:
  `/tmp/param_eq_phase_probe.py --idx 100` and `--idx 140` split parse,
  initial before-cost extraction, pass input registration, analysis,
  rewrites, graph-size checks, pass extraction, pop, and render/decode.
- Key measurement for appended140:
  binary accounted wall `7.38s`, container accounted wall `11.01s`.
  Binary pass extraction was slower (`4.85s`) than container pass extraction
  (`2.28s`), so custom-cost extraction is not the relative container slowdown
  for this case.
  Container was slower in pass input registration (`2.20s` vs `0.70s`),
  analysis (`1.63s` vs `0.03s`), rewrites (`1.49s` vs `0.64s`), parse
  (`1.21s` vs `0.16s`), and before-cost extraction (`1.80s` vs `0.89s`).
  Container RunReport totals had only `8146` matches vs binary `109202`, but
  much higher merge/rebuild time (`0.99s` vs `0.02s`) and higher reported
  search/apply (`1.25s` vs `0.60s`).
- Falsified hypothesis:
  "containers are slower because custom-cost extraction dominates" is false on
  appended100 and appended140. Container extraction and callback time are lower
  than binary; the slowdown is in registering/map-lowering the extracted
  container current plus container analysis/rebuild/rewrite work.
- Rule-level probe:
  the top direct container rule cost was the whole-polynomial coefficient
  divisor witness rule:
  `a*M + b*N -> a * (M + (b/a)*N)`-style scaling generalized over a small
  polynomial. It took about `0.29s` on appended100 and `0.56s` on appended140.
  Temporarily disabling it reduced direct rewrite time but did not produce a
  clean total win: appended100 container pipeline was about `5.11s` vs `5.14s`
  baseline, while appended140 was about `7.87s` vs `7.71s` baseline and
  regressed params from `774` to `775`. The temporary guard was reverted.
- Secondary probe:
  a private exact-root let path that binds the pass input with
  `expr_to_let=False` reduced container `pass_let` substantially
  (`1.15s -> 0.34s` on appended100, `2.20s -> 0.53s` on appended140), with
  identical params in those probes. However it shifted some work into
  analysis/rebuild/extraction, so net report improvement was modest:
  appended100 container pipeline `5.14s -> 4.81s`;
  appended140 `7.71s -> 7.37s`. This is a promising overhead reduction, not a
  full explanation for the remaining slowdown.
- Current conclusion:
  the container representation helps the e-graph size and parameter quality,
  but the large appended expressions stress a different cost center: each
  container node carries larger map payloads, and the container analysis rules
  repeatedly fold/merge those payloads. Total e-node count is therefore not a
  good proxy for runtime on these cases. The next safe implementation target is
  the exact-root pass input helper, because it attacks measured overhead without
  changing rewrite semantics. The remaining larger issue is container
  analysis/rebuild over map payloads; removing a single broad rewrite rule is
  not supported by the evidence.

## 2026-05-04: relaxed parameter-parity tradeoff probes

- Status:
  diagnosis only. Temporary rule guards were reverted.
- Question:
  if container results do not need to always match or beat binary parameter
  counts, which levers most reduce runtime while keeping a reasonable
  better/same/worse split?
- Probe:
  `/tmp/param_eq_tradeoff_eval.py` ran a mixed sample of 14 canonical rows
  (`0,1,2,3,4,5,6,7,8,9,10,20,40,80`) and four synthetic appended rows
  (`8,40,79,100`). It reports container-vs-binary parameter outcomes,
  report-time ratios, and e-graph-size ratios.
- Baseline current container scheduler `30/20`:
  `5 / 13 / 0` parameter better/same/worse, mean report ratio `2.10`, total
  report ratio `1.01`, mean size ratio `0.28`. Appended rows were still
  parameter-better, but appended40/79/100 were about `1.10x`, `1.10x`, and
  `1.16x` binary runtime.
- Scheduler-only probes:
  `10/10`: `4 / 13 / 1` better/same/worse, total report ratio `0.52`.
  Appended runtime ratios dropped to about `0.26x`, `0.31x`, `0.57x`,
  `0.77x`; appended40 was the only worse parameter case in this sample
  (`+1` param).
  `5/10`: also `4 / 13 / 1`, total report ratio `0.52`. Appended runtime
  ratios were about `0.20x`, `0.33x`, `0.47x`, `0.65x`; appended40 again
  regressed by `+1` param.
  `1/5`: much faster (`0.43` total report ratio) but too lossy:
  `1 / 11 / 6` better/same/worse, with all four appended rows worse
  (`+2`, `+7`, `+8`, `+4` params).
- Rule ablation probes:
  narrowing broad representative coefficient factoring from `poly.length <= 4`
  to `<= 2` did not help the mixed sample. It kept `5 / 13 / 0`
  better/same/worse but total report ratio worsened to `1.09`; combined with
  `5/10` it was effectively the same as scheduler-only `5/10`.
  Temporarily disabling repeated scalar coefficient-subset factoring also did
  not help this mixed sample. It kept `5 / 13 / 0`, total report ratio stayed
  about `1.00`, and appended100 got slower (`1.40x` binary).
- Current conclusion:
  if parity can be relaxed, the highest-impact low-complexity lever is the
  container scheduler budget, not deleting one of the current broad container
  rules. `5/10` looks like the best measured starting point for "mostly keeps
  quality, makes larger synthetic sums much faster": it gives one worse case
  in this mixed sample while making all appended sample rows faster than
  binary. `1/5` crosses the quality cliff. Rule removals are less attractive
  because their costs are context-dependent: they can reduce direct match time
  but often shift work into analysis/rebuild/extraction or lose parameter wins
  without improving total runtime.

## 2026-05-04: strict score-improving representative-scale probe

- Status:
  diagnosis only. Temporary rule swap was reverted.
- Probe:
  disabled the broad representative coefficient factoring rule
  `poly.length <= 4; coef == poly2[poly2.pick_key()]` and enabled the parked
  strict `map_best_common_float_scale(poly)` rule, which only returns a scale
  when it locally decreases the approximate paid-float count.
- Same mixed sample as the relaxed tradeoff probe:
  14 canonical rows plus appended `8,40,79,100`.
- Result with the normal `30/20` container scheduler:
  current baseline: `5 / 13 / 0` parameter better/same/worse,
  mean report ratio `2.10`, total report ratio `1.01`, mean size ratio `0.28`.
  strict score-improving scale: `5 / 10 / 3`, mean report ratio `1.94`,
  total report ratio `1.22`, mean size ratio `0.21`.
  It made three canonical rows worse by `+1` param and made the total report
  time worse on this sample despite a lower mean ratio.
- Result with the faster `5/10` container scheduler:
  scheduler-only `5/10`: `4 / 13 / 1`, mean report ratio `1.94`,
  total report ratio `0.52`.
  strict score-improving scale plus `5/10`: `4 / 10 / 4`, mean report ratio
  `1.81`, total report ratio `0.52`.
  It did not improve total time over scheduler-only `5/10`, and it increased
  parameter regressions from one row to four rows.
- Current conclusion:
  the strict score-improving replacement is not a good performance tradeoff in
  the current rule suite. It reduces some local e-graph size, but it removes
  non-local reachability witnesses that later rules use, while not improving
  total runtime beyond what a lower scheduler budget already achieves. If
  parameter parity is relaxed, scheduler `5/10` is still the cleaner fast-mode
  knob than this semantic cut.
