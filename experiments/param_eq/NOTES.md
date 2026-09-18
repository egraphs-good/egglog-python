# Param-Eq research handoff

## What is retained

The binary pipeline is the fidelity baseline for de França and Kronberger's
parameter-reducing rewrite method. It uses a lexicographic extraction cost:
noninteger floating constants first, then expression nodes. Each outer pass
builds equivalences, extracts the best expression, and repeats once if the
representative changed.

The container variant represents monomials and polynomials as nested
`Map`/`BigRat` values. It was introduced to make associative/commutative
polynomial structure canonical and to avoid enumerating every binary tree.
Only the general higher-order container operations needed by the retained
pipelines and public demonstrations should remain in `egglog-experimental`.
Argument-order adaptations use lambda-created primitives; flipped duplicate
builtins are not part of the retained design.

The final public-case ablation retained small-polynomial coefficient factoring,
Horner factoring, and exact nested-polynomial flattening from the container
basic rules, together with constant/index collection and polynomial
normalization from the analysis rules. The specialized integer-residual,
subset-scale, and corpus-tail rules were unnecessary for all three public cases,
so the installed pipeline no longer references their primitives.
Literal float exponents are converted on the Python side with
`float.as_integer_ratio()`, constant-map collisions use a general old-value
merge, and rational float powers are composed from ordinary exponentiation and
`BigRat.to_f64()`.

Container lowering distributes a literal scalar or one monomial into a single
polynomial and combines coefficient collisions. Polynomial-by-polynomial and
monomial-over-polynomial forms stay nested. This makes the public Equation 4
case canonical without broadening the nested-flatten rewrite or recreating its
factor/flatten cycle.

Both variants now use ordinary persistent upstream backoff. This is close to,
but not identical to, the prototype scheduler and is a documented fidelity
boundary rather than an unverified claim of exact replication.

The binary repeated-monomial public stress case is the known boundary for the
retained 30-round inner limit. When it reaches that limit, CI still verifies
every sample point of its extracted expression, and the corpus runner records
it as `iteration_limit` instead of including it in numeric summaries. CI also
accepts saturation if scheduler behavior improves. Revisit scheduler parity
before resuming corpus measurements if the limit still recurs.

## Semantic limits

The retained rules target real-valued expressions on inputs where the source
and every introduced subexpression are defined. Guards cover the literal and
structural domain boundaries needed by the retained cases, but the pipeline has
no general sign or interval analysis. CI compares every configured finite
sample point; those checks are regression evidence, not a universal proof of
equivalence. The binary rules omit cancellations whose only justification is
current e-class disequality: a later merge could invalidate that test while
leaving an incorrect finite result. The remaining guarded logarithm rules have
the same positive-input domain on both sides; their disequality guard only
avoids introducing an already undefined term. The container representation
still combines equal bases algebraically, so expressions with no defined
inputs, such as `(x - x) / (x - x)`, are outside the comparison contract and
may normalize differently between representations. Before broadening that
contract, add monotone per-e-class definedness/nonzero tracking. The container
variant fails explicitly if coefficient normalization becomes non-finite.
Treat external-corpus measurements the same way.

## Performance evidence worth preserving

The last pre-pause corpus artifacts repeatedly showed a smaller final container
e-graph alongside a slower wall time. Profiling indicated several reasons worth
testing later:

- final `egraph_total_size` is not peak or cumulative work;
- constructing rules/declarations and materializing higher-order operations can
  dominate small final graphs;
- embedded `Map` payload and custom extraction costs are not represented by a
  simple row count;
- some exact flatten/scale rules create large intermediate polynomial spaces or
  normalization cycles.

These are hypotheses supported by local probes, not causal or portable
performance conclusions. The stale row-level artifacts and chronological debug
transcript were deliberately removed, and no result CSVs remain checked in. A
final dependency-compatible aggregation must write and verify them before
publishing numerical corpus results.

## Rejected or parked directions

- Exact flattening and representative-scale rules improved isolated shapes but
  created cycles or bad tail cases.
- Increasing the container match budget restored some reachability but made the
  long tail worse.
- Final e-graph size alone was not a useful stop or performance proxy.
- Rank-miss-specific rules were not retained as a second undocumented rule set.
- The common-float-scale and integer-residual campaigns are recorded only as
  rejected directions here; their commented implementations and specialized
  helper APIs were deliberately removed from the installed package.

## Restart checklist

1. Record the Python, `egglog`, and `egglog-experimental` commits and rebuild the
   extension in release mode. The runner records the clean source checkouts and
   hashes the loaded native extension; the hash detects a changed executable
   but does not itself prove which checkout produced it.
2. Run the three public cases in both variants and keep their independent
   numeric checks green. Record whether binary `repeated_monomial` still reaches
   `iteration_limit`; the other reports should be `saturated`. Resolve a
   recurring limit before publishing new corpus measurements.
3. Set `EGGLOG_PARAM_EQ_DATA_DIR` to the private archive and
   `EGGLOG_PARAM_EQ_EXPECTED_ARCHIVE_SHA256` to the stable value recovered from
   private research records. The runner refuses an absent or mismatched hash.
4. Run binary and container rows with the same time/memory limits and inspect
   every iteration-limit/timeout/memory-limit/error count.
   Optionally run `make -C experiments/param_eq haskell`; this live baseline
   compiles its temporary runner once, forces both result counts inside the
   timed region, and requires Stack plus the external Haskell checkout. Its raw
   output is diagnostic only; no tracked Haskell aggregate path is maintained.
5. Generate aggregate-only outputs and verify the manifest and paired hashes.
   Confirm its Python and Rust versions, platform, CPU, execution mode,
   requested/effective workers, ordering seed, loaded-extension hash, and
   per-variant status counts;
   aggregation rejects dirty or unidentified dependency worktrees and a dirty
   egglog-python checkout.
6. Investigate peak/cumulative work and extraction payload costs before adding
   more rewrite rules.
7. Obtain corrected license/redistribution permission before publishing any
   source or row-level material.
