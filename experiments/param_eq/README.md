# Param-Eq paused-research harness

This directory preserves the optional corpus experiment without installing it
as part of `egglog`. The reusable expression domain and simplifier live in
`python/egglog/exp/param_eq`; this directory owns external-data loading,
isolated row execution, resource limits, aggregation, and the research handoff.

The work is paused. The bounded public demonstrations remain maintained in CI,
while the private 714-row corpus is not run automatically.
The checked-in result CSVs contain headers only as schema examples; a manifest
is created only by a final dependency-compatible aggregate run.

## Provenance and redistribution boundary

This work reimplements the method published by Fabrício Olivetti de França and
Gabriel Kronberger in [Reducing Overparameterization of Symbolic Regression
Models with Equality Saturation](https://doi.org/10.1145/3583131.3590346).
Fabrício provided the original Haskell experiment repository and a separate
`pandoc-symreg` archive in personal correspondence. Those private files were
used for behavioral validation but are not redistributed here. The Python
implementation reproduces the published method without copying the prototype's
source text; checked-in result artifacts contain only aggregate measurements
and no source expressions.

The supplied archive has placeholder copyright/author metadata, so attribution
alone is not sufficient redistribution permission. Keep it outside this
repository and configure it explicitly:

```bash
export EGGLOG_PARAM_EQ_DATA_DIR=/absolute/path/to/param-eq-haskell
export EGGLOG_PARAM_EQ_EXPECTED_ARCHIVE_SHA256=<privately-recorded-sha256>
```

The expected hash is intentionally not checked into this public tree because
it fingerprints private material. Recover it from private research records
before resuming. If no prior value can be recovered, compute the current
candidate with `external_archive_hash()` only after independently verifying the
archive, store that value privately, and use it for every subsequent run.

The loader applies the retained-paper policy by name: FEAT is omitted, SRjl and
GOMEA are displayed as PySR and GP-GOMEA, known unusable raw rows are omitted,
and rows without rank data are excluded.

## Installed stress demo

Run either retained representation on one public expression:

```bash
python -m egglog.exp.param_eq --expr '2.3 * (3.7*x0 + 5.1*x1) / 7.9' --variant container
```

The restricted Python-like syntax accepts finite numeric literals, variables,
`+`, `-`, `*`, `/`, literal exponents, and the unary functions `abs`, `exp`,
`log`, `sqrt`, `plog`, `square`, and `cube`. The JSON report has status
`saturated` when every inner schedule can stop, or `iteration_limit` when the
retained 30-round boundary is reached. In either case the extracted expression
is available for independent checking; only saturated corpus rows contribute
to aggregates. The container variant rejects inputs whose coefficient
normalization produces a non-finite `f64` value. See `NOTES.md` for fidelity
and semantic limits.

## External corpus commands

```bash
make -C experiments/param_eq smoke
make -C experiments/param_eq binary
make -C experiments/param_eq container
make -C experiments/param_eq haskell
make -C experiments/param_eq aggregate
```

`binary` and `container` are useful for focused work and write expression-free
row metrics only under the ignored `results/raw/` directory. `aggregate` runs
the paired mode, alternating variant order by stable row hash, validates
identities, configuration, and input hashes, then replaces the tracked
aggregate CSVs and manifest. Never add files from `results/raw/`, the external
archive, source expressions, extracted expressions, or private absolute paths.

The raw `external_archive_sha256` column intentionally repeats one hash of the
corpus inputs, Haskell source modules, and Stack/Cabal lock/configuration files;
it is not a hash of an individual expression. Raw rows
also record the Python and Rust versions, platform, CPU, declared debug/release
mode, requested and memory-capped worker counts, stable ordering seed, clean
source-checkout commits, and the SHA-256 of the loaded native extension.
Aggregation refuses to publish when those fields disagree, a recorded
worktree is dirty, the current repository is not the recorded producer commit,
or an identity is missing. The native hash identifies the code that executed;
the source commits remain procedural provenance, so rebuild the extension from
those clean checkouts before running. Set `EXECUTION_MODE=debug` on the make
command only when the installed extension was actually built in debug mode;
the default is `release`.

Full timing comparisons are single-machine exploratory measurements. The
runner isolates each row, records iteration limits, timeouts, and errors instead
of dropping them, and alternates binary/container order by stable row hash when
`--variant both` is used. Aggregate rows retain separate counts for iteration,
timeout, memory, and execution failures. Ratio summaries omit pairs whose
binary denominator is zero and expose the remaining sample as `n_ratio`.

The optional `haskell` target compiles one temporary runner against the
author-supplied implementation, then executes it once per isolated row. The
one-time Stack/GHC build is outside the per-row timer and has its own 600-second
guard; override that setup boundary with `--haskell-build-timeout-sec` when
calling the runner directly. The generated program looks up rows by public
metadata rather than embedding expression text, forces the input counts before
starting the clock, and forces both result counts before stopping it. Stack is
not installed or invoked by CI; program generation and expression-free output
parsing are unit tested. Haskell has no container representation, so this route
supports only `--variant binary`. Its expression-free CSV remains a local raw
diagnostic: `aggregate` does not currently publish a Haskell aggregate. Add
and validate a separate aggregate path before presenting those measurements.
