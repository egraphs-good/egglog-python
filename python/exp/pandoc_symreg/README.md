# Pandoc-Symreg EqSat Replication

This folder packages a self-contained Egglog replication of the `pandoc-symreg` equality-saturation pipeline, followed by a multiset-based A/C experiment.

The source repo used for provenance is the local clone at `/Users/saul/p/pandoc-symreg`. Runtime does not depend on that checkout. The copied rule families come from `/Users/saul/p/pandoc-symreg/src/Data/SRTree/EqSat.hs`:

- `rewritesBasic`
- `constReduction`
- `constFusion`
- `rewritesFun`

The reusable implementation lives in `/Users/saul/p/egg-smol-python/python/egglog/exp/pandoc_symreg.py`.

## Chosen witnesses

- `erro:1`
  - Small sanity case that does reduce parameter count under the copied EqSat pipeline.
- `problems:4`
  - Small readable A/C-heavy example inside `Abs(Log(Exp(...)))`.
- `examples/feynman_I_6_2.hl:11`
  - Larger A/C stress case with additive and multiplicative structure around nonlinear atoms.
- `examples/example.pysr:3`
  - Optional extra stress case.

## Commands

Run the core comparison set:

```bash
uv run --project /Users/saul/p/egg-smol-python python /Users/saul/p/egg-smol-python/python/exp/pandoc_symreg/run_compare.py
```

Run only the dramatic witness:

```bash
uv run --project /Users/saul/p/egg-smol-python python /Users/saul/p/egg-smol-python/python/exp/pandoc_symreg/run_compare.py --witness dramatic
```

Run only the binary replication:

```bash
uv run --project /Users/saul/p/egg-smol-python python /Users/saul/p/egg-smol-python/python/exp/pandoc_symreg/run_compare.py --mode binary
```

Execute the tutorial doc-script directly:

```bash
uv run --project /Users/saul/p/egg-smol-python python /Users/saul/p/egg-smol-python/docs/tutorials/pandoc_symreg_ac_multiset.py
```

Run the focused tests:

```bash
uv run --project /Users/saul/p/egg-smol-python pytest /Users/saul/p/egg-smol-python/python/tests/test_pandoc_symreg.py
```

## Current conclusions

The baseline Egglog replication is working on the selected examples and uses the paper-aligned post-extraction metric: convert non-integer float constants to parameters, then count parameters and compare to Jacobian rank.

Current observed results from the implementation:

- `erro:1` reproduces the kind of parameter reduction reported in the paper.
  - Binary EqSat reduces the parameter count from `4` to `2`, which is a `0.5` reduction ratio.
  - Multisets preserve the same simplified result and shrink total e-graph size from `17` to `14`.
- `problems:4` is a readable A/C stress case rather than a positive parameter-reduction case.
  - Binary and multiset runs extract the same final expression and both keep the parameter count at `2`.
  - Multisets reduce total e-graph size from `49` to `18`.
- `feynman_I_6_2.hl:11` is the strongest current A/C stress witness.
  - Binary and multiset runs keep the parameter count at `11` and the same extracted cost `107`.
  - Multisets reduce total e-graph size from `338` to `59`.
  - Runtime is not yet better: the current multiset pipeline is still slower on this witness because it containerizes A/C structure and then reruns the binary rules to preserve downstream simplifications.
- `example.pysr:3` shows the same pattern.
  - Total size drops from `75` to `51`, but the parameter count stays at `10`.

Across all of the current witnesses, sampled numerical error is `0.0` between the original expression and the extracted result.

## Notes on the multiset path

The multiset implementation is intentionally partial.

- It ports additive and multiplicative A/C structure into `sum_(MultiSet[Term])` and `product_(MultiSet[Term])`.
- It ports constant combining and one distributive expansion rule into the container world.
- It then reruns the copied binary rules on the extracted result so the rest of the EqSat pipeline still fires.

That means the multiset section is already useful for measuring blow-up reduction, but it is not yet a pure container-native replacement for every downstream rule.
