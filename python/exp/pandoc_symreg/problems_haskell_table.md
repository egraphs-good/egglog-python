# Haskell `problems` Table From `pandoc-symreg`

This note records a direct run against the Haskell source implementation in `/Users/saul/p/pandoc-symreg`, using the built `pandoc-symreg` executable from that repo.

## What was run

- Input file: `/Users/saul/p/pandoc-symreg/problems`
- Parser mode: `tir`
- Output mode: `latex`
- Simplifier: Haskell `Data.SRTree.EqSat.simplifyEqSat` via `--simplify`
- Safety guard: one row at a time, no parallelism, hard timeout per row

The timeout used for the final run below was `30` seconds per row.

## Reproduction sketch

These are the rough commands, not a verbatim log.

Build the Haskell executable:

```bash
cd /path/to/pandoc-symreg
stack build
```

Render the full file without simplification:

```bash
stack exec -- pandoc-symreg --from tir --to latex --input problems
```

To test simplification safely, run one line at a time with a timeout instead of passing the whole file to `--simplify`:

```bash
line="$(sed -n "${ROW}p" problems)"
printf '%s\n' "$line" > /tmp/one-problem.tir
stack exec -- pandoc-symreg --from tir --to latex --input /tmp/one-problem.tir --simplify
```

Wrap that last command in a timeout and repeat for each row. That is how the table below was produced.

## Result summary

Every row parsed and rendered to LaTeX immediately.

Every row still timed out under Haskell EqSat simplification, even with a `30s` timeout:

- row 1: timeout at `30.543s`
- row 2: timeout at `30.310s`
- row 3: timeout at `30.451s`
- row 4: timeout at `30.408s`
- row 5: timeout at `30.615s`

## GitHub / MathJax rendering version

GitHub's documented math rendering uses inline `$...$`, block `$$...$$`, or fenced `math` blocks. To avoid table-cell rendering issues, the expressions below use fenced `math` blocks.

### Row 1

Input:

```math
\left(2.82238 + \left(3.092415 * \left(\operatorname{sin}(\operatorname{log}(\left |0.0\right |)) * \left(\left(-0.162842 * x_{, 2}\right) - \left(0.116404 * x_{, 1}\right)\right)\right)\right)\right)
```

Simplified output: timeout after `30.543s`

### Row 2

Input:

```math
\operatorname{sin}(\operatorname{log}(0.0))
```

Simplified output: timeout after `30.310s`

### Row 3

Input:

```math
\left(-1.0 * \operatorname{exp}(\operatorname{log}(\left |\left(-1.3 * \left(x_{, 1} - \left(1.2 * x_{, 2}\right)\right)\right)\right |))\right)
```

Simplified output: timeout after `30.451s`

### Row 4

Input:

```math
\left(-1.0 * \operatorname{exp}(\operatorname{log}(\left |\left(\left(-1.3 * x_{, 1}\right) + \left(1.56 * x_{, 2}\right)\right)\right |))\right)
```

Simplified output: timeout after `30.408s`

### Row 5

Input:

```math
\left(-1.0 * \operatorname{exp}(\operatorname{log}(\left |\left(\left(0.256 * x_{, 3}\right) + \left(-0.2561 * x_{, 2}\right)\right)\right |))\right)
```

Simplified output: timeout after `30.615s`
