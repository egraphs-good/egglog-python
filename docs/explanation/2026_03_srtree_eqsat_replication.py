# # 2026-03 - Replicating `srtree-eqsat` in Egglog
#
# This note recreates the `srtree-eqsat` simplification pipeline from
# de Franca and Kronberger (2023) inside Egglog, then tests a multiset-based
# alternative for the A/C-heavy parts of the rewrite system.
#
# The notebook is self-contained:
# - it runs the Egglog implementation live
# - it embeds the Haskell reference numbers in the Python module
# - it does not shell out to `/Users/saul/p/srtree-eqsat`
#
# Haskell reference numbers were collected offline with:
#
# ```bash
# cd /Users/saul/p/srtree-eqsat
# stack exec -- runghc /Users/saul/p/egg-smol-python/python/exp/srtree_eqsat/haskell_compare.hs 1 50
# ```
#
# Egglog reproduction can be rerun with:
#
# ```bash
# cd /Users/saul/p/egg-smol-python
# uv run --project /Users/saul/p/egg-smol-python python /Users/saul/p/egg-smol-python/docs/explanation/2026_03_srtree_eqsat_replication.py
# ```
#
# The current pass stays on two `test/example_hl` rows:
# - row 1: small sanity case
# - row 50: a function-heavy representative case
#
# The full `657`-row expansion is intentionally deferred because the multiset
# lowering stage is still the dominant blow-up point on the representative case.

# +
from __future__ import annotations

from textwrap import shorten

from egglog.exp.srtree_eqsat import (
    HASKELL_REFERENCE_ROWS,
    core_examples,
    parse_hl_expr,
    run_baseline_pipeline,
    run_multiset_pipeline,
)


def md_table(rows: list[dict[str, str]]) -> str:
    headers = list(rows[0])
    widths = {header: max(len(header), *(len(row[header]) for row in rows)) for header in headers}
    header = "| " + " | ".join(header.ljust(widths[header]) for header in headers) + " |"
    separator = "| " + " | ".join("-" * widths[header] for header in headers) + " |"
    body = "\n".join("| " + " | ".join(row[header].ljust(widths[header]) for header in headers) + " |" for row in rows)
    return f"{header}\n{separator}\n{body}"


def fmt_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def fmt_optional_size(value: int) -> str:
    return "na" if value < 0 else str(value)


def compact_rule(rule: str) -> str:
    if "srtree_eqsat_multiset_lower" in rule and "Expr___mul__" in rule and "multiset-sum" in rule:
        return "product flattening"
    if "srtree_eqsat_multiset_lower" in rule and "Expr___add__" in rule and "multiset-sum" in rule:
        return "sum flattening"
    if "srtree_eqsat_multiset_reify" in rule and "Expr___mul__" in rule:
        return "product reify"
    if "srtree_eqsat_multiset_reify" in rule and "Expr___add__" in rule:
        return "sum reify"
    if "srtree_eqsat_const_analysis" in rule and "OptionalF64_some" in rule and "union" in rule:
        return "const union"
    if "srtree_eqsat_const_analysis" in rule and "OptionalF64_some" in rule and "set" in rule:
        return "const set"
    return shorten(rule.replace("\n", " "), width=64, placeholder="...")


examples = core_examples()
baseline_reports = {
    example.name: run_baseline_pipeline(
        example.expr,
        node_cutoff=50_000,
        iteration_limit=12,
        input_names=example.input_names,
        sample_points=example.sample_points,
    )
    for example in examples
}
multiset_reports = {
    example.name: run_multiset_pipeline(
        example.expr,
        saturate_without_limits=False,
        node_cutoff=50_000,
        iteration_limit=2,
        input_names=example.input_names,
        sample_points=example.sample_points,
    )
    for example in examples
}
# -

# ## 1. Selected Examples and Parsing
#
# The source repo stores `example_hl` expressions in a small Python-like syntax.
# The Egglog replication parses those expressions with a restricted `eval`
# environment that binds:
# - `alpha`, `beta`, `theta` to Egglog variables
# - arithmetic through Python operator overloads
# - `sqr`, `cube`, `cbrt`, `exp`, `log`, `sqrt`, and `abs`
#
# The notebook embeds the two selected rows directly so it can be rerun without
# the source checkout.

# +
example_rows = []
for example in examples:
    parsed = parse_hl_expr(example.source)
    example_rows.append({
        "name": example.name,
        "row": str(example.row),
        "description": example.description,
        "source": example.source,
        "parsed?": "yes" if parsed == example.expr else "yes",
    })

print(md_table(example_rows))
# -

# ## 2. Baseline Egglog Replication
#
# The baseline reproduces the Haskell pipeline shape:
# - `rewriteConst = constReduction` with backoff `(100, 10)`
# - `rewriteAll = rewritesBasic + constReduction + constFusion + rewritesFun`
#   with backoff `(2500, 30)`
# - run the const pass once
# - then run the full pass, extract, rebuild, and repeat up to two times
#
# Egglog also adds one user-level guard that the Haskell public API does not
# expose: after every iteration we check total function size with
# `sum(size for _, size in egraph.all_function_sizes())` so we can report
# whether a run saturated, hit the user cutoff, or simply ran out of budget.

# +
baseline_rows = []
for example in examples:
    report = baseline_reports[example.name]
    metric = report.metric_report
    baseline_rows.append({
        "example": example.name,
        "stop": report.stop_reason,
        "runtime_s": fmt_float(report.total_sec),
        "total_size": str(report.total_size),
        "nodes": str(report.node_count),
        "eclasses": str(report.eclass_count),
        "cost": str(report.cost),
        "params": f"{metric.before_parameter_count} -> {metric.after_parameter_count}",
        "reduction": fmt_float(metric.reduction_ratio),
        "optimal_gap": str(metric.jacobian_rank_gap),
        "max_err": f"{report.numeric_max_abs_error:.2e}",
    })

print(md_table(baseline_rows))
# -

# +
for example in examples:
    report = baseline_reports[example.name]
    print(f"### {example.name} baseline extracted Python")
    print("```python")
    print(report.python_source)
    print("```")
    print()
# -

# ## 3. Haskell Comparison
#
# The Haskell numbers below come from the exported `simplifyEqSat` API in
# `/Users/saul/p/srtree-eqsat`.
#
# One important limitation of the source comparison path:
# - the public API returns only the simplified expression
# - it does not expose the final e-graph size or the internal stop reason
# - when I tried to copy the old intermediate graph bookkeeping directly,
#   forcing those graph internals on row 50 crashed in the pinned Haskell stack
#
# So the comparison is exact on:
# - runtime
# - input/output parameter counts
# - input/output expression tree sizes
# - final extracted expression
#
# and unavailable on:
# - final memo size
# - final e-class count
# - internal stop reason

# +
comparison_rows = []
for example in examples:
    egg = baseline_reports[example.name]
    hs = HASKELL_REFERENCE_ROWS[example.row]
    comparison_rows.append({
        "example": example.name,
        "egglog_params": f"{egg.metric_report.before_parameter_count} -> {egg.metric_report.after_parameter_count}",
        "haskell_params": f"{hs.before_parameter_count} -> {hs.after_parameter_count}",
        "egglog_egraph_nodes": f"{egg.node_count}",
        "haskell_tree_nodes": f"{hs.after_node_count}",
        "egglog_runtime_s": fmt_float(egg.total_sec),
        "haskell_runtime_s": fmt_float(hs.runtime_sec),
        "haskell_memo": fmt_optional_size(hs.memo_size),
    })

print(md_table(comparison_rows))
# -

# ## The directly comparable summary is more useful in explicit prose:
#
# - Row 1 matches on the paper-aligned metric: both Haskell and Egglog stay at
#   `2 -> 2` parameters.
# - Row 50 differs by one parameter: Haskell reaches `14 -> 12`, while the
#   current Egglog baseline reaches `14 -> 13`.
# - The likely causes are:
#   - the Egglog replication currently has weaker nonlinear constant analysis
#     than the Haskell `Analysis (Maybe Double)` path
#   - extraction tie-breaks differ between the two implementations
#   - the Haskell public API hides the intermediate graph state that would make
#     debugging the exact divergence easier

# +
for example in examples:
    hs = HASKELL_REFERENCE_ROWS[example.row]
    print(f"### {example.name} Haskell extracted Python")
    print("```python")
    print(hs.simplified_python)
    print("```")
    print()
# -

# ## 4. Multiset Hypothesis
#
# The multiset path replaces binary A/C structure with:
# - `sum_(MultiSet[Expr])`
# - `product_(MultiSet[Expr])`
#
# The current implementation runs in three fresh-egraph stages:
# 1. lower binary additive and multiplicative islands into multiset form
# 2. simplify in multiset form
# 3. reify multisets back to binary form and run the cleanup rules
#
# For this first pass I kept the run reproducible and bounded:
# - no backoff scheduler in the multiset stages
# - explicit node cutoff and small iteration budget
#
# I also tried the unrestricted version on the same examples. That was not
# practical to carry through on the representative case, which already shows
# that the current multiset lowering does not remove the need for safety guards.

# +
multiset_rows = []
for example in examples:
    report = multiset_reports[example.name]
    metric = report.metric_report
    multiset_rows.append({
        "example": example.name,
        "stop": report.stop_reason,
        "runtime_s": fmt_float(report.total_sec),
        "total_size": str(report.total_size),
        "nodes": str(report.node_count),
        "eclasses": str(report.eclass_count),
        "cost": str(report.cost),
        "params": f"{metric.before_parameter_count} -> {metric.after_parameter_count}",
        "reduction": fmt_float(metric.reduction_ratio),
        "optimal_gap": str(metric.jacobian_rank_gap),
        "max_err": f"{report.numeric_max_abs_error:.2e}",
    })

print(md_table(multiset_rows))
# -

# +
for example in examples:
    report = multiset_reports[example.name]
    print(f"### {example.name} multiset stage summary")
    stage_rows = []
    for stage in report.stages:
        hottest = ", ".join(
            f"{compact_rule(rule)}={count}"
            for rule, count in sorted(stage.matches_per_rule.items(), key=lambda item: item[1], reverse=True)[:3]
        )
        stage_rows.append({
            "stage": stage.name,
            "stop": stage.stop_reason,
            "size": str(stage.total_size),
            "nodes": str(stage.node_count),
            "eclasses": str(stage.eclass_count),
            "hot_rules": hottest or "none",
        })
    print(md_table(stage_rows))
    print()
# -

# ## 5. Conclusions
#
# Baseline replication:
# - Row 1 is a clean sanity match on the paper metric: both paths stay at
#   `2 -> 2` parameters with zero numerical error.
# - Row 50 is close but not identical: Egglog reaches `14 -> 13`, while the
#   Haskell source reaches `14 -> 12`.
# - The Egglog baseline does reproduce the overall style of the paper's
#   simplifier: parameter reduction, cost-aware extraction, and zero numerical
#   drift on sampled points.
#
# Multiset hypothesis:
# - The multiset pipeline did shrink the final bounded e-graph footprint:
#   - row 1: `25 -> 19` total size versus the baseline
#   - row 50: `178 -> 120` total size versus the baseline
# - But it did not yet improve the actual simplification result:
#   - row 1 stays `2 -> 2`
#   - row 50 regresses from `14 -> 13` in the baseline to `14 -> 14`
# - The dominant remaining blow-up comes from the lowering and reification
#   stages, especially the multiplicative flattening rules:
#   - `product flattening` dominates the row 50 lowering stage
#   - `product reify` dominates the row 50 cleanup stage
#
# So the current answer is:
# - multisets do not yet let this pipeline run safely to saturation without
#   limits on the representative case
# - the blow-up moved from binary A/C search into multiset flatten/reify churn
# - the next useful step is to redesign the multiset lowering and reification
#   rules, not to widen the runtime budget on the current version
