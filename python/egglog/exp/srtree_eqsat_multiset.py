"""
Experimental multiset variant of the srtree-eqsat translation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast

import egglog
from egglog import *
from egglog.exp import srtree_eqsat as base

__all__ = [
    "multiset_lower_rules",
    "multiset_reify_rules",
    "multiset_simplify_rules",
    "product_",
    "run_multiset_pipeline",
    "sum_",
]

multiset_language_rules = ruleset(name="srtree_eqsat_multiset_lang")
multiset_lower_rules = ruleset(name="srtree_eqsat_multiset_lower")
multiset_simplify_rules = ruleset(name="srtree_eqsat_multiset_simplify")
multiset_reify_rules = ruleset(name="srtree_eqsat_multiset_reify")


@function(ruleset=multiset_language_rules)
def sum_(xs: MultiSetLike[base.Num, base.NumLike]) -> base.Num: ...  # type: ignore[empty-body]


@function(ruleset=multiset_language_rules)
def product_(xs: MultiSetLike[base.Num, base.NumLike]) -> base.Num: ...  # type: ignore[empty-body]


@multiset_lower_rules.register
def _multiset_lower(
    x: base.Num,
    y: base.Num,
    xs: MultiSet[base.Num],
    ys: MultiSet[base.Num],
) -> Iterable[RewriteOrRule]:
    yield rewrite(x + y, subsume=True).to(sum_(MultiSet(x, y)))
    yield rewrite(x * y, subsume=True).to(product_(MultiSet(x, y)))
    yield rule(eq(x).to(sum_(xs)), eq(y).to(sum_(ys))).then(union(x + y).with_(sum_(xs + ys)))
    yield rule(eq(x).to(product_(xs)), eq(y).to(product_(ys))).then(union(x * y).with_(product_(xs + ys)))


@multiset_simplify_rules.register
def _multiset_simplify(
    x: base.Num,
    y: base.Num,
    z: base.Num,
    xs: MultiSet[base.Num],
    ys: MultiSet[base.Num],
    zs: MultiSet[base.Num],
    common: MultiSet[base.Num],
    i: f64,
    j: f64,
) -> Iterable[RewriteOrRule]:
    yield rewrite(sum_(MultiSet[base.Num]())).to(base._zero())
    yield rewrite(product_(MultiSet[base.Num]())).to(base._one())
    yield rule(eq(x).to(sum_(xs)), xs.length() == i64(1)).then(union(x).with_(xs.pick()))
    yield rule(eq(x).to(product_(xs)), xs.length() == i64(1)).then(union(x).with_(xs.pick()))
    yield rule(eq(x).to(sum_(xs)), xs.contains(base._zero()), xs.length() > 1).then(
        union(x).with_(sum_(xs.remove(base._zero())))
    )
    yield rule(eq(x).to(product_(xs)), xs.contains(base._one()), xs.length() > 1).then(
        union(x).with_(product_(xs.remove(base._one())))
    )
    yield rule(eq(x).to(product_(xs)), xs.contains(base._zero())).then(union(x).with_(base._zero()))
    yield rule(
        eq(x).to(sum_(xs)),
        eq(y).to(base.Num(i)),
        xs.contains(y),
        eq(z).to(base.Num(j)),
        xs.remove(y).contains(z),
    ).then(union(x).with_(sum_(xs.remove(y).remove(z).insert(base.Num(i + j)))))
    yield rule(
        eq(x).to(product_(xs)),
        eq(y).to(base.Num(i)),
        xs.contains(y),
        eq(z).to(base.Num(j)),
        xs.remove(y).contains(z),
    ).then(union(x).with_(product_(xs.remove(y).remove(z).insert(base.Num(i * j)))))
    yield rule(
        eq(x).to(sum_(xs)),
        eq(y).to(product_(ys)),
        eq(z).to(product_(zs)),
        xs.contains(y),
        xs.remove(y).contains(z),
        eq(common).to(ys & zs),
        common.length() > 0,
    ).then(
        union(x).with_(
            sum_(
                xs.remove(y)
                .remove(z)
                .insert(product_(common.insert(sum_(MultiSet(product_(ys - common), product_(zs - common))))))
            )
        )
    )


@multiset_reify_rules.register
def _multiset_reify(
    x: base.Num,
    y: base.Num,
    xs: MultiSet[base.Num],
    ys: MultiSet[base.Num],
) -> Iterable[RewriteOrRule]:
    yield rewrite(sum_(MultiSet[base.Num]())).to(base._zero())
    yield rewrite(product_(MultiSet[base.Num]())).to(base._one())
    yield rule(eq(x).to(sum_(xs)), xs.length() == i64(1)).then(union(x).with_(xs.pick()))
    yield rule(eq(x).to(product_(xs)), xs.length() == i64(1)).then(union(x).with_(xs.pick()))
    yield rule(eq(x).to(sum_(xs)), xs.length() > 1, eq(y).to(xs.pick()), eq(ys).to(xs.remove(y))).then(
        union(x).with_(y + sum_(ys))
    )
    yield rule(eq(x).to(product_(xs)), xs.length() > 1, eq(y).to(xs.pick()), eq(ys).to(xs.remove(y))).then(
        union(x).with_(y * product_(ys))
    )


def _multiset_cleanup_rules() -> egglog.Schedule:
    return base.const_analysis_rules | base.const_reduction_rules | base.const_fusion_rules | base.fun_rules


def _multiset_notes(stages: Sequence[base.StageReport]) -> tuple[str, ...]:
    notes: list[str] = []
    first = stages[0]
    if first.stop_reason == "saturated":
        notes.append("Multiset lowering saturated without backoff or node limits.")
    else:
        notes.append(f"Multiset lowering stopped with {first.stop_reason}.")
    for stage in stages:
        hot_rules = sorted(stage.matches_per_rule.items(), key=lambda item: item[1], reverse=True)[:3]
        if hot_rules:
            notes.append(f"{stage.name} hottest rules: {', '.join(f'{name}={count}' for name, count in hot_rules)}")
    return tuple(notes)


def run_multiset_pipeline(
    num: base.Num,
    *,
    saturate_without_limits: bool = True,
    node_cutoff: int | None = None,
    iteration_limit: int | None = None,
    input_names: Sequence[str] = ("alpha", "beta", "theta"),
    sample_points: Sequence[Sequence[float]] | None = None,
) -> base.PipelineReport:
    del input_names
    iteration_limit = iteration_limit or 80
    stage1 = base._run_stage(
        "multiset_lower",
        num,
        base.const_analysis_rules | multiset_lower_rules,
        node_cutoff=None if saturate_without_limits else node_cutoff,
        iteration_limit=iteration_limit,
        scheduler=None,
    )
    stage2 = base._run_stage(
        "multiset_simplify",
        stage1.extracted,
        base.const_analysis_rules | multiset_simplify_rules | base.fun_rules,
        node_cutoff=None if saturate_without_limits else node_cutoff,
        iteration_limit=iteration_limit,
        scheduler=None,
    )
    stage3_rules = cast("Any", _multiset_cleanup_rules()) | multiset_reify_rules
    stage3 = base._run_stage(
        "multiset_reify_cleanup",
        stage2.extracted,
        stage3_rules,
        node_cutoff=None if saturate_without_limits else node_cutoff,
        iteration_limit=iteration_limit,
        scheduler=None,
    )
    stages = (stage1, stage2, stage3)
    selection = (
        base.SampleSelection(
            tuple(tuple(float(value) for value in row) for row in sample_points),
            "provided",
            len(sample_points),
        )
        if sample_points is not None
        else base._choose_domain_safe_sample_points(num, ("alpha", "beta", "theta"))
    )
    metrics = base._pipeline_metrics(
        num, stage3.extracted, input_names=("alpha", "beta", "theta"), sample_points=selection.points
    )
    numeric_max_abs_error, numeric_status = base._numeric_max_abs_error(
        num,
        stage3.extracted,
        input_names=("alpha", "beta", "theta"),
        sample_points=selection.points,
    )
    return base.PipelineReport(
        mode="multiset",
        stages=stages,
        extracted=stage3.extracted,
        cost=stage3.cost,
        total_size=stage3.total_size,
        node_count=stage3.node_count,
        eclass_count=stage3.eclass_count,
        stop_reason=stage3.stop_reason,
        rendered=base.render_num(stage3.extracted),
        metric_report=metrics,
        numeric_max_abs_error=numeric_max_abs_error,
        numeric_status=numeric_status if selection.status == "ok" else selection.status,
        notes=_multiset_notes(stages),
    )
