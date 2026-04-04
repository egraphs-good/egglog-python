"""Inspect one retained param-eq case through the Egglog replication pipeline."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

from egglog import EGraph, f64
from egglog.deconstruct import get_callable_args
from egglog.egraph import Ruleset, UnstableCombinedRuleset
from egglog.exp.param_eq import pipeline as param_eq_hegg
from egglog.exp.param_eq.paths import GOLDEN_PATH

GOLDEN_FIXTURE = GOLDEN_PATH


def _load_case_source(case_id: str | None, expr: str | None) -> tuple[str, str]:
    if expr is not None:
        return expr, expr
    if case_id is None:
        msg = "Either --case-id or --expr is required"
        raise ValueError(msg)
    cases = json.loads(GOLDEN_FIXTURE.read_text())["cases"]
    case = next(case for case in cases if case["case_id"] == case_id)
    return str(case["source"]), case_id


def _schedule_for_variant(variant: str):
    basic_rules = {
        "add_comm": param_eq_hegg.basic_add_comm_rules,
        "mul_comm": param_eq_hegg.basic_mul_comm_rules,
        "add_assoc": param_eq_hegg.basic_add_assoc_rules,
        "mul_assoc": param_eq_hegg.basic_mul_assoc_rules,
        "product_regroup": param_eq_hegg.basic_product_regroup_rules,
        "other": param_eq_hegg.basic_other_rules,
    }

    if variant.startswith("baseline_without_"):
        disabled = variant.removeprefix("baseline_without_")
        selected_basic: Ruleset | UnstableCombinedRuleset | None = None
        for name, ruleset in basic_rules.items():
            if name == disabled:
                continue
            selected_basic = ruleset if selected_basic is None else selected_basic | ruleset
        if selected_basic is None:
            msg = f"No rewrite rules left after disabling {disabled}"
            raise ValueError(msg)
        rewrite_schedule = param_eq_hegg.run(
            selected_basic | param_eq_hegg.fun_rules, scheduler=param_eq_hegg.scheduler
        )
        analysis_round = param_eq_hegg.analysis_schedule.saturate() + rewrite_schedule
        return param_eq_hegg.scheduler.scope(analysis_round.saturate(stop_when_no_updates=True))
    if variant == "baseline":
        return param_eq_hegg.total_ruleset
    if variant == "baseline_with_add_comm":
        rewrite_schedule = param_eq_hegg.run(
            param_eq_hegg.basic_rules | param_eq_hegg.fun_rules, scheduler=param_eq_hegg.scheduler
        )
        analysis_round = param_eq_hegg.analysis_schedule.saturate() + rewrite_schedule
        return param_eq_hegg.scheduler.scope(analysis_round.saturate(stop_when_no_updates=True))
    if variant == "single_pair":
        return param_eq_hegg.analysis_rewrite_round
    if variant == "two_rounds":
        return param_eq_hegg.scheduler.scope(
            param_eq_hegg.analysis_rewrite_round + param_eq_hegg.analysis_rewrite_round
        )
    msg = f"Unsupported variant: {variant}"
    raise ValueError(msg)


def _render_optional(value: param_eq_hegg.OptionalF64) -> dict[str, float | str | None]:
    match get_callable_args(value, param_eq_hegg.OptionalF64.some):
        case (inner,) if isinstance(inner, f64):
            return {"kind": "some", "value": float(inner.value)}
        case _:
            return {"kind": "none", "value": None}


def _top_mapping(mapping: dict[str, object], limit: int = 10) -> list[tuple[str, float]]:
    normalized: list[tuple[str, float]] = []
    for key, value in mapping.items():
        if hasattr(value, "total_seconds"):
            normalized.append((key, float(value.total_seconds())))
        elif isinstance(value, int | float):
            normalized.append((key, float(value)))
        else:
            msg = f"Unsupported run-report value for {key}: {value!r}"
            raise TypeError(msg)
    normalized.sort(key=lambda item: item[1], reverse=True)
    return normalized[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id")
    parser.add_argument("--expr")
    parser.add_argument(
        "--variant",
        choices=(
            "baseline",
            "baseline_with_add_comm",
            "single_pair",
            "two_rounds",
            "baseline_without_add_comm",
            "baseline_without_mul_comm",
            "baseline_without_add_assoc",
            "baseline_without_mul_assoc",
            "baseline_without_product_regroup",
        ),
        default="baseline",
    )
    args = parser.parse_args()

    source, label = _load_case_source(args.case_id, args.expr)
    expr = param_eq_hegg.parse_expression(source)
    schedule = _schedule_for_variant(args.variant)

    egraph = EGraph()
    egraph.register(expr)

    start = time.perf_counter()
    run_report = egraph.run(schedule)
    elapsed = time.perf_counter() - start
    extracted, cost = egraph.extract(expr, include_cost=True)
    extracted_analysis = egraph.extract(param_eq_hegg.const_value(expr))
    total_size = sum(size for _, size in egraph.all_function_sizes())
    payload = json.loads(egraph._serialize().to_json())
    ops = Counter(node["op"] for node in payload.get("nodes", {}).values())

    print(
        json.dumps(
            {
                "case": label,
                "variant": args.variant,
                "elapsed_sec": elapsed,
                "updated": bool(run_report.updated),
                "can_stop": bool(getattr(run_report, "can_stop", False)),
                "rendered": param_eq_hegg.render_num(extracted),
                "cost": int(cost),
                "analysis": _render_optional(extracted_analysis),
                "total_size": total_size,
                "node_count": len(payload.get("nodes", {})),
                "eclass_count": len(payload.get("class_data", {})),
                "top_ops": ops.most_common(10),
                "top_rule_matches": _top_mapping(getattr(run_report, "num_matches_per_rule", {})),
                "top_rule_times": _top_mapping(getattr(run_report, "search_and_apply_time_per_rule", {})),
                "top_ruleset_times": _top_mapping(getattr(run_report, "search_and_apply_time_per_ruleset", {})),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
