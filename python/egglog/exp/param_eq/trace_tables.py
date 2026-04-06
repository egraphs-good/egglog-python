"""Local e-graph snapshot helpers for stepwise param-eq trace comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import pandas as pd

from egglog import EGraph, bindings
from egglog.declarations import (
    CallDecl,
    ClassMethodRef,
    ClassVariableRef,
    FunctionRef,
    InitRef,
    LitDecl,
    MethodRef,
    TypedExprDecl,
    ValueDecl,
)
from egglog.pretty import JustTypeRef
from egglog.egraph import _CostModel, default_cost_model
from egglog.exp.param_eq.pipeline import Num, const_value, render_num

JsonValue: TypeAlias = object

NUM_SORT = "egglog.exp.param_eq.pipeline.Num"
OPTIONAL_F64_SORT = "OptionalF64"


@dataclass(frozen=True)
class SnapshotTables:
    """One trace snapshot serialized as metadata plus table-shaped records."""

    metadata: dict[str, JsonValue]
    tables: dict[str, list[dict[str, JsonValue]]]

    def to_pandas(self) -> dict[str, pd.DataFrame]:
        """Return the stored table records as pandas data frames."""
        return {name: pd.DataFrame(rows) for name, rows in self.tables.items()}

    def to_jsonable(self) -> dict[str, JsonValue]:
        """Return a JSON-serializable payload."""
        return {
            "metadata": self.metadata,
            "tables": self.tables,
        }

    def write_json(self, path: Path) -> None:
        """Write the snapshot payload to one JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_jsonable(), indent=2, sort_keys=False) + "\n")


def _analysis_to_json(text: str) -> dict[str, JsonValue]:
    if text in {"Nothing", "OptionalF64.none"}:
        return {"kind": "none", "value": None}
    if text.startswith("Just "):
        value_text = text.removeprefix("Just ").strip()
    elif text.startswith("OptionalF64.some(") and text.endswith(")"):
        value_text = text.removeprefix("OptionalF64.some(")[:-1].strip()
    else:
        return {"kind": "raw", "value": text}
    return {"kind": "some", "value": float(value_text)}


def _analysis_key(value: object) -> str:
    if isinstance(value, dict):
        kind = value.get("kind")
        if kind == "none":
            return "none"
        if kind == "some":
            return f"some:{float(value['value']):.12g}"
    return json.dumps(value, sort_keys=True)


def _canonical_numeric(value: object) -> str:
    text = str(value).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    try:
        return str(float(text))
    except ValueError:
        return text


def _optional_exprs_to_analysis(exprs: tuple[CallDecl, ...]) -> dict[str, JsonValue]:
    if not exprs:
        return {"kind": "raw", "value": "[]"}
    raw_fallback = str(exprs[0])
    for expr in exprs:
        match expr:
            case CallDecl(callable=ClassVariableRef(var_name="none"), args=()):
                return {"kind": "none", "value": None}
            case CallDecl(callable=ClassMethodRef(method_name="some"), args=(TypedExprDecl(expr=LitDecl(value=value)),)):
                if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                    return {"kind": "some", "value": float(value)}
    return {"kind": "raw", "value": raw_fallback}


def _extract_best_by_value(egraph: EGraph, *, value: bindings.Value, tp: JustTypeRef) -> tuple[str, int]:
    egg_sort = egraph._state.type_ref_to_egg(tp)
    cost_model = _CostModel(default_cost_model, egraph).to_bindings_cost_model()
    extractor = cast(Any, bindings.Extractor([egg_sort], egraph._state.egraph, cost_model))
    termdag = bindings.TermDag()
    cost, term = extractor.extract_best(egraph._state.egraph, termdag, value, egg_sort)
    extracted = egraph._from_termdag(termdag, term, tp)
    return render_num(extracted), int(cost)


def _num_class_signatures(
    *,
    num_exprs_by_value: dict[str, tuple[CallDecl, ...]],
    num_analysis_by_value: dict[str, dict[str, JsonValue]],
    num_best_by_value: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    cache: dict[str, str] = {}
    semantic_cache: dict[str, str] = {}
    active: set[str] = set()

    def class_signature(class_id: str, *, include_best_expr: bool) -> str:
        current_cache = cache if include_best_expr else semantic_cache
        if class_id in current_cache:
            return current_cache[class_id]
        if class_id in active:
            return f"Cycle({class_id})"
        active.add(class_id)
        exprs = num_exprs_by_value[class_id]
        analysis = num_analysis_by_value.get(class_id, {"kind": "none", "value": None})
        members = sorted(member_signature(expr, include_best_expr=include_best_expr) for expr in exprs)
        prefix = f"{_analysis_key(analysis)}|"
        if include_best_expr:
            best_expr = num_best_by_value.get(class_id, "<missing>")
            prefix = f"{prefix}best:{best_expr}|"
        signature = f"{prefix}{'||'.join(members)}"
        current_cache[class_id] = signature
        active.remove(class_id)
        return signature

    def arg_signature(arg: TypedExprDecl) -> str:
        match arg.expr:
            case ValueDecl(value):
                return class_signature(str(value), include_best_expr=current_include_best_expr)
            case LitDecl(value):
                if isinstance(value, str):
                    return f'"{value}"'
                return _canonical_numeric(value)
            case CallDecl() as call:
                return member_signature(call, include_best_expr=current_include_best_expr)
            case _:
                return str(arg.expr)

    def member_signature(call: CallDecl, *, include_best_expr: bool) -> str:  # noqa: C901, PLR0911
        match call.callable:
            case InitRef(ident=ident) if ident.name == "Num":
                return f"Const({arg_signature(call.args[0])})"
            case ClassMethodRef(ident=ident, method_name="var") if ident.name == "Num":
                return f"Var({arg_signature(call.args[0]).strip('\"')})"
            case MethodRef(ident=ident, method_name="__add__") if ident.name == "Num":
                return f"Add({arg_signature(call.args[0])},{arg_signature(call.args[1])})"
            case MethodRef(ident=ident, method_name="__sub__") if ident.name == "Num":
                return f"Sub({arg_signature(call.args[0])},{arg_signature(call.args[1])})"
            case MethodRef(ident=ident, method_name="__mul__") if ident.name == "Num":
                return f"Mul({arg_signature(call.args[0])},{arg_signature(call.args[1])})"
            case MethodRef(ident=ident, method_name="__truediv__") if ident.name == "Num":
                return f"Div({arg_signature(call.args[0])},{arg_signature(call.args[1])})"
            case MethodRef(ident=ident, method_name="__pow__") if ident.name == "Num":
                return f"Pow({arg_signature(call.args[0])},{arg_signature(call.args[1])})"
            case MethodRef(ident=ident, method_name="exp") if ident.name == "Num":
                return f"Exp({arg_signature(call.args[0])})"
            case MethodRef(ident=ident, method_name="log") if ident.name == "Num":
                return f"Log({arg_signature(call.args[0])})"
            case MethodRef(ident=ident, method_name="sqrt") if ident.name == "Num":
                return f"Sqrt({arg_signature(call.args[0])})"
            case MethodRef(ident=ident, method_name="__abs__") if ident.name == "Num":
                return f"Abs({arg_signature(call.args[0])})"
            case _:
                callable_name = (
                    call.callable.method_name
                    if isinstance(call.callable, (MethodRef, ClassMethodRef))
                    else call.callable.ident.name
                    if hasattr(call.callable, "ident")
                    else str(call.callable)
                )
                return f"{callable_name}({','.join(arg_signature(arg) for arg in call.args)})"

    current_include_best_expr = True
    with_best = {}
    for class_id in sorted(num_exprs_by_value):
        current_include_best_expr = True
        with_best[class_id] = class_signature(class_id, include_best_expr=True)
    without_best = {}
    for class_id in sorted(num_exprs_by_value):
        current_include_best_expr = False
        without_best[class_id] = class_signature(class_id, include_best_expr=False)
    return (
        with_best,
        without_best,
    )


def build_egglog_snapshot(
    egraph: EGraph,
    *,
    root: Num,
    metadata: dict[str, JsonValue],
) -> SnapshotTables:
    """Build a comparable local trace snapshot from an Egglog e-graph."""
    frozen = egraph._egraph.freeze()
    frozen_decl = egraph.freeze().decl
    payload = json.loads(egraph._serialize().to_json())

    functions: list[dict[str, JsonValue]] = []
    rows: list[dict[str, JsonValue]] = []
    for function_name, function in frozen.functions.items():
        input_sorts = [str(sort) for sort in function.input_sorts]
        functions.append({
            "function_name": function_name,
            "input_sorts": input_sorts,
            "output_sort": function.output_sort,
            "is_let_binding": function.is_let_binding,
            "row_count": len(function.rows),
        })
        for row_index, row in enumerate(function.rows, start=1):
            rows.append({
                "function_name": function_name,
                "row_index": row_index,
                "subsumed": row.subsumed,
                "inputs": [str(value) for value in row.inputs],
                "output": str(row.output),
            })

    node_counts_by_class: dict[str, int] = {}
    for node in payload.get("nodes", {}).values():
        eclass = str(node["eclass"])
        node_counts_by_class[eclass] = node_counts_by_class.get(eclass, 0) + 1

    optional_analysis_by_value = {
        str(value): _optional_exprs_to_analysis(exprs)
        for value, (tp, exprs) in frozen_decl.e_classes.items()
        if str(tp) == OPTIONAL_F64_SORT
    }
    num_analysis_by_value: dict[str, dict[str, JsonValue]] = {}
    for call, out in frozen_decl.sets.items():
        match call:
            case CallDecl(
                callable=FunctionRef(ident=ident),
                args=(TypedExprDecl(expr=ValueDecl(value=num_value)),),
            ) if ident.name == "const_value":
                match out.expr:
                    case ValueDecl(value=analysis_value):
                        num_analysis_by_value[str(num_value)] = optional_analysis_by_value.get(
                            str(analysis_value),
                            {"kind": "raw", "value": str(out)},
                        )
                    case _:
                        num_analysis_by_value[str(num_value)] = {"kind": "raw", "value": str(out)}

    num_exprs_by_value = {
        str(value): exprs
        for value, (tp, exprs) in frozen_decl.e_classes.items()
        if str(tp) == "Num"
    }
    num_best_by_value: dict[str, str] = {}
    num_cost_by_value: dict[str, int] = {}
    for value, (tp, _) in frozen_decl.e_classes.items():
        if str(tp) != "Num":
            continue
        best_expr, best_cost = _extract_best_by_value(egraph, value=value, tp=tp)
        num_best_by_value[str(value)] = best_expr
        num_cost_by_value[str(value)] = best_cost
    num_signatures_by_value, num_semantic_signatures_by_value = _num_class_signatures(
        num_exprs_by_value=num_exprs_by_value,
        num_analysis_by_value=num_analysis_by_value,
        num_best_by_value=num_best_by_value,
    )

    classes = [
        {
            "class_id": str(value),
            "type": str(tp),
            "node_count": len(exprs),
            "analysis": num_analysis_by_value.get(str(value), {"kind": "none", "value": None}),
            "best_expr": num_best_by_value[str(value)],
            "best_cost": num_cost_by_value[str(value)],
            "semantic_signature": num_semantic_signatures_by_value[str(value)],
            "signature": num_signatures_by_value[str(value)],
        }
        for value, (tp, exprs) in frozen_decl.e_classes.items()
        if str(tp) == "Num"
    ]

    nodes = [
        {
            "node_id": node_id,
            "class_id": str(node["eclass"]),
            "op": str(node["op"]),
            "children": [str(child) for child in node.get("children", [])],
            "cost": float(node["cost"]),
            "subsumed": bool(node["subsumed"]),
        }
        for node_id, node in payload.get("nodes", {}).items()
    ]

    extracted, extracted_cost = egraph.extract(root, include_cost=True)
    try:
        root_analysis = str(egraph.extract(const_value(root)))
    except Exception:
        root_analysis = "OptionalF64.none"
    root_rows = [
        {
            "root_expr": render_num(root),
            "extracted_expr": render_num(extracted),
            "extracted_cost": int(extracted_cost),
            "root_analysis": _analysis_to_json(root_analysis),
        }
    ]

    snapshot_metadata = {
        **metadata,
        "class_count": len(classes),
        "node_count": sum(cast(int, class_row["node_count"]) for class_row in classes),
        "serialized_class_count": len(payload.get("class_data", {})),
        "serialized_node_count": len(payload.get("nodes", {})),
        "root_analysis": _analysis_to_json(root_analysis),
        "root_extracted_expr": render_num(extracted),
        "root_extracted_cost": int(extracted_cost),
    }
    return SnapshotTables(
        metadata=snapshot_metadata,
        tables={
            "functions": functions,
            "rows": rows,
            "classes": classes,
            "nodes": nodes,
            "root": root_rows,
        },
    )


def read_snapshot(path: Path) -> SnapshotTables:
    """Read one snapshot JSON file back into the local data class."""
    payload = json.loads(path.read_text())
    metadata = payload["metadata"]
    tables = payload["tables"]
    assert isinstance(metadata, dict)
    assert isinstance(tables, dict)
    return SnapshotTables(metadata=metadata, tables=tables)
