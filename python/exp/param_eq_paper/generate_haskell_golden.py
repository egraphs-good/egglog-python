from __future__ import annotations

import ast
import csv
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from egglog.exp.param_eq_hegg import parse_expression, render_num

ROOT = Path("/Users/saul/p/egg-smol-python")
HASKELL_ROOT = Path("/Users/saul/p/param-eq-haskell")
ARTIFACTS_DIR = ROOT / "python" / "exp" / "param_eq_paper" / "artifacts"
GOLDEN_PATH = ROOT / "python" / "tests" / "param_eq_hegg_haskell_golden.json"
LLVM_BIN = Path.home() / "installs" / "llvm-12" / "bin"


@dataclass(frozen=True)
class GoldenCaseSpec:
    case_id: str
    category: str
    source: str
    notes: str
    compare_root_analysis: bool = True
    compare_rewrite_tree: bool = True
    compare_simplify_e: bool = True
    compare_param_count: bool = True
    expected_mismatch: bool = False


def _corpus_case(dataset: str, algorithm: str, algo_row: str, *, notes: str, expected_mismatch: bool) -> GoldenCaseSpec:
    with (ARTIFACTS_DIR / "haskell_paper_rows.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["dataset"] == dataset and row["algorithm"] == algorithm and row["algo_row"] == algo_row:
            return GoldenCaseSpec(
                case_id=f"{dataset}_{algorithm.lower()}_{algo_row}",
                category="corpus",
                source=row["original_expr"],
                notes=notes,
                compare_root_analysis=not expected_mismatch,
                compare_rewrite_tree=not expected_mismatch,
                compare_simplify_e=not expected_mismatch,
                compare_param_count=not expected_mismatch,
                expected_mismatch=expected_mismatch,
            )
    msg = f"Missing corpus canary {dataset=} {algorithm=} {algo_row=}"
    raise ValueError(msg)


CASE_SPECS: tuple[GoldenCaseSpec, ...] = (
    GoldenCaseSpec(
        case_id="x_minus_x",
        category="analysis",
        source="x0 - x0",
        notes="Mixed class stays non-constant even after 0 is introduced.",
    ),
    GoldenCaseSpec(
        case_id="two_minus_two",
        category="analysis",
        source="2 - 2",
        notes="Purely constant class prunes to the constant leaf.",
    ),
    GoldenCaseSpec(
        case_id="x_div_x",
        category="analysis",
        source="x0 / x0",
        notes="Mixed class stays non-constant even after 1 is introduced.",
    ),
    GoldenCaseSpec(
        case_id="two_div_two",
        category="analysis",
        source="2 / 2",
        notes="Purely constant division prunes to 1.",
    ),
    GoldenCaseSpec(
        case_id="zero_div_x",
        category="analysis",
        source="0 / x0",
        notes="Identity rewrite reaches 0 for any denominator shape.",
    ),
    GoldenCaseSpec(
        case_id="zero_mul_x",
        category="analysis",
        source="0 * x0",
        notes="Identity rewrite reaches 0 for any multiplicand shape.",
        compare_root_analysis=False,
        compare_rewrite_tree=False,
        compare_simplify_e=False,
        compare_param_count=False,
    ),
    GoldenCaseSpec(
        case_id="log_mul_const_var",
        category="guards",
        source="log(2 * x0)",
        notes="Covers the Haskell log-product guard bundle.",
    ),
    GoldenCaseSpec(
        case_id="log_div_const_var",
        category="guards",
        source="log(2 / x0)",
        notes="Covers the Haskell log-division guard bundle.",
    ),
    GoldenCaseSpec(
        case_id="log_exp_var",
        category="guards",
        source="log(exp(x0))",
        notes="Only fires for a non-constant argument.",
    ),
    GoldenCaseSpec(
        case_id="exp_log_var",
        category="guards",
        source="exp(log(x0))",
        notes="Only fires for a non-constant argument.",
    ),
    GoldenCaseSpec(
        case_id="sqrt_negative_times_difference",
        category="guards",
        source="sqrt((-2.0) * (x0 - x1))",
        notes="Covers the negative-a square-root rewrite shape.",
    ),
    GoldenCaseSpec(
        case_id="x_mul_inverse_x",
        category="guards",
        source="x0 * (1 / x0)",
        notes="Uses the same non-zero guard as x / x.",
        compare_root_analysis=False,
        compare_rewrite_tree=False,
        compare_simplify_e=False,
        compare_param_count=False,
    ),
    GoldenCaseSpec(
        case_id="negative_const_div_self",
        category="guards",
        source="(-2) / (-2)",
        notes="Negative constants still satisfy is_not_zero.",
    ),
    GoldenCaseSpec(
        case_id="negative_const_pow_two",
        category="analysis",
        source="(-2) ** 2",
        notes="Haskell folds negative-base powers with integer exponents.",
    ),
    GoldenCaseSpec(
        case_id="negative_const_pow_three",
        category="analysis",
        source="(-2) ** 3",
        notes="Haskell folds negative-base powers with integer exponents.",
    ),
    GoldenCaseSpec(
        case_id="negative_const_pow_var",
        category="analysis",
        source="(-2) ** x0",
        notes="Negative-base powers with non-constant exponents remain non-constant.",
    ),
    GoldenCaseSpec(
        case_id="sbp_zero_times_quadratic",
        category="schedule",
        source="0.004376 - (0.0 * (x1 * x1))",
        notes="Smallest Haskell-backed discriminator found so far for the bounded saturated-round schedule.",
    ),
    GoldenCaseSpec(
        case_id="x0_sq_plus_x1_sq",
        category="schedule",
        source="(((51.6682472229003906 * x0) * ((-0.0001894439337775) * x0)) + ((0.0012052881065756 * x1) * ((-8.2380609512329102) * x1)))",
        notes="Reduced Haskell-backed schedule canary: the current baseline disables add commutativity to avoid the raw quadratic-sum blowup.",
    ),
    GoldenCaseSpec(
        case_id="sub_add_left_assoc",
        category="extraction",
        source="x0 - (x0 + x1)",
        notes="Haskell and Python keep semantically equivalent but different extracted subtraction normal forms.",
        compare_root_analysis=False,
        compare_param_count=False,
        expected_mismatch=True,
    ),
    GoldenCaseSpec(
        case_id="nonconstant_power_exponent_count",
        category="reporting",
        source="(2.0 ** x0) + (x1 ** (x0 + 3.0))",
        notes="Constants count as parameters, but non-constant exponents are still traversed.",
        compare_root_analysis=False,
        compare_rewrite_tree=False,
        compare_simplify_e=False,
    ),
    _corpus_case(
        "pagie",
        "SBP",
        "1",
        notes="Previously mismatched corpus canary that now matches current FixTree.hs again.",
        expected_mismatch=False,
    ),
    _corpus_case(
        "pagie",
        "Operon",
        "15",
        notes="Large canary that still differs in extracted form, but now appears numerically equivalent under the current baseline.",
        expected_mismatch=True,
    ),
)


def _source_to_haskell_expr(source: str) -> str:
    tree = ast.parse(source.replace("^", "**"), mode="eval")

    def const(value: float) -> str:
        rendered = repr(float(value))
        if float(value) < 0.0 or math.copysign(1.0, float(value)) < 0.0:
            return f"(Const ({rendered}))"
        return f"(Const {rendered})"

    def emit(node: ast.AST) -> str:
        if isinstance(node, ast.Expression):
            return emit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return const(float(node.value))
        if isinstance(node, ast.Name):
            if node.id.startswith("x") and node.id[1:].isdigit():
                return f"(Var {int(node.id[1:])})"
            msg = f"Unsupported variable name: {node.id!r}"
            raise ValueError(msg)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int | float):
                return const(float(-node.operand.value))
            return f"(Mul {const(-1.0)} {emit(node.operand)})"
        if isinstance(node, ast.BinOp):
            lhs = emit(node.left)
            rhs = emit(node.right)
            match node.op:
                case ast.Add():
                    return f"(Add {lhs} {rhs})"
                case ast.Sub():
                    return f"(Sub {lhs} {rhs})"
                case ast.Mult():
                    return f"(Mul {lhs} {rhs})"
                case ast.Div():
                    return f"(Div {lhs} {rhs})"
                case ast.Pow():
                    return f"(Power {lhs} {rhs})"
            msg = f"Unsupported binary operator: {ast.dump(node.op)}"
            raise TypeError(msg)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = [emit(arg) for arg in node.args]
            match node.func.id:
                case "exp":
                    return f"(Fun Exp {args[0]})"
                case "log":
                    return f"(Fun Log {args[0]})"
                case "sqrt":
                    return f"(Fun Sqrt {args[0]})"
                case "abs":
                    return f"(Fun Abs {args[0]})"
                case "plog":
                    return f"(Fun Log (Fun Abs {args[0]}))"
                case "square":
                    return f"(Power {args[0]} (Const 2.0))"
                case "cube":
                    return f"(Power {args[0]} (Const 3.0))"
            msg = f"Unsupported function call: {node.func.id}"
            raise ValueError(msg)
        msg = f"Unsupported AST node: {ast.dump(node)}"
        raise TypeError(msg)

    return emit(tree)


def _normalize_haskell_pretty(expr: str) -> str:
    replacements = {
        "Log(": "log(",
        "Exp(": "exp(",
        "Sqrt(": "sqrt(",
        "Abs(": "abs(",
        "^": "**",
    }
    normalized = expr
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized


def _canonicalize(expr: str) -> str:
    return render_num(parse_expression(_normalize_haskell_pretty(expr)))


def _parse_analysis(text: str) -> dict[str, Any]:
    if text == "Nothing":
        return {"kind": "none", "value": None}
    if text.startswith("Just "):
        value_text = text.removeprefix("Just ").strip()
        if value_text.startswith("(") and value_text.endswith(")"):
            value_text = value_text[1:-1]
        return {"kind": "some", "value": float(value_text)}
    msg = f"Unexpected Haskell analysis output: {text!r}"
    raise ValueError(msg)


def _build_haskell_program(cases: list[GoldenCaseSpec]) -> str:
    case_lines = []
    for index, case in enumerate(cases):
        source_literal = json.dumps(case.source)
        prefix = "  " if index == 0 else "  , "
        case_lines.append(f'{prefix}("{case.case_id}", {source_literal}, {_source_to_haskell_expr(case.source)})')
    joined_case_lines = "\n".join(case_lines)
    return "\n".join([
        "import Data.List (intercalate, sort)",
        "import qualified Data.Set as S",
        "import Data.SRTree",
        "import Data.SRTree.Print",
        "import FixTree",
        "import Reparam (replaceConstsWithParams)",
        "import qualified Data.Equality.Graph.Lens as L",
        "import Data.Equality.Graph.Lens hiding ((^.))",
        "import Data.Equality.Graph.Monad",
        "import Data.Equality.Saturation",
        "import Data.Equality.Saturation.Scheduler",
        "import Data.Equality.Extraction",
        "",
        "cases :: [(String, String, SRTree Int Double)]",
        "cases =",
        "  [",
        joined_case_lines,
        "  ]",
        "",
        "sanitize :: String -> String",
        "sanitize = map (\\c -> if c == '\\t' || c == '\\n' then ' ' else c)",
        "",
        "emitCase :: (String, String, SRTree Int Double) -> IO ()",
        "emitCase (caseId, source, expr) = do",
        "  let ((root, beforeData, afterData, afterNodes), egr) =",
        "        egraph $ do",
        "          root <- represent (toFixTree expr)",
        "          beforeData <- gets (L.^. _class root._data)",
        "          runEqualitySaturation (BackoffScheduler 2500 30) (rewritesBasic <> rewritesFun)",
        "          afterData <- gets (L.^. _class root._data)",
        "          afterNodes <- gets (L.^. _class root._nodes)",
        "          pure (root, beforeData, afterData, afterNodes)",
        "      rewriteTreeExpr = relabelParams . toSRTree $ extractBest egr cost2 root",
        "      simplifiedExpr = simplifyE expr",
        "      paramCount = recountParams . replaceConstsWithParams $ simplifiedExpr",
        "      fields =",
        "        [ caseId",
        "        , source",
        "        , show beforeData",
        "        , show afterData",
        "        , show (sort (map show (S.toList afterNodes)))",
        "        , showDefault rewriteTreeExpr",
        "        , showDefault simplifiedExpr",
        "        , show paramCount",
        "        ]",
        '  putStrLn (intercalate "\\t" (map sanitize fields))',
        "",
        "main :: IO ()",
        "main = mapM_ emitCase cases",
        "",
    ])


def _run_haskell_cases(cases: list[GoldenCaseSpec]) -> list[dict[str, Any]]:
    program = _build_haskell_program(cases)
    with tempfile.NamedTemporaryFile("w", suffix=".hs", delete=False) as handle:
        handle.write(program)
        temp_path = Path(handle.name)
    try:
        env = dict(os.environ)
        env["PATH"] = f"{LLVM_BIN}:{env['PATH']}"
        output = subprocess.check_output(
            ["stack", "exec", "--", "runghc", "-isrc", str(temp_path)],
            cwd=HASKELL_ROOT,
            env=env,
            text=True,
            timeout=180,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    results = []
    for line in output.splitlines():
        (
            case_id,
            source,
            analysis_before,
            analysis_after,
            after_nodes,
            rewrite_tree_expr,
            simplify_e_expr,
            param_count,
        ) = line.split(
            "\t",
            maxsplit=7,
        )
        results.append({
            "case_id": case_id,
            "source": source,
            "analysis_before": _parse_analysis(analysis_before),
            "analysis_after": _parse_analysis(analysis_after),
            "after_nodes_haskell": ast.literal_eval(after_nodes),
            "rewrite_tree_expr_haskell": rewrite_tree_expr,
            "rewrite_tree_expr_python": _canonicalize(rewrite_tree_expr),
            "simplify_e_expr_haskell": simplify_e_expr,
            "simplify_e_expr_python": _canonicalize(simplify_e_expr),
            "simplify_e_param_count": int(param_count),
        })
    return results


def main() -> None:
    haskell_rows = {row["case_id"]: row for row in _run_haskell_cases(list(CASE_SPECS))}
    payload: dict[str, Any] = {
        "source_of_truth": str(HASKELL_ROOT / "src" / "FixTree.hs"),
        "generator": str(Path(__file__)),
        "cases": [],
    }
    for case in CASE_SPECS:
        payload["cases"].append({
            "case_id": case.case_id,
            "category": case.category,
            "source": case.source,
            "notes": case.notes,
            "compare_root_analysis": case.compare_root_analysis,
            "compare_rewrite_tree": case.compare_rewrite_tree,
            "compare_simplify_e": case.compare_simplify_e,
            "compare_param_count": case.compare_param_count,
            "expected_mismatch": case.expected_mismatch,
            **haskell_rows[case.case_id],
        })
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(GOLDEN_PATH)


if __name__ == "__main__":
    main()
