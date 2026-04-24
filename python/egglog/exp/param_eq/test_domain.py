from collections.abc import Iterable

import rich.progress
from hypothesis import example, given
from hypothesis import strategies as st

from .domain import *

I64_STRATEGY = st.integers(-(2**63), 2**63 - 1)
EXPR_STRATEGY = st.recursive(
    st.sampled_from(["x", "y", "z"])
    | I64_STRATEGY.map(str)
    | st.floats(1e-4 + 1, 1e16 - 1, allow_infinity=False, allow_nan=False).map(str),
    lambda children: (
        st.builds(
            lambda op, e: f"{op}({e})",
            st.sampled_from(["-", "exp", "log", "abs", "square", "cube", "sqrt", "plog"]),
            children,
        )
        | st.builds(lambda exp, base: f"({base} ** {exp})", I64_STRATEGY, children)
        | st.builds(
            lambda op, l, r: f"({l} {op} {r})",
            st.sampled_from(["+", "-", "*", "/"]),
            children,
            children,
        )
    ),
)


def _test_expr(expr: str) -> Iterable[str]:
    """
    Verifies that a couple loops of binary <-> container extraction are consistent with each other, and that the costs match up eventually.

    This is to check that pretty printing and cost modeling are consistent between the binary and container forms, which is important for debugging and for ensuring that the cost model is correctly guiding the search.
    """
    # 1. x + x
    binary, binary_cost = EGraph().extract(
        validate_is_binary(parse_expression(expr)), include_cost=True, cost_model=param_cost_model
    )
    assert EGraph().extract(parse_expression(render_num(binary))) == binary, (
        f"Parsing the rendered binary expression should give back the same expression\nOriginal: {binary}\nRendered: {render_num(binary)}\nParsed: {parse_expression(render_num(binary))}"
    )
    # 2. {{x: 1}: 2}
    container, container_cost = EGraph().extract(
        binary_to_containers(binary), cost_model=container_cost_model, include_cost=True
    )
    validate_is_containers(container)
    # 3. 2 * x
    binary_2, binary_2_cost = EGraph().extract(
        validate_is_binary(containers_to_binary(container)),
        cost_model=param_cost_model,
        include_cost=True,
    )
    # 4. {{x: 1, 2: 1}: 1}
    container_2, container_2_cost = EGraph().extract(
        binary_to_containers(binary_2),
        cost_model=container_cost_model,
        include_cost=True,
    )
    validate_is_containers(container_2)
    # 5. 2 * x
    binary_3, binary_3_cost = EGraph().extract(
        validate_is_binary(containers_to_binary(container_2)),
        cost_model=param_cost_model,
        include_cost=True,
    )
    # 6. {{x: 1, 2: 1}: 1}
    validate_is_containers(EGraph().extract(binary_to_containers(binary_3)))

    assert binary_3_cost == container_2_cost, (
        f"Cost mismatch between decoded binary and container forms\nBinary: {binary_3}\nContainer: {container_2}"
    )
    assert container_cost.floats == binary_2_cost.floats, (
        f"Cost mismatch between binary and container forms\nBinary: {binary}\nBinary 2: {binary_2}"
    )
    yield (f"# {expr}")
    yield (f"# n_params={binary_cost.floats}")
    yield (f"{binary}")
    yield (f"# n_params={container_cost.floats}")
    yield (f"{container}\n")


@example("x - y")
@example("-x - y")
@example("x - 2 * y")
@example("x - y - y")
@example("x - 1 / y")
@example("x - 2.5 / y")
@example("-x + (y - x)")
@example("-(x + y)")
@example("(-1) * (x / y)")
@example("-(-x)")
@example("square(-1)")
@given(EXPR_STRATEGY)
def test_expr(expr: str):
    list(_test_expr(expr))


EXPRS = [
    "x * y + z",
    "x * (y + z)",
    "x * y * z",
    "(x ** 2) / y",
    "x + 1.0 * y",
    "x + -1.0 * y",
    "x - y",
    "-x",
    "1 / x",
    "2 / x",
    "x / 2",
    "x / 1",
    "x ** 1/2",
    "0.2306440753250631 + (0.03139967317000205)*(x1) + (-1.1634241022901022 + (exp(exp(-1 + x0)))**(-1))*(1.2522488356336676 - (exp((x0)*((exp(exp(-1 + x0)))**(-1)))))",
    "(x0**2 - (0.04106910574307527*x0 + 0.043582355979073722*x1*(x1 - 4.735723943783631) + 0.01496006509706177 + I*pi)*exp(x0**2) + 0.00822065460724008)*exp(-x0**2)",
    "exp((plog((((plog((x1+0.385))*((-0.328/(-0.612))^3))/(-0.379))-plog((((-0.479/(-0.246/(-0.358)))-((-0.289/(-0.327))^3))-(x1/(-0.045-(((-0.464/x0)/x0)^2)))))))*((-0.293/(-0.389))^3)))",
    "(-0.7330341374049288 * x1 * (1.1635766746115828 * x0 * (x0 - 1.096491354684671 * x1 + 0.09649135468467125 * exp(x1) + 0.065716650770683) - 3.3628776435387486 * x0 - x1 + 0.5423590312635699) - 0.02765235981387666 * x1 + 0.02765235981387666 * exp(x0 ** 2.0) + 0.02765235981387666 * exp(x1) + 0.09299150260917513) / (-1.0 * x1 + exp(x0 ** 2.0) + exp(x1) + 3.3628776435387486)",
    "-0.8529414239783971 * (x1 * (x0 * (-2.824404573652556 + (x0 + x1 / -0.9119998946891651 + exp(x1) / 10.363622764629516)) + x1 / -1.1635766746115828 + 0.4661137019136418)) / (exp(x1) - x1 + exp(x0 ** 2.0) + 3.3628776435387486) + 0.02765235981387666",
]


def test_parsing(snapshot_py):
    lines = ["from egglog import *", "from egglog.exp.param_eq.domain import *", ""]

    for expr in rich.progress.track(EXPRS, description="Testing parsing of expressions..."):
        lines.extend(_test_expr(expr))

    assert snapshot_py == "\n".join(lines)
