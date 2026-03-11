---
file_format: mystnb
---

# How-to guides

## Parsing and running program strings

You can provide your program in a special DSL language. Parse it with
{meth}`egglog.bindings.EGraph.parse_program` and run the resulting commands with
{meth}`egglog.bindings.EGraph.run_program`:

```{code-cell}
from egglog.bindings import EGraph

egraph = EGraph()
commands = egraph.parse_program("(check (= (+ 1 2) 3))")
commands
```

```{code-cell}
egraph.run_program(*commands)
```

## Debugging a high-level e-graph

When a rule does not fire or an equality appears unexpectedly, the fastest tools
to reach for are:

- {meth}`egglog.egraph.EGraph.run` for per-run match counts and timings.
- {meth}`egglog.egraph.EGraph.function_values` for inspecting function tables.
- {meth}`egglog.egraph.EGraph.display` for visualizing the current e-graph.
- {meth}`egglog.egraph.EGraph.saturate` for stepping to a fixpoint while
  printing extracted expressions.

```{code-cell}
from __future__ import annotations

from egglog import *


class Math(Expr):
    def __init__(self, value: i64Like) -> None: ...

    def __add__(self, other: Math) -> Math: ...


@function
def score(x: Math) -> i64: ...


debug_rules = ruleset()


@debug_rules.register
def _(i: i64, j: i64):
    yield rewrite(Math(i) + Math(j)).to(Math(i + j))
```

Start with a normal run report when you want to see which rules matched:

```{code-cell}
egraph = EGraph()
expr = egraph.let("expr", Math(2) + Math(3))
egraph.register(set_(score(expr)).to(5))

report = egraph.run(debug_rules)
report.num_matches_per_rule
```

Use `function_values(...)` to inspect the rows currently stored for a function:

```{code-cell}
egraph.function_values(score)
```

Use `display(...)` to look at the current e-graph structure:

```{code-cell}
egraph.display(graphviz=True)
```

Use `saturate(...)` when you want to keep running until nothing changes while
printing the extracted form of an expression after each iteration:

```{code-cell}
egraph = EGraph()
expr = egraph.let("expr", Math(2) + Math(3))
egraph.saturate(debug_rules, expr=expr, max=2, visualize=False)
```

For a structural snapshot that can be turned back into replayable high-level
actions, see the `freeze()` section in
{doc}`reference/python-integration`.
