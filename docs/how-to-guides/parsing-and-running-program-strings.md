---
file_format: mystnb
---

# Parsing and running program strings

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

When the program is already a string, use
{meth}`egglog.bindings.EGraph.parse_and_run_program` to parse and execute it in
one call. Supplying a filename adds that source name to parse and runtime error
locations:

```{code-cell}
egraph.parse_and_run_program(
    "(function double (i64) i64 :no-merge)\n(set (double 2) 4)",
    filename="generated.egg",
)
```

The low-level binding can also record successfully executed command batches.
Recording is opt-in and preserves program order:

```{code-cell}
recording_egraph = EGraph(record=True)
recording_egraph.parse_and_run_program("(let $answer 42)\n(check (= $answer 42))")
print(recording_egraph.commands())
```

If a batch fails after executing an earlier command, Egglog keeps the mutation
made by that prefix, but the failed batch is not appended to `commands()`. Code
that needs transactional behavior should use a separate e-graph or explicit
`push`/`pop` boundaries.
