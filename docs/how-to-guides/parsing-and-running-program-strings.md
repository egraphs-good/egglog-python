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
