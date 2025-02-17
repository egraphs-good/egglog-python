---
file_format: mystnb
---

# How-to guides

## Parsing and running program strings

You can provide your program in a special DSL language. You can parse this with {meth}`egglog.bindings.EGraph.parse_program` and then run the result with You can parse this with {meth}`egglog.bindings.EGraph.run_program`::

```{code-cell}
from egglog.bindings import EGraph

egraph = EGraph()
commands = egraph.parse_program("(check (= (+ 1 2) 3))")
commands
```

```{code-cell}
egraph.run_program(*commands)
```
