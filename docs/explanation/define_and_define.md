```{post} 2022-11-29
:author: Saul
```

# Define function vs define action.

There is both a top level `define` method as well as a `define` action type. What is the difference between them? The method does accept a cost, otherwise they both accept a string and an expression. Let's look at the implementations.

The top level `define` method declares a function that takes no args with the name of the value, and inserts the expression as the definition when calling it.

Whereas the other define, inserts into the locals table the expression.

```python
In [22]: egraph.let("z", Lit(Int(1)))

In [23]: egraph.extract_expr(Var("z"))
Out[23]: (0, Lit(Int(1)), [])

In [24]: egraph.print_function("z", 10)
Out[24]: '(z) -> 1\n'

In [25]: egraph.eval_actions(Let("a", Lit(Int(1))))

In [26]: egraph.print_function("a", 10)
---------------------------------------------------------------------------
EggSmolError                              Traceback (most recent call last)
Cell In [26], line 1
----> 1 egraph.print_function("a", 10)

EggSmolError: Unbound symbol a

In [27]: egraph.extract_expr(Var("a"))
---------------------------------------------------------------------------
EggSmolError                              Traceback (most recent call last)
Cell In [27], line 1
----> 1 egraph.extract_expr(Var("a"))

EggSmolError: Errors:
Unbound symbol a
```

So how does one access a local variable?
