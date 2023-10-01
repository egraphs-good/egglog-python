---
file_format: mystnb
---

# Python Integration

Alongside [the support for builtin `egglog` functionality](./egglog-translation.md), `egglog` also provides functionality to more easily integrate with the Python ecosystem.

## Custom Python Objects

We define a custom "primitive sort" (i.e. a builtin type) for `PyObject`s. This allows us to store any Python object in the e-graph.

### Saving Python Objects

To create an expression of type `PyObject`, we have to use the `egraph.save_object` method. This method takes a Python object and returns an expression of type `PyObject`.

```{code-cell} python
from __future__ import annotations
from egglog import *

egraph = EGraph()
one = egraph.save_object(1)
one
```

We see that this as saved internally as a pointer to the Python object. For hashable objects like `int` we store two integers, a hash of the type and a has of the value.

We can also store unhashable objects in the e-graph like lists.

```{code-cell} python
egraph.save_object([1, 2, 3])
```

We see that this is stored with one number, simply the `id` of the object.

```{admonition} Mutable Objects
:class: warning

While it is possible to store unhashable objects in the e-graph, you have to be careful defining any rules which create new unhashable objects. If each time a rule is run, it creates a new object, then the e-graph will never saturate.

Creating hashable objects is safer, since while the rule might create new Python objects each time it executes, they should have the same hash, i.e. be equal, so that the e-graph can saturate.
```

### Retrieving Python Objects

The inverse of `egraph.save_object` is `egraph.load_object`. This takes an expression of type `PyObject` and returns the Python object it represents.

```{code-cell} python
egraph.load_object(one)
```

### Builtin methods

Currently, we only support a few methods on `PyObject`s, but we plan to add more in the future.

Conversion to/from a string:

```{code-cell} python
egraph.extract(egraph.save_object('hi').to_string())
```

```{code-cell} python
egraph.load_object(egraph.extract(PyObject.from_string("1")))
```

Conversion from an int:

```{code-cell} python
egraph.load_object(egraph.extract(PyObject.from_int(1)))
```

We also support evaling arbitrary Python bode, given some locals and globals. This technically allows us to implement any Python method:

```{code-cell} python
egraph.load_object(egraph.extract(py_eval("1 + 2")))
```

Execing Python code is also supported. In this case, the return value will be the updated globals dict, which will be copied first before using.

```{code-cell} python
egraph.load_object(egraph.extract(py_exec("x = 1 + 2")))
```

Alongside this, we support a function `dict_update` method, which can allow you to combine some local local egglog expressions alongside, say, the locals and globals of the Python code you are evaling.

```{code-cell} python
# Need this from our globals()
def my_add(a, b):
    return a + b

locals_expr = egraph.save_object(locals())
globals_expr = egraph.save_object(globals())
# Need `one` to map to the expression for `1` not the Python object of the expression
amended_globals = globals_expr.dict_update(PyObject.from_string("one"), one)
evalled = py_eval("my_add(one,  2)", locals_expr, amended_globals)
assert egraph.load_object(egraph.extract(evalled)) == 3
```

### Simpler Eval

Instead of using the above low level primitive for evaling, there is a higher level wrapper function, `egraph.eval_fn`.

It takes in a Python function and converts it to a function of PyObjects, by using `py_eval`
under the hood.

The above code code be re-written like this:

```{code-cell} python
def my_add(a, b):
    return a + b

evalled = egraph.eval_fn(lambda a: my_add(a, 2))(one)
assert egraph.load_object(egraph.extract(evalled)) == 3
```

#### Custom Type Promotion

Similar to how an `int` can be automatically upcasted to an `i64`, we also support registering conversion to your custom types. For example:

```{code-cell} python
converter(i64, Math, Math)
converter(String, Math, Math.var)

Math(2) + i64(30) + String("x")
# equal to
Math(2) + Math(i64(30)) + Math.var(String("x"))
```

Regstering a conversion from A to B will also register all transitively reachable conversions from A to B, so you can also use:

```{code-cell} python
Math(2) + 30 + "x"
```

If you want to have this work with the static type checker, you can define your own `Union` type, which MUST include
have the Egglog class as the first item in the union. For example, in this case you could then define:

```{code-cell} python
from typing import Union
MathLike = Union[Math, i64Like, StringLike]

@egraph.function
def some_math_fn(x: MathLike) -> MathLike:
    ...

some_math_fn(10)
```

## "Preserved" methods

You can use the the `@egraph.method(preserve=True)` decorator to mark a method as "preserved", meaning that calling it will actually execute the body of the function and a coresponding egglog function will not be created,

Normally, all methods defined on a egglog `Expr` will ignore their bodies and simply build an expression object based on the arguments.

However, there are times in Python when you need the return type of a method to be an instance of a particular Python type, and some similar acting expression won't cut it.

For example, let's say you are implementing a `Bool` expression, but you want to be able to use it in `if` statements in Python. That means it needs to define a `__bool__` methods which returns a Python `bool`, based on evaluating the expression.

```{code-cell} python
@egraph.class_
class Bool(Expr):
    @egraph.method(preserve=True)
    def __bool__(self) -> bool:
        # Add this expression converted to a Python object to the e-graph
        egraph.register(self)
        # Run until the e-graph saturates
        egraph.run(run().saturate())
        # Extract the Python object from the e-graph
        return egraph.load_object(egraph.extract(self.to_py()))

    def to_py(self) -> PyObject:
        ...

    def __or__(self, other: Bool) -> Bool:
        ...

TRUE = egraph.constant("TRUE", Bool)
FALSE = egraph.constant("FALSE", Bool)


@egraph.register
def _bool(x: Bool):
    return [
        set_(TRUE.to_py()).to(egraph.save_object(True)),
        set_(FALSE.to_py()).to(egraph.save_object(False)),
        rewrite(TRUE | x).to(TRUE),
        rewrite(FALSE | x).to(x),
    ]
```

Now whenever the `__bool__` method is called, it will actually execute the body of the function, and return a Python `bool` based on the result.

```{code-cell} python
if TRUE | FALSE:
    print("True!")
```
