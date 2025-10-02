---
file_format: mystnb
---

# Python Specific Functionality

Alongside [the support for builtin `egglog` functionality](./egglog-translation.md), `egglog` also provides functionality to more easily integrate with the Python ecosystem.

## Retrieving Values

If you have an egglog value, you might want to convert it from an expression to a native Python object. This is done through a number of helper functions:

For a primitive value (like `i64`, `f64`, `Bool`, `String`, or `PyObject`), use `get_literal_value(expr)` or the `.value` property:

```{code-cell} python
from __future__ import annotations

from egglog import *

assert get_literal_value(i64(42)) == 42
assert get_literal_value(i64(42) + i64(1)) == None # This is because i64(42) + i64(1) is a call expression, not a literal
assert i64(42).value == 42
assert get_literal_value(f64(3.14)) == 3.14
assert Bool(True).value is True
assert String("hello").value == "hello"
assert PyObject([1,2,3]).value == [1,2,3]
```

To check if an expression is a let value and get its name, use `get_let_name(expr)`:

```{code-cell} python
x = EGraph().let("my_var", i64(1))
assert get_let_name(x) == "my_var"
```

To check if an expression is a variable and get its name, use `get_var_name(expr)`:

```{code-cell} python
from egglog import var, get_var_name
v = var("x", i64)
assert get_var_name(v) == "x"
```

For a callable (method, function, classmethod, or constructor), use `get_callable_fn(expr)` to get the underlying Python function:

```{code-cell} python
expr = i64(1) + i64(2)
fn = get_callable_fn(expr)
assert fn == i64.__add__
```

To get the arguments to a callable, use `get_callable_args(expr)`. If you want to match against a specific callable, use `get_callable_args(expr, fn)`, where `fn` is the Python function you want to match against. This will return `None` if the callable does not match the function, and if it does match, the args will be properly typed:

```{code-cell} python
assert get_callable_args(expr) == (i64(1), i64(2))

assert get_callable_args(expr, i64.__add__) == (i64(1), i64(2))
assert get_callable_args(expr, i64.__sub__) == None
```

### Pattern Matching

You can use Python's structural pattern matching (`match`/`case`) to destructure egglog primitives:

```{code-cell} python
x = i64(5)
match i64(5):
    case i64(i):
        print(f"Integer literal: {i}")
```

You can add custom support for pattern matching against your classes by adding `__match_args__` to your class:

```python
class MyExpr(Expr):
    def __init__(self, value: StringLike): ...

    __match_args__ = ("value",)

    @method(preserve=True)
    @property
    def value(self) -> str:
        match get_callable_args(self, MyExpr):
            case (String(value),):
                return value
        raise ExprValueError(self, "MyExpr")

match MyExpr("hello"):
    case MyExpr(value):
        print(f"Matched MyExpr with value: {value}")
```

## Python Object Sort

We define a custom "primitive sort" (i.e. a builtin type) for `PyObject`s. This allows us to store any Python object in the e-graph.

### Saving Python Objects

To create an expression of type `PyObject`, we call the call the constructor with any Python object. It will
save a reference to the object:

```{code-cell} python
PyObject(1)
```

We see that this as saved internally as a pointer to the Python object. For hashable objects like `int` we store two integers, a hash of the type and a has of the value.

We can also store unhashable objects in the e-graph like lists.

```{code-cell} python
lst = PyObject([1, 2, 3])
lst
```

We see that this is stored with one number, simply the `id` of the object.

```{admonition} Mutable Objects
:class: warning

While it is possible to store unhashable objects in the e-graph, you have to be careful defining any rules which create new unhashable objects. If each time a rule is run, it creates a new object, then the e-graph will never saturate.

Creating hashable objects is safer, since while the rule might create new Python objects each time it executes, they should have the same hash, i.e. be equal, so that the e-graph can saturate.
```

### Retrieving Python Objects

Like other primitives, we can retrieve the Python object from the e-graph by using the `.value` property:

```{code-cell} python
assert lst.value == [1, 2, 3]
```

### Builtin methods

Currently, we only support a few methods on `PyObject`s, but we plan to add more in the future.

Conversion to/from a string:

```{code-cell} python
EGraph().extract(PyObject('hi').to_string())
```

```{code-cell} python
EGraph().extract(PyObject.from_string("1"))
```

Conversion from an int:

```{code-cell} python
EGraph().extract(PyObject.from_int(1))
```

We also support evaluating arbitrary Python code, given some locals and globals. This technically allows us to implement any Python method:

```{code-cell} python
EGraph().extract(py_eval("1 + 2"))
```

Executing Python code is also supported. In this case, the return value will be the updated globals dict, which will be copied first before using.

```{code-cell} python
EGraph().extract(py_exec("x = 1 + 2"))
```

Alongside this, we support a function `dict_update` method, which can allow you to combine some local egglog expressions alongside, say, the locals and globals of the Python code you are evaluating.

```{code-cell} python
# Need this from our globals()
def my_add(a, b):
    return a + b

amended_globals = PyObject(globals()).dict_update("one", 1)
evalled = py_eval("my_add(one,  2)", locals(), amended_globals)
assert EGraph().extract(evalled).value == 3
```

### Simpler Eval

Instead of using the above low level primitive for evaluating, there is a higher level wrapper function, `py_eval_fn`.

It takes in a Python function and converts it to a function of PyObjects, by using `py_eval` under the hood.

The above code code be re-written like this:

```{code-cell} python
def my_add(a, b):
    return a + b

evalled = py_eval_fn(lambda a: my_add(a, 2))(1)
assert EGraph().extract(evalled).value == 3
```

## Functions

(type-promotion)=

### Type Promotion

Similar to how an `int` can be automatically upcasted to an `i64`, we also support registering conversion to your custom types. For example:

```{code-cell} python
class Math(Expr):
    def __init__(self, x: i64Like) -> None: ...

    @classmethod
    def var(cls, name: StringLike) -> Math: ...

    def __add__(self, other: Math) -> Math: ...
    def __mul__(self, other: Math) -> Math: ...

converter(i64, Math, Math)
converter(String, Math, Math.var)

Math(2) + i64(30) + String("x")
# equal to
Math(2) + Math(i64(30)) + Math.var(String("x"))
```

You can also specify a "cost" for a conversion, which will be used to determine which conversion to use when multiple are possible. For example `convert(i64, Math, 10)`.

Registering a conversion from A to B will also register all transitively reachable conversions from A to B, so you can also use:

```{code-cell} python
Math(2) + 30 + "x"
```

If you want to have this work with the static type checker, you can define your own `Union` type, which MUST include
the `Expr` class as the first item in the union. For example, in this case you could then define:

```{code-cell} python
MathLike = Math | i64Like | StringLike

@function
def some_math_fn(x: MathLike) -> MathLike:
    ...

some_math_fn(10)
```

### Keyword arguments

All arguments for egg functions must be declared positional or keyword (the default argument type) currently. You can pass arguments variably or also as keyword arguments:

```{code-cell} python
# egg: (function bar (i64 i64) i64)
@function
def bar(a: i64Like, b: i64Like) -> i64:
    pass

# egg: (bar 1 2)
bar(1, 2)
bar(b=2, a=1)
```

### Default arguments

Default argument values are also supported. They are not translated to egglog definition, which has no notion of optional values. Instead, they are added to args when the functions is called.

```{code-cell} python
# egg: (function bar (i64 i64) i64)
@function
def baz(a: i64Like, b: i64Like=i64(0)) -> i64:
    pass

# egg: (baz 1 0)
baz(1)
```

## Methods

When defining a custom class, you are free to use any method names you like.

### Builtin Methods

Most of the Python special dunder (= "double under") methods are supported as well:

- `__lt__`
- `__le__`
- `__eq__`
- `__ne__`
- `__ne__`
- `__gt__`
- `__ge__`
- `__add__`
- `__sub__`
- `__mul__`
- `__matmul__`
- `__truediv__`
- `__floordiv__`
- `__mod__`
- `__pow__`
- `__lshift__`
- `__rshift__`
- `__and__`
- `__xor__`
- `__or__`
- `__pos__`
- `__neg__`
- `__invert__`
- `__getitem__`
- `__call__`
- `__setitem__`
- `__delitem__`

Currently `__divmod__` is not supported, since it returns multiple results.

Also these methods are currently used in the runtime class and cannot be overridden currently, although we could change this
if the need arises:

- `__getattr__`
- `__repr__`
- `__str__`
- `_ipython_display_`
- `__dir__`
- `__getstate__`
- `__setstate__`

### "Preserved" methods

You can use the `@method(preserve=True)` decorator to mark a method as "preserved", meaning that calling it will actually execute the body of the function and a corresponding egglog function will not be created,

Normally, all methods defined on a egglog `Expr` will ignore their bodies and simply build an expression object based on the arguments.

However, there are times in Python when you need the return type of a method to be an instance of a particular Python type, and some similar acting expression won't cut it.

For example, let's say you are implementing a `Bool` expression, but you want to be able to use it in `if` statements in Python. That means it needs to define a `__bool__` methods which returns a Python `bool`, based on evaluating the expression.

```{code-cell} python
egraph = EGraph()
class Boolean(Expr):
    @method(preserve=True)
    def __bool__(self) -> bool:
        # Add this expression converted to a Python object to the e-graph
        egraph.register(self)
        # Run until the e-graph saturates
        egraph.run(10)
        # Extract the Python object from the e-graph
        value = egraph.extract(self)
        if value == TRUE:
            return True
        elif value == FALSE:
            return False
        raise ExprValueError(self, "Boolean expression must be TRUE or FALSE")

    def __or__(self, other: Boolean) -> Boolean:
        ...

TRUE = constant("TRUE", Boolean)
FALSE = constant("FALSE", Boolean)


@egraph.register
def _bool(x: Boolean):
    return [
        rewrite(TRUE | x).to(TRUE),
        rewrite(FALSE | x).to(x),
    ]
```

Now whenever the `__bool__` method is called, it will actually execute the body of the function, and return a Python `bool` based on the result.

```{code-cell} python
if TRUE | FALSE:
    print("True!")
```

Note that the following list of methods are only supported as "preserved" since they have to return a specific Python object type:

- `__bool__`
- `__len__`
- `__complex_`
- `__int_`
- `__float_`
- `__hash_`
- `__iter_`
- `__index__`

If you want to register additional methods as always preserved and defined on the `Expr` class itself, if needed
instead of the normal mechanism which relies on `__getattr__`, you can call `egglog.define_expr_method(name: str)`,
with the name of a method. This is only needed for third party code that inspects the type object itself to see if a
method is defined instead of just attempting to call it.

### Binary Method Conversions

For [rich comparison methods](https://docs.python.org/3/reference/datamodel.html#object.__lt__) (like `__lt__`, `__le__`, `__eq__`, etc.) and [binary numeric methods](https://docs.python.org/3/reference/datamodel.html#object.__add__) (like `__add__`, `__sub__`, etc.), some more advanced conversion logic is needed to ensure they are converted properly. We add the `__r<name>__` methods for all expressions so that we can handle either position they are placed in.

If we have two values `lhs` and `rhs`, we will try to find the minimum cost conversion for both of them, and then call the method on the converted values.
If both are expression instances, we will convert at most one of them. However, if one is an expression and the other
is a different Python value (like an `int`), we will consider all possible conversions of both arguments to find the minimum.

```{code-cell} python
class Int(Expr):
    def __init__(self, i: i64Like) -> None:
        ...

    @classmethod
    def var(cls, name: StringLike) -> Int:
        ...

    def __add__(self, other: Int) -> Int:
        ...


class Float(Expr):
    def __init__(self, i: f64Like) -> None:
        ...

    @classmethod
    def var(cls, name: StringLike) -> Float:
        ...

    @classmethod
    def from_int(cls, i: Int) -> Float:
        ...

    def __add__(self, other: Float) -> Float:
        ...


converter(f64, Float, Float)
converter(Int, Float, Float.from_int)


assert str(-1.0 + Int.var("x")) == "Float(-1.0) + Float.from_int(Int.var(\"x\"))"
```

### Mutating arguments

In order to support Python functions and methods which mutate their arguments, you can pass in the `mutate_first_arg` keyword argument to the `@function` decorator and the `mutates_self` argument to the `@method` decorator. This will cause the first argument to be mutated in place, instead of being copied.

```{code-cell} python
from copy import copy


class Int(Expr):
    def __init__(self, i: i64Like) -> None:
        ...

    def __add__(self, other: Int) -> Int:  # type: ignore[empty-body]
        ...

@function(mutates_first_arg=True)
def incr(x: Int) -> None:
    ...

i = var("i", Int)
incr_i = copy(i)
incr(incr_i)

x = Int(10)
incr(x)
mutate_egraph = EGraph()
mutate_egraph.register(rewrite(incr_i).to(i + Int(1)), x)
mutate_egraph.run(10)
mutate_egraph.check(eq(x).to(Int(10) + Int(1)))
mutate_egraph
```

Any function which mutates its first argument must return `None`. In egglog, this is translated into a function which
returns the type of its first argument.

Note that dunder methods such as `__setitem__` will automatically be marked as mutating their first argument.

## Functions as Values

In Python, functions are first class objects, and can be passed around as values. You can use the builtin `Callable`
type annotation to specify that a function is expected as an argument. You can then pass egglog functions directly
and call them with rewrite rules. For example, here is how you could define a `MathList` class which supports mapping:

```{code-cell} python
from collections.abc import Callable
from typing import ClassVar

class MathList(Expr):
    EMPTY: ClassVar[MathList]

    def append(self, x: Math) -> MathList: ...

    def map(self, fn: Callable[[Math], Math]) -> MathList: ...

@ruleset
def math_list_ruleset(xs: MathList, x: Math, f: Callable[[Math], Math]):
    yield rewrite(MathList.EMPTY.map(f)).to(MathList.EMPTY)
    yield rewrite(xs.append(x).map(f)).to(xs.map(f).append(f(x)))
```

To support partial application, you can use the builtin `functools.partial` function:

```{code-cell} python
from functools import partial

x = MathList.EMPTY.append(Math(1))
added_two = x.map(partial(Math.__add__, Math(2)))

check_eq(added_two, MathList.EMPTY.append(Math(2) + Math(1)), math_list_ruleset.saturate())
```

Note that this is all built on the [unstable function support added as a sort to egglog](https://github.com/egraphs-good/egglog/pull/348).
While this sort is exposed directly at the high level with the `UnstableFn` class, we don't reccomend depending on it directly, and instead
using the builtin Python type annotations. This will allow us to change the implementation in the future without breaking user code.

### Unwrapped functions

We also support using normal python functions, either named or anonymous, as values. These will automatically be wrapped as egglog functions when passed to a function which expects an egglog function.

```{code-cell} python
x = MathList.EMPTY.append(Math(1))
added_two = x.map(lambda x: x + Math(2))
check_eq(added_two, MathList.EMPTY.append(Math(1) + Math(2)), (math_list_ruleset + run()) * 10)
```

Their definition will be added to the default rulset, unless they are defined in the body of a function themselves or
in a rule function:

```{code-cell} python
@function(ruleset=math_list_ruleset)
def map_add_two(x: MathList) -> MathList:
    return x.map(lambda x: x + Math(2))

check_eq(map_add_two(MathList.EMPTY.append(Math(1))), MathList.EMPTY.append(Math(1) + Math(2)), math_list_ruleset.saturate())
```

Their name will just be the body of the function, so that two anonymous functions with the same body will be considered equal.

```{code-cell} python
added_two
```

## Default Replacements

When defining a function or a constant, you can also provide a default replacement value. This is useful when
you might want both the original value and the replaced value in the e-graph, so that later rules could reference either.

```{code-cell} python
@function
def math_float(f: f64Like) -> Math:
    ...


# Can add a default replacement value for a constants
pi = constant("pi", Math, math_float(3.14))


# or for a function by providing a body
@function
def square(x: Math) -> Math:
    return x * x

# thse rewrites will be added to the e-graph under the default ruleset
egraph = EGraph()
egraph.register(pi)
egraph.register(square(Math.var('x')))
egraph.run(1)
egraph.check(eq(pi).to(math_float(3.14)))
egraph.check(eq(square(Math.var('x'))).to(Math.var('x') * Math.var('x')))
egraph
```

This is equivalent to adding the rewrite rules to the e-graph directly, like this, but just more succinct:

```python
x  = var("x", Math)
egraph.register(rewrite(pi).to(math_float(3.14)))
egraph.register(rewrite(square(x)).to(x * x))
```

You can also specify a ruleset to add the rewrites to, by passing in the `ruleset` keyword argument:

```{code-cell} python
math_ruleset = ruleset()

e_constant = constant("e", Math, math_float(2.71), ruleset=math_ruleset)

@function(ruleset=math_ruleset)
def cube(x: Math) -> Math:
    return x * x * x


egraph.register(e_constant)
egraph.register(cube(Math.var('x')))
egraph.run(math_ruleset)
egraph.check(eq(e_constant).to(math_float(2.71)))
egraph.check(eq(cube(Math.var('x'))).to(Math.var('x') * Math.var('x') * Math.var('x')))
```

### Default Replacement for Classes

In classes, you can also provide a default replacement value for constants and methods, and an optional ruleset on the class constructor:

```{code-cell} python
other_math_ruleset = ruleset()


class WrappedMath(Expr, ruleset=other_math_ruleset):
    PI: ClassVar[Math] = math_float(3.14)

    def __init__(self, x: Math) -> None: ...

    def double(self) -> WrappedMath:
        return self + self

    def __add__(self, other: WrappedMath) -> WrappedMath: ...

x = WrappedMath(WrappedMath.PI).double()
egraph = EGraph()
egraph.register(x)
egraph.run(other_math_ruleset * 2)
egraph.check(eq(x).to(WrappedMath(math_float(3.14)) + WrappedMath(math_float(3.14))))
egraph
```

## Visualization

The default renderer for the e-graph in a Jupyter Notebook [an interactive Javascript visualizer](https://github.com/egraphs-good/egraph-visualizer):

```{code-cell} python
egraph
```

You can also customize the visualization through using the <inv:egglog.EGraph.display> method:

```{code-cell} python
egraph.display()
```

If you would like to visualize the progression of the e-graph over time, you can use the <inv:egglog.EGraph.saturate> method to
run a number of iterations and then visualize the e-graph at each step:

```{code-cell} python
egraph = EGraph()
egraph.register(Math(2) + Math(100))
i, j = vars_("i j", i64)
r = ruleset(
    rewrite(Math(i) + Math(j)).to(Math(i + j)),
)
egraph.saturate(r)
```

## Custom Cost Models

By default, when extracting from the e-graph, we use a simple cost model, that looks at the costs assigned to each
function and any custom costs set with `set_cost`, and finds the lowest cost expression looking at the total tree size.

Custom cost models are also supported, which can be passed into `extract` as the `cost_model` keyword argument. They
are defined as functions followed the `CostModel` protocol, that take in an e-graph, an expression, and the costs of the children, and return the total cost of that expression. Costs don't have to be integers, they can be any type that supports comparison.

There are a few builtin cost models:

- `default_cost_model`: The default cost model, which uses integer costs and sums them up.
- `greedy_dag_cost_model(inner_cost_model=default_cost_model)`: A cost model which uses a greedy DAG algorithm to find the lowest cost expression, allowing for shared sub-expressions. It takes in another cost model to use for the base costs of each expression.

Note that when passed into your cost model, the expression won't be a full tree. Instead, only the top level call be present, and all of it's arguments will be opaque "value" expressions, representing e-classes in the e-graph. You can't do much with them except use them to construct other expression to pass into `egraph.lookup_function_value` to get the resulting value of a call with those arguments. The only exception is all builtin types, like ints, vecs, strings, etc. will be fully evaluated recursively, so they can be matched against.

For example, here is a cost model that has a boolean cost if the value is even or not:

```{code-cell} python
def is_even_cost_model(egraph: EGraph, expr: Expr, children_costs: list[bool]) -> bool:
    from egglog import i64  # noqa: PLC0415

    match expr:
        case i64(i):
            return i % 2 == 0
    return False
assert EGraph().extract(i64(10), include_cost=True, cost_model=is_even_cost_model) == (i64(10), True)

assert EGraph().extract(i64(5), include_cost=True, cost_model=is_even_cost_model) == (i64(5), False)
```
