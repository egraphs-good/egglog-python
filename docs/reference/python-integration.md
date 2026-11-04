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

Exact numeric constructors accept Python integers, and `Rational` and `BigRat`
also accept `Fraction` or two integer numerator/denominator arguments. These
host values are available through `.value` immediately:

```{code-cell} python
from fractions import Fraction

assert BigInt(10**100).value == 10**100
assert Rational(1, -2).value == Fraction(-1, 2)
assert BigRat(Fraction(2, 3)).value == Fraction(2, 3)
assert EGraph().extract(BigInt(10**100) + 1).value == 10**100 + 1
assert EGraph().extract(BigInt.from_string("123")).value == 123
```

Expression arguments and primitive operations such as `from_string` remain
symbolic. Extract them before reading `.value`; this property only decodes
value payloads and does not evaluate constructor calls or arithmetic. Invalid
host fractions can fail during construction, while symbolic-operation errors
occur when the engine executes them. Exact numeric host constructor arguments
are positional.

`join` concatenates any number of strings, including none. String addition,
replacement, integer conversion, and Boolean operations build expressions;
extract them to inspect the computed value:

```{code-cell} python
graph = EGraph()
assert graph.extract(join()).value == ""
assert graph.extract(join("hello", " ", String("world"))).value == "hello world"
assert graph.extract((String("abc") + "!").replace("b", "x")).value == "axc!"
assert graph.extract(i64(-42).to_string()).value == "-42"
assert graph.extract(~Bool(False) & (Bool(False) | True)).value is True
assert graph.extract(Bool(True).implies(False)).value is False
```

`bool(Bool(False))` is `False`. Inspecting the truth value of a computed Bool
raises `ExprValueError` until it is extracted. `Unit()` has no `.value` or
`.eval()`; its fact truth check raises `ValueError` because it is not a
structural equality.

`Maybe[T].value` is `None` or a symbolic child of type `T`. It inspects direct
`none`/`some` construction and extracted values without evaluating the child;
`Some(None)` therefore remains distinct from an outer `None`:

```{code-cell} python
none = Maybe[i64].none()
some = Maybe[i64].some(i64(7))
assert none.value is None
assert some.value.value == 7
assert Maybe[Maybe[i64]].some(none).value.value is None
assert EGraph().extract(some.unwrap_or(i64(2))).value == 7
```

`unwrap()` is undefined for `None`. `unwrap_or(default)` evaluates its default
even when the value is present. Python `None` converts to `Maybe[T].none()` when
the expected type supplies `T`.

Direct `Pair`, `Vec`, `Set`, `MultiSet` and `Maybe` construction, and host dict
conversion to `Map`, store their children as data. Inspection returns shallow
Python containers of expression wrappers; it does not execute symbolic children:

```{code-cell} python
symbolic = i64(1) / 0
values = Vec(symbolic, i64(2))
assert values.value[0] == symbolic
assert Pair(symbolic, i64(3)).value[0] == symbolic
assert Maybe[i64].some(symbolic).value == symbolic
mapping = convert({i64(1) + 1: i64(3), i64(2): i64(4)}, Map[i64, i64])
assert len(mapping.value) == 2
assert {key.value: value.value for key, value in EGraph().extract(mapping).value.items()} == {2: 4}
```

`Map.from_pairs`, `insert`, `push`, indexing and other operations remain
symbolic, including when their inputs are direct data. Extract their results
before reading `.value`. Native execution determines set deduplication and map
key collisions; local inspection uses the expression wrappers' structural
identity. Constructor data has no callable for `get_callable_fn` or
`get_callable_args`; use `.value` to inspect its children.

`EGraph.let` stores a value in a graph-local binding and returns an expression
that reads it. Names are exact: `"x"`, `"$x"` and `""` are distinct. Reassigning
a name requires the same sort, and the initializer reads its previous value.
Use `get_let_name(expr)` to inspect the binding name:

```{code-cell} python
bindings_graph = EGraph()
x = bindings_graph.let("my_var", i64(1))
assert get_let_name(x) == "my_var"
bindings_graph.let("my_var", x + 1)
bindings_graph.check(eq(x).to(i64(2)))
bindings_graph.register(set_(x).to(x + 1))
bindings_graph.run(seq(let("my_var", x + 1)) * 2)
bindings_graph.check(eq(x).to(i64(5)))
```

Bindings occupy their own namespace, separate from functions and lexical
variables. `let(name, expr)` authors an assignment for `register` or `seq`;
assignments cannot appear in rule heads. A failed initializer retains the old
binding value and any completed operand effects. `freeze()` preserves both
populated and unset bindings; a saved binding expression reads the current value
in whichever graph executes it.

To check if an expression is a variable and get its name, use `get_var_name(expr)`:

```{code-cell} python
from egglog import var, get_var_name
v = var("x", i64)
assert get_var_name(v) == "x"
```

Variables have one canonical name, selected by `egg_name or name or "_"`.
Both `str()` and `get_var_name()` expose this name; `egg_name` does not retain
a separate Python display identity. Empty names and aliases fall back as shown:

```{code-cell} python
assert get_var_name(var("shown", i64, egg_name="wire")) == "wire"
assert str(var("", i64)) == "_"
assert get_var_name(var("x", i64, egg_name="")) == "x"
```

Within a query, independently created variables with the same canonical name
join, and their sorts must agree. The typed name `"_"` is also a named variable,
not a fresh wildcard. Some canonical names (including `"_"`, `"0"` and `"@x"`)
work in direct protobuf queries but cannot be rendered as source. With
`save_egglog_string=True`, those source errors arise when requesting
`as_egglog_string`, without preventing engine execution.

This differs from the legacy backend's implicit underscore prefix:

| Variable pair | Legacy join | Canonical-name join |
| --- | --- | --- |
| `var("x", i64)` and `var("shown", i64, egg_name="_x")` | Same | Different |
| `var("x", i64)` and `var("shown", i64, egg_name="x")` | Different | Same |

Aliases are exact canonical names now; use `egg_name="x"` to join `var("x", i64)`.
Separate queries and function bodies retain their own binders.

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

To create an expression of type `PyObject`, call the constructor with any Python object. The value is immediately
serialized with `cloudpickle.dumps`. The e-graph stores that snapshot rather than a live reference.

```{code-cell} python
from dataclasses import dataclass

@dataclass
class MyObject:
    a: int = 10

PyObject(MyObject())
```

The new serialization approach works for both hashable and unhashable Python values, and no longer depends on
their `id()`. Subsequent inserts of equal values round-trip through `cloudpickle` so the e-graph can identify and
merge them by value.

```{admonition} Serialization requirements
:class: note

`PyObject` relies on `cloudpickle`. Any object you store must be serializable by `cloudpickle.dumps`; objects such
as open file handles, generators, or extension types that `cloudpickle` cannot handle will raise an error when you
try to construct a `PyObject`.
```

Only load trusted Python objects. Accessing `.value` or `get_literal_value`,
displaying a `PyObject` in a public Python expression, and executing its native
operations may deserialize its pickle and run Python code. Generic protobuf
record inspection, copying, and structural comparison keep Custom payloads opaque.
Public display preserves the Python representation of the object; it does not
evaluate a symbolic Egglog expression to obtain that representation.

### Retrieving Python Objects

Like other primitives, we can retrieve a Python object by using the `.value` property. Deserialization happens on
every access, so you receive a fresh copy each time rather than the original object.

```{code-cell} python
original = {"count": 1}
expr = PyObject(original)

restored = expr.value
assert restored == original
assert restored is not original

# Mutating the copy does not affect the stored value.
restored["count"] = 2
assert expr.value == {"count": 1}
```

### Builtin methods

Currently, we only support a few methods on `PyObject`s, but we plan to add more in the future. Each builtin
deserializes its inputs, performs the operation in Python, and then serializes the result back into a new
`PyObject`, so previously stored values remain unchanged.

### Calling Stored Python Functions

Any `PyObject` whose value is callable can now be invoked directly. Each positional argument is first
converted into a `PyObject`, the call executes inside Python, and the result is serialized back into a new
`PyObject` expression.

```{code-cell} python
scale = PyObject(lambda x, *, factor=1: x * factor)
result = scale(21)

assert EGraph().extract(result).value == 21
```

When you already have the arguments packaged up, use `call_extended` to forward explicit `*args`/`**kwargs`
collections. Use this form for keyword arguments, which the simple call does not support:

```{code-cell} python
args = PyObject((10,))
kwargs = PyObject({"factor": 3})

assert EGraph().extract(scale.call_extended(args, kwargs)).value == 30
```

### Expected Primitive Failure

A Python callback can raise `PrimitiveFailure` (imported from `egglog`) when an
input has no result. Subclasses have the same meaning. In a rule query this is
an ordinary non-match; direct evaluation fails with the native evaluation error.
It does not roll back earlier effects or retain a Python callback error, and
later callbacks can still run when native execution continues.

Returning Python `None` is different: it is a successful `PyObject(None)` value.
Other callback exceptions propagate as their original Python exception objects,
not as expected primitive failures.

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

Executing Python code is also supported. The return value is the updated copy of
the locals dictionary. Omitted globals and locals use symbolic `PyObject.dict()`
defaults; neither operation mutates a dictionary supplied by the caller.

```{code-cell} python
EGraph().extract(py_exec("x = 1 + 2"))
```

Alongside this, we support a function `dict_update` method, which can allow you to combine some local egglog expressions alongside, say, the locals and globals of the Python code you are evaluating.

```{code-cell} python
# Need this from our globals()
def my_add(a, b):
    return a + b

amended_globals = PyObject({"my_add": my_add}).dict_update("one", 1)
evalled = py_eval("my_add(one,  2)", {}, amended_globals)
assert EGraph().extract(evalled).value == 3
```

### `py_eval_fn` (deprecated)

The previous helper `py_eval_fn` is still available for backward compatibility, but it now simply returns
`PyObject(fn)` and is deprecated. Prefer constructing a `PyObject` directly and calling it as shown above.

`py_eval_fn` returns a callable that mirrors the old behaviour, so existing code continues to work while you migrate.

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

When defining converters for a custom `Expr` sort, prefer registering conversions from egglog primitive sorts such as `i64`, `f64`, and `String` rather than directly from Python builtins like `int`, `float`, and `str`. The builtin promotions already handle those Python values transitively, so keeping the custom converters at the egglog-sort layer makes the promotion path clearer and usually leads to cleaner `...Like` aliases such as `Math | i64Like | f64Like | StringLike`.

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

You can use the `@method(preserve=True)` decorator to mark a method as
"preserved", meaning that calling it executes ordinary Python behavior and no
corresponding egglog function is created.

A bodyless method written with `...` builds an egglog call. A body-defined
method is lowered eagerly, or as a rewrite when an eqsort method has an
explicit ruleset, as described in the
[function translation rules](egglog-translation.md#functions-vs-constructors).
Use `preserve=True` when the method must instead return an ordinary Python
value or perform ordinary Python behavior.

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

Preserved methods now support binary operators as well. You can mark implementations like `__add__` or `__lt__` as
preserved when you need their Python behaviour, and the runtime will call your preserved implementation for both the
regular and reflected (`__r*__`) variants.

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

In order to support Python functions and methods which mutate their arguments, use the `mutates_first_arg` keyword argument on `@function` and the `mutates_self` keyword argument on `@method`. The runtime treats the mutated receiver as the semantic return value of the egglog call. A body lowers eagerly unless it is attached to an explicit `ruleset`, in which case the updated expression becomes that callable's default rewrite.

Inside the Python implementation you can call `__replace_expr__` on an `Expr` instance to swap out its underlying egglog expression in-place. This keeps any existing Python references in sync while still allowing the e-graph to reason about the mutated value. The same helper works for methods that run immediately with `@method(preserve=True)`.

Any function which mutates its first argument must return `None`. In egglog, this is translated into a function which
returns the type of its first argument.

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

# The update can instead be defined as an eager body:
@function(mutates_first_arg=True)
def incr_other(x: Int) -> None:
    x.__replace_expr__(x + Int(1))
x = Int(10)
incr_other(x)
mutate_egraph = EGraph()
incremented = mutate_egraph.let("incremented", x)
mutate_egraph.check(eq(incremented).to(Int(10) + Int(1)))
mutate_egraph
```

The body executes eagerly, so it does not add a rewrite or require a `run()`.
The `let` action registers the resulting e-class value before the read-only
check. Use an explicit `ruleset=` when the body should remain a rewrite.

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

`UnstableFn[Result, *Arguments]` uses the result type first. For a named function
value, `.value` inspects it without running egglog: it returns the callable, or
a `functools.partial` containing its captured prefix. Captured expressions stay
symbolic until the function value is constructed in the engine; reading `.value`
does not evaluate them. A nonliteral function expression must be extracted before
its `.value` is available. For an anonymous or computed function value, `.value`
returns the symbolic callable itself; invoking it authors an application without
running its body in Python. Mutating callables cannot be partially applied.

A free `@function` with concrete argument and result sorts, fixed
positional-or-keyword parameters, a body, and no explicit `ruleset` defines eager
computation, including when its result is an equality sort. Its
Python body runs once with symbolic arguments when the declaration is first
needed. The resulting calls and table reads execute when egglog applies the
function; authoring or inspecting a function value does not execute them. A
bodyless declaration still defines a table or constructor. Named eager bodies
can call other named eager functions and capture positional prefixes with
`UnstableFn`; free query variables and recursive bodies are not supported.

### Unwrapped functions

We also support using normal python functions, either named or anonymous, as values. These will automatically be wrapped as egglog functions when passed to a function which expects an egglog function.

```{code-cell} python
x = MathList.EMPTY.append(Math(1))
added_two = x.map(lambda x: x + Math(2))
check_eq(added_two, MathList.EMPTY.append(Math(1) + Math(2)), math_list_ruleset.saturate())
```

Converting the callback to `Callable`/`UnstableFn` authors a Lambda body with
explicit symbolic captures. Captures are evaluated when the function value is
constructed in the engine; its body runs on application. It does not add a
rewrite to the default ruleset or require a separate `run()` step. An enclosing declared function still follows
the ordinary lowering rules; for example, an explicit `ruleset` makes this
outer body rewrite-backed:

```{code-cell} python
@function(ruleset=math_list_ruleset)
def map_add_two(x: MathList) -> MathList:
    return x.map(lambda x: x + Math(2))

check_eq(map_add_two(MathList.EMPTY.append(Math(1))), MathList.EMPTY.append(Math(1) + Math(2)), math_list_ruleset.saturate())
```

## Default Replacements

The full lowering matrix for functions, constructors, primitives, and defaults is documented in
[Translation to/from egglog](egglog-translation.md#functions-vs-constructors). This section focuses on the
Python-only ergonomics for the explicit-`ruleset` case: when you pass a `ruleset`, the body/default is added as
a rewrite into that ruleset instead of being lowered eagerly.

This is useful when you want the declared name and the replacement body to both remain available to later rewrite rules.

You can specify a ruleset for a default replacement by passing the `ruleset` keyword argument:

```{code-cell} python
math_ruleset = ruleset()

@function
def math_float(value: f64Like) -> Math: ...


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

This rewrite-backed path is only available for eqsort-returning bodies and defaults. Primitive-returning defaults lower
eagerly and cannot use an explicit `ruleset`.

When `subsume=True` is allowed for that callable shape, it applies on this rewrite-backed path as well.

Constants without defaults can use merge functions because they lower as zero-argument function-style declarations:

```{code-cell} python
best_score = constant("best_score", i64, merge=lambda old, new: old.max(new))

egraph = EGraph()
egraph.register(set_(best_score).to(i64(1)), set_(best_score).to(i64(2)))
egraph.check(eq(best_score).to(i64(2)))
```

Constants with eager or rewrite-backed defaults cannot also use `merge`.

### Default Replacement for Classes

In classes, a `ruleset=` on the class means default method and class-variable bodies are also added to that ruleset as rewrites:

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

## Debugging and Inspection

When a rule does not fire or an equality appears unexpectedly, the most useful
high-level inspection methods are `run`, `stats`, `function_values`, `freeze`,
`display`, and `saturate`.

```{code-cell} python
from __future__ import annotations

from egglog import *


class DebugMath(Expr):
    def __init__(self, value: i64Like) -> None: ...

    def __add__(self, other: DebugMath) -> DebugMath: ...


@function
def score(x: DebugMath) -> i64: ...


debug_rules = ruleset()


@debug_rules.register
def _(i: i64, j: i64):
    yield rewrite(DebugMath(i) + DebugMath(j)).to(DebugMath(i + j))
```

### `run`

Use {meth}`egglog.egraph.EGraph.run` to execute a schedule and inspect the
two semantic flags in its generated `RunResult`:

```{code-cell} python
egraph = EGraph()
expr = egraph.let("expr", DebugMath(2) + DebugMath(3))
egraph.register(set_(score(expr)).to(5))

result = egraph.run(debug_rules)
result.updated, result.can_stop
```

`result.updated` records whether the run changed the database. The separate
`result.can_stop` flag is true only when the run observed no changes and its
scheduler has no deferred work that requires another iteration.

Detailed metric collection is off by default. Request an opaque backend document
with `result, diagnostic = egraph.run_with_diagnostics(debug_rules)`.
`diagnostic.format` identifies its format and `diagnostic.content` carries the
text or JSON. Its backend fields are diagnostic data rather than a portable
report API. `egraph.explain(*groups)` inspects plans for already installed group
names without executing rules; omitting groups selects the default group.

Schedules author sequencing, capped repetition, saturation, BackOff bindings
and conjunctive `until` queries as canonical records without an engine.
Local scheduler scopes retain their explicit placement. Persistent BackOff
bindings and scheduler progress follow graph push/pop.
Repetition stops early when its body permits stopping; a count is a cap rather
than a promise to execute every iteration. Zero runs no body or stopping check.
Constructing a schedule does not install rules; printing it can materialize deferred
rule generators, and later appended rules remain visible when it is run.

### Live observations

Pass `on_output=callback` to `run` or `run_with_diagnostics` to consume owned
protobuf response chunks while execution is in progress. For snapshots,
`schedule.with_snapshots()` appends one Freeze after the selected schedule.
Place that schedule inside a loop to choose the observation cadence, for example
`debug_rules.with_snapshots() * 3`. `EGraph.saturate()` uses these live snapshots
for its existing per-round visualization. Notebook widgets update as snapshots
arrive; the standalone browser view opens the collected history after completion.

The callback is synchronous, so a slow consumer pauses execution without a
growing queue. Return `False` to cancel at a safe command boundary. Completed
effects and delivered observations remain available after cancellation or a
later error; cancellation does not roll back. Inspect owned snapshots in the
callback; operations on the executing graph are rejected. Callback exceptions
retain their original Python identity. Buffered calls collect the same canonical
stream. Input history continues to record inputs only.

### Exporting input history as source

Create with `save_egglog_string=True` or `record_history=True`, then request
`as_egglog_string` to render the current graph's accepted input history using the
shared Rust Source renderer. Export is lazy: unsupported source forms raise at
export without preventing engine execution. Custom payloads remain inert.

Pop restores the parent's history prefix and omits the abandoned branch; no
push/pop commands are added to the input Program. Accepted failed requests
retain their whole input, including unexecuted suffixes. The text therefore does
not promise exact replay after failure, and failures do not disable later engine
operations. Observation-only Freeze commands are omitted from source while their
setup is retained. `close()` is a compatibility no-op; source needs no separate
finalization. Initial `seminaive=False` is supported for execution/history but
source export reports the unsupported creation context.

### `stats`

Use {meth}`egglog.egraph.EGraph.stats` for a cumulative backend `Diagnostic` in
the same opaque format:

```{code-cell} python
stats = egraph.stats()
stats.format
```

### `function_values`

Use {meth}`egglog.egraph.EGraph.function_values` to inspect the current rows in a
function table. This accepts relations, constructors, bodyless functions, and
bodyless constants; eager and builtin primitives do not have tables to inspect:

```{code-cell} python
egraph.function_values(score)
```

If any selected argument or output has no extractable representative,
`function_values` raises `ValueError` identifying the table, zero-based row, and
argument index or output. It does not return a partial dictionary or an
`Unextractable` placeholder. `length=0` selects no rows and returns `{}`.

For row counts without extracting values, use
{meth}`egglog.egraph.EGraph.function_size` or
{meth}`egglog.egraph.EGraph.all_function_sizes`. Counts include subsumed rows and
rows without extractable representatives. `function_size` registers an untouched
table and returns zero; `all_function_sizes` only observes registered tables.

### `freeze`

Use {meth}`egglog.egraph.EGraph.freeze` to snapshot the current state into a
generated protobuf `EGraphSnapshot`. The snapshot owns its reconstruction
`Program` and logical options, independently of the live handle:

```{code-cell} python
frozen = egraph.freeze()
type(frozen.program).__name__
```

`frozen.to_binary()` serializes the snapshot without materializing host values.
It records logical state rather than request history or scheduler progress.
The shared snapshot contains semantic definitions, types and values. Python
names and call conventions remain in the frontend context; a detached snapshot
does not carry enough information to reconstruct an identical Python API.
Snapshots retain documentation and rule labels, but omit source files and spans,
including origins in saved rules and callable bodies. Restoring a snapshot does
not restore those source locations; later inputs can supply their own locations.
Builtin signatures are also separate: the snapshot refers to the configured
backend's canonical keys and needs that builtin environment for frontend typing.

### `history`

Create an {class}`egglog.egraph.EGraph` with `record_history=True` or
`save_egglog_string=True` to retain accepted input programs, then retrieve one owned accumulated `Program` with
{meth}`egglog.egraph.EGraph.history`. Recording is off
by default; requesting disabled history raises `ValueError`. Retrieval itself
does not append a command.

Retrieve history before destroying the graph; destruction releases its retained
inputs. Other live graphs and clones retain their own histories.

History remaps each accepted program's references into shared arenas while
preserving command order, node sharing and input source locations. It includes
observations such as
`freeze` and accepted inputs whose execution later fails. Validation and
preparation rejects are omitted. `pop` restores the saved history prefix,
discarding inputs from the consumed branch. History contains inputs rather than
outputs or outcomes and does not promise deterministic replay or an engine
checkpoint.
Like snapshots, history contains no Python presentation map or unused default
templates. Defaults enter an input program when a call actually omits them.

Both flags enable the same native input store, which also supports lazy
`as_egglog_string` export. Export limitations do not alter engine admission or
invalidate retained inputs. Structured history supports `seminaive=False` even
though source export cannot encode that initial creation setting.

### `display`

{meth}`egglog.egraph.EGraph.display` projects the current protobuf snapshot into
the interactive widget, or Graphviz when called with `graphviz=True`. Equality
alternatives, function/relation rows, shared containers and cycles come directly
from its reconstruction records. An empty stored class is displayed as `∅`;
this is a display placeholder, not a constructor inserted into the graph.

`max_functions`, `max_calls_per_function`, `split_primitive_outputs`,
`split_functions` and `n_inline_leaves` affect only the visualization. Omitted or
truncated tables emit warnings; filtering cannot truncate or mutate the snapshot.
Backend temporary functions are already absent from the logical export, so
`include_temporary_functions` does not change it. Opaque host values and costs
remain opaque; visualization does not unpickle or evaluate them.

### `saturate`

Use {meth}`egglog.egraph.EGraph.saturate` to keep running until the schedule
reports no graph changes or deferred scheduler work, while printing the
extracted form after each step. By default the widget also shows snapshot
history, including the final observed failure prefix if a step raises. Use
`visualize=False` to disable the widget:

```{code-cell} python
egraph = EGraph()
expr = egraph.let("expr", DebugMath(2) + DebugMath(100))
egraph.saturate(debug_rules, expr=expr, max=2, visualize=False)
```

Common pitfalls when authoring rules:

- Primitive container sorts like `Vec[...]` should not be merged or unioned.
- Guard vector indexing rules with bounds checks (`0 <= k < vs.length()`).
- Ensure rules that subtract from lengths only fire when the length is proven
  positive.

## Extraction and Cost Models

{meth}`egglog.egraph.EGraph.extract` accepts `extractor="tree"` (the
default) or `extractor="greedy-dag"`. Tree extraction charges each occurrence
of a subexpression. Greedy-DAG extraction charges shared subexpressions once
within the result; it is a heuristic rather than a globally optimal DAG
extractor. The public `ExtractionMode` alias contains these two values.
With `include_cost=True`, `extract` returns `(expression, cost)`; custom model
costs are returned directly rather than through a wrapper object.

### Dynamic Costs

Ordinary single-root tree extraction uses Egglog's default tree cost model.
When no explicit `cost_model` is supplied, registering a row cost with
`set_cost`, selecting greedy-DAG extraction, or using multi-root extraction or
`keep_best` selects the experimental dynamic cost model. A row cost then
overrides that node's marginal cost; otherwise the model falls back to costs
declared on callables and then the backend default. An explicit `TreeCostModel`
or `DagCostModel` takes precedence over those dynamic row costs.
Dynamic row costs must be nonnegative.

### Multiple Roots

{meth}`egglog.egraph.EGraph.extract_multiple` returns up to `n` variants for
one expression. Passing a non-empty sequence performs one extraction for all
roots and returns one variant list per root in the same order:

```{code-block} python
variants = egraph.extract_multiple(expr, 3, extractor="tree")
variants_by_root = egraph.extract_multiple([expr1, expr2], 3, extractor="greedy-dag")
variants_with_costs = egraph.extract_multiple(expr, 3, include_cost=True, cost_model=model)
```

An inner list may be empty when its root has no extractable variant. The
variant count must be positive, and the sequence form rejects an empty input.
The roots share extraction preparation, but every root and variant is costed
independently; sharing between separate roots does not reduce either cost. The
sequence form uses dynamic costs when no explicit model is supplied; the
single-expression form follows the ordinary `extract` selection described
above. Both forms accept a custom `cost_model`; greedy-DAG extraction requires
a `DagCostModel`. With `include_cost=True`, each variant is an
`(expression, cost)` pair. Default-model costs are integers, and custom-model
costs retain the model's result type. An unencodable requested cost raises an
error, just as it does for `extract`.

### Custom Tree Cost Models

A `TreeCostModel` is a callable that receives the e-graph, one expression
node, and the total costs of its immediate children, then returns the total
cost of that expression. Cost values may be any totally ordered type. The old
`CostModel` name remains as a compatibility alias for this protocol.
Models should normally return a cost no smaller than any child; non-monotone
models are responsible for avoiding cycles in the extracted term.

The expression passed to a custom model contains only its top-level call.
Arguments representing e-classes are opaque value expressions, although
builtin values such as numbers and containers are reconstructed recursively.
Use {meth}`egglog.egraph.EGraph.lookup_function_value` when a model needs to
inspect a table registered before extraction using those callback arguments
directly. A same-e-graph lookup cannot evaluate a newly derived key while
extraction is holding the graph read-only; such a lookup raises `ValueError`.
Opaque class values belong to the current observation; they cannot identify
live classes in another lookup, extraction request, or e-graph. Portable
primitive results can be reused. Outside callbacks, lookup reads a fresh snapshot
without evaluating keys or inserting rows; the table must already be registered,
and missing keys or rows return `None`.
Use `egraph.register(expr)` for materialization, or
`saved = egraph.let("saved", original_call)` to capture a live read for later
lookup or extraction. A binding reads its stored value, not its initializer, so
source-row updates or deletion do not retarget it. This deliberately replaces
the earlier prototype's implicit key materialization and raw-value retention.

For example, this model uses a boolean cost for whether an `i64` is even:

```{code-cell} python
def is_even_cost_model(egraph: EGraph, expr: BaseExpr, children_costs: list[bool]) -> bool:
    from egglog import i64  # noqa: PLC0415

    match expr:
        case i64(i):
            return i % 2 == 0
    return False
assert EGraph().extract(i64(10), include_cost=True, cost_model=is_even_cost_model) == (i64(10), True)

assert EGraph().extract(i64(5), include_cost=True, cost_model=is_even_cost_model) == (i64(5), False)
```

A `TreeCostModel` can only be used with the tree extractor. Passing one with
`extractor="greedy-dag"` raises `TypeError`, because a callback that returns
total child costs does not expose the marginal costs needed to account for
sharing.

### Additive DAG Cost Models

`DagCostModel(marginal_cost, identity)` is a frozen model that can be used with
either extraction mode. Its callback returns the cost of the current node
without its children or container elements. The backend combines those values
with Python `+`.

Cost values must be effectively immutable and totally ordered. Addition must
be associative, commutative, and monotone, with `identity` as a two-sided
identity. Under tree extraction the values are added once per occurrence;
under greedy-DAG extraction they are added once per selected shared node.
When using tree extraction, recursive e-classes must not contain a
cost-improving cycle or cost computation may not terminate.

```{code-block} python
model = DagCostModel(
    marginal_cost=lambda egraph, node: default_cost_model(egraph, node, []),
    identity=0,
)

tree_result, tree_cost = egraph.extract(expr, include_cost=True, cost_model=model)
dag_result, dag_cost = egraph.extract(
    expr,
    include_cost=True,
    cost_model=model,
    extractor="greedy-dag",
)
```

### Keeping Only the Best Representatives

{meth}`egglog.egraph.EGraph.keep_best` compacts table-backed callables using
dynamic costs:

```{code-block} python
egraph.keep_best(target, other_target, extractor="greedy-dag")
```

Each target must be a constructor, relation, bodyless function, or bodyless
constant with a backend table; eager and builtin primitives are rejected.

This operation is destructive. It clears all old rows, then rebuilds the
extracted rows of the requested callables and any constructor rows needed to
represent their values. Unreferenced rows remain absent. Declarations and
cost-table identities remain available, although their rows are cleared.
Bindings returned by {meth}`egglog.egraph.EGraph.let` use their own typed slot
namespace and cannot be selected as callable targets. Compaction can unset their
values while retaining the typed slots. Reading an unset binding does not
reinitialize it; popping a saved scope restores its stored bindings.
Snapshot-owned lookup class values remain observation-local, independently of
compaction. The same `EGraph` can
continue registering new actions and running rules afterward. Call this method
only when dropping all unselected table state is intended.

## Thread safety

Python threads can construct expressions concurrently and run independent
`EGraph` instances after single-threaded setup. Importing `egglog` on
free-threaded CPython does not re-enable the GIL.

On the defining thread, register representative expressions with a setup
e-graph, including the conversions, parameterized types, and constants workers
will use. Load each shared ruleset with `setup_graph.run(0, ruleset=rules)`.
This resolves lazy declarations and rule generators without running the rules.
Do not define or update classes, functions, converters, or rulesets during
concurrent use.

Serialize all access to a shared e-graph. Serialize access to shared expressions
when any thread mutates them. Python callbacks must synchronize shared mutable state.
The experimental `set_array_api_egraph` and `set_any_expr_egraph` contexts affect
all threads: other threads must not use the corresponding API while a context
is active. See the [low-level bindings](bindings.md#thread-safety) for their
thread-safety requirements.

## Ruleset snapshots

Pass ruleset objects directly to `run(...)` or `EGraph.run(...)`. A schedule
reads each mutable ruleset when it is submitted, so later `append` and
`register` calls are included in subsequent runs. Anonymous collections become
literal selections; named collections replace the corresponding engine binding
with their complete current contents. Anonymous combinations select their component literals and names in order.
A named combined collection captures its current members, without installing
a live dependency between its name and its components.

`EGraph.register(...)` keeps accumulating rules in its default collection across
calls. Registering actions preserves their order, while rule definitions are
installed as setup before those actions. Replacing a collection changes future
membership; it preserves effects and matching progress from earlier runs.
