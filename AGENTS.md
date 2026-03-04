

# Debug notes for egglog / python bindings (array_api focus)

This is a short scratchpad of things learned while debugging `python/tests/test_array_api.py::test_jit[lda]`.

## How evaluation works (python bindings)
- `Boolean.to_bool` is a primitive `Bool` output. If no rule sets it, extraction fails with `lookup of function ... Boolean_to_bool`.
- `try_evaling(...)` in `python/egglog/exp/array_api.py` first tries `egraph.extract(prim_expr)`, then registers the expr and runs `array_api_schedule`. If that fails it retries with a fresh `EGraph` to avoid cross-expression contradictions. Panics inside rules (e.g., `False cannot equal True`) still bubble.
- A common failure path is: a boolean expression doesn’t reduce to `TRUE/FALSE` so `to_bool` never gets set.

## Vecs, to_vec, and illegal merges
- `Vec[...]` is a primitive sort in egg-smol. You cannot rewrite or union two Vec values safely; this causes `Illegal merge attempted for function ..._to_vec` or `Cannot union values of sort Vec[...]`.
- `TupleInt.to_vec` was wired via `set_(ti.to_vec).to(vs)` when `ti == TupleInt.from_vec(vs)`. If multiple Vec representations become equal (e.g., `vec-of` vs `append`), this can cause illegal merges.
- If you ever add `array_api_vec_to_cons_ruleset` to the main schedule, it can create multiple Vec representations and re-trigger illegal merges. Keeping vec-to-cons rules out of the main schedule avoids this class of panic.
- As a temporary mitigation I added a merge on `TupleInt.to_vec` to keep the old Vec (`@method(merge=lambda old,_new: old)`), but this can mask real inconsistencies.

## Tuple/Vec helpers that unblock evaluation
- To allow extraction to reason about tuple shapes, these rewrites are useful:
  - `TupleInt.from_vec(vs).length() -> Int(vs.length())`
  - `TupleInt.from_vec(vs)[Int(k)] -> vs[k]` guarded by `k >= 0` and `k < vs.length()`
  - `TupleValue.from_vec(vs).length() -> Int(vs.length())`
- Guard `vec-get` accesses; unguarded rules can produce `vec-get failed` panics when indices are not provably in-bounds.

## LDA failure shape (what expression was stuck)
- The failing boolean in `test_jit[lda]` eventually reduces to the sklearn priors check:
  - `abs(sum(astype(unique_counts(y)[1], dtype) / 150.0) - 1.0) > 1e-5`
- The system needs a rule that normalizes `sum(astype(unique_counts(x)[1], dtype) / Float.from_int(x.size))` to `1.0`.
- Without that, the boolean never reduces to `TRUE/FALSE`, so `Boolean.to_bool` lookup fails.

## concat / unique_values simplifications
- `concat(TupleNDArray.EMPTY.append(x)) -> x` exists, but `concat(TupleNDArray.from_vec(vs))` with `vs.length()==1` also needs a rewrite to avoid leaving a nested `concat` term.
- `unique_values(x)` rewrites to `NDArray.vector(possible_values(x.index(ALL_INDICES)))`, which pushes the problem into `possible_values(...)` and tuple lengths. If `possible_values(...)` doesn’t reduce to a concrete `TupleValue`, shape queries can get stuck.

## Common panic/error messages and likely causes
- `lookup of function ... Boolean_to_bool failed`: no rewrite set the primitive `Bool` for that boolean expression.
- `Illegal merge attempted for function ..._to_vec`: two distinct Vec values were merged for a function output.
- `vec-get failed`: a rewrite attempted `vs[k]` without a provable `0 <= k < vs.length()`.
- `False cannot equal True`: a contradictory rule chain produced both booleans in the same e-class.

## Files worth checking first
- `python/egglog/exp/array_api.py`: rulesets, `try_evaling`, tuple/vec conversions, ndarray ops.
- `python/egglog/egraph_state.py`: error paths and how rule commands are compiled.
- `python/egglog/exp/program_gen.py`: example of a safe `@method(merge=...)` on properties.

# Context7

Use context7 MCP server on https://context7.com/egraphs-good/egglog-python to understand docs.

Here is a copy of the summary

# egglog Python Library

egglog is a Python library providing high-level bindings to the Rust [egglog](https://github.com/egraphs-good/egglog/) library, enabling e-graph-based equality saturation in Python. E-graphs are a powerful data structure that efficiently represents many equivalent programs simultaneously, making them ideal for program optimization, theorem proving, and symbolic computation tasks.

The library offers a Pythonic API for defining custom expression types, rewrite rules, and Datalog-style relational rules. It supports automatic term extraction based on cost models, union-find operations for equivalence classes, and built-in primitive types (integers, floats, strings, rationals, vectors, sets, maps). The core workflow involves creating an EGraph, registering expressions and rewrite rules, running saturation, and extracting optimized results.

## EGraph - Core E-Graph Operations

The EGraph class is the central data structure that maintains equivalence classes of expressions. It supports registering expressions, running rewrite rules, checking facts, and extracting optimal expressions based on cost models.

```python
from egglog import *

# Create an EGraph instance
egraph = EGraph()

# Define a custom expression type
class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...
    @classmethod
    def var(cls, name: StringLike) -> Num: ...
    def __add__(self, other: Num) -> Num: ...
    def __mul__(self, other: Num) -> Num: ...

# Register expressions with let() to get references
expr1 = egraph.let("expr1", Num(2) * (Num.var("x") + Num(3)))
expr2 = egraph.let("expr2", Num(6) + Num(2) * Num.var("x"))

# Register expressions directly
egraph.register(Num(1) + Num(2))

# Run rules for 10 iterations
egraph.run(10)

# Extract the lowest-cost equivalent expression
result = egraph.extract(expr1)

# Extract with cost information
result, cost = egraph.extract(expr1, include_cost=True)

# Check if expressions are equivalent
egraph.check(expr1 == expr2)  # Raises if not equivalent

# Check that expressions are NOT equivalent
egraph.check_fail(Num(1) == Num(2))

# Push/pop state for backtracking
egraph.push()
egraph.register(Num(100))
egraph.pop()  # Reverts state

# Context manager for automatic push/pop
with egraph:
    egraph.run(5)
    result = egraph.extract(expr1)
# State reverted here
```

## Expr - Defining Custom Expression Types

Subclass Expr to define domain-specific languages with typed constructors, methods, and operators. Methods become egglog functions automatically.

```python
from egglog import *
from typing import TypeAlias, ClassVar

# Define a math expression type
class Math(Expr):
    # Constructor with i64 parameter
    def __init__(self, value: i64Like) -> None: ...

    # Class method constructor
    @classmethod
    def var(cls, name: StringLike) -> Math: ...

    # Operators become egglog functions
    def __add__(self, other: Math) -> Math: ...
    def __mul__(self, other: Math) -> Math: ...
    def __truediv__(self, other: Math) -> Math: ...

    # Property-style functions
    @property
    def is_zero(self) -> Unit: ...

    # Class variables (constants)
    ZERO: ClassVar[Math]

# Type alias for automatic conversion
MathLike: TypeAlias = Math | i64Like | StringLike

# Register converters so 2 + Math(3) works
converter(i64, Math, Math)
converter(String, Math, Math.var)

# Usage
egraph = EGraph()
x = Math.var("x")
expr = x * 2 + 3  # Automatically converts 2 and 3 to Math
egraph.register(expr)
```

## rewrite() - Defining Rewrite Rules

The rewrite() function creates conditional rewrite rules that transform expressions when patterns match. Rules are added to rulesets and executed during egraph.run().

```python
from egglog import *

class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...
    @classmethod
    def var(cls, name: StringLike) -> Num: ...
    def __add__(self, other: Num) -> Num: ...
    def __mul__(self, other: Num) -> Num: ...

converter(i64, Num, Num)
converter(String, Num, Num.var)

egraph = EGraph()

# Create pattern variables
a, b, c = vars_("a b c", Num)
i, j = vars_("i j", i64)

# Register rewrite rules using decorator pattern
@egraph.register
def _(x: Num, y: Num, z: Num):
    # Commutativity
    yield rewrite(x + y).to(y + x)
    yield rewrite(x * y).to(y * x)
    # Associativity
    yield rewrite(x + (y + z)).to((x + y) + z)
    yield rewrite(x * (y * z)).to((x * y) * z)
    # Distributivity
    yield rewrite(x * (y + z)).to((x * y) + (x * z))

# Constant folding with primitive operations
@egraph.register
def _(i: i64, j: i64):
    yield rewrite(Num(i) + Num(j)).to(Num(i + j))
    yield rewrite(Num(i) * Num(j)).to(Num(i * j))

# Conditional rewrites
@egraph.register
def _(x: Num, y: Num):
    # Only rewrite x/x to 1 if x.is_nonzero fact exists
    yield rewrite(x / x).to(Num(1), x.is_nonzero)

# Bidirectional rewrites (works both directions)
@egraph.register
def _(x: Num, y: Num, z: Num):
    yield birewrite(x + (y + z)).to((x + y) + z)

# Run rules and verify equivalence
expr1 = egraph.let("e1", Num(2) * (Num.var("x") + Num(3)))
expr2 = egraph.let("e2", Num(6) + Num(2) * Num.var("x"))
egraph.run(10)
egraph.check(expr1 == expr2)
```

## rule() - Defining Datalog-Style Rules

The rule() function creates Datalog-style rules with arbitrary actions (not just rewrites). Rules query facts and trigger actions like union, set, or panic.

```python
from egglog import *

# Define relations for graph connectivity
edge = relation("edge", i64, i64)
path = relation("path", i64, i64)

egraph = EGraph()

# Insert edge facts
egraph.register(
    edge(i64(1), i64(2)),
    edge(i64(2), i64(3)),
    edge(i64(3), i64(4)),
)

# Datalog rules: derive path from edges
@egraph.register
def _(a: i64, b: i64, c: i64):
    # Base case: edge implies path
    yield rule(edge(a, b)).then(path(a, b))
    # Transitive case: path + edge implies longer path
    yield rule(path(a, b), edge(b, c)).then(path(a, c))

# Run until saturation (fixed point)
egraph.run(run().saturate())

# Check derived facts
egraph.check(path(i64(1), i64(4)))

# Functions with merge semantics (e.g., shortest path)
@function(merge=lambda old, new: old.min(new))
def dist(from_: i64Like, to: i64Like) -> i64: ...

egraph = EGraph()
egraph.register(set_(dist(1, 2)).to(i64(10)))
egraph.register(set_(dist(2, 3)).to(i64(5)))

@egraph.register
def _(a: i64, b: i64, c: i64, ab: i64, bc: i64):
    yield rule(dist(a, b) == ab).then(set_(dist(a, a)).to(i64(0)))
    yield rule(dist(a, b) == ab, dist(b, c) == bc).then(
        set_(dist(a, c)).to(ab + bc)
    )

egraph.run(run().saturate())
egraph.check(eq(dist(1, 3)).to(15))
```

## ruleset() - Organizing and Scheduling Rules

Rulesets group related rules together for controlled execution. Schedules compose rulesets with operations like saturation, repetition, and sequencing.

```python
from egglog import *

class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...
    @classmethod
    def var(cls, name: StringLike) -> Num: ...
    def __add__(self, other: Num) -> Num: ...
    def __mul__(self, other: Num) -> Num: ...

converter(i64, Num, Num)
converter(String, Num, Num.var)

# Create separate rulesets for different concerns
analysis_rules = ruleset()
optimization_rules = ruleset()

# Add rules to analysis ruleset
@analysis_rules.register
def _(x: Num, y: Num):
    yield rewrite(Num(0) + x).to(x)
    yield rewrite(Num(1) * x).to(x)

# Add rules to optimization ruleset
@optimization_rules.register
def _(x: Num, y: Num, z: Num):
    yield rewrite(x + y).to(y + x)
    yield rewrite(x * (y + z)).to(x * y + x * z)

egraph = EGraph()
expr = egraph.let("expr", Num(1) * (Num.var("x") + Num(0)))

# Run analysis to saturation, then optimize
egraph.run(analysis_rules.saturate() + optimization_rules * 3)

# Complex schedules with repetition
schedule = (analysis_rules.saturate() + optimization_rules) * 10
egraph.run(schedule)

# Use backoff scheduler to prevent rule explosion
egraph.run(run(optimization_rules, scheduler=back_off(match_limit=1000)) * 10)

# Combine multiple rulesets
combined = analysis_rules | optimization_rules
egraph.run(combined.saturate())

# Run until facts are true
x = var("x", Num)
egraph.run(10, eq(expr).to(Num.var("x")), ruleset=optimization_rules)
```

## Built-in Types - Primitives and Containers

egglog provides built-in primitive types (i64, f64, String, Bool, Rational, BigInt, BigRat) and container types (Vec, Set, Map, MultiSet) with full operator support.

```python
from egglog import *
from fractions import Fraction

egraph = EGraph()

# Integer operations
x = i64(10)
y = i64(3)
egraph.register(x + y, x - y, x * y, x / y, x % y)
egraph.check(i64(10) + i64(3) == i64(13))

# Comparisons return Unit (existence facts)
egraph.check(i64(5) > i64(3))
egraph.check(i64(3) <= i64(5))

# Evaluate primitives to Python values
result = int(i64(10) + i64(5))  # Returns 15
assert result == 15

# Float operations
f = f64(3.14) * f64(2.0)
egraph.register(f)

# String operations
s = String("hello") + String(" world")
egraph.register(s)
egraph.check(eq(s).to(String("hello world")))

# Rational numbers
r = Rational(1, 2) + Rational(1, 3)  # 5/6
assert r.value == Fraction(5, 6)

# Vectors (immutable lists)
v = Vec(i64(1), i64(2), i64(3))
egraph.register(v.push(i64(4)))  # Returns new Vec
egraph.check(v.length() == i64(3))
egraph.check(v[0] == i64(1))

# Convert to Python
python_tuple = v.value  # Returns (i64(1), i64(2), i64(3))

# Sets with set operations
s1 = Set(i64(1), i64(2), i64(3))
s2 = Set(i64(2), i64(3), i64(4))
egraph.register(s1 | s2)  # Union
egraph.register(s1 & s2)  # Intersection
egraph.register(s1 - s2)  # Difference

# Maps (dictionaries)
m = Map[i64, String].empty().insert(i64(1), String("one"))
egraph.register(m)
egraph.check(m[i64(1)] == String("one"))
```

## function() - Defining Custom Functions

The @function decorator creates egglog functions with optional merge semantics, costs, and default rewrites from function bodies.

```python
from egglog import *

# Simple function declaration
@function
def fib(n: i64Like) -> i64: ...

# Function with merge semantics (keeps minimum)
@function(merge=lambda old, new: old.min(new))
def shortest_path(from_: i64Like, to: i64Like) -> i64: ...

# Function with custom egglog name
@function(egg_fn="my-custom-name")
def custom_func(x: i64Like) -> i64: ...

# Function with cost for extraction
@function(cost=10)
def expensive_op(x: i64Like) -> i64: ...

# Function with default rewrite (body becomes rewrite rule)
@function(ruleset=my_ruleset, subsume=True)
def double(x: i64Like) -> i64:
    return x + x  # Automatically creates: rewrite(double(x)).to(x + x)

# Unextractable function (won't appear in extracted terms)
@function(unextractable=True)
def helper(x: i64Like) -> i64: ...

egraph = EGraph()

# Use set_ to define function values
egraph.register(
    set_(fib(0)).to(i64(0)),
    set_(fib(1)).to(i64(1)),
)

# Rules can derive function values
@egraph.register
def _(n: i64, a: i64, b: i64):
    yield rule(
        fib(n) == a,
        fib(n + 1) == b
    ).then(set_(fib(n + 2)).to(a + b))

egraph.run(run().saturate())
egraph.check(eq(fib(10)).to(55))

# Query function values
values = egraph.function_values(fib, length=10)  # Returns dict of fib values
```

## constant() and var() - Creating Named Values

The constant() function creates named zero-argument functions, while var() creates pattern variables for use in rewrite rules.

```python
from egglog import *

class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...
    def __add__(self, other: Num) -> Num: ...
    def __mul__(self, other: Num) -> Num: ...

converter(i64, Num, Num)

# Create a named constant
x = constant("x", Num)
y = constant("y", Num)

egraph = EGraph()
egraph.register(x + y)

# Constants with default values (creates rewrite to the value)
pi = constant("pi", Num, Num(314))  # pi rewrites to Num(314)

# Pattern variables for rules
a = var("a", Num)
b = var("b", Num)

# Multiple variables at once
p, q, r = vars_("p q r", Num)

# Use in rewrite rules
@egraph.register
def _(a: Num, b: Num):
    yield rewrite(a + b).to(b + a)

# Variables can have custom egglog names
custom_var = var("myvar", Num, egg_name="custom-egglog-name")
```

## union() and set_() - E-Graph Actions

Actions modify the e-graph by unioning expressions (making them equivalent) or setting function values.

```python
from egglog import *

class Term(Expr):
    def __init__(self, name: StringLike) -> None: ...
    def f(self) -> Term: ...

@function
def cost(t: Term) -> i64: ...

egraph = EGraph()

a = egraph.let("a", Term("a"))
b = egraph.let("b", Term("b"))

# Union makes two expressions equivalent
egraph.register(union(a).with_(b))
egraph.check(a == b)  # Now equivalent

# set_ assigns a value to a function call
egraph.register(set_(cost(a)).to(i64(10)))
egraph.check(eq(cost(a)).to(10))

# Actions in rules
@egraph.register
def _(x: Term, y: Term):
    # When f(x) == y, union x with y
    yield rule(x.f() == y).then(union(x).with_(y))

    # Set cost based on structure
    yield rule(x.f() == y, cost(y) == i64(5)).then(
        set_(cost(x)).to(i64(6))
    )

# Multiple actions in one rule
@egraph.register
def _(x: Term, y: Term, c: i64):
    yield rule(x.f() == y, cost(y) == c).then(
        set_(cost(x)).to(c + 1),
        union(x.f().f()).with_(y)
    )
```

## UnstableFn - First-Class Functions

UnstableFn provides first-class function values that can be passed around, partially applied, and called within the e-graph.

```python
from egglog import *
from functools import partial

class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...
    def __add__(self, other: Num) -> Num: ...

converter(i64, Num, Num)

# Define a function that takes a callable
@function
def apply_twice(f: UnstableFn[Num, Num], x: Num) -> Num: ...

egraph = EGraph()

# Create function values from lambdas
add_one = UnstableFn(lambda x: x + Num(1))

# Partial application
add_to_ten = UnstableFn(Num.__add__, Num(10))

# Register and use
egraph.register(apply_twice(add_one, Num(5)))

# Rules with higher-order functions
@egraph.register
def _(f: UnstableFn[Num, Num], x: Num):
    yield rewrite(apply_twice(f, x)).to(f(f(x)))

egraph.run(10)

# Map over collections
v = Vec(Num(1), Num(2), Num(3))
# MultiSet supports map
ms = MultiSet(Num(1), Num(2))
mapped = ms.map(lambda x: x + Num(10))

# Get the underlying callable
fn_value = add_one.value  # Returns the lambda
```

## Extraction and Cost Models

Extract optimal expressions from the e-graph using configurable cost models. Custom costs enable domain-specific optimization.

```python
from egglog import *

class Op(Expr):
    def __init__(self, value: i64Like) -> None: ...

    @method(cost=1)  # Low cost
    def __add__(self, other: Op) -> Op: ...

    @method(cost=10)  # High cost - prefer add over mul
    def __mul__(self, other: Op) -> Op: ...

converter(i64, Op, Op)

egraph = EGraph()

@egraph.register
def _(x: Op):
    yield rewrite(x * Op(2)).to(x + x)  # Cheaper alternative

expr = egraph.let("expr", Op.var("x") * Op(2))
egraph.run(10)

# Basic extraction (uses AST size by default)
result = egraph.extract(expr)

# Extraction with cost
result, cost = egraph.extract(expr, include_cost=True)

# Extract multiple variants
variants = egraph.extract_multiple(expr, n=5)

# Custom per-node costs using set_cost
class Matrix(Expr):
    def __init__(self, rows: i64Like, cols: i64Like) -> None: ...
    def __matmul__(self, other: Matrix) -> Matrix: ...

    @property
    def row(self) -> i64: ...
    @property
    def col(self) -> i64: ...

@egraph.register
def _(x: Matrix, y: Matrix, r: i64, m: i64, c: i64):
    # Cost based on matrix dimensions
    yield rule(
        y @ z,
        r == y.row,
        m == y.col,
        c == z.col
    ).then(set_cost(y @ z, r * m * c))

# Custom cost model function
def my_cost_model(egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
    base = 1  # Base cost per node
    return base + sum(children_costs)

result, cost = egraph.extract(expr, include_cost=True, cost_model=my_cost_model)

# DAG cost model (counts shared subexpressions once)
dag_model = greedy_dag_cost_model()
result, dag_cost = egraph.extract(expr, include_cost=True, cost_model=dag_model)
```

## PyObject - Python Interop

PyObject enables embedding arbitrary Python objects in the e-graph, supporting dynamic code evaluation and integration with Python libraries.

```python
from egglog import *

egraph = EGraph()

# Wrap any Python object
obj = PyObject([1, 2, 3])
egraph.register(obj)

# Get the Python value back
python_list = obj.value  # Returns [1, 2, 3]

# Call PyObjects as functions
fn = PyObject(lambda x, y: x + y)
result = fn(PyObject(1), PyObject(2))

# Python eval in the e-graph
code_result = py_eval("1 + 2")  # Returns PyObject(3)

# With globals/locals
py_eval("x + 1", globals=PyObject({"x": 10}))

# Execute Python code
py_exec("y = x * 2", locals=PyObject({"x": 5}))

# Pattern matching on PyObject
match obj:
    case PyObject(value=v):
        print(f"Got value: {v}")

# Convert functions to callable PyObjects
@function
def apply_python(f: PyObject, x: PyObject) -> PyObject:
    return f(x)

# Rules with Python objects
@egraph.register
def _(f: PyObject, x: PyObject, result: PyObject):
    yield rule(
        apply_python(f, x) == result
    ).then(
        # Can trigger Python-side effects via PyObject
        set_(some_log(x)).to(result)
    )
```

## Summary

egglog provides a powerful framework for equality saturation and term rewriting in Python. The primary use case is program optimization, where expressions are added to an e-graph, equivalence-preserving rewrite rules are applied to saturation, and then the optimal expression is extracted using a cost model. This pattern applies to compiler optimization, theorem proving, symbolic mathematics, and domain-specific language implementations.

Integration patterns typically involve: (1) defining a custom Expr subclass representing your domain's AST, (2) registering converters for ergonomic Python-to-egglog type coercion, (3) defining rewrite rules in rulesets for modular scheduling, (4) combining analysis rules (which derive facts) with optimization rules (which add equivalent terms), and (5) extracting results using appropriate cost models. The library seamlessly integrates with Python's type system, supports Datalog-style relational queries for complex analyses, and provides first-class functions via UnstableFn for higher-order transformations.
# egglog Python Library

egglog is a Python library providing high-level bindings to the Rust [egglog](https://github.com/egraphs-good/egglog/) library, enabling e-graph-based equality saturation in Python. E-graphs are a powerful data structure that efficiently represents many equivalent programs simultaneously, making them ideal for program optimization, theorem proving, and symbolic computation tasks.

The library offers a Pythonic API for defining custom expression types, rewrite rules, and Datalog-style relational rules. It supports automatic term extraction based on cost models, union-find operations for equivalence classes, and built-in primitive types (integers, floats, strings, rationals, vectors, sets, maps). The core workflow involves creating an EGraph, registering expressions and rewrite rules, running saturation, and extracting optimized results.

## EGraph - Core E-Graph Operations

The EGraph class is the central data structure that maintains equivalence classes of expressions. It supports registering expressions, running rewrite rules, checking facts, and extracting optimal expressions based on cost models.

```python
from egglog import *

# Create an EGraph instance
egraph = EGraph()

# Define a custom expression type
class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...
    @classmethod
    def var(cls, name: StringLike) -> Num: ...
    def __add__(self, other: Num) -> Num: ...
    def __mul__(self, other: Num) -> Num: ...

# Register expressions with let() to get references
expr1 = egraph.let("expr1", Num(2) * (Num.var("x") + Num(3)))
expr2 = egraph.let("expr2", Num(6) + Num(2) * Num.var("x"))

# Register expressions directly
egraph.register(Num(1) + Num(2))

# Run rules for 10 iterations
egraph.run(10)

# Extract the lowest-cost equivalent expression
result = egraph.extract(expr1)

# Extract with cost information
result, cost = egraph.extract(expr1, include_cost=True)

# Check if expressions are equivalent
egraph.check(expr1 == expr2)  # Raises if not equivalent

# Check that expressions are NOT equivalent
egraph.check_fail(Num(1) == Num(2))

# Push/pop state for backtracking
egraph.push()
egraph.register(Num(100))
egraph.pop()  # Reverts state

# Context manager for automatic push/pop
with egraph:
    egraph.run(5)
    result = egraph.extract(expr1)
# State reverted here
```

## Expr - Defining Custom Expression Types

Subclass Expr to define domain-specific languages with typed constructors, methods, and operators. Methods become egglog functions automatically.

```python
from egglog import *
from typing import TypeAlias, ClassVar

# Define a math expression type
class Math(Expr):
    # Constructor with i64 parameter
    def __init__(self, value: i64Like) -> None: ...

    # Class method constructor
    @classmethod
    def var(cls, name: StringLike) -> Math: ...

    # Operators become egglog functions
    def __add__(self, other: Math) -> Math: ...
    def __mul__(self, other: Math) -> Math: ...
    def __truediv__(self, other: Math) -> Math: ...

    # Property-style functions
    @property
    def is_zero(self) -> Unit: ...

    # Class variables (constants)
    ZERO: ClassVar[Math]

# Type alias for automatic conversion
MathLike: TypeAlias = Math | i64Like | StringLike

# Register converters so 2 + Math(3) works
converter(i64, Math, Math)
converter(String, Math, Math.var)

# Usage
egraph = EGraph()
x = Math.var("x")
expr = x * 2 + 3  # Automatically converts 2 and 3 to Math
egraph.register(expr)
```

## rewrite() - Defining Rewrite Rules

The rewrite() function creates conditional rewrite rules that transform expressions when patterns match. Rules are added to rulesets and executed during egraph.run().

```python
from egglog import *

class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...
    @classmethod
    def var(cls, name: StringLike) -> Num: ...
    def __add__(self, other: Num) -> Num: ...
    def __mul__(self, other: Num) -> Num: ...

converter(i64, Num, Num)
converter(String, Num, Num.var)

egraph = EGraph()

# Create pattern variables
a, b, c = vars_("a b c", Num)
i, j = vars_("i j", i64)

# Register rewrite rules using decorator pattern
@egraph.register
def _(x: Num, y: Num, z: Num):
    # Commutativity
    yield rewrite(x + y).to(y + x)
    yield rewrite(x * y).to(y * x)
    # Associativity
    yield rewrite(x + (y + z)).to((x + y) + z)
    yield rewrite(x * (y * z)).to((x * y) * z)
    # Distributivity
    yield rewrite(x * (y + z)).to((x * y) + (x * z))

# Constant folding with primitive operations
@egraph.register
def _(i: i64, j: i64):
    yield rewrite(Num(i) + Num(j)).to(Num(i + j))
    yield rewrite(Num(i) * Num(j)).to(Num(i * j))

# Conditional rewrites
@egraph.register
def _(x: Num, y: Num):
    # Only rewrite x/x to 1 if x.is_nonzero fact exists
    yield rewrite(x / x).to(Num(1), x.is_nonzero)

# Bidirectional rewrites (works both directions)
@egraph.register
def _(x: Num, y: Num, z: Num):
    yield birewrite(x + (y + z)).to((x + y) + z)

# Run rules and verify equivalence
expr1 = egraph.let("e1", Num(2) * (Num.var("x") + Num(3)))
expr2 = egraph.let("e2", Num(6) + Num(2) * Num.var("x"))
egraph.run(10)
egraph.check(expr1 == expr2)
```

## rule() - Defining Datalog-Style Rules

The rule() function creates Datalog-style rules with arbitrary actions (not just rewrites). Rules query facts and trigger actions like union, set, or panic.

```python
from egglog import *

# Define relations for graph connectivity
edge = relation("edge", i64, i64)
path = relation("path", i64, i64)

egraph = EGraph()

# Insert edge facts
egraph.register(
    edge(i64(1), i64(2)),
    edge(i64(2), i64(3)),
    edge(i64(3), i64(4)),
)

# Datalog rules: derive path from edges
@egraph.register
def _(a: i64, b: i64, c: i64):
    # Base case: edge implies path
    yield rule(edge(a, b)).then(path(a, b))
    # Transitive case: path + edge implies longer path
    yield rule(path(a, b), edge(b, c)).then(path(a, c))

# Run until saturation (fixed point)
egraph.run(run().saturate())

# Check derived facts
egraph.check(path(i64(1), i64(4)))

# Functions with merge semantics (e.g., shortest path)
@function(merge=lambda old, new: old.min(new))
def dist(from_: i64Like, to: i64Like) -> i64: ...

egraph = EGraph()
egraph.register(set_(dist(1, 2)).to(i64(10)))
egraph.register(set_(dist(2, 3)).to(i64(5)))

@egraph.register
def _(a: i64, b: i64, c: i64, ab: i64, bc: i64):
    yield rule(dist(a, b) == ab).then(set_(dist(a, a)).to(i64(0)))
    yield rule(dist(a, b) == ab, dist(b, c) == bc).then(
        set_(dist(a, c)).to(ab + bc)
    )

egraph.run(run().saturate())
egraph.check(eq(dist(1, 3)).to(15))
```

## ruleset() - Organizing and Scheduling Rules

Rulesets group related rules together for controlled execution. Schedules compose rulesets with operations like saturation, repetition, and sequencing.

```python
from egglog import *

class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...
    @classmethod
    def var(cls, name: StringLike) -> Num: ...
    def __add__(self, other: Num) -> Num: ...
    def __mul__(self, other: Num) -> Num: ...

converter(i64, Num, Num)
converter(String, Num, Num.var)

# Create separate rulesets for different concerns
analysis_rules = ruleset()
optimization_rules = ruleset()

# Add rules to analysis ruleset
@analysis_rules.register
def _(x: Num, y: Num):
    yield rewrite(Num(0) + x).to(x)
    yield rewrite(Num(1) * x).to(x)

# Add rules to optimization ruleset
@optimization_rules.register
def _(x: Num, y: Num, z: Num):
    yield rewrite(x + y).to(y + x)
    yield rewrite(x * (y + z)).to(x * y + x * z)

egraph = EGraph()
expr = egraph.let("expr", Num(1) * (Num.var("x") + Num(0)))

# Run analysis to saturation, then optimize
egraph.run(analysis_rules.saturate() + optimization_rules * 3)

# Complex schedules with repetition
schedule = (analysis_rules.saturate() + optimization_rules) * 10
egraph.run(schedule)

# Use backoff scheduler to prevent rule explosion
egraph.run(run(optimization_rules, scheduler=back_off(match_limit=1000)) * 10)

# Combine multiple rulesets
combined = analysis_rules | optimization_rules
egraph.run(combined.saturate())

# Run until facts are true
x = var("x", Num)
egraph.run(10, eq(expr).to(Num.var("x")), ruleset=optimization_rules)
```

## Built-in Types - Primitives and Containers

egglog provides built-in primitive types (i64, f64, String, Bool, Rational, BigInt, BigRat) and container types (Vec, Set, Map, MultiSet) with full operator support.

```python
from egglog import *
from fractions import Fraction

egraph = EGraph()

# Integer operations
x = i64(10)
y = i64(3)
egraph.register(x + y, x - y, x * y, x / y, x % y)
egraph.check(i64(10) + i64(3) == i64(13))

# Comparisons return Unit (existence facts)
egraph.check(i64(5) > i64(3))
egraph.check(i64(3) <= i64(5))

# Evaluate primitives to Python values
result = int(i64(10) + i64(5))  # Returns 15
assert result == 15

# Float operations
f = f64(3.14) * f64(2.0)
egraph.register(f)

# String operations
s = String("hello") + String(" world")
egraph.register(s)
egraph.check(eq(s).to(String("hello world")))

# Rational numbers
r = Rational(1, 2) + Rational(1, 3)  # 5/6
assert r.value == Fraction(5, 6)

# Vectors (immutable lists)
v = Vec(i64(1), i64(2), i64(3))
egraph.register(v.push(i64(4)))  # Returns new Vec
egraph.check(v.length() == i64(3))
egraph.check(v[0] == i64(1))

# Convert to Python
python_tuple = v.value  # Returns (i64(1), i64(2), i64(3))

# Sets with set operations
s1 = Set(i64(1), i64(2), i64(3))
s2 = Set(i64(2), i64(3), i64(4))
egraph.register(s1 | s2)  # Union
egraph.register(s1 & s2)  # Intersection
egraph.register(s1 - s2)  # Difference

# Maps (dictionaries)
m = Map[i64, String].empty().insert(i64(1), String("one"))
egraph.register(m)
egraph.check(m[i64(1)] == String("one"))
```

## function() - Defining Custom Functions

The @function decorator creates egglog functions with optional merge semantics, costs, and default rewrites from function bodies.

```python
from egglog import *

# Simple function declaration
@function
def fib(n: i64Like) -> i64: ...

# Function with merge semantics (keeps minimum)
@function(merge=lambda old, new: old.min(new))
def shortest_path(from_: i64Like, to: i64Like) -> i64: ...

# Function with custom egglog name
@function(egg_fn="my-custom-name")
def custom_func(x: i64Like) -> i64: ...

# Function with cost for extraction
@function(cost=10)
def expensive_op(x: i64Like) -> i64: ...

# Function with default rewrite (body becomes rewrite rule)
@function(ruleset=my_ruleset, subsume=True)
def double(x: i64Like) -> i64:
    return x + x  # Automatically creates: rewrite(double(x)).to(x + x)

# Unextractable function (won't appear in extracted terms)
@function(unextractable=True)
def helper(x: i64Like) -> i64: ...

egraph = EGraph()

# Use set_ to define function values
egraph.register(
    set_(fib(0)).to(i64(0)),
    set_(fib(1)).to(i64(1)),
)

# Rules can derive function values
@egraph.register
def _(n: i64, a: i64, b: i64):
    yield rule(
        fib(n) == a,
        fib(n + 1) == b
    ).then(set_(fib(n + 2)).to(a + b))

egraph.run(run().saturate())
egraph.check(eq(fib(10)).to(55))

# Query function values
values = egraph.function_values(fib, length=10)  # Returns dict of fib values
```

## constant() and var() - Creating Named Values

The constant() function creates named zero-argument functions, while var() creates pattern variables for use in rewrite rules.

```python
from egglog import *

class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...
    def __add__(self, other: Num) -> Num: ...
    def __mul__(self, other: Num) -> Num: ...

converter(i64, Num, Num)

# Create a named constant
x = constant("x", Num)
y = constant("y", Num)

egraph = EGraph()
egraph.register(x + y)

# Constants with default values (creates rewrite to the value)
pi = constant("pi", Num, Num(314))  # pi rewrites to Num(314)

# Pattern variables for rules
a = var("a", Num)
b = var("b", Num)

# Multiple variables at once
p, q, r = vars_("p q r", Num)

# Use in rewrite rules
@egraph.register
def _(a: Num, b: Num):
    yield rewrite(a + b).to(b + a)

# Variables can have custom egglog names
custom_var = var("myvar", Num, egg_name="custom-egglog-name")
```

## union() and set_() - E-Graph Actions

Actions modify the e-graph by unioning expressions (making them equivalent) or setting function values.

```python
from egglog import *

class Term(Expr):
    def __init__(self, name: StringLike) -> None: ...
    def f(self) -> Term: ...

@function
def cost(t: Term) -> i64: ...

egraph = EGraph()

a = egraph.let("a", Term("a"))
b = egraph.let("b", Term("b"))

# Union makes two expressions equivalent
egraph.register(union(a).with_(b))
egraph.check(a == b)  # Now equivalent

# set_ assigns a value to a function call
egraph.register(set_(cost(a)).to(i64(10)))
egraph.check(eq(cost(a)).to(10))

# Actions in rules
@egraph.register
def _(x: Term, y: Term):
    # When f(x) == y, union x with y
    yield rule(x.f() == y).then(union(x).with_(y))

    # Set cost based on structure
    yield rule(x.f() == y, cost(y) == i64(5)).then(
        set_(cost(x)).to(i64(6))
    )

# Multiple actions in one rule
@egraph.register
def _(x: Term, y: Term, c: i64):
    yield rule(x.f() == y, cost(y) == c).then(
        set_(cost(x)).to(c + 1),
        union(x.f().f()).with_(y)
    )
```

## UnstableFn - First-Class Functions

UnstableFn provides first-class function values that can be passed around, partially applied, and called within the e-graph.

```python
from egglog import *
from functools import partial

class Num(Expr):
    def __init__(self, value: i64Like) -> None: ...
    def __add__(self, other: Num) -> Num: ...

converter(i64, Num, Num)

# Define a function that takes a callable
@function
def apply_twice(f: UnstableFn[Num, Num], x: Num) -> Num: ...

egraph = EGraph()

# Create function values from lambdas
add_one = UnstableFn(lambda x: x + Num(1))

# Partial application
add_to_ten = UnstableFn(Num.__add__, Num(10))

# Register and use
egraph.register(apply_twice(add_one, Num(5)))

# Rules with higher-order functions
@egraph.register
def _(f: UnstableFn[Num, Num], x: Num):
    yield rewrite(apply_twice(f, x)).to(f(f(x)))

egraph.run(10)

# Map over collections
v = Vec(Num(1), Num(2), Num(3))
# MultiSet supports map
ms = MultiSet(Num(1), Num(2))
mapped = ms.map(lambda x: x + Num(10))

# Get the underlying callable
fn_value = add_one.value  # Returns the lambda
```

## Extraction and Cost Models

Extract optimal expressions from the e-graph using configurable cost models. Custom costs enable domain-specific optimization.

```python
from egglog import *

class Op(Expr):
    def __init__(self, value: i64Like) -> None: ...

    @method(cost=1)  # Low cost
    def __add__(self, other: Op) -> Op: ...

    @method(cost=10)  # High cost - prefer add over mul
    def __mul__(self, other: Op) -> Op: ...

converter(i64, Op, Op)

egraph = EGraph()

@egraph.register
def _(x: Op):
    yield rewrite(x * Op(2)).to(x + x)  # Cheaper alternative

expr = egraph.let("expr", Op.var("x") * Op(2))
egraph.run(10)

# Basic extraction (uses AST size by default)
result = egraph.extract(expr)

# Extraction with cost
result, cost = egraph.extract(expr, include_cost=True)

# Extract multiple variants
variants = egraph.extract_multiple(expr, n=5)

# Custom per-node costs using set_cost
class Matrix(Expr):
    def __init__(self, rows: i64Like, cols: i64Like) -> None: ...
    def __matmul__(self, other: Matrix) -> Matrix: ...

    @property
    def row(self) -> i64: ...
    @property
    def col(self) -> i64: ...

@egraph.register
def _(x: Matrix, y: Matrix, r: i64, m: i64, c: i64):
    # Cost based on matrix dimensions
    yield rule(
        y @ z,
        r == y.row,
        m == y.col,
        c == z.col
    ).then(set_cost(y @ z, r * m * c))

# Custom cost model function
def my_cost_model(egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
    base = 1  # Base cost per node
    return base + sum(children_costs)

result, cost = egraph.extract(expr, include_cost=True, cost_model=my_cost_model)

# DAG cost model (counts shared subexpressions once)
dag_model = greedy_dag_cost_model()
result, dag_cost = egraph.extract(expr, include_cost=True, cost_model=dag_model)
```

## PyObject - Python Interop

PyObject enables embedding arbitrary Python objects in the e-graph, supporting dynamic code evaluation and integration with Python libraries.

```python
from egglog import *

egraph = EGraph()

# Wrap any Python object
obj = PyObject([1, 2, 3])
egraph.register(obj)

# Get the Python value back
python_list = obj.value  # Returns [1, 2, 3]

# Call PyObjects as functions
fn = PyObject(lambda x, y: x + y)
result = fn(PyObject(1), PyObject(2))

# Python eval in the e-graph
code_result = py_eval("1 + 2")  # Returns PyObject(3)

# With globals/locals
py_eval("x + 1", globals=PyObject({"x": 10}))

# Execute Python code
py_exec("y = x * 2", locals=PyObject({"x": 5}))

# Pattern matching on PyObject
match obj:
    case PyObject(value=v):
        print(f"Got value: {v}")

# Convert functions to callable PyObjects
@function
def apply_python(f: PyObject, x: PyObject) -> PyObject:
    return f(x)

# Rules with Python objects
@egraph.register
def _(f: PyObject, x: PyObject, result: PyObject):
    yield rule(
        apply_python(f, x) == result
    ).then(
        # Can trigger Python-side effects via PyObject
        set_(some_log(x)).to(result)
    )
```

## Summary

egglog provides a powerful framework for equality saturation and term rewriting in Python. The primary use case is program optimization, where expressions are added to an e-graph, equivalence-preserving rewrite rules are applied to saturation, and then the optimal expression is extracted using a cost model. This pattern applies to compiler optimization, theorem proving, symbolic mathematics, and domain-specific language implementations.

Integration patterns typically involve: (1) defining a custom Expr subclass representing your domain's AST, (2) registering converters for ergonomic Python-to-egglog type coercion, (3) defining rewrite rules in rulesets for modular scheduling, (4) combining analysis rules (which derive facts) with optimization rules (which add equivalent terms), and (5) extracting results using appropriate cost models. The library seamlessly integrates with Python's type system, supports Datalog-style relational queries for complex analyses, and provides first-class functions via UnstableFn for higher-order transformations.
