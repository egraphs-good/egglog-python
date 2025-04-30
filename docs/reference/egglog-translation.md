---
file_format: mystnb
---

# Translation to/from egglog

The high level bindings available at the top module (`egglog`) expose most of the functionality of the `egglog` text format. This guide explains how to translate between the two.

Any EGraph can also be converted to egglog with the `egraph.as_egglog_string` property, as long as it was created with `Egraph(save_egglog_string=True)`.

## Unsupported features

The currently unsupported features are:

- `(output ...)`: No examples in the tests, so not sure how this works.

## Builtin Types

The builtin types of Unit, String, Int, Map, and Rational are all exposed as Python classes.

These can be imported from `egglog` can be instantiated using the class constructor from the equivalent Python type. Many of the functions on them are mapped to Python operators. For example, the `>>` operator is mapped to `__rshift__` so it can be used as `a >> b` in Python.

```{code-cell} python
from __future__ import annotations
from egglog import *

# egg: (+ 10 2)
i64(10) + i64(2)
```

```{code-cell} python
# egg: (+ (rational 1 2)  (rational 2 1))
Rational(i64(1), i64(2)) / Rational(i64(2), i64(1))
```

These types are also all checked statically with MyPy, so for example, if you try to add a `String` and a `i64`, you will get a type error.

### Type Promotion

Since it is cumbersome to have to wrap every Python literal in the corresponding egglog type, we also support converting automatically from the Python primitives to these types when they are passed as arguments. The above could be written as:

```{code-cell} python
i64(10) + 2
```

```{code-cell} python
Rational(1, 2) / Rational(2, 1)
```

### `!=` Operator

The `!=` function in egglog works on any two types with the same sort. In Python, this is mapped to the `ne` function:

```{code-cell} python
# egg: (!= 10 2)
ne(i64(10)).to(i64(2))
```

This is a two part function so that we can statically check both sides are the same type.

## Declaring Sorts

Users can declare their own sorts in Python by subclassing the `Expr` class:

```{code-cell} python
# egg: (datatype Math)
class Math(Expr):
    pass
```

By default, the egg sort name is generated from the Python class name. You can override this if you wish with the `egg_sort` keyword argument:

```{code-cell} python
class Math(Expr, egg_sort="Math2"):
    pass
```

### Parameterized sorts

In egglog, the builtin `Map` sort can be parameterized with the key and value sorts. In Python, we can use the generic typing syntax to do the same:

```{code-cell} python
# egg: (sort MyMap (Map i64 String))
MyMap = Map[i64, String]

# egg: (map-insert (map-empty) 1 "one")
MyMap.empty().insert(i64(1), String("one"))
```

Since the generic types in the `Map` sort as specified with the `Generic` class, all of the methods will be statically checked, to make sure the right key/value types are used.

This doesn't require any custom type analysis on our part, only using Python's built in annotations with generic types.

## Declaring Functions

In egglog, the most general way to declare a function is with the `(function ...)` command. In Python, we can use the `@function` decorator on a function with no body. The arg and return types are inferred from the function signature:

```{code-cell} python
# egg: (function fib (i64) i64)
@function
def fib(n: i64Like) -> i64:
    pass
```

Note that instead of using `i64` as the argument type, we used `i64Like` which is `i64 | int`. This allows us statically to declare that this function can take integers as well which will be upcasted to `i64` automatically.

The `function` decorator supports a number of options as well, which can be passed as keyword arguments, that correspond to the options in the egglog command:

- `egg_fn`: The name of the function in egglog. By default, this is the same as the Python function name.
- `cost`: The cost of the function. By default, this is 1.
- `merge`: A function to merge the results of the function. This must be a function that takes two arguments of the return type, the old and the new, and returns a single value of the return type.

```{code-cell} python
# egg: (function foo () i64 :cost 10 :merge (max old new))
@function(egg_fn="foo", cost=10, merge=lambda old, new: old.max(new))
def my_foo() -> i64:
    pass
```

The static types on the decorator preserve the type of the underlying function, so that they can all be checked statically.

### Functions vs Constructors

Egglog has changed how it handles functions, seperating them into two seperate commands:

- `function` which can include a `merge` expression.
- `constructor` which can include a cost and requires the result to be an "eqsort" aka a non builtin type.

Since this was added after the Python API was first created, we added support to automatically choose between the two based on the return type of the function and whether a merge function is provided. If the return type is a builtin type, it will be a `function`, otherwise it will be a `constructor`, unless it has a merge function
provided then it will always be a `function`.

### Datatype functions

In egglog, the `(datatype ...)` command can also be used to declare functions. All of the functions declared in this block return the type of the declared datatype. Similarly, in Python, any methods of an `Expr` will be registered automatically. These
can be either instance methods (including any supported `__` method), class methods, or the `__init__` method. The return type of these functions is inferred from the return type of the function. Additionally, any supported keyword argument for the `@function` decorator can be used here as well, by using the `@method` decorator to add values.

Note that by default, the egg name for any method is the Python class name combined with the method name. This allows us to define two classes with the same method name, with different signatures, that map to different egglog functions.

```{code-cell} python
# egg:
# (datatype Math
#   (Num i64)
#   (Var String)
#   (Add Math Math)
#   (Mul Math Math)
#   (Neg Math))

class Math(Expr):
    @method(egg_fn="Num")
    def __init__(self, v: i64Like):
        ...
    @method(egg_fn="Var")
    @classmethod
    def var(cls, v: StringLike) -> Math:
        ...

    @method(egg_fn="Add")
    def __add__(self, other: Math) -> Math:
        ...

    @method(egg_fn="Mul")
    def __mul__(self, other: Math) -> Math:
        ...

    @method(egg_fn="Neg")
    @property
    def neg(self) -> Math:
        ...

# egg: (Neg (Mul (Num 2) (Add (Var "x") (Num 3)))))
(Math(2) * (Math.var("x") + Math(3))).neg
```

As shown above, we can also use the `@classmethod` and `@property` decorators to define class methods and properties.

For more information on how to define methods, see the [Python Integration](python-integration.md) guide.

### Declarations

In egglog, the `(declare ...)` command is syntactic sugar for a nullary function. In Python, these can be declare either as class variables or with the toplevel `constant` function:

```{code-cell} python
# egg:
# (datatype Boolean)
# (function or (Boolean Boolean) Boolean)
# (declare True Boolean)
# (declare False Boolean)
# (or True False)
#

from typing import ClassVar

class Boolean(Expr):
    TRUE: ClassVar[Boolean]

    def __or__(self, other: Boolean) -> Boolean:
        ...

FALSE = constant("False", Boolean)
Boolean.TRUE | FALSE
```

### Relations

The `(relation ...)` command is syntactic sugar for a function that returns the `Unit` type. This can be declared in Python with the `relation` function:

```{code-cell} python
# egg: (relation path (i64 i64))
#      (path 1 2)
path = relation("path", i64, i64)
path(i64(1), i64(2))
```

The correct function type (in this case it would be `Callable[[i64, i64], Unit]`) is inferred from the arguments to the `relation` function, so that it can be checked statically.

## Running Actions

To run actions in Python, they are passed as arguments to the `egraph.register` function. We have constructors to create each kind of action. They are created and registered in this way, so that we can use the same syntax for executing them on the top level egraph as we do for defining them as results for rules.

Here are examples of all the actions:

### Let

```{code-cell} python
egraph = EGraph()
# egg: (let x 1)
egraph.register(let("x", i64(1)))
```

### Set

```{code-cell} python
# egg: (set (fib 0) 0)
egraph.register(set_(fib(0)).to(i64(0)))
# egg: (set (fib 1) 1)
egraph.register(set_(fib(1)).to(i64(1)))
```

For `set_`, we use a fluent API of `set(...).to(...)`, so that we can type check that the two values match statically.

### Delete

```{code-cell} python
# egg: (delete (fib 0))
egraph.register(delete(fib(0)))
```

### Union

```{code-cell} python
# egg: (union (or True False) True)
egraph.register(union(Boolean.TRUE | FALSE).with_(Boolean.TRUE))
```

Similar to the `set` function, this uses a fluent API, so that we can verify the types statically.

### Expr as an action

```{code-cell} python
# re-set after deletion:
egraph.register(set_(fib(0)).to(i64(1)))

# egg: (fib 0)
egraph.register(fib(0))
```

### Panic

```{code-cell} python
# egg: (panic "This is an error")
try:
    EGraph().register(panic("This is an error"))
except BaseException as e:
    print(e)
```

## Defining Rules

To define rules in Python, we create a rule with the `rule(*facts).then(*actions) (rule ...)` command in egglog.

```{code-cell} python
# egg:
# (rule ((= f0 (fib x))
#        (= f1 (fib (+ x 1))))
#       ((set (fib (+ x 2)) (+ f0 f1))))
f0, f1, x = vars_("f0 f1 x", i64)
egraph.register(
    rule(
        eq(f0).to(fib(x)),
        eq(f1).to(fib(x + 1)),
    ).then(set_(fib(x + 2)).to(f0 + f1))
)
```

### Variables

Unlike in egglog, variables must be declared before being use and must be given a type. They need a type both so that they can be checked statically and also so that we know what types are used to understand what how the names of the egg functions correspond to the method names.

### Facts

Facts can either be created with the `eq` function or with `Unit` expressions.

The `eq` function is also fluent, similar to `set` and `union`, so that we can verify that the types match statically.

### Rulesets

Rulesets can be generated in Python with the `ruleset([*rules], [name])` function and used by registering rules with them:

```{code-cell} python
# egg: (relation edge (i64 i64))
edge = relation("edge", i64, i64)

# egg: (ruleset path)
# (rule ((edge x y))
#       ((path x y)) :ruleset path)
x, y = vars_("x y", i64)
path_ruleset = ruleset(rule(edge(x, y)).then(path(x, y)), name="path")
```

### Rewrites

Rewrites in egglog are syntactic sugar for rules. In Python, we can use the `rewrite(expr).to(expr, *when)` function to create a rule that rewrites the first expression to the second expression when the `when` expressions are true, like the `(rewrite ...)` command in egglog.

```{code-cell} python
# egg: (rewrite (Add a b) (Add b a))
a, b = vars_("a b", Math)
egraph.register(rewrite(a + b).to(b + a))
```

Since it uses a fluent API, static type checkers can verify that the type of the first expression matches the type of the second expression.

The `(birewrite ...)` command in egglog is syntactic sugar for creating two rewrites, one in each direction. In Python, we can use the `birewrite(expr).to(expr, *when)` function to create two rules that rewrite in each direction.

### Using functions to define vars

Instead of defining variables with `vars_`, we can also use functions to define variables. This can be more succinct
and also will make sure the variables won't be used outside of the scope of the function.

```{code-cell} python
# egg: (rewrite (Mul a b) (Mul b a))
# egg: (rewrite (Add a b) (Add b a))

@egraph.register
def _math(a: Math, b: Math):
    yield rewrite(a * b).to(b * a)
    yield rewrite(a + b).to(b + a)
```

## Running

To run the egraph, we can use the `egraph.run()` function. This will run all the default rules until a fixed point is reached, or until a timeout is reached.

```{code-cell} python
# egg: (run 5)
egraph.run(5)
```

Facts can be passed after the timeout to only run until those facts are reached:

```{code-cell} python
# egg: (run 10000 :until (fib 10))
egraph.run(10000, eq(fib(7)).to(i64(13)))
```

Rulesets can be run as well, by calling the `run` method on them:

```{code-cell} python
# egg: (run 10 :ruleset path)
egraph.run(10, ruleset=path_ruleset)
```

After a run, you get a run report, with some timing information as well as whether things were updated.

### Schedules

The `egraph.run` function also takes a `schedule` argument, which corresponds to the `(run-schedule ...)` command in egglog. A schedule can be either:

- A run configuration, created with `run(limit=..., ruleset=..., *until)`, corresponding to `(run ...)` in egglog
- Saturating an existing schedule, by calling the `schedule.saturate()` method, corresponding to `(saturate ...)` in egglog
- A sequence of sequences run one after the other, created with `seq(*schedules)`, corresponding to `(seq ...)` in egglog
- Repeating a schedule some number of times, created with `schedule * n`, corresponding to `(repeat ...)` in egglog

We can show an example of this by translating the `schedule-demo.egg` to Python:

```
; Step with alternating feet, left before right
(relation left (i64))
(relation right (i64))

(left 0)
(right 0)

(ruleset step-left)
(rule ((left x) (right x))
      ((left (+ x 1)))
      :ruleset step-left)

(ruleset step-right)
(rule ((left x) (right y) (= x (+ y 1)))
      ((right x))
      :ruleset step-right)

(run-schedule
      (repeat 10
            (saturate step-right)
            (saturate step-left)))

; We took 10 steps with the left, but the right couldn't go the first round,
; so we took only 9 steps with the right.
(check (left 10))
(check (right 9))
(fail (check (left 11)))
(fail (check (right 10)))
```

```{code-cell} python
left = relation("left", i64)
right = relation("right", i64)

x, y = vars_("x y", i64)

step_left = ruleset(
    rule(
        left(x),
        right(x),
    ).then(left(x + 1))
)
step_right = ruleset(
    rule(
        left(x),
        right(y),
        eq(x).to(y + 1),
    ).then(right(x))
)

step_egraph = EGraph()
step_egraph.register(left(i64(0)), right(i64(0)))
step_egraph.run(
    seq(
        step_right.saturate(),
        step_left.saturate(),
    ) * 10
)
```

```{code-cell} python
step_egraph.check(left(i64(10)), right(i64(9)))
step_egraph.check_fail(left(i64(11)), right(i64(10)))
```

## Check

The `(check ...)` command to verify that some facts are true, can be translated to Python with the `egraph.check` function:

```{code-cell} python
# egg: (check (= (fib 7) 13))
egraph.check(eq(fib(1)).to(i64(1)))
```

## Extract

The `(extract ...)` command in egglog translates to the `egraph.extract` method, returning the lowest cost expression:

```{code-cell} python
# egg: (extract (fib 1))
egraph.extract(fib(1))
```

If you want to see the cost as well, pass in the `include_cost` flag:

```{code-cell} python
egraph.extract(fib(1), include_cost=True)
```

Multiple items can also be extracted, returning a list of the lowest cost expressions, with `egraph.extract_multiple`:

```{code-cell} python
a, b, c = vars_("a b c", Math)
i, j = vars_("i j", i64)
egraph.register(
    rewrite(a * b).to(b * a),
    rewrite(a + b).to(b + a),
    rewrite(a * (b * c)).to((a * b) * c),
    rewrite(a * (b + c)).to((a * b) + (a * c)),
    rewrite(Math(i) + Math(j)).to(Math(i + j)),
    rewrite(Math(i) * Math(j)).to(Math(i * j)),
)

# egg:
# (define y (Add (Num 6) (Mul (Num 2) (Var "x")))
# (run 10)
# (extract y :variants 2)
y = egraph.let("y", Math(6) + Math(2) * Math.var("x"))
egraph.run(10)
# TODO: For some reason this is extracting temp vars
# egraph.extract_multiple(y, 2)
egraph
```

### Simplify

The `(simplify ...)` command in egglog translates to the `egraph.simplify` method, which combines running a schedule and extracting:

```{code-cell} python
# egg: (simplify (Mul (Num 6) (Add (Num 2) (Mul (Var "x") (Num 2)))) 20)
egraph.simplify(Math(6) * (Math(2) + Math.var("x") * Math(2)), 20)
```

## Push/Pop

The `(push)` and `(pop)` commands in egglog can be translated to the context manager on the `egraph` object:

```{code-cell} python
# egg:
# (push)
# (union (Num 0) (Num 1))
# (check (= (Num 0) (Num 1)))
# (pop)
# (fail (check (= (Num 0) (Num 1))))

with egraph:
    egraph.register(union(Math(0)).with_(Math(1)))
    egraph.check(eq(Math(0)).to(Math(1)))
egraph.check_fail(eq(Math(0)).to(Math(1)))
```

## Include

The `(include <path>)` command is used to add modularity, by allowing you to pull in the source from another egglog file into the current file.

In Python, we can instead just import the desired types, functions, and rulesets and use them in our EGraph.
