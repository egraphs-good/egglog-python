---
file_format: mystnb
---

# Translation to/from egglog

The high level bindings available at the top module (`egglog`) expose most of the functionality of the `egglog` text format. This guide explains how to translate between the two.

Any EGraph can also be converted to egglog with the `egraph.as_egglog_string` property, as long as it was created with `EGraph(save_egglog_string=True)`.

## Builtin Types

Builtin sorts including `Unit`, `String`, `i64`, `f64`, `BigInt`, `BigRat`,
`Map`, `Set`, and `Vec` are exposed as Python classes.

These can be imported from `egglog` can be instantiated using the class constructor from the equivalent Python type. Many of the functions on them are mapped to Python operators. For example, the `>>` operator is mapped to `__rshift__` so it can be used as `a >> b` in Python.

```{code-cell} python
from __future__ import annotations
from egglog import *

# egg: (+ 10 2)
i64(10) + i64(2)
```

```{code-cell} python
# egg: (+ (bigrat (bigint 1)  (bigint 2))  (big-rat (bigint 2)  (bigint 1)))
BigRat(1, 2) / BigRat(2, 1)
```

These types are also all checked statically with MyPy, so for example, if you try to add a `String` and a `i64`, you will get a type error.

### Type Promotion

Since it is cumbersome to have to wrap every Python literal in the corresponding egglog type, we also support converting automatically from the Python primitives to these types when they are passed as arguments. The above could be written as:

```{code-cell} python
i64(10) + 2
```

```{code-cell} python
BigRat(1, 2) / BigRat(2, 1)
```

The floating-point sort also exposes the backend's `exp()`, `log()`, and
`sqrt()` primitives. `BigRat.to_i64()` is partial: it is defined only when the
rational value is an integer that fits in `i64`. As with other partial
primitives, undefined use in a rule fact skips that match, while undefined use
in an action is an error.

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

### Generic container operations

`Pair[L, R]` and `Maybe[T]` expose Egglog's generic product and optional-value
patterns in Python. `catch(lambda: expression)` converts an undefined partial
primitive call, such as a missing map lookup, into `Maybe.none()` instead of
failing the surrounding expression.

```{code-cell} python
pair = Pair(i64(1), String("one"))
pair.left, pair.right

present = Maybe[i64].some(1)
missing = catch(lambda: Map[i64, String].empty()[1])
present, missing
```

Maps have a general `map_fold_kv` primitive. The higher-level
`map_filter_kv`, `map_map_values`, `map_merge_with`, `Map.keys()`, and
`Map.pick_key()` operations are composed from that fold and ordinary
container operations, so callbacks may be regular Egglog lambdas:

```{code-cell} python
numbers = Map[i64, i64].empty().insert(1, 10).insert(2, 20)
map_map_values(lambda _key, value: value + 1, numbers)
```

Map folding uses opaque, e-graph-local `Value` order, not a semantic ordering
promised for arbitrary e-class keys. Prefer order-independent callbacks. An
undefined filter predicate skips that entry;
an undefined transform, merge callback, or fold callback makes the whole
operation undefined.

## Declaring Functions

In egglog, the most general way to declare a function is with the `(function ...)` command. In Python, we can use the `@function` decorator on a function with no body. The arg and return types are inferred from the function signature:

```{code-cell} python
# egg: (function fib (i64) i64)
@function
def fib(n: i64Like) -> i64:
    pass
```

Note that instead of using `i64` as the argument type, we used `i64Like` which is `i64 | int`. This allows us statically to declare that this function can take integers as well which will be upcasted to `i64` automatically.

The `function` decorator also accepts keyword arguments that map to backend features. Which ones are valid depends on how
the callable lowers, as described in [Functions vs Constructors](#functions-vs-constructors):

- `egg_fn`: The name of the function in egglog. By default, this is the same as the Python function name.
- `merge`: A function to merge the results of function-style declarations. This must take the old and new return values and
  return a single value of the same type.
- `cost`: The extraction cost for constructor-style declarations.

```{code-cell} python
# egg: (function foo () i64 :merge (max old new))
@function(egg_fn="foo", merge=lambda old, new: old.max(new))
def my_foo() -> i64:
    pass
```

The static types on the decorator preserve the type of the underlying function, so that they can all be checked statically.

### Functions vs Constructors

The Python bindings follow the backend split in egglog:

- primitive-returning callables use function-style lowering
- eqsort-returning callables use constructor-style lowering

That is not a Python-only policy choice. It comes from which backend features exist on each command:

- function-style declarations support `merge`
- constructor-style declarations support `cost` and `unextractable`
- `subsume` only applies to rewrite-backed bodies, so it only makes sense with an explicit `ruleset`

Python automatically infers which backend lowering to use from the callable shape. In practice, Python declarations can
lower to a `function`, a `constructor`, or an eager `primitive` depending on the return kind and whether a body/default
is present.

For bodies and defaults, the canonical lowering mapping is:

| Python shape | Lowering |
| --- | --- |
| primitive return, no body | lower to `function` |
| primitive return, body | lower to eager `primitive` |
| eqsort return, no body, no `merge` | lower to `constructor` |
| eqsort return, no body, with `merge` | lower to `function` |
| eqsort return, body, no `ruleset` | lower to eager `primitive` |
| eqsort return, body, explicit `ruleset` | lower to `constructor` plus rewrite-backed body |

Constants and class-variable defaults are just zero-arg bodies/defaults, so they follow the same split based on their
declared return type:

- no-default constants lower like zero-arg declarations, so primitive-returning constants lower as functions, while
  eqsort-returning constants lower as constructors unless `merge` forces function-style lowering
- eqsort-returning defaults lower eagerly without a `ruleset`, and lower to rewrite-backed defaults with an explicit `ruleset`
- primitive-returning defaults lower eagerly, and cannot use an explicit `ruleset`

Options follow that same backend split:

- no-body function-style declarations may use `merge`
- builtin declarations are primitive/function-style only
- constructor-style declarations may use `cost` and `unextractable`
- `subsume` is only valid when an eqsort-returning body is lowered through an explicit `ruleset`
- `egg_fn` and mutating arguments are supported in every case
- direct top-level `@function(ruleset=...)` declarations require a body
- `constant(..., ruleset=...)` declarations require an eqsort-returning default
- `constant(..., merge=...)` declarations must not provide a default
- class-level `ruleset=` is still valid shorthand for attaching rewrite-backed eqsort method and class-variable defaults

For the Python ergonomics of attaching rewrite-backed bodies/defaults to an explicit `ruleset`, see
[Python Integration](python-integration.md#default-replacements).

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

You can also pass initial actions directly to the high-level constructor:

```{code-cell} python
egraph = EGraph(
    let("x", i64(1)),
    set_(fib(0)).to(i64(0)),
)
```

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

### Set Cost

You can also set the cost of individual values, like the egglog experimental feature, to override the default cost from constructing a function:

```{code-cell} python
# egg: (set-cost (fib 0) 1)
egraph.register(set_cost(fib(0), 1))
```

This will be taken into account when extracting. Any value that can be converted to an `i64` is supported as a cost,
so dynamic costs can be created in rules.

It does this by creating a new table for each function you set the cost for that maps the arguments to an i64.

_Note: Unlike in egglog, where you have to declare which functions support custom costs, in Python all functions
are automatically registered to create a custom cost table when they are constructed_

You can also get the cost of a function with `get_cost`, which will return an `i64` if one has already been set.

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

Rules use semi-naive evaluation by default. A rule whose higher-order callback
must read tables populated during the same run can opt into naive evaluation
with `rule(..., eval_mode="naive")`. The third mode,
`eval_mode="unsafe-seminaive"`, skips semi-naive validation and should only be
used when the rule is known to be valid under that evaluation strategy.

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
path_ruleset = ruleset(rule(edge(x, y)).then(path(x, y)), name="path_ruleset")
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

@EGraph().register
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
# egg: (run 10000 :until (fib 7))
egraph.run(10000, fib(7))
```

Rulesets can be run as well, by calling the `run` method on them:

```{code-cell} python
# egg: (run 10 :ruleset path)
egraph.run(10, ruleset=path_ruleset)
```

After a run, you get a run report, with some timing information as well as whether things were updated.

### Schedules

`EGraph.run` can run either a bounded number of iterations or a full schedule.
The bounded form:

```python
egraph.run(5)
```

is shorthand for running the default ruleset five times. You can also pass a
ruleset and optional stop facts:

```python
egraph.run(10, ruleset=path_ruleset)
egraph.run(10000, fib(7))
```

For more control, pass a `Schedule` object. Schedules correspond to egglog's
`(run-schedule ...)` command and are composed from these Python forms:

| Python | egglog | Meaning |
| --- | --- | --- |
| `run()` | `(run)` | Run the default ruleset once. |
| `run(ruleset)` | `(run ruleset)` | Run one named ruleset once. |
| `run(ruleset, fact)` | `(run ruleset :until fact)` | Run until the fact is reached. |
| `schedule.saturate()` | `(saturate schedule)` | Repeat until the schedule stops changing the e-graph. |
| `schedule * n` | `(repeat n schedule)` | Repeat a schedule exactly `n` times. |
| `left + right` | `(seq left right)` | Run two schedules in order. |
| `seq(a, b, c)` | `(seq a b c)` | Run any number of schedules in order. |

Rulesets are schedules, so `egraph.run(path_ruleset)` runs `path_ruleset` once.
For readability, prefer `run(path_ruleset)` when you are composing a larger
schedule and `egraph.run(10, ruleset=path_ruleset)` when all you need is a
bounded run.

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

#### Backoff Scheduler

The custom backoff scheduler can delay rules that produce too many matches in a
single scheduler iteration. Create one with
`bo = back_off(match_limit=None, ban_length=None)`, then pass it to
`run(ruleset, *facts, scheduler=bo)`.

- `match_limit`: per-rule threshold of matches allowed in a single scheduler iteration. If a rule produces more matches than the threshold, that rule is temporarily banned.
- `ban_length`: initial ban duration (in scheduler iterations). While banned, that rule is skipped.
- Exponential backoff: each time a rule is banned, both the threshold and ban length double for that rule. After `times_banned` bans, the effective threshold is `match_limit << times_banned` and the ban duration is `ban_length << times_banned`.
- Fast-forwarding: when any rule is banned, the scheduler fast-forwards by the minimum remaining ban to unban at least one rule before checking for termination again.
- Defaults: match_limit defaults to 1000; ban_length defaults to 5.
- `:until` support: custom scheduler runs can use at most one non-equality fact as the stop condition. Equality stop facts and multiple stop facts raise `ValueError`.

For example, this egglog code:

```
(run-schedule
    (let-scheduler bo (back-off :match-limit 10))
    (repeat 10 (run-with bo step_right)))
```

Is translated as:

```{code-cell} python
step_egraph.run(
    run(step_right, scheduler=back_off(match_limit=10)) * 10
)
```

By default the scheduler will be created before any other schedules are run.
To control where it is instantiated explicitly, use `bo.scope(<schedule>)`, where it will be created before everything in `<schedule>`.

So the previous is equivalent to:

```{code-cell} python
bo = back_off(match_limit=10)
step_egraph.run(
    bo.scope(run(step_right, scheduler=bo) * 10)
)
```

If you wanted to create the scheduler inside the repeated schedule, you can do:

```{code-cell} python
bo = back_off(match_limit=10)
step_egraph.run(
    bo.scope(run(step_right, scheduler=bo)) * 10
)
```

This would be equivalent to this egglog:

```
(run-schedule
    (repeat 10
        (let-scheduler bo (back-off :match-limit 10))
        (run-with bo step_right)))
```

That distinction matters because a scheduler carries state. Hoisting one
scheduler outside `* 10` lets its `times_banned` counters accumulate across all
ten runs. Placing `bo.scope(...)` inside `* 10` creates a fresh scheduler each
time, so every iteration starts with the initial `match_limit` and `ban_length`.

The scheduler bindings above are local to one call to `EGraph.run`. To carry a
scheduler's ban state across separate calls on the same e-graph, mark it as
persistent:

```{code-cell} python
bo = back_off(match_limit=10).persistent()
step_egraph.run(run(step_right, scheduler=bo))
step_egraph.run(run(step_right, scheduler=bo))
```

The scheduler is registered once on that e-graph and reused by both calls. A
persistent scheduler has its own identity, so deriving it from another
scheduler configuration does not alias that configuration's local state.
High-level `EGraph.saturate()` also waits for `RunReport.can_stop`, so a
no-change round does not discard work deferred by a persistent scheduler.

## Check

The `(check ...)` command to verify that some facts are true, can be translated to Python with the `egraph.check` function:

```{code-cell} python
# egg: (check (= (fib 7) 13))
egraph.check(eq(fib(1)).to(i64(1)))
```

Low-level proof-mode commands such as `Prove`, `ProveExists`, and
`ProveExistsOutput` are exposed through the bindings layer, but the high-level
Python API does not yet support a complete proof workflow.

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

## Function Sizes

The `(print-size <function name>?)` command is translated into either `egraph.function_size(fn)` to get the number of
rows in one table-backed callable or `egraph.all_function_sizes()` to list the sizes of all registered function tables.
Relations, constructors, and bodyless functions have tables; eager and builtin primitives do not:

```{code-cell} python
# (function-size Math)
egraph.function_size(Math)
```

```{code-cell} python
# (function-size)
egraph.all_function_sizes()
```

## Overall Statistics

The `(print-stats)` command is translated into `egraph.stats()` to get overall statistics about the EGraph.

```{code-cell} python
# (print-stats)
egraph.stats()
```

## Function Values

The `print-function` command is translated into `egraph.function_values(fn, [length]?)` to get the rows of a
table-backed callable. As with `function_size`, eager and builtin primitives cannot be inspected this way.

```{code-cell} python
# (print-function fib 3)
egraph.function_values(fib, length=3)
```

## Include

The `(include <path>)` command is used to add modularity, by allowing you to pull in the source from another egglog file into the current file.

In Python, we can instead just import the desired types, functions, and rulesets and use them in our EGraph.
