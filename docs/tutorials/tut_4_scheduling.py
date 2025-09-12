# # 04 - Scheduling
#
# _[This tutorial is translated from egglog.](https://egraphs-good.github.io/egglog-tutorial/04-scheduling.html)_
#
#  In this lesson, we will learn how to use `run-schedule` to improve the performance of egglog.
#  We start by using the same language as the previous lesson.

# mypy: disable-error-code="empty-body"
from __future__ import annotations
from collections.abc import Iterable
from egglog import *
from tut_3_analysis import Num, zero, one, upper_bound, lower_bound, two


#  ## Rulesets
#
#  Different from lesson 3, we organize our rules into "rulesets"
#  A ruleset is exactly what it sounds like; a set of rules.
#  We can declare rulesets using the `ruleset` method.

optimizations = ruleset()
analysis = ruleset()

#  We can add rules to rulesets by calling the `register` method on the ruleset instead of the egraph.
#
#  We can run rulesets using `run(ruleset)`, or `run()` for running the default ruleset.
#
#  Here, we add `<=` rules to the `analysis` ruleset, because they don't add new `Num` nodes to the e-graph.


@analysis.register
def _(
    e1: Num, e2: Num, e3: Num, n1: BigRat, n2: BigRat, x: String, e1a: Num, e1b: Num, e2a: Num, e2b: Num
) -> Iterable[RewriteOrRule]:
    yield rule(e1 <= e2, e2 <= e3).then(e1 <= e3)
    yield rule(e1 == Num(n1), e2 == Num(n2), n1 <= n2).then(e1 <= e2)
    yield rule(e1 == Num.var(x)).then(e1 <= e1)  # noqa: PLR0124
    yield rule(
        e1 == (e1a + e1b),
        e2 == (e2a + e2b),
        e1a <= e2a,
        e1b <= e2b,
    ).then(e1 <= e2)


#  In contrast, the following axiomatic rules are doing optimizations, so we add them to the `optimizations` ruleset.


@optimizations.register
def _(x: Num, y: Num, z: Num, a: BigRat, b: BigRat) -> Iterable[RewriteOrRule]:
    yield birewrite(x + (y + z)).to((x + y) + z)
    yield birewrite(x * (y * z)).to((x * y) * z)
    yield rewrite(x + y).to(y + x)
    yield rewrite(x * y).to(y * x)
    yield rewrite(x * (y + z)).to((x * y) + (x * z))
    yield rewrite(x + zero).to(x)
    yield rewrite(x * one).to(x)
    yield rewrite(Num(a) + Num(b)).to(Num(a + b))
    yield rewrite(Num(a) * Num(b)).to(Num(a * b))


#  Here we add the rest of the rules from the last section, but tagged with the appropriate rulesets.


@analysis.register
def _(e: Num, n: BigRat, e1: Num, e2: Num, u1: BigRat, u2: BigRat, l1: BigRat, l2: BigRat) -> Iterable[RewriteOrRule]:
    yield rule(e <= Num(n)).then(set_(upper_bound(e)).to(n))
    yield rule(Num(n) <= e).then(set_(lower_bound(e)).to(n))
    yield rule(
        e == (e1 + e2),
        upper_bound(e1) == u1,
        upper_bound(e2) == u2,
    ).then(set_(upper_bound(e)).to(u1 + u2))
    yield rule(
        e == (e1 + e2),
        lower_bound(e1) == l1,
        lower_bound(e2) == l2,
    ).then(set_(lower_bound(e)).to(l1 + l2))
    yield rule(
        e == (e1 * e2),
        l1 == lower_bound(e1),
        l2 == lower_bound(e2),
        u1 == upper_bound(e1),
        u2 == upper_bound(e2),
    ).then(
        set_(lower_bound(e)).to((l1 * l2).min((l1 * u2).min((u1 * l2).min(u1 * u2)))),
        set_(upper_bound(e)).to((l1 * l2).max((l1 * u2).max((u1 * l2).max(u1 * u2)))),
    )
    yield rule(e == e1 * e1).then(set_(lower_bound(e)).to(zero))
    yield rule(lower_bound(e) > zero).then(e.non_zero)
    yield rule(upper_bound(e) < zero).then(e.non_zero)


#  Finally, we have optimization rules that depend on the analysis rules we defined above.


@optimizations.register
def _(e: Num, e2: Num) -> Iterable[RewriteOrRule]:
    yield rewrite(e / e).to(Num(one), e.non_zero)
    yield rewrite(e * (e2 / e)).to(e2, e.non_zero)


#  Now consider the following program, which consists of a long sequence of additions _inside_
#  a cancelling division.
egraph = EGraph()
addition_chain = egraph.let("addition_chain", "a" + ("b" + ("c" + ("d" + ("e" + Num.var("f"))))))
nonzero_expr = egraph.let("nonzero_expr", Num(one) + (Num(one) + (Num(one) + (Num(one) + Num(two)))))
expr = egraph.let("expr", nonzero_expr * (addition_chain / nonzero_expr))

#  We want the following check to pass after running the rules.

egraph.check_fail(expr == addition_chain)

#  To make this check pass, we have to first discover that `nonzero_expr` is indeed non-zero,
#  which allows the rule from `x * (y / x)` to `y` to fire.
#  On the other hand, if we apply the optimization rules, we risk the exponential blowup from
#  the associative and commutative permutations of the `addition_chain`.
#
#  Therefore, if we try to run both rulesets directly, egglog will spend lots of effort reassociating and
#  commuting the terms in the `addition_chain`, even though the optimization that we actually
#  want to run only takes one iteration. However, that optimization requires knowing a fact
#  that takes multiple iterations to compute (propagating lower- and upper-bounds
#  through `nonzero_expr`). We can build a more efficient *schedule*.

#  ## Schedules

#  Our schedule starts by saturating the analysis rules, fully propagating the `non_zero` information _without_
#  adding any e-nodes to the e-graph.

egraph.run(analysis.saturate())

#  Then, just run one iteration of the `optimizations` ruleset.

egraph.run(optimizations)

#  Or equivalently,
#
# ```python
# egraph.run(analysis.saturate() + optimizations)
# ```
#
# This makes our check pass
egraph.check(expr == addition_chain)

#  While the above program is effective at optimizing that specific program, it would fail if
#  we had a slightly more complex program where we had to interleave the optimizations and analyses
#  to derive the optimal program.
#  For expressing more complex schedules like these, `egglog` supports a scheduling sub-language,
#  with primitives `repeat`, `seq`, `saturate`, and `run`.


#  The idea behind the following schedule is to always saturate analyses before running optimizations.
#  This combination is wrapped in a `repeat` block to give us control over how long to run egglog.
#  With `repeat 1` it is the same schedule as before, but now we can increase the iteration
#  count if we want to optimize harder with more time and space budget.

egraph.run((analysis.saturate() + optimizations) * 2)


#  Running more iterations does not help our above example per se,
#  but if we had started with a slightly more complex program to optimize...

egraph = EGraph()
addition_chain = egraph.let("addition_chain", "a" + ("b" + ("c" + ("d" + ("e" + Num.var("f"))))))
x_times_zero = egraph.let("x_times_zero", Num.var("x") * zero)
nonzero_expr = egraph.let("nonzero_expr", Num(one) + (Num(one) + (Num(one) + (Num(one) + x_times_zero))))
expr = egraph.let("expr", nonzero_expr * (addition_chain / nonzero_expr))

#  For the purpose of this example, we add this rule


@optimizations.register
def _(x: Num) -> Iterable[RewriteOrRule]:
    yield rewrite(x * zero).to(Num(zero))


#  To prove `expr` is equivalent to `addition_chain` by applying the cancellation law,
#  we need to prove `nonzero_expr` is nonzero, which requires proving
#  `x_times_zero`'s bound.
#  To show `x_times_zero`'s bound, we need to apply an optimization rule to rewrite
#  it to 0.
#  In other words, this requires running analyses in between two runs of optimization rules
#  (the cancellation law and `*`'s identity law)

# Therefore, only running our schedule with one iteration (`repeat 1`) does not give us the optimal program.
# Note that here we used the context manager of e-graph, which calls `egraph.push()` and `egraph.pop()` automatically,
# to create a copy of the e-graph to run our schedule on, which is then reverted at the end.

with egraph:
    egraph.run(analysis.saturate() + optimizations)
    extracted = egraph.extract(expr)
extracted

#  Instead, we need to increase the iteration number.
with egraph:
    egraph.run((analysis.saturate() + optimizations) * 2)
    extracted = egraph.extract(expr)
extracted

# ## Using custom schedulers

#  However, sometimes just having an iteration number does not give you enough control.
#  For example, for many rules, such as associativity and commutativity (AC), the size of the e-graph grows hyper-exponentially
#  with respect to the number of iterations.

#  Let's go back to this example, and run for five iterations.
# (push)
with egraph:
    egraph.run((analysis.saturate() + optimizations) * 5)
    assert egraph.function_size(Num.__mul__) == 582

#  At iteration 5, the `Mul` function has size 582. However, if we bump that to 6,
#  the size of the `Mul` function will increase to 13285! Therefore, the iteration number is too coarse
#  of a granularity for defining the search space.

#  To this end, egglog provides a scheduler mechanism. A scheduler can decide which matches are important and need to be applied,
#  while others can be delayed or skipped. To use scheduler, pass it in as the `scheduler` argument to `run`.
#
#  Currently, `egglog-experimental` implements one scheduler, `back_off`. The idea of `back_off` is that it will ban a rule from applying if that rule grows the
#  e-graph too fast. The decision to ban is based on a threshold, which is initially small and increases as rules are banned.
#  This scheduler works well when the ruleset contains explosive rules like AC.

#  In this example, the back-off scheduler can prevent the associativity rule
#  from dominating the equality saturation: when the the associativity rule (or any other rule) is fired too much,
#  the scheduler will automatically ban this rule for a few iterations, so that other rules can catch up.

egraph.run(run(optimizations, scheduler=back_off()) * 10)
egraph.function_size(Num.__mul__)


# Note that any scheudler which doesn't have an explicit scope is bound to the outer loop like:
#
# ```python
# bo = back_off()
# egraph.run(bo.scope(run(optimizations, scheduler=bo) * 10))
# ```


#  It is important that the scheduler `bo` is instantiated outside the `repeat` loop, since each scheduler carries some state that is updated
#  when run. For example, the following schedule has a very different semantics than the schedule above.
#
# ```python
# bo = back_off()
# egraph.run(bo.scope(run(optimizations, scheduler=bo)) * 10)
# ```
#
#  This schedule instantiates a (fresh) `back-off` scheduler for each `run-with`, so the ten iterations of rulesets are all run
#  with the initial configuration of the `back-off` scheduler, which has a very low threshold for banning rules.
