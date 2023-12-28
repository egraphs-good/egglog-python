# Implicit Definitions

## Abstract

## Motiviation

- Remove modules because they are confusing
- Make API smaller
- Allow global use without egraph
- remove star imports

## Examples

```python
from __future__ import annotations
import egglog

@egglog
class Num:
    def __init__(self, value: i64Like) -> None: ...

    @classmethod
    def var(cls, name: StringLike) -> Num: ...

    def __add__(self, other: Num) -> Num: ...

    def __mul__(self, other: Num) -> Num: ...


a, b, c = vars_("a b c", Num)
i, j = vars_("i j", i64)
rules = egglog.Ruleset(
    rewrite(a + b).to(b + a),
    rewrite(a * (b + c)).to((a * b) + (a * c)),
    rewrite(Num(i) + Num(j)).to(Num(i + j)),
    rewrite(Num(i) * Num(j)).to(Num(i * j)),
)


# expr1 = 2 * (x + 3)
expr1 = Num(2) * (Num.var("x") + Num(3))
# expr2 = 6 + 2 * x
expr2 = Num(6) + Num(2) * Num.var("x")


egraph = egglog.EGraph()
egraph.add(expr1, expr2)
egraph.run(rules * 10)
egraph.check(eq(expr1).to(expr2))
egraph
```

## Specification

All function/type wrappers will move to the global `egglog` module, removing the need for modules.

<!-- What to call this?? -->

- `egglog[T: Callable | Type].class_(f: T) -> T`
- `egglog[T: Callable | Type].method(f: T) -> T`
- `egglog[T: Callable | Type].function(f: T) -> T`

- `egglog.Ruleset(*rules: Rule | Rewrite | () -> Rules, egg_name: bool | None = None) -> egglog.Ruleset`
- `ruleset.add(*rules: Rule | Rewrite | () -> Rules) -> None`

- `egglog.var(name: str, tp: type[T: Expr]) -> T`
- `egglog.vars_(name: str, tp: type[T: Expr]) -> List[T]`

- `egglog.EGraph(seminaive: bool = True, save_egglog_string: bool = False) -> egglog.EGraph`
- `egraph.register(*exprs: Expr | Union | Set | Delete) -> None`
- `egraph.check(*facts: Expr | Eq) -> None`
- `egraph.run(schedule: Schedule = DefaultSchedule) -> None`
- `egraph.extract[T: Expr](x: T) -> T`
- `egraph.extract_multiple[T: Expr](x: T, max: int) -> List[T]`
- `egraph.simplify[T: Expr](x: T, schedule: Schedule = DefaultSchedule) -> T`
- `egraph.__enter__() -> egglog.EGraph`
- `egraph.display(**kwargs) -> None`

Add()
Check()
Run()
Extract()
ExtractMultiple()
Display()
Push()
Pop()

-> Iter[SingleExtract, MultipleExtract, GraphViz]

Removes:

- Let, Modules

## Other Considerations
