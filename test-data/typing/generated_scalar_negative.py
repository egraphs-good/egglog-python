"""Each marked public misuse must independently produce its diagnostic code."""

from egglog import Bool, BoolLike, String, StringLike, Unit, Vec, f64, function, i64, i64Like, join, ne, ruleset

String(1)  # error: arg-type
String()  # error: call-arg
String("a", "b")  # error: call-arg
Bool(1)  # error: arg-type
Bool()  # error: call-arg
Unit(None)  # error: call-arg
Unit().value  # error: attr-defined
Unit().eval()  # error: attr-defined
join(i64(1))  # error: arg-type
join(strings="x")  # error: call-arg
String("a") + Bool(False)  # error: operator
String("abc").replace("b")  # error: call-arg
String("abc").replace("b", "x", "extra")  # error: call-arg
String("abc").replace(False, "x")  # error: arg-type
Bool(True) | String("x")  # error: operator
Bool(True).__or__()  # error: call-arg
Bool(True).__or__(False, True)  # error: call-arg
Bool(True).__invert__(False)  # error: call-arg
Bool(True).implies(String("x"))  # error: arg-type
ne(i64(1)).to(String("1"))  # error: arg-type
ne(Vec[i64](i64(1))).to(Vec[String](String("1")))  # error: arg-type
ne(i64(1)).to()  # error: call-arg
ne(i64(1)).to(i64(2), i64(3))  # error: call-arg
wrong_ne: Bool = ne(i64(1)).to(i64(2))  # error: assignment


@function
def scalar_message(prefix: StringLike = "x", enabled: BoolLike = False) -> String: ...


scalar_message(i64(1))  # error: arg-type
scalar_message("x", String("false"))  # error: arg-type


@function
def bad_scalar_default(value: BoolLike = "false") -> Bool: ...  # error: assignment


i64("bad")  # error: arg-type
i64(1.5)  # error: arg-type
f64("bad")  # error: arg-type
i64()  # error: call-arg
i64(1, 2)  # error: call-arg
i64(1) + f64(2.0)  # error: operator
f64(1.0) + i64(2)  # error: operator
i64(1).__add__()  # error: call-arg
i64(1).__add__(2, 3)  # error: call-arg
wrong_value: str = i64(1).value  # error: assignment


@function
def score(x: i64Like, offset: i64Like = 2) -> i64: ...


score(f64(1.0))  # error: arg-type
score()  # error: call-arg
score(1, 2, 3)  # error: call-arg
score(1, unknown=2)  # error: call-arg


@function(cost=1)  # error: type-var
def bad_cost() -> i64: ...


@function(ruleset=ruleset(), subsume=True)  # error: type-var
def bad_ruleset() -> f64: ...


bad_element: Vec[int]  # error: type-var
bad_arity: Vec[i64, f64]  # error: type-arg
