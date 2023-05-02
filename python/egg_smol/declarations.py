"""
Data only descriptions of the components of an egraph and the expressions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union, cast

from typing_extensions import assert_never

from . import bindings

__all__ = [
    "Declarations",
    "JustTypeRef",
    "ClassTypeVarRef",
    "TypeRefWithVars",
    "TypeOrVarRef",
    "FunctionRef",
    "MethodRef",
    "ClassMethodRef",
    "ClassVariableRef",
    "CallableRef",
    "ConstantRef",
    "constant_function_decl",
    "FunctionDecl",
    "VarDecl",
    "LitType",
    "LitDecl",
    "CallDecl",
    "ExprDecl",
    "tp_and_expr_decl_from_egg",
    "ClassDecl",
    "RewriteDecl",
    "EqDecl",
    "FactDecl",
    "fact_decl_to_egg",
    "RuleDecl",
    "LetDecl",
    "SetDecl",
    "DeleteDecl",
    "UnionDecl",
    "PanicDecl",
    "ActionDecl",
    "action_decl_to_egg",
]
# Special methods which we might want to use as functions
# Mapping to the operator they represent for pretty printing them
# https://docs.python.org/3/reference/datamodel.html
BINARY_METHODS = {
    "__lt__": "<",
    "__le__": "<=",
    "__eq__": "==",
    "__ne__": "!=",
    "__gt__": ">",
    "__ge__": ">=",
    # Numeric
    "__add__": "+",
    "__sub__": "-",
    "__mul__": "*",
    "__matmul__": "@",
    "__truediv__": "/",
    "__floordiv__": "//",
    "__mod__": "%",
    "__divmod__": "divmod",
    "__pow__": "**",
    "__lshift__": "<<",
    "__rshift__": ">>",
    "__and__": "&",
    "__xor__": "^",
    "__or__": "|",
}
UNARY_METHODS = {
    "__pos__": "+",
    "__neg__": "-",
    "__invert__": "~",
}


@dataclass
class Declarations:
    _functions: dict[str, FunctionDecl] = field(default_factory=dict)
    _classes: dict[str, ClassDecl] = field(default_factory=dict)
    _constants: dict[str, FunctionDecl] = field(default_factory=dict)

    # Bidirectional mapping between egg function names and python callable references.
    # Note that there are possibly mutliple callable references for a single egg function name, like `+`
    # for both int and rational classes.
    _egg_fn_to_callable_refs: defaultdict[str, set[CallableRef]] = field(default_factory=lambda: defaultdict(set))
    _callable_ref_to_egg_fn: dict[CallableRef, str] = field(default_factory=dict)

    # Bidirectional mapping between egg sort names and python type references.
    _egg_sort_to_type_ref: dict[str, JustTypeRef] = field(default_factory=dict)
    _type_ref_to_egg_sort: dict[JustTypeRef, str] = field(default_factory=dict)

    def _get_function_decl(self, ref: CallableRef) -> FunctionDecl:
        if isinstance(ref, FunctionRef):
            return self._functions[ref.name]
        elif isinstance(ref, MethodRef):
            return self._classes[ref.class_name].methods[ref.method_name]
        elif isinstance(ref, ClassMethodRef):
            return self._classes[ref.class_name].class_methods[ref.method_name]
        elif isinstance(ref, ConstantRef):
            return self._constants[ref.name]
        elif isinstance(ref, ClassVariableRef):
            return self._classes[ref.class_name].class_variables[ref.variable_name]
        assert_never(ref)

    def _set_function_decl(self, ref: CallableRef, decl: FunctionDecl) -> None:
        """
        Sets a function declaration for the given callable reference.
        """
        if isinstance(ref, FunctionRef):
            if ref.name in self._functions:
                raise ValueError(f"Function {ref.name} already registered")
            self._functions[ref.name] = decl
        elif isinstance(ref, MethodRef):
            if ref.method_name in self._classes[ref.class_name].methods:
                raise ValueError(f"Method {ref.class_name}.{ref.method_name} already registered")
            self._classes[ref.class_name].methods[ref.method_name] = decl
        elif isinstance(ref, ClassMethodRef):
            if ref.method_name in self._classes[ref.class_name].class_methods:
                raise ValueError(f"Class method {ref.class_name}.{ref.method_name} already registered")
            self._classes[ref.class_name].class_methods[ref.method_name] = decl
        elif isinstance(ref, ConstantRef):
            if ref.name in self._constants:
                raise ValueError(f"Constant {ref.name} already registered")
            self._constants[ref.name] = decl
        elif isinstance(ref, ClassVariableRef):
            if ref.variable_name in self._classes[ref.class_name].class_variables:
                raise ValueError(f"Class variable {ref.class_name}.{ref.variable_name} already registered")
            self._classes[ref.class_name].class_variables[ref.variable_name] = decl
        else:
            assert_never(ref)

    def register_class(self, name: str, n_type_vars: int, egg_sort: Optional[str]) -> Iterable[bindings._Command]:
        # Register class first
        if name in self._classes:
            raise ValueError(f"Class {name} already registered")
        decl = ClassDecl(n_type_vars=n_type_vars)
        self._classes[name] = decl
        return self._register_sort(JustTypeRef(name), egg_sort)

    def _register_sort(self, ref: JustTypeRef, egg_name: Optional[str] = None) -> Iterable[bindings._Command]:
        """
        Register a sort with the given name. If no name is given, one is generated.

        If this is a type called with generic args, register the generic args as well.
        """
        if ref in self._type_ref_to_egg_sort:
            if egg_name and self._type_ref_to_egg_sort[ref] != egg_name:
                raise ValueError(f"Type {ref} is already registered with egg name {self._type_ref_to_egg_sort[ref]}")
            return []
        egg_name = egg_name or ref.generate_egg_name()
        if egg_name in self._egg_sort_to_type_ref:
            raise ValueError(f"Sort {egg_name} is already registered.")
        self._egg_sort_to_type_ref[egg_name] = ref
        self._type_ref_to_egg_sort[ref] = egg_name
        return ref.to_commands(self)

    def register_callable(
        self, ref: CallableRef, fn_decl: FunctionDecl, egg_name: Optional[str], generate_commands: bool = True
    ) -> Iterable[bindings._Command]:
        """
        Registers a callable with the given egg name. The callable's function needs to be registered
        first.
        """
        egg_name = egg_name or ref.generate_egg_name()
        self.register_callable_ref(ref, egg_name)
        self._set_function_decl(ref, fn_decl)
        return fn_decl.to_commands(self, egg_name) if generate_commands else []

    def register_callable_ref(self, ref: CallableRef, egg_name: str) -> None:
        """
        Registers a callable reference with the given egg name. The callable's function needs to be registered
        first.
        """
        if ref in self._callable_ref_to_egg_fn:
            raise ValueError(f"Callable ref {ref} already registered")
        self._callable_ref_to_egg_fn[ref] = egg_name
        self._egg_fn_to_callable_refs[egg_name].add(ref)

    def get_egg_fn(self, ref: CallableRef) -> str:
        return self._callable_ref_to_egg_fn[ref]


def constant_function_decl(type_ref: JustTypeRef) -> FunctionDecl:
    """
    Create a function decleartion for a constant function. This is similar to how egg-smol compiles
    the `constant` command.
    """
    # Divide high cost by 10 to not overflow the cost field.
    return FunctionDecl(arg_types=(), return_type=type_ref.to_var(), cost=int(bindings.HIGH_COST / 10))


# Have two different types of type refs, one that can include vars recursively and one that cannot.
# We only use the one with vars for classmethods and methods, and the other one for egg references as
# well as runtime values.
@dataclass(frozen=True)
class JustTypeRef:
    name: str
    args: tuple[JustTypeRef, ...] = ()

    def generate_egg_name(self) -> str:
        """
        Generates an egg sort name for this type reference by linearizing the type.
        """
        if not self.args:
            return self.name
        args = "_".join(a.generate_egg_name() for a in self.args)
        return f"{self.name}__{args}"

    def to_commands(self, decls: Declarations) -> Iterable[bindings._Command]:
        """
        Register this type with the egg solver.
        """
        egg_name = decls._type_ref_to_egg_sort[self]
        for arg in self.args:
            yield from decls._register_sort(arg)
        arg_sorts = [cast("bindings._Expr", bindings.Var(decls._type_ref_to_egg_sort[a])) for a in self.args]
        yield bindings.Sort(egg_name, (self.name, arg_sorts) if arg_sorts else None)

    def to_var(self) -> TypeRefWithVars:
        return TypeRefWithVars(self.name, tuple(a.to_var() for a in self.args))

    def pretty(self) -> str:
        if not self.args:
            return self.name
        args = ", ".join(a.pretty() for a in self.args)
        return f"{self.name}[{args}]"


@dataclass(frozen=True)
class ClassTypeVarRef:
    """
    A class type variable represents one of the types of the class, if it is a generic
    class.
    """

    index: int

    def to_just(self) -> JustTypeRef:
        raise NotImplementedError("egg-smol does not support generic classes yet.")


@dataclass(frozen=True)
class TypeRefWithVars:
    name: str
    args: tuple[TypeOrVarRef, ...] = ()

    def to_just(self) -> JustTypeRef:
        return JustTypeRef(self.name, tuple(a.to_just() for a in self.args))


TypeOrVarRef = Union[ClassTypeVarRef, TypeRefWithVars]


@dataclass(frozen=True)
class FunctionRef:
    name: str

    def generate_egg_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class MethodRef:
    class_name: str
    method_name: str

    def generate_egg_name(self) -> str:
        return f"{self.class_name}__{self.method_name}"


@dataclass(frozen=True)
class ClassMethodRef:
    class_name: str
    method_name: str

    def to_egg(self, decls: Declarations) -> str:
        return decls._callable_ref_to_egg_fn[self]

    def generate_egg_name(self) -> str:
        return f"{self.class_name}__{self.method_name}"


@dataclass(frozen=True)
class ConstantRef:
    name: str

    def generate_egg_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class ClassVariableRef:
    class_name: str
    variable_name: str

    def generate_egg_name(self) -> str:
        return f"{self.class_name}__{self.variable_name}"


CallableRef = Union[FunctionRef, MethodRef, ClassMethodRef, ConstantRef, ClassVariableRef]


@dataclass(frozen=True)
class FunctionDecl:
    # TODO: Add arg name to arg so can call with keyword arg
    arg_types: tuple[TypeOrVarRef, ...]
    return_type: TypeOrVarRef
    cost: Optional[int] = None
    default: Optional[ExprDecl] = None
    merge: Optional[ExprDecl] = None
    merge_action: tuple[ActionDecl, ...] = ()

    def to_commands(self, decls: Declarations, egg_name: str) -> Iterable[bindings._Command]:
        just_arg_types = [a.to_just() for a in self.arg_types]
        for a in just_arg_types:
            yield from decls._register_sort(a)
        just_return_type = self.return_type.to_just()
        yield from decls._register_sort(just_return_type)

        egg_fn_decl = bindings.FunctionDecl(
            egg_name,
            # Remove all vars from the type refs, raising an errory if we find one,
            # since we cannot create egg functions with vars
            bindings.Schema(
                [decls._type_ref_to_egg_sort[a] for a in just_arg_types],
                decls._type_ref_to_egg_sort[just_return_type],
            ),
            self.default.to_egg(decls) if self.default else None,
            self.merge.to_egg(decls) if self.merge else None,
            [action_decl_to_egg(decls, a) for a in self.merge_action],
            self.cost,
        )
        yield bindings.Function(egg_fn_decl)


@dataclass(frozen=True)
class VarDecl:
    name: str

    @classmethod
    def from_egg(cls, var: bindings.Var) -> tuple[JustTypeRef, LitDecl]:
        raise NotImplementedError("Cannot turn var into egg type because typing unknown.")

    def to_egg(self, _decls: Declarations) -> bindings.Var:
        return bindings.Var(self.name)

    def pretty(self, **kwargs) -> str:
        return self.name


LitType = Union[int, str, float, None]


@dataclass(frozen=True)
class LitDecl:
    value: LitType

    @classmethod
    def from_egg(cls, lit: bindings.Lit) -> tuple[JustTypeRef, LitDecl]:
        if isinstance(lit.value, bindings.Int):
            return JustTypeRef("i64"), cls(lit.value.value)
        if isinstance(lit.value, bindings.String):
            return JustTypeRef("String"), cls(lit.value.value)
        if isinstance(lit.value, bindings.F64):
            return JustTypeRef("f64"), cls(lit.value.value)
        elif isinstance(lit.value, bindings.Unit):
            return JustTypeRef("unit"), cls(None)
        assert_never(lit.value)

    def to_egg(self, _decls: Declarations) -> bindings.Lit:
        if self.value is None:
            return bindings.Lit(bindings.Unit())
        if isinstance(self.value, int):
            return bindings.Lit(bindings.Int(self.value))
        if isinstance(self.value, float):
            return bindings.Lit(bindings.F64(self.value))
        if isinstance(self.value, str):
            return bindings.Lit(bindings.String(self.value))
        assert_never(self.value)

    def pretty(self, wrap_lit=True, **kwargs) -> str:
        """
        Returns a string representation of the literal.

        :param wrap_lit: If True, wraps the literal in a call to the literal constructor.
        """
        if self.value is None:
            return "unit()"
        if isinstance(self.value, int):
            return f"i64({self.value})" if wrap_lit else str(self.value)
        if isinstance(self.value, float):
            return f"f64({self.value})" if wrap_lit else str(self.value)
        if isinstance(self.value, str):
            return f"String({repr(self.value)})" if wrap_lit else repr(self.value)
        assert_never(self.value)


@dataclass(frozen=True)
class CallDecl:
    callable: CallableRef
    args: tuple[ExprDecl, ...] = ()
    # type parameters that were bound to the callable, if it is a classmethod
    bound_tp_params: Optional[tuple[JustTypeRef, ...]] = None

    def __post_init__(self):
        if self.bound_tp_params and not isinstance(self.callable, ClassMethodRef):
            raise ValueError("Cannot bind type parameters to a non-class method callable.")

    @classmethod
    def from_egg(cls, decls: Declarations, call: bindings.Call) -> tuple[JustTypeRef, CallDecl]:
        from .type_constraint_solver import TypeConstraintSolver

        results = [tp_and_expr_decl_from_egg(decls, a) for a in call.args]
        arg_types = tuple(r[0] for r in results)
        arg_decls = tuple(r[1] for r in results)

        # Find the first callable ref that matches the call
        for callable_ref in decls._egg_fn_to_callable_refs[call.name]:
            tcs = TypeConstraintSolver()
            fn_decl = decls._get_function_decl(callable_ref)
            return_tp = tcs.infer_return_type(fn_decl.arg_types, fn_decl.return_type, arg_types)
            return return_tp, cls(callable_ref, arg_decls)
        raise ValueError(f"Could not find callable ref for call {call}")

    def to_egg(self, decls: Declarations) -> bindings.Call:
        """Convert a Call to an egg Call."""
        egg_fn = decls._callable_ref_to_egg_fn[self.callable]
        return bindings.Call(egg_fn, [a.to_egg(decls) for a in self.args])

    def pretty(self, parens=True, **kwargs) -> str:
        """
        Pretty print the call.

        :param parens: If true, wrap the call in parens if it is a binary or unary method call.
        """
        ref, args = self.callable, list(self.args)
        if isinstance(ref, FunctionRef):
            fn_str = ref.name
        elif isinstance(ref, ClassMethodRef):
            tp_ref = JustTypeRef(ref.class_name, self.bound_tp_params or ())
            if ref.method_name == "__init__":
                fn_str = tp_ref.pretty()
            else:
                fn_str = f"{tp_ref.pretty()}.{ref.method_name}"
        elif isinstance(ref, MethodRef):
            name = ref.method_name
            slf, *args = args
            if name in UNARY_METHODS:
                return f"{UNARY_METHODS[name]}{slf.pretty()}"
            elif name in BINARY_METHODS:
                assert len(args) == 1
                expr = f"{slf.pretty()} {BINARY_METHODS[name]} {args[0].pretty(wrap_lit=False)}"
                return expr if not parens else f"({expr})"
            elif name == "__getitem__":
                assert len(args) == 1
                return f"{slf.pretty()}[{args[0].pretty(wrap_lit=False)}]"
            elif name == "__call__":
                return f"{slf.pretty()}({', '.join(a.pretty(wrap_lit=False) for a in args)})"
            fn_str = f"{slf.pretty()}.{name}"
        elif isinstance(ref, ConstantRef):
            return ref.name
        elif isinstance(ref, ClassVariableRef):
            return f"{ref.class_name}.{ref.variable_name}"
        else:
            assert_never(ref)
        return f"{fn_str}({', '.join(a.pretty(wrap_lit=False) for a in args)})"


def test_expr_pretty():
    assert VarDecl("x").pretty() == "x"
    assert LitDecl(42).pretty() == "i64(42)"
    assert LitDecl("foo").pretty() == 'String("foo")'
    assert LitDecl(None).pretty() == "unit()"
    assert CallDecl(FunctionRef("foo"), (VarDecl("x"),)).pretty() == "foo(x)"
    assert CallDecl(FunctionRef("foo"), (VarDecl("x"), VarDecl("y"), VarDecl("z"))).pretty() == "foo(x, y, z)"
    assert CallDecl(MethodRef("foo", "__add__"), (VarDecl("x"), VarDecl("y"))).pretty() == "x + y"
    assert CallDecl(MethodRef("foo", "__getitem__"), (VarDecl("x"), VarDecl("y"))).pretty() == "x[y]"
    assert CallDecl(ClassMethodRef("foo", "__init__"), (VarDecl("x"), VarDecl("y"))).pretty() == "foo(x, y)"
    assert CallDecl(ClassMethodRef("foo", "bar"), (VarDecl("x"), VarDecl("y"))).pretty() == "foo.bar(x, y)"
    assert CallDecl(MethodRef("foo", "__call__"), (VarDecl("x"), VarDecl("y"))).pretty() == "x(y)"
    assert (
        CallDecl(
            ClassMethodRef("Map", "__init__"),
            (),
            (JustTypeRef("i64"), JustTypeRef("unit")),
        ).pretty()
        == "Map[i64, unit]()"
    )


ExprDecl = Union[VarDecl, LitDecl, CallDecl]


def tp_and_expr_decl_from_egg(decls: Declarations, expr: bindings._Expr) -> tuple[JustTypeRef, ExprDecl]:
    if isinstance(expr, bindings.Var):
        return VarDecl.from_egg(expr)
    if isinstance(expr, bindings.Lit):
        return LitDecl.from_egg(expr)
    if isinstance(expr, bindings.Call):
        return CallDecl.from_egg(decls, expr)
    assert_never(expr)


@dataclass
class ClassDecl:
    methods: dict[str, FunctionDecl] = field(default_factory=dict)
    class_methods: dict[str, FunctionDecl] = field(default_factory=dict)
    class_variables: dict[str, FunctionDecl] = field(default_factory=dict)
    n_type_vars: int = 0


@dataclass(frozen=True)
class RewriteDecl:
    lhs: ExprDecl
    rhs: ExprDecl
    conditions: tuple[FactDecl, ...]

    def to_egg(self, decls: Declarations) -> bindings.Rewrite:
        return bindings.Rewrite(
            self.lhs.to_egg(decls),
            self.rhs.to_egg(decls),
            [fact_decl_to_egg(decls, c) for c in self.conditions],
        )


@dataclass(frozen=True)
class EqDecl:
    exprs: tuple[ExprDecl, ...]

    def to_egg(self, decls: Declarations) -> bindings.Eq:
        return bindings.Eq([e.to_egg(decls) for e in self.exprs])


FactDecl = Union[ExprDecl, EqDecl]


def fact_decl_to_egg(decls: Declarations, fact: FactDecl) -> bindings._Fact:
    if isinstance(fact, EqDecl):
        return fact.to_egg(decls)
    return bindings.Fact(fact.to_egg(decls))


@dataclass(frozen=True)
class RuleDecl:
    head: tuple[ActionDecl, ...]
    body: tuple[FactDecl, ...]

    def to_egg(self, decls: Declarations) -> bindings.Rule:
        return bindings.Rule(
            [action_decl_to_egg(decls, a) for a in self.head],
            [fact_decl_to_egg(decls, f) for f in self.body],
        )


@dataclass(frozen=True)
class LetDecl:
    name: str
    value: ExprDecl

    def to_egg(self, decls: Declarations) -> bindings.Let:
        return bindings.Let(self.name, self.value.to_egg(decls))


@dataclass(frozen=True)
class SetDecl:
    call: CallDecl
    rhs: ExprDecl

    def to_egg(self, decls: Declarations) -> bindings.Set:
        return bindings.Set(
            decls._callable_ref_to_egg_fn[self.call.callable],
            [a.to_egg(decls) for a in self.call.args],
            self.rhs.to_egg(decls),
        )


@dataclass(frozen=True)
class DeleteDecl:
    call: CallDecl

    def to_egg(self, decls: Declarations) -> bindings.Delete:
        return bindings.Delete(
            decls._callable_ref_to_egg_fn[self.call.callable], [a.to_egg(decls) for a in self.call.args]
        )


@dataclass(frozen=True)
class UnionDecl:
    lhs: ExprDecl
    rhs: ExprDecl

    def to_egg(self, decls: Declarations) -> bindings.Union:
        return bindings.Union(self.lhs.to_egg(decls), self.rhs.to_egg(decls))


@dataclass(frozen=True)
class PanicDecl:
    message: str

    def to_egg(self, _decls: Declarations) -> bindings.Panic:
        return bindings.Panic(self.message)


ActionDecl = Union[LetDecl, SetDecl, DeleteDecl, UnionDecl, PanicDecl, ExprDecl]


def action_decl_to_egg(decls: Declarations, action: ActionDecl) -> bindings._Action:
    if isinstance(action, (CallDecl, LitDecl, VarDecl)):
        return bindings.Expr_(action.to_egg(decls))
    return action.to_egg(decls)
