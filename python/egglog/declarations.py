"""
Data only descriptions of the components of an egraph and the expressions.

We seperate it it into two pieces, the references the declerations, so that we can report mutually recursive types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, Union, runtime_checkable

from typing_extensions import Self, assert_never

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


__all__ = [
    "Declarations",
    "DeclerationsLike",
    "DelayedDeclerations",
    "upcast_declerations",
    "Declarations",
    "JustTypeRef",
    "ClassTypeVarRef",
    "TypeRefWithVars",
    "TypeOrVarRef",
    "MethodRef",
    "ClassMethodRef",
    "FunctionRef",
    "ConstantRef",
    "ClassVariableRef",
    "PropertyRef",
    "CallableRef",
    "FunctionDecl",
    "RelationDecl",
    "ConstantDecl",
    "CallableDecl",
    "VarDecl",
    "PyObjectDecl",
    "PartialCallDecl",
    "LitType",
    "LitDecl",
    "CallDecl",
    "ExprDecl",
    "TypedExprDecl",
    "ClassDecl",
    "RulesetDecl",
    "CombinedRulesetDecl",
    "SaturateDecl",
    "RepeatDecl",
    "SequenceDecl",
    "RunDecl",
    "ScheduleDecl",
    "EqDecl",
    "ExprFactDecl",
    "FactDecl",
    "LetDecl",
    "SetDecl",
    "ExprActionDecl",
    "ChangeDecl",
    "UnionDecl",
    "PanicDecl",
    "ActionDecl",
    "RewriteDecl",
    "BiRewriteDecl",
    "RuleDecl",
    "RewriteOrRuleDecl",
    "ActionCommandDecl",
    "CommandDecl",
    "SpecialFunctions",
    "FunctionSignature",
    "DefaultReplacement",
]


@dataclass
class DelayedDeclerations:
    __egg_decls_thunk__: Callable[[], Declarations]

    @property
    def __egg_decls__(self) -> Declarations:
        return self.__egg_decls_thunk__()


@runtime_checkable
class HasDeclerations(Protocol):
    @property
    def __egg_decls__(self) -> Declarations: ...


DeclerationsLike: TypeAlias = Union[HasDeclerations, None, "Declarations"]


def upcast_declerations(declerations_like: Iterable[DeclerationsLike]) -> list[Declarations]:
    d = []
    for l in declerations_like:
        if l is None:
            continue
        if isinstance(l, HasDeclerations):
            d.append(l.__egg_decls__)
        elif isinstance(l, Declarations):
            d.append(l)
        else:
            assert_never(l)
    return d


@dataclass
class Declarations:
    _functions: dict[str, FunctionDecl | RelationDecl] = field(default_factory=dict)
    _constants: dict[str, ConstantDecl] = field(default_factory=dict)
    _classes: dict[str, ClassDecl] = field(default_factory=dict)
    _rulesets: dict[str, RulesetDecl | CombinedRulesetDecl] = field(default_factory=lambda: {"": RulesetDecl([])})

    @classmethod
    def create(cls, *others: DeclerationsLike) -> Declarations:
        others = upcast_declerations(others)
        if not others:
            return Declarations()
        first, *rest = others
        if not rest:
            return first
        new = first.copy()
        new.update(*rest)
        return new

    def copy(self) -> Declarations:
        new = Declarations()
        new |= self
        return new

    def update(self, *others: DeclerationsLike) -> None:
        for other in others:
            self |= other

    def __or__(self, other: DeclerationsLike) -> Declarations:
        result = self.copy()
        result |= other
        return result

    def __ior__(self, other: DeclerationsLike) -> Self:
        if other is None:
            return self
        if isinstance(other, HasDeclerations):
            other = other.__egg_decls__
        other.update_other(self)
        return self

    def update_other(self, other: Declarations) -> None:
        """
        Updates the other decl with these values in palce.
        """
        other._functions |= self._functions
        other._classes |= self._classes
        other._constants |= self._constants
        other._rulesets |= self._rulesets

    def get_callable_decl(self, ref: CallableRef) -> CallableDecl:
        match ref:
            case FunctionRef(name):
                return self._functions[name]
            case ConstantRef(name):
                return self._constants[name]
            case MethodRef(class_name, method_name):
                return self._classes[class_name].methods[method_name]
            case ClassVariableRef(class_name, name):
                return self._classes[class_name].class_variables[name]
            case ClassMethodRef(class_name, name):
                return self._classes[class_name].class_methods[name]
            case PropertyRef(class_name, property_name):
                return self._classes[class_name].properties[property_name]
        assert_never(ref)

    def has_method(self, class_name: str, method_name: str) -> bool | None:
        """
        Returns whether the given class has the given method, or None if we cant find the class.
        """
        if class_name in self._classes:
            return method_name in self._classes[class_name].methods
        return None

    def get_class_decl(self, name: str) -> ClassDecl:
        return self._classes[name]


@dataclass
class ClassDecl:
    egg_name: str | None = None
    type_vars: tuple[str, ...] = ()
    builtin: bool = False
    class_methods: dict[str, FunctionDecl] = field(default_factory=dict)
    # These have to be seperate from class_methods so that printing them can be done easily
    class_variables: dict[str, ConstantDecl] = field(default_factory=dict)
    methods: dict[str, FunctionDecl] = field(default_factory=dict)
    properties: dict[str, FunctionDecl] = field(default_factory=dict)
    preserved_methods: dict[str, Callable] = field(default_factory=dict)


@dataclass(frozen=True)
class RulesetDecl:
    rules: list[RewriteOrRuleDecl]

    # Make hashable so when traversing for pretty-fying we can know which rulesets we have already
    # made into strings
    def __hash__(self) -> int:
        return hash((type(self), tuple(self.rules)))


@dataclass(frozen=True)
class CombinedRulesetDecl:
    rulesets: tuple[str, ...]


# Have two different types of type refs, one that can include vars recursively and one that cannot.
# We only use the one with vars for classmethods and methods, and the other one for egg references as
# well as runtime values.
@dataclass(frozen=True)
class JustTypeRef:
    name: str
    args: tuple[JustTypeRef, ...] = ()

    def to_var(self) -> TypeRefWithVars:
        return TypeRefWithVars(self.name, tuple(a.to_var() for a in self.args))

    def __str__(self) -> str:
        if self.args:
            return f"{self.name}[{', '.join(str(a) for a in self.args)}]"
        return self.name


##
# Type references with vars
##


@dataclass(frozen=True)
class ClassTypeVarRef:
    """
    A class type variable represents one of the types of the class, if it is a generic class.
    """

    name: str

    def to_just(self) -> JustTypeRef:
        msg = "egglog does not support generic classes yet."
        raise NotImplementedError(msg)

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TypeRefWithVars:
    name: str
    args: tuple[TypeOrVarRef, ...] = ()

    def to_just(self) -> JustTypeRef:
        return JustTypeRef(self.name, tuple(a.to_just() for a in self.args))

    def __str__(self) -> str:
        if self.args:
            return f"{self.name}[{', '.join(str(a) for a in self.args)}]"
        return self.name


TypeOrVarRef: TypeAlias = ClassTypeVarRef | TypeRefWithVars

##
# Callables References
##


@dataclass(frozen=True)
class FunctionRef:
    name: str


@dataclass(frozen=True)
class ConstantRef:
    name: str


@dataclass(frozen=True)
class MethodRef:
    class_name: str
    method_name: str


@dataclass(frozen=True)
class ClassMethodRef:
    class_name: str
    method_name: str


@dataclass(frozen=True)
class ClassVariableRef:
    class_name: str
    var_name: str


@dataclass(frozen=True)
class PropertyRef:
    class_name: str
    property_name: str


CallableRef: TypeAlias = FunctionRef | ConstantRef | MethodRef | ClassMethodRef | ClassVariableRef | PropertyRef


##
# Callables
##


@dataclass(frozen=True)
class RelationDecl:
    arg_types: tuple[JustTypeRef, ...]
    # List of defaults. None for any arg which doesn't have one.
    arg_defaults: tuple[ExprDecl | None, ...]
    egg_name: str | None

    def to_function_decl(self) -> FunctionDecl:
        return FunctionDecl(
            FunctionSignature(
                arg_types=tuple(a.to_var() for a in self.arg_types),
                arg_names=tuple(f"__{i}" for i in range(len(self.arg_types))),
                arg_defaults=self.arg_defaults,
                return_type=TypeRefWithVars("Unit"),
            ),
            egg_name=self.egg_name,
            default=LitDecl(None),
        )


# TODO: Move this to a type of replacement instead of part of constant or function


@dataclass(frozen=True)
class DefaultReplacement:
    """
    The default replacement for the function.

    If the ruleset is not specified, it is the default ruleset.
    """

    expr: ExprDecl
    ruleset: str | None


@dataclass(frozen=True)
class ConstantDecl:
    """
    Same as `(declare)` in egglog
    """

    type_ref: JustTypeRef
    default_replacement: DefaultReplacement | None
    egg_name: str | None = None

    def to_function_decl(self) -> FunctionDecl:
        return FunctionDecl(
            FunctionSignature(return_type=self.type_ref.to_var()),
            default_replacement=self.default_replacement,
            egg_name=self.egg_name,
        )


# special cases for partial function creation and application, which cannot use the normal python rules
SpecialFunctions: TypeAlias = Literal["fn-partial", "fn-app"]


@dataclass(frozen=True)
class FunctionSignature:
    arg_types: tuple[TypeOrVarRef, ...] = ()
    arg_names: tuple[str, ...] = ()
    # List of defaults. None for any arg which doesn't have one.
    arg_defaults: tuple[ExprDecl | None, ...] = ()
    # If None, then the first arg is mutated and returned
    return_type: TypeOrVarRef | None = None
    var_arg_type: TypeOrVarRef | None = None

    @property
    def semantic_return_type(self) -> TypeOrVarRef:
        """
        The type that is returned by the function, which wil be in the first arg if it mutates it.
        """
        return self.return_type or self.arg_types[0]

    @property
    def mutates(self) -> bool:
        return self.return_type is None


@dataclass(frozen=True)
class FunctionDecl:
    signature: FunctionSignature | SpecialFunctions = field(default_factory=FunctionSignature)
    default_replacement: DefaultReplacement | None = None
    # Egg params
    builtin: bool = False
    egg_name: str | None = None
    cost: int | None = None
    default: ExprDecl | None = None
    on_merge: tuple[ActionDecl, ...] = ()
    merge: ExprDecl | None = None
    unextractable: bool = False

    def to_function_decl(self) -> FunctionDecl:
        return self


CallableDecl: TypeAlias = RelationDecl | ConstantDecl | FunctionDecl

##
# Expressions
##


@dataclass(frozen=True)
class VarDecl:
    name: str


@dataclass(frozen=True)
class PyObjectDecl:
    value: object

    def __hash__(self) -> int:
        """Tries using the hash of the value, if unhashable use the ID."""
        try:
            return hash((type(self.value), self.value))
        except TypeError:
            return id(self.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PyObjectDecl):
            return False
        return self.parts == other.parts

    @property
    def parts(self) -> tuple[type, object]:
        return (type(self.value), self.value)


LitType: TypeAlias = int | str | float | bool | None


@dataclass(frozen=True)
class LitDecl:
    value: LitType

    def __hash__(self) -> int:
        """
        Include type in has so that 1.0 != 1
        """
        return hash(self.parts)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LitDecl):
            return False
        return self.parts == other.parts

    @property
    def parts(self) -> tuple[type, LitType]:
        return (type(self.value), self.value)


@dataclass(frozen=True)
class CallDecl:
    callable: CallableRef
    # TODO: Can I make these not typed expressions?
    args: tuple[TypedExprDecl, ...] = ()
    # type parameters that were bound to the callable, if it is a classmethod
    # Used for pretty printing classmethod calls with type parameters
    bound_tp_params: tuple[JustTypeRef, ...] | None = None

    def __post_init__(self) -> None:
        if self.bound_tp_params and not isinstance(self.callable, ClassMethodRef):
            msg = "Cannot bind type parameters to a non-class method callable."
            raise ValueError(msg)

    def __hash__(self) -> int:
        return self._cached_hash

    @cached_property
    def _cached_hash(self) -> int:
        return hash((self.callable, self.args, self.bound_tp_params))

    def __eq__(self, other: object) -> bool:
        # Override eq to use cached hash for perf
        if not isinstance(other, CallDecl):
            return False
        return hash(self) == hash(other)


@dataclass(frozen=True)
class PartialCallDecl:
    """
    A partially applied function aka a function sort.

    Note it does not need to have any args, in which case it's just a function pointer.

    Seperated from the call decl so it's clear it is translated to a `unstable-fn` call.
    """

    call: CallDecl


ExprDecl: TypeAlias = VarDecl | LitDecl | CallDecl | PyObjectDecl | PartialCallDecl


@dataclass(frozen=True)
class TypedExprDecl:
    tp: JustTypeRef
    expr: ExprDecl

    def descendants(self) -> list[TypedExprDecl]:
        """
        Returns a list of all the descendants of this expression.
        """
        l = [self]
        if isinstance(self.expr, CallDecl):
            for a in self.expr.args:
                l.extend(a.descendants())
        return l


##
# Schedules
##


@dataclass(frozen=True)
class SaturateDecl:
    schedule: ScheduleDecl


@dataclass(frozen=True)
class RepeatDecl:
    schedule: ScheduleDecl
    times: int


@dataclass(frozen=True)
class SequenceDecl:
    schedules: tuple[ScheduleDecl, ...]


@dataclass(frozen=True)
class RunDecl:
    ruleset: str
    until: tuple[FactDecl, ...] | None


ScheduleDecl: TypeAlias = SaturateDecl | RepeatDecl | SequenceDecl | RunDecl

##
# Facts
##


@dataclass(frozen=True)
class EqDecl:
    tp: JustTypeRef
    exprs: tuple[ExprDecl, ...]


@dataclass(frozen=True)
class ExprFactDecl:
    typed_expr: TypedExprDecl


FactDecl: TypeAlias = EqDecl | ExprFactDecl

##
# Actions
##


@dataclass(frozen=True)
class LetDecl:
    name: str
    typed_expr: TypedExprDecl


@dataclass(frozen=True)
class SetDecl:
    tp: JustTypeRef
    call: CallDecl
    rhs: ExprDecl


@dataclass(frozen=True)
class ExprActionDecl:
    typed_expr: TypedExprDecl


@dataclass(frozen=True)
class ChangeDecl:
    tp: JustTypeRef
    call: CallDecl
    change: Literal["delete", "subsume"]


@dataclass(frozen=True)
class UnionDecl:
    tp: JustTypeRef
    lhs: ExprDecl
    rhs: ExprDecl


@dataclass(frozen=True)
class PanicDecl:
    msg: str


ActionDecl: TypeAlias = LetDecl | SetDecl | ExprActionDecl | ChangeDecl | UnionDecl | PanicDecl


##
# Commands
##


@dataclass(frozen=True)
class RewriteDecl:
    tp: JustTypeRef
    lhs: ExprDecl
    rhs: ExprDecl
    conditions: tuple[FactDecl, ...]
    subsume: bool


@dataclass(frozen=True)
class BiRewriteDecl:
    tp: JustTypeRef
    lhs: ExprDecl
    rhs: ExprDecl
    conditions: tuple[FactDecl, ...]


@dataclass(frozen=True)
class RuleDecl:
    head: tuple[ActionDecl, ...]
    body: tuple[FactDecl, ...]
    name: str | None


RewriteOrRuleDecl: TypeAlias = RewriteDecl | BiRewriteDecl | RuleDecl


@dataclass(frozen=True)
class ActionCommandDecl:
    action: ActionDecl


CommandDecl: TypeAlias = RewriteOrRuleDecl | ActionCommandDecl
