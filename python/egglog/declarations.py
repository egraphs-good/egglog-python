"""
Data only descriptions of the components of an egraph and the expressions.

We seperate it it into two pieces, the references the declerations, so that we can report mutually recursive types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Protocol, TypeAlias, Union, runtime_checkable

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
    "LitType",
    "LitDecl",
    "CallDecl",
    "ExprDecl",
    "TypedExprDecl",
    "ClassDecl",
    "RulesetDecl",
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
    "DeleteDecl",
    "UnionDecl",
    "PanicDecl",
    "ActionDecl",
    "RewriteDecl",
    "BiRewriteDecl",
    "RuleDecl",
    "RewriteOrRuleDecl",
    "ActionCommandDecl",
    "CommandDecl",
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


# TODO: Make all ClassDecls take deferred type refs, which return new decls when resolving.


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
    _rulesets: dict[str, RulesetDecl] = field(default_factory=dict)

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

    # def set_callable_decl(self, ref: CallableRef, decl: CallableDecl) -> None:
    #     """
    #     Sets a function declaration for the given callable reference.
    #     """
    #     match ref:
    #         case FunctionRef(name):
    #             if name in self._functions:
    #                 raise ValueError(f"Function {name} already registered")
    #             self._functions[name] = decl
    #         case MethodRef(class_name, method_name):
    #             assert isinstance(decl, FunctionDecl | RelationDecl)
    #             methods = self._classes[class_name].methods
    #             if method_name in methods:
    #                 raise ValueError(f"Method {class_name}.{method_name} already registered")
    #             methods[method_name] = decl
    #         case ClassMethodRef(class_name, method_name):
    #             if method_name in self._classes[class_name].class_methods:
    #                 raise ValueError(f"Class method {class_name}.{method_name} already registered")
    #             self._classes[class_name].class_methods[method_name] = decl
    #         case PropertyRef(class_name, property_name):
    #             if property_name in self._classes[class_name].properties:
    #                 raise ValueError(f"Property {class_name}.{property_name} already registered")
    #             self._classes[class_name].properties[property_name] = decl
    #         case _:
    #             assert_never(ref)

    def get_callable_decl(self, ref: CallableRef) -> CallableDecl:
        match ref:
            case ConstantRef(name) | FunctionRef(name):
                return self._functions[name]
            case MethodRef(class_name, method_name):
                return self._classes[class_name].methods[method_name]
            case ClassVariableRef(class_name, name) | ClassMethodRef(class_name, name):
                return self._classes[class_name].class_methods[name]
            case PropertyRef(class_name, property_name):
                return self._classes[class_name].properties[property_name]
        assert_never(ref)

    # def set_constant_type(self, ref: ConstantCallableRef, tp: JustTypeRef) -> None:
    #     match ref:
    #         case ConstantRef(name):
    #             if name in self._constants:
    #                 raise ValueError(f"Constant {name} already registered")
    #             self._constants[name] = tp
    #         case ClassVariableRef(class_name, variable_name):
    #             if variable_name in self._classes[class_name].class_variables:
    #                 raise ValueError(f"Class variable {class_name}.{variable_name} already registered")
    #             self._classes[class_name].class_variables[variable_name] = tp
    #         case _:
    #             assert_never(ref)

    # def register_callable_ref(self, ref: CallableRef, egg_name: str) -> None:
    #     """
    #     Registers a callable reference with the given egg name.

    #     The callable's function needs to be registered first.
    #     """
    #     if ref in self._callable_ref_to_egg_fn:
    #         raise ValueError(f"Callable ref {ref} already registered")
    #     self._callable_ref_to_egg_fn[ref] = egg_name
    #     self._egg_fn_to_callable_refs[egg_name].add(ref)

    # def get_callable_refs(self, egg_name: str) -> Iterable[CallableRef]:
    #     return self._egg_fn_to_callable_refs[egg_name]

    # def get_egg_fn(self, ref: CallableRef) -> str:
    #     return self._callable_ref_to_egg_fn[ref]

    # def get_egg_sort(self, ref: JustTypeRef) -> str:
    #     return self._type_ref_to_egg_sort[ref]

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


@dataclass
class RulesetDecl:
    rules: list[RewriteOrRuleDecl]

    # Make hashable so when traversing for pretty-fying we can know which rulesets we have already
    # made into strings
    def __hash__(self) -> int:
        return hash((type(self), tuple(self.rules)))


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
            arg_types=tuple(a.to_var() for a in self.arg_types),
            arg_names=tuple(f"__{i}" for i in range(len(self.arg_types))),
            arg_defaults=self.arg_defaults,
            return_type=TypeRefWithVars("Unit"),
            egg_name=self.egg_name,
            default=LitDecl(None),
        )


@dataclass(frozen=True)
class ConstantDecl:
    """
    Same as `(declare)` in egglog
    """

    type_ref: JustTypeRef
    egg_name: str | None = None

    def to_function_decl(self) -> FunctionDecl:
        return FunctionDecl(
            arg_types=(),
            arg_names=(),
            arg_defaults=(),
            return_type=self.type_ref.to_var(),
            egg_name=self.egg_name,
        )


@dataclass(frozen=True)
class FunctionDecl:
    # All args are delayed except for relations converted to function decls
    arg_types: tuple[TypeOrVarRef, ...]
    arg_names: tuple[str, ...]
    # List of defaults. None for any arg which doesn't have one.
    arg_defaults: tuple[ExprDecl | None, ...]
    # If None, then the first arg is mutated and returned
    return_type: TypeOrVarRef | None
    var_arg_type: TypeOrVarRef | None = None

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

    @property
    def semantic_return_type(self) -> TypeOrVarRef:
        """
        The type that is returned by the function, which wil be in the first arg if it mutates it.
        """
        return self.return_type or self.arg_types[0]

    @property
    def mutates(self) -> bool:
        return self.return_type is None


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


LitType: TypeAlias = int | str | float | bool | None


@dataclass(frozen=True)
class LitDecl:
    value: LitType


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


ExprDecl: TypeAlias = VarDecl | LitDecl | CallDecl | PyObjectDecl


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
    exprs: tuple[ExprDecl, ...]


@dataclass(frozen=True)
class ExprFactDecl:
    expr: ExprDecl


FactDecl: TypeAlias = EqDecl | ExprFactDecl

##
# Actions
##


@dataclass(frozen=True)
class LetDecl:
    name: str
    expr: ExprDecl


@dataclass(frozen=True)
class SetDecl:
    call: CallDecl
    rhs: ExprDecl


@dataclass(frozen=True)
class ExprActionDecl:
    expr: ExprDecl


@dataclass(frozen=True)
class DeleteDecl:
    call: CallDecl


@dataclass(frozen=True)
class UnionDecl:
    lhs: ExprDecl
    rhs: ExprDecl


@dataclass(frozen=True)
class PanicDecl:
    msg: str


ActionDecl: TypeAlias = LetDecl | SetDecl | ExprActionDecl | DeleteDecl | UnionDecl | PanicDecl


##
# Commands
##


@dataclass(frozen=True)
class RewriteDecl:
    lhs: ExprDecl
    rhs: ExprDecl
    conditions: tuple[FactDecl, ...]


@dataclass(frozen=True)
class BiRewriteDecl:
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
