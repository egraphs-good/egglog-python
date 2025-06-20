"""
Data only descriptions of the components of an egraph and the expressions.

We seperate it it into two pieces, the references the declerations, so that we can report mutually recursive types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, TypeAlias, TypeVar, Union, cast, runtime_checkable
from weakref import WeakValueDictionary

from typing_extensions import Self, assert_never

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


__all__ = [
    "ActionCommandDecl",
    "ActionDecl",
    "BiRewriteDecl",
    "CallDecl",
    "CallableDecl",
    "CallableRef",
    "ChangeDecl",
    "ClassDecl",
    "ClassMethodRef",
    "ClassTypeVarRef",
    "ClassVariableRef",
    "CombinedRulesetDecl",
    "CommandDecl",
    "ConstantDecl",
    "ConstantRef",
    "ConstructorDecl",
    "Declarations",
    "Declarations",
    "DeclerationsLike",
    "DefaultRewriteDecl",
    "DelayedDeclerations",
    "EqDecl",
    "ExprActionDecl",
    "ExprDecl",
    "ExprFactDecl",
    "FactDecl",
    "FunctionDecl",
    "FunctionRef",
    "FunctionSignature",
    "HasDeclerations",
    "InitRef",
    "JustTypeRef",
    "LetDecl",
    "LitDecl",
    "LitType",
    "MethodRef",
    "PanicDecl",
    "PartialCallDecl",
    "PropertyRef",
    "PyObjectDecl",
    "RelationDecl",
    "RepeatDecl",
    "RewriteDecl",
    "RewriteOrRuleDecl",
    "RuleDecl",
    "RulesetDecl",
    "RunDecl",
    "SaturateDecl",
    "ScheduleDecl",
    "SequenceDecl",
    "SetDecl",
    "SpecialFunctions",
    "TypeOrVarRef",
    "TypeRefWithVars",
    "TypedExprDecl",
    "UnionDecl",
    "UnnamedFunctionRef",
    "VarDecl",
    "replace_typed_expr",
    "upcast_declerations",
]


@dataclass
class DelayedDeclerations:
    __egg_decls_thunk__: Callable[[], Declarations] = field(repr=False)

    @property
    def __egg_decls__(self) -> Declarations:
        thunk = self.__egg_decls_thunk__
        try:
            return thunk()
        # Catch attribute error, so that it isn't bubbled up as a missing attribute and fallbacks on `__getattr__`
        # instead raise explicitly
        except AttributeError as err:
            msg = f"Cannot resolve declarations for {self}"
            raise RuntimeError(msg) from err


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
    _unnamed_functions: set[UnnamedFunctionRef] = field(default_factory=set)
    _functions: dict[str, FunctionDecl | RelationDecl | ConstructorDecl] = field(default_factory=dict)
    _constants: dict[str, ConstantDecl] = field(default_factory=dict)
    _classes: dict[str, ClassDecl] = field(default_factory=dict)
    _rulesets: dict[str, RulesetDecl | CombinedRulesetDecl] = field(default_factory=lambda: {"": RulesetDecl([])})

    @property
    def default_ruleset(self) -> RulesetDecl:
        ruleset = self._rulesets[""]
        assert isinstance(ruleset, RulesetDecl)
        return ruleset

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
        self.update_other(new)
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
        # Must combine rulesets bc the empty ruleset might be different, bc DefaultRewriteDecl
        # is added to functions.
        combined_default_rules: set[RewriteOrRuleDecl] = {*self.default_ruleset.rules, *other.default_ruleset.rules}
        other._rulesets |= self._rulesets
        other._rulesets[""] = RulesetDecl(list(combined_default_rules))

    def get_callable_decl(self, ref: CallableRef) -> CallableDecl:  # noqa: PLR0911
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
            case InitRef(class_name):
                init_fn = self._classes[class_name].init
                assert init_fn, f"Class {class_name} does not have an init function."
                return init_fn
            case UnnamedFunctionRef():
                return ConstructorDecl(ref.signature)

        assert_never(ref)

    def set_function_decl(
        self,
        ref: FunctionRef | MethodRef | ClassMethodRef | PropertyRef | InitRef,
        decl: FunctionDecl | ConstructorDecl,
    ) -> None:
        match ref:
            case FunctionRef(name):
                self._functions[name] = decl
            case MethodRef(class_name, method_name):
                self._classes[class_name].methods[method_name] = decl
            case ClassMethodRef(class_name, name):
                self._classes[class_name].class_methods[name] = decl
            case PropertyRef(class_name, property_name):
                self._classes[class_name].properties[property_name] = decl
            case InitRef(class_name):
                self._classes[class_name].init = decl
            case _:
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

    def get_paramaterized_class(self, name: str) -> TypeRefWithVars:
        """
        Returns a class reference with type parameters, if the class is paramaterized.
        """
        type_vars = self._classes[name].type_vars
        return TypeRefWithVars(name, type_vars)


@dataclass
class ClassDecl:
    egg_name: str | None = None
    type_vars: tuple[ClassTypeVarRef, ...] = ()
    builtin: bool = False
    init: ConstructorDecl | FunctionDecl | None = None
    class_methods: dict[str, FunctionDecl | ConstructorDecl] = field(default_factory=dict)
    # These have to be seperate from class_methods so that printing them can be done easily
    class_variables: dict[str, ConstantDecl] = field(default_factory=dict)
    methods: dict[str, FunctionDecl | ConstructorDecl] = field(default_factory=dict)
    properties: dict[str, FunctionDecl | ConstructorDecl] = field(default_factory=dict)
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

# mapping of name and module of resolved typevars to runtime values
# so that when spitting them back out again can use same instance
# since equality is based on identity not value
_RESOLVED_TYPEVARS: dict[ClassTypeVarRef, TypeVar] = {}


@dataclass(frozen=True)
class ClassTypeVarRef:
    """
    A class type variable represents one of the types of the class, if it is a generic class.
    """

    name: str
    module: str

    def to_just(self) -> JustTypeRef:
        msg = f"{self}: egglog does not support generic classes yet."
        raise NotImplementedError(msg)

    def __str__(self) -> str:
        return str(self.to_type_var())

    @classmethod
    def from_type_var(cls, typevar: TypeVar) -> ClassTypeVarRef:
        res = cls(typevar.__name__, typevar.__module__)
        _RESOLVED_TYPEVARS[res] = typevar
        return res

    def to_type_var(self) -> TypeVar:
        return _RESOLVED_TYPEVARS[self]


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
class UnnamedFunctionRef:
    """
    A reference to a function that doesn't have a name, but does have a body.
    """

    # tuple of var arg names and their types
    args: tuple[TypedExprDecl, ...]
    res: TypedExprDecl

    @property
    def signature(self) -> FunctionSignature:
        arg_types = []
        arg_names = []
        for a in self.args:
            arg_types.append(a.tp.to_var())
            assert isinstance(a.expr, VarDecl)
            arg_names.append(a.expr.name)
        return FunctionSignature(
            arg_types=tuple(arg_types),
            arg_names=tuple(arg_names),
            arg_defaults=(None,) * len(self.args),
            return_type=self.res.tp.to_var(),
        )

    @property
    def egg_name(self) -> None | str:
        return None


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
class InitRef:
    class_name: str


@dataclass(frozen=True)
class ClassVariableRef:
    class_name: str
    var_name: str


@dataclass(frozen=True)
class PropertyRef:
    class_name: str
    property_name: str


CallableRef: TypeAlias = (
    FunctionRef
    | ConstantRef
    | MethodRef
    | ClassMethodRef
    | InitRef
    | ClassVariableRef
    | PropertyRef
    | UnnamedFunctionRef
)


##
# Callables
##


@dataclass(frozen=True)
class RelationDecl:
    arg_types: tuple[JustTypeRef, ...]
    # List of defaults. None for any arg which doesn't have one.
    arg_defaults: tuple[ExprDecl | None, ...]
    egg_name: str | None

    @property
    def signature(self) -> FunctionSignature:
        return FunctionSignature(
            arg_types=tuple(a.to_var() for a in self.arg_types),
            arg_names=tuple(f"__{i}" for i in range(len(self.arg_types))),
            arg_defaults=self.arg_defaults,
            return_type=TypeRefWithVars("Unit"),
        )


@dataclass(frozen=True)
class ConstantDecl:
    """
    Same as `(declare)` in egglog
    """

    type_ref: JustTypeRef
    egg_name: str | None = None

    @property
    def signature(self) -> FunctionSignature:
        return FunctionSignature(return_type=self.type_ref.to_var())


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
    # Whether to reverse args when emitting to egglog
    reverse_args: bool = False

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
    builtin: bool = False
    egg_name: str | None = None
    merge: ExprDecl | None = None


@dataclass(frozen=True)
class ConstructorDecl:
    signature: FunctionSignature = field(default_factory=FunctionSignature)
    egg_name: str | None = None
    cost: int | None = None
    unextractable: bool = False


CallableDecl: TypeAlias = RelationDecl | ConstantDecl | FunctionDecl | ConstructorDecl

##
# Expressions
##


@dataclass(frozen=True)
class VarDecl:
    name: str
    # Differentiate between let bound vars and vars created in rules so that they won't shadow in egglog, by adding a prefix
    is_let: bool


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

    # pool objects for faster __eq__
    _args_to_value: ClassVar[WeakValueDictionary[tuple[object, ...], CallDecl]] = WeakValueDictionary({})

    def __new__(cls, *args: object, **kwargs: object) -> Self:
        """
        Pool CallDecls so that they can be compared by identity more quickly.

        Neccessary bc we search for common parents when serializing CallDecl trees to egglog to
        only serialize each sub-tree once.
        """
        # normalize the args/kwargs to a tuple so that they can be compared
        callable = args[0] if args else kwargs["callable"]
        args_ = args[1] if len(args) > 1 else kwargs.get("args", ())
        bound_tp_params = args[2] if len(args) > 2 else kwargs.get("bound_tp_params")

        normalized_args = (callable, args_, bound_tp_params)
        try:
            return cast("Self", cls._args_to_value[normalized_args])
        except KeyError:
            res = super().__new__(cls)
            cls._args_to_value[normalized_args] = res
            return res

    def __post_init__(self) -> None:
        if self.bound_tp_params and not isinstance(self.callable, ClassMethodRef | InitRef):
            msg = "Cannot bind type parameters to a non-class method callable."
            raise ValueError(msg)

    def __hash__(self) -> int:
        return self._cached_hash

    @cached_property
    def _cached_hash(self) -> int:
        return hash((self.callable, self.args, self.bound_tp_params))

    def __eq__(self, other: object) -> bool:
        return self is other

    def __ne__(self, other: object) -> bool:
        return self is not other


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


def replace_typed_expr(typed_expr: TypedExprDecl, replacements: Mapping[TypedExprDecl, TypedExprDecl]) -> TypedExprDecl:
    """
    Replace all the typed expressions in the given typed expression with the replacements.
    """
    # keep track of the traversed expressions for memoization
    traversed: dict[TypedExprDecl, TypedExprDecl] = {}

    def _inner(typed_expr: TypedExprDecl) -> TypedExprDecl:
        if typed_expr in traversed:
            return traversed[typed_expr]
        if typed_expr in replacements:
            res = replacements[typed_expr]
        else:
            match typed_expr.expr:
                case CallDecl(callable, args, bound_tp_params) | PartialCallDecl(
                    CallDecl(callable, args, bound_tp_params)
                ):
                    new_args = tuple(_inner(a) for a in args)
                    call_decl = CallDecl(callable, new_args, bound_tp_params)
                    res = TypedExprDecl(
                        typed_expr.tp,
                        call_decl if isinstance(typed_expr.expr, CallDecl) else PartialCallDecl(call_decl),
                    )
                case _:
                    res = typed_expr
        traversed[typed_expr] = res
        return res

    return _inner(typed_expr)


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
    left: ExprDecl
    right: ExprDecl


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


@dataclass(frozen=True)
class DefaultRewriteDecl:
    ref: CallableRef
    expr: ExprDecl
    subsume: bool


RewriteOrRuleDecl: TypeAlias = RewriteDecl | BiRewriteDecl | RuleDecl | DefaultRewriteDecl


@dataclass(frozen=True)
class ActionCommandDecl:
    action: ActionDecl


CommandDecl: TypeAlias = RewriteOrRuleDecl | ActionCommandDecl
