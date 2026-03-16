"""
Data only descriptions of the components of an egraph and the expressions.

We separate it it into two pieces, the references the declarations, so that we can report mutually recursive types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache, cached_property
from itertools import chain, repeat
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    Union,
    assert_never,
    cast,
    runtime_checkable,
)
from uuid import UUID
from weakref import WeakValueDictionary

from .bindings import Value

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


__all__ = [
    "ActionCommandDecl",
    "ActionDecl",
    "BackOffDecl",
    "BiRewriteDecl",
    "CallDecl",
    "CallableDecl",
    "CallableRef",
    "ChangeDecl",
    "ClassDecl",
    "ClassMethodRef",
    "ClassVariableRef",
    "CombinedRulesetDecl",
    "CommandDecl",
    "ConstantDecl",
    "ConstantRef",
    "ConstructorDecl",
    "Declarations",
    "Declarations",
    "DeclarationsLike",
    "DefaultRewriteDecl",
    "DelayedDeclarations",
    "DummyDecl",
    "EGraphDecl",
    "EqDecl",
    "ExprActionDecl",
    "ExprDecl",
    "ExprFactDecl",
    "FactDecl",
    "FunctionDecl",
    "FunctionRef",
    "FunctionSignature",
    "GetCostDecl",
    "HasDeclarations",
    "Ident",
    "InitRef",
    "JustTypeRef",
    "LetDecl",
    "LetRefDecl",
    "LetSchedulerDecl",
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
    "SetCostDecl",
    "SetDecl",
    "SpecialFunctions",
    "TypeOrVarRef",
    "TypeRefWithVars",
    "TypeVarError",
    "TypeVarRef",
    "TypedExprDecl",
    "UnboundVarDecl",
    "UnionDecl",
    "UnnamedFunctionRef",
    "ValueDecl",
    "collect_unbound_vars",
    "replace_typed_expr",
    "upcast_declarations",
]


@dataclass(match_args=False)
class DelayedDeclarations:
    __egg_decls_thunk__: Callable[[], Declarations] = field(repr=False)

    @property
    def __egg_decls__(self) -> Declarations:
        thunk = self.__egg_decls_thunk__
        try:
            return thunk()
        # Catch attribute error, so that it isn't bubbled up as a missing attribute and fallbacks on `__getattr__`
        # instead raise explicitly
        except AttributeError as err:
            msg = f"Cannot resolve declarations for {self}: {err}"
            raise RuntimeError(msg) from err


@runtime_checkable
class HasDeclarations(Protocol):
    @property
    def __egg_decls__(self) -> Declarations: ...


DeclarationsLike: TypeAlias = Union[HasDeclarations, None, "Declarations"]


def upcast_declarations(declarations_like: Iterable[DeclarationsLike]) -> list[Declarations]:
    d = []
    for l in declarations_like:
        if l is None:
            continue
        if isinstance(l, HasDeclarations):
            d.append(l.__egg_decls__)
        elif isinstance(l, Declarations):
            d.append(l)
        else:
            assert_never(l)
    return d


@dataclass(frozen=True)
class Ident:
    name: str
    module: str | None = None

    def __str__(self) -> str:
        if self.module:
            return f"{self.module}.{self.name}"
        return self.name

    @classmethod
    def builtin(cls, name: str) -> Ident:
        return cls(name, "egglog.builtins")


default_ruleset_identifier = Ident("")


@dataclass
class Declarations:
    _unnamed_functions: set[UnnamedFunctionRef] = field(default_factory=set)
    _functions: dict[Ident, FunctionDecl | RelationDecl | ConstructorDecl] = field(default_factory=dict)
    _constants: dict[Ident, ConstantDecl] = field(default_factory=dict)
    _classes: dict[Ident, ClassDecl] = field(default_factory=dict)
    _rulesets: dict[Ident, RulesetDecl | CombinedRulesetDecl] = field(
        default_factory=lambda: {default_ruleset_identifier: RulesetDecl([])}
    )

    @property
    def default_ruleset(self) -> RulesetDecl:
        ruleset = self._rulesets[default_ruleset_identifier]
        assert isinstance(ruleset, RulesetDecl)
        return ruleset

    @classmethod
    def create(cls, *others: DeclarationsLike) -> Declarations:
        others = upcast_declarations(others)
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

    def update(self, *others: DeclarationsLike) -> None:
        for other in others:
            self |= other

    def __or__(self, other: DeclarationsLike) -> Declarations:
        result = self.copy()
        result |= other
        return result

    def __ior__(self, other: DeclarationsLike) -> Self:
        if other is None:
            return self
        if isinstance(other, HasDeclarations):
            other = other.__egg_decls__
        other.update_other(self)
        return self

    def update_other(self, other: Declarations) -> None:
        """
        Updates the other decl with these values in place.
        """
        other._functions |= self._functions
        other._classes |= self._classes
        other._constants |= self._constants
        # Must combine rulesets bc the empty ruleset might be different, bc DefaultRewriteDecl
        # is added to functions.
        combined_default_rules: set[RewriteOrRuleDecl] = {*self.default_ruleset.rules, *other.default_ruleset.rules}
        other._rulesets |= self._rulesets
        other._rulesets[default_ruleset_identifier] = RulesetDecl(list(combined_default_rules))

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

    def check_binary_method_with_types(self, method_name: str, self_type: JustTypeRef, other_type: JustTypeRef) -> bool:
        """
        Checks if the class has a binary method compatible with the given types.
        """
        vars: dict[TypeVarRef, JustTypeRef] = {}
        if callable_decl := self._classes[self_type.ident].methods.get(method_name):
            match callable_decl.signature:
                case FunctionSignature((self_arg_type, other_arg_type)) if self_arg_type.matches_just(
                    vars, self_type
                ) and other_arg_type.matches_just(vars, other_type):
                    return True
        return False

    def check_binary_method_with_self_type(self, method_name: str, self_type: JustTypeRef) -> JustTypeRef | None:
        """
        Checks if the class has a binary method with the given name and self type. Returns the other type if it exists.
        """
        vars: dict[TypeVarRef, JustTypeRef] = {}
        class_decl = self._classes.get(self_type.ident)
        if class_decl is None:
            return None
        if callable_decl := class_decl.methods.get(method_name):
            match callable_decl.signature:
                case FunctionSignature((self_arg_type, other_arg_type)) if self_arg_type.matches_just(vars, self_type):
                    return other_arg_type.to_just(vars)
        return None

    def check_binary_method_with_other_type(self, method_name: str, other_type: JustTypeRef) -> Iterable[JustTypeRef]:
        """
        Returns the types which are compatible with the given binary method name and other type.
        """
        for class_decl in self._classes.values():
            vars: dict[TypeVarRef, JustTypeRef] = {}
            if callable_decl := class_decl.methods.get(method_name):
                match callable_decl.signature:
                    case FunctionSignature((self_arg_type, other_arg_type)) if other_arg_type.matches_just(
                        vars, other_type
                    ):
                        yield self_arg_type.to_just(vars)

    def get_class_decl(self, ident: Ident) -> ClassDecl:
        return self._classes[ident]

    def get_parameterized_class(self, ident: Ident) -> TypeRefWithVars:
        """
        Returns a class reference with type parameters, if the class is parameterized.
        """
        type_vars = self._classes[ident].type_vars
        return TypeRefWithVars(ident, type_vars)


@dataclass
class ClassDecl:
    egg_name: str | None = None
    type_vars: tuple[TypeVarRef, ...] = ()
    builtin: bool = False
    init: ConstructorDecl | FunctionDecl | None = None
    class_methods: dict[str, FunctionDecl | ConstructorDecl] = field(default_factory=dict)
    # These have to be separate from class_methods so that printing them can be done easily
    class_variables: dict[str, ConstantDecl] = field(default_factory=dict)
    methods: dict[str, FunctionDecl | ConstructorDecl] = field(default_factory=dict)
    properties: dict[str, FunctionDecl | ConstructorDecl] = field(default_factory=dict)
    preserved_methods: dict[str, Callable] = field(default_factory=dict)
    match_args: tuple[str, ...] = field(default=())
    doc: str | None = field(default=None)


@dataclass(frozen=True)
class RulesetDecl:
    rules: list[RewriteOrRuleDecl]

    # Make hashable so when traversing for pretty-fying we can know which rulesets we have already
    # made into strings
    def __hash__(self) -> int:
        return hash((type(self), tuple(self.rules)))


@dataclass(frozen=True)
class CombinedRulesetDecl:
    rulesets: tuple[Ident, ...]


T_expr_decl = TypeVar("T_expr_decl", bound="ExprDecl")


@dataclass(frozen=True)
class EGraphDecl:
    """
    State of an e-graph, which when re-added to a new e-graph will reconstruct the same e-graph, given the same Declarations.

    All the expressions in here may reference values which appear in the `e_classes` mapping.
    """

    # Mapping from top level let binding names to their types and expressions
    let_bindings: dict[str, TypedExprDecl] = field(default_factory=dict)
    # Mapping from egglog values representing e-classes to all the expressions in that e-class
    e_classes: dict[Value, tuple[JustTypeRef, tuple[CallDecl, ...]]] = field(default_factory=dict)
    # Mapping from function calls to the values they are set to
    sets: dict[CallDecl, TypedExprDecl] = field(default_factory=dict)
    # Top-level expr actions such as relation facts.
    expr_actions: tuple[TypedExprDecl, ...] = field(default=())
    # Mapping from function calls to the set costs.
    costs: dict[CallDecl, tuple[JustTypeRef, int]] = field(default_factory=dict)
    # Set of values which are subsumed
    subsumed: tuple[tuple[JustTypeRef, CallDecl], ...] = field(default=())

    def __hash__(self) -> int:
        return hash((
            type(self),
            tuple(self.let_bindings.items()),
            tuple((value, tp, exprs) for value, (tp, exprs) in self.e_classes.items()),
            tuple(self.sets.items()),
            self.expr_actions,
            tuple(self.costs.items()),
            self.subsumed,
        ))

    @cached_property
    def to_actions(self) -> list[ActionDecl]:  # noqa: C901
        """
        Converts this egraph decl to a list of actions that can be executed to reconstruct the egraph.

        Converts all e-classes to grounded terms + unions.

        Currently does not support cycles or empty e-classes.
        """
        # First fill up the e_class_grounded_term for all e_classes
        # by iteratively adding grounded terms for e-classes which have a grounded term until no more progress can be  made.

        # mapping from e-class to a grounded term in that e-class
        e_class_grounded_term: dict[Value, CallDecl] = {}

        def is_grounded(expr: ExprDecl) -> bool:
            """
            Checks if the given expression is grounded, meaning any values recursively in it have grounded terms in their e-classes.
            """
            match expr:
                case LetRefDecl(name):
                    raise ValueError(f"Cannot have unexpanded let bindings in egraph decl: {name}")
                case UnboundVarDecl(_):
                    msg = "Cannot have unbound variables in egraph decl"
                    raise ValueError(msg)
                case CallDecl(_, args, _):
                    return all(is_grounded(a.expr) for a in args)
                case LitDecl(_) | PyObjectDecl(_):
                    return True
                case PartialCallDecl(call):
                    return is_grounded(call)
                case DummyDecl():
                    msg = "Cannot have dummy decls in egraph decl"
                    raise ValueError(msg)
                case ValueDecl(value):
                    return value in e_class_grounded_term
                case GetCostDecl():
                    msg = "Cannot have GetCostDecl in egraph decl"
                    raise ValueError(msg)
                case _:
                    assert_never(expr)

        made_progress = True
        while made_progress:
            made_progress = False
            for e_class, (_, exprs) in self.e_classes.items():
                if e_class in e_class_grounded_term:
                    continue
                for expr in exprs:
                    if is_grounded(expr):
                        e_class_grounded_term[e_class] = expr
                        made_progress = True
                        break

        # call declarations already emitted as part of other actions.
        emitted_call_decls = set[CallDecl]()

        @cache
        def to_grounded(expr: ExprDecl) -> ExprDecl:
            """
            Converts the given expression to a grounded term, by replacing any values in it with their grounded terms.
            """
            match expr:
                case LetRefDecl(name):
                    raise ValueError(f"Cannot have unexpanded let bindings in egraph decl: {name}")
                case UnboundVarDecl(_):
                    msg = "Cannot have unbound variables in egraph decl"
                    raise ValueError(msg)
                case CallDecl(callable, args, bound_tp_params):
                    emitted_call_decls.add(expr)
                    new_args = tuple(TypedExprDecl(a.tp, to_grounded(a.expr)) for a in args)
                    return CallDecl(callable, new_args, bound_tp_params)
                case LitDecl(_) | PyObjectDecl(_):
                    return expr
                case PartialCallDecl(call):
                    return PartialCallDecl(cast("CallDecl", to_grounded(call)))
                case DummyDecl():
                    msg = "Cannot have dummy decls in egraph decl"
                    raise ValueError(msg)
                case ValueDecl(value):
                    if value not in e_class_grounded_term:
                        raise ValueError(f"Value {value} does not have a grounded term in egraph decl")
                    return to_grounded(e_class_grounded_term[value])
                case GetCostDecl():
                    msg = "Cannot have GetCostDecl in egraph decl"
                    raise ValueError(msg)
                case _:
                    assert_never(expr)

        # calls that are in e-classes with only one value, so wouldn't be added as a union and might need
        # to be added as a single expr action if they don't show up anywhere else
        single_e_class_calls: list[tuple[JustTypeRef, CallDecl]] = []

        # Now add all e-classes as actions.
        actions: list[ActionDecl] = []
        for e_class, (tp, exprs) in self.e_classes.items():
            chosen_term = e_class_grounded_term[e_class]
            if len(exprs) == 1:
                single_e_class_calls.append((tp, chosen_term))
                continue

            grounded_chosen_term = to_grounded(chosen_term)
            for expr in exprs:
                if expr == chosen_term:
                    continue
                actions.append(UnionDecl(tp, grounded_chosen_term, to_grounded(expr)))
        actions.extend(
            LetDecl(name, TypedExprDecl(typed_expr.tp, to_grounded(typed_expr.expr)))
            for name, typed_expr in self.let_bindings.items()
        )
        actions.extend(
            SetDecl(set_expr.tp, cast("CallDecl", to_grounded(call)), to_grounded(set_expr.expr))
            for call, set_expr in self.sets.items()
        )
        actions.extend(
            ExprActionDecl(TypedExprDecl(typed_expr.tp, to_grounded(typed_expr.expr)))
            for typed_expr in self.expr_actions
        )
        actions.extend(
            SetCostDecl(tp, cast("CallDecl", to_grounded(call)), LitDecl(cost))
            for call, (tp, cost) in self.costs.items()
        )
        actions.extend(ChangeDecl(tp, cast("CallDecl", to_grounded(call)), "subsume") for tp, call in self.subsumed)

        # Now add any remaining call    s that weren't part of any other actions
        actions.extend(
            ExprActionDecl(TypedExprDecl(tp, to_grounded(expr)))
            for (tp, expr) in single_e_class_calls
            if expr not in emitted_call_decls
        )

        return actions


# Have two different types of type refs, one that can include vars recursively and one that cannot.
# We only use the one with vars for classmethods and methods, and the other one for egg references as
# well as runtime values.
@dataclass(frozen=True)
class JustTypeRef:
    ident: Ident
    args: tuple[JustTypeRef, ...] = ()

    def to_var(self) -> TypeRefWithVars:
        return TypeRefWithVars(self.ident, tuple(a.to_var() for a in self.args))

    def __str__(self) -> str:
        if self.args:
            return f"{self.ident.name}[{', '.join(str(a) for a in self.args)}]"
        return str(self.ident.name)


##
# Type references with vars
##

# mapping of name and module of resolved typevars to runtime values
# so that when spitting them back out again can use same instance
# since equality is based on identity not value
_RESOLVED_TYPEVARS: dict[TypeVarRef, TypeVar] = {}


class TypeVarError(RuntimeError):
    """Error when trying to resolve a type variable that doesn't exist."""


@dataclass(frozen=True)
class TypeVarRef:
    """
    A generic type variable reference.
    """

    ident: Ident

    def to_just(self, vars: dict[TypeVarRef, JustTypeRef] | None = None) -> JustTypeRef:
        if vars is None or self not in vars:
            raise TypeVarError(f"Cannot convert type variable {self} to concrete type without variable bindings")
        return vars[self]

    def __str__(self) -> str:
        return str(self.to_type_var())

    @classmethod
    def from_type_var(cls, typevar: TypeVar) -> TypeVarRef:
        res = cls(Ident(typevar.__name__, typevar.__module__))
        _RESOLVED_TYPEVARS[res] = typevar
        return res

    def to_type_var(self) -> TypeVar:
        return _RESOLVED_TYPEVARS[self]

    def matches_just(self, vars: dict[TypeVarRef, JustTypeRef], other: JustTypeRef) -> bool:
        """
        Checks if this type variable matches the given JustTypeRef, including type variables.
        """
        if self in vars:
            return vars[self] == other
        vars[self] = other
        return True

    @property
    def vars(self) -> set[TypeVarRef]:
        """
        Returns all type variables in this type reference.
        """
        return {self}


@dataclass(frozen=True)
class TypeRefWithVars:
    ident: Ident
    args: tuple[TypeOrVarRef, ...] = ()

    def to_just(self, vars: dict[TypeVarRef, JustTypeRef] | None = None) -> JustTypeRef:
        return JustTypeRef(self.ident, tuple(a.to_just(vars) for a in self.args))

    def __str__(self) -> str:
        if self.args:
            return f"{self.ident.name}[{', '.join(str(a) for a in self.args)}]"
        return str(self.ident.name)

    def matches_just(self, vars: dict[TypeVarRef, JustTypeRef], other: JustTypeRef) -> bool:
        """
        Checks if this type reference matches the given JustTypeRef, including type variables.
        """
        return (
            self.ident == other.ident
            and len(self.args) == len(other.args)
            and all(a.matches_just(vars, b) for a, b in zip(self.args, other.args, strict=True))
        )

    @property
    def vars(self) -> set[TypeVarRef]:
        """
        Returns all type variables in this type reference.
        """
        vars = set[TypeVarRef]()
        for arg in self.args:
            vars.update(arg.vars)
        return vars


TypeOrVarRef: TypeAlias = TypeVarRef | TypeRefWithVars

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
            assert isinstance(a.expr, UnboundVarDecl)
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
    ident: Ident


@dataclass(frozen=True)
class ConstantRef:
    ident: Ident


@dataclass(frozen=True)
class MethodRef:
    ident: Ident
    method_name: str


@dataclass(frozen=True)
class ClassMethodRef:
    ident: Ident
    method_name: str


@dataclass(frozen=True)
class InitRef:
    ident: Ident


@dataclass(frozen=True)
class ClassVariableRef:
    ident: Ident
    var_name: str


@dataclass(frozen=True)
class PropertyRef:
    ident: Ident
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
            return_type=TypeRefWithVars(Ident.builtin("Unit")),
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

    @property
    def arg_vars(self) -> set[TypeVarRef]:
        """
        Returns all type variables in the argument types.
        """
        vars = set[TypeVarRef]()
        for arg in self.arg_types:
            vars.update(arg.vars)
        if self.var_arg_type:
            vars.update(self.var_arg_type.vars)
        return vars

    @property
    def all_args(self) -> Iterable[TypeOrVarRef]:
        """
        Returns all argument types, including var args.
        """
        return chain(self.arg_types, (repeat(self.var_arg_type) if self.var_arg_type else []))


@dataclass(frozen=True)
class FunctionDecl:
    signature: FunctionSignature | SpecialFunctions = field(default_factory=FunctionSignature)
    builtin: bool = False
    egg_name: str | None = None
    merge: ExprDecl | None = None
    doc: str | None = None


@dataclass(frozen=True)
class ConstructorDecl:
    signature: FunctionSignature = field(default_factory=FunctionSignature)
    egg_name: str | None = None
    cost: int | None = None
    unextractable: bool = False
    doc: str | None = None


CallableDecl: TypeAlias = RelationDecl | ConstantDecl | FunctionDecl | ConstructorDecl

##
# Expressions
##


@dataclass(frozen=True)
class UnboundVarDecl:
    name: str
    egg_name: str | None = None


@dataclass(frozen=True)
class DummyDecl:
    pass


@dataclass(frozen=True)
class LetRefDecl:
    name: str


@dataclass(frozen=True)
class PyObjectDecl:
    pickled: bytes


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
    bound_tp_params: tuple[JustTypeRef, ...] = ()

    # pool objects for faster __eq__
    _args_to_value: ClassVar[WeakValueDictionary[tuple[object, ...], CallDecl]] = WeakValueDictionary({})

    def __new__(cls, *args: object, **kwargs: object) -> Self:
        """
        Pool CallDecls so that they can be compared by identity more quickly.

        Necessary bc we search for common parents when serializing CallDecl trees to egglog to
        only serialize each sub-tree once.
        """
        # normalize the args/kwargs to a tuple so that they can be compared
        callable = args[0] if args else kwargs["callable"]
        args_ = args[1] if len(args) > 1 else kwargs.get("args", ())
        bound_tp_params = args[2] if len(args) > 2 else kwargs.get("bound_tp_params", ())

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

    Separated from the call decl so it's clear it is translated to a `unstable-fn` call.
    """

    call: CallDecl


@dataclass(frozen=True)
class GetCostDecl:
    callable: CallableRef
    args: tuple[TypedExprDecl, ...]


@dataclass(frozen=True)
class ValueDecl:
    value: Value


ExprDecl: TypeAlias = (
    DummyDecl
    | UnboundVarDecl
    | LetRefDecl
    | LitDecl
    | CallDecl
    | PyObjectDecl
    | PartialCallDecl
    | ValueDecl
    | GetCostDecl
)


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


def collect_unbound_vars(typed_expr: TypedExprDecl) -> set[TypedExprDecl]:
    """
    Returns the set of all unbound vars
    """
    seen = set[TypedExprDecl]()
    unbound_vars = set[TypedExprDecl]()

    def visit(typed_expr: TypedExprDecl) -> None:
        if typed_expr in seen:
            return
        seen.add(typed_expr)
        match typed_expr.expr:
            case CallDecl(_, args) | PartialCallDecl(CallDecl(_, args)):
                for arg in args:
                    visit(arg)
            case UnboundVarDecl(_):
                unbound_vars.add(typed_expr)

    visit(typed_expr)
    return unbound_vars


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
    ruleset: Ident
    until: tuple[FactDecl, ...] | None
    scheduler: BackOffDecl | None = None


@dataclass(frozen=True)
class LetSchedulerDecl:
    scheduler: BackOffDecl
    inner: ScheduleDecl


ScheduleDecl: TypeAlias = SaturateDecl | RepeatDecl | SequenceDecl | RunDecl | LetSchedulerDecl


@dataclass(frozen=True)
class BackOffDecl:
    id: UUID
    match_limit: int | None
    ban_length: int | None


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


@dataclass(frozen=True)
class SetCostDecl:
    tp: JustTypeRef
    expr: CallDecl
    cost: ExprDecl


ActionDecl: TypeAlias = LetDecl | SetDecl | ExprActionDecl | ChangeDecl | UnionDecl | PanicDecl | SetCostDecl


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
