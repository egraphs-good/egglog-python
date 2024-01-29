"""
Data only descriptions of the components of an egraph and the expressions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from inspect import Parameter, Signature
from typing import TYPE_CHECKING, Protocol, TypeAlias, Union, runtime_checkable

from typing_extensions import Self, assert_never

from . import bindings

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


__all__ = [
    "Declarations",
    "DeclerationsLike",
    "upcast_decleratioons",
    "JustTypeRef",
    "ClassTypeVarRef",
    "TypeRefWithVars",
    "TypeOrVarRef",
    "FunctionRef",
    "MethodRef",
    "ClassMethodRef",
    "ClassVariableRef",
    "FunctionCallableRef",
    "PropertyRef",
    "CallableRef",
    "ConstantRef",
    "FunctionDecl",
    "VarDecl",
    "LitType",
    "PyObjectDecl",
    "LitDecl",
    "CallDecl",
    "ExprDecl",
    "TypedExprDecl",
    "ClassDecl",
    "PrettyContext",
    "GLOBAL_PY_OBJECT_SORT",
]

# Create a global sort for python objects, so we can store them without an e-graph instance
# Needed when serializing commands to egg commands when creating modules
GLOBAL_PY_OBJECT_SORT = bindings.PyObjectSort()

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
    # TODO: Support divmod, with tuple return value
    # "__divmod__": "divmod",
    # TODO: Three arg power
    "__pow__": "**",
    "__lshift__": "<<",
    "__rshift__": ">>",
    "__and__": "&",
    "__xor__": "^",
    "__or__": "|",
}
REFLECTED_BINARY_METHODS = {
    "__radd__": "__add__",
    "__rsub__": "__sub__",
    "__rmul__": "__mul__",
    "__rmatmul__": "__matmul__",
    "__rtruediv__": "__truediv__",
    "__rfloordiv__": "__floordiv__",
    "__rmod__": "__mod__",
    "__rpow__": "__pow__",
    "__rlshift__": "__lshift__",
    "__rrshift__": "__rshift__",
    "__rand__": "__and__",
    "__rxor__": "__xor__",
    "__ror__": "__or__",
}
UNARY_METHODS = {
    "__pos__": "+",
    "__neg__": "-",
    "__invert__": "~",
}


@runtime_checkable
class HasDeclerations(Protocol):
    @property
    def __egg_decls__(self) -> Declarations:
        ...


DeclerationsLike: TypeAlias = Union[HasDeclerations, None, "Declarations"]


def upcast_decleratioons(declerations_like: Iterable[DeclerationsLike]) -> list[Declarations]:
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
    _functions: dict[str, FunctionDecl] = field(default_factory=dict)
    _classes: dict[str, ClassDecl] = field(default_factory=dict)
    _constants: dict[str, JustTypeRef] = field(default_factory=dict)

    # Bidirectional mapping between egg function names and python callable references.
    # Note that there are possibly mutliple callable references for a single egg function name, like `+`
    # for both int and rational classes.
    _egg_fn_to_callable_refs: defaultdict[str, set[CallableRef]] = field(default_factory=lambda: defaultdict(set))
    _callable_ref_to_egg_fn: dict[CallableRef, str] = field(default_factory=dict)

    # Bidirectional mapping between egg sort names and python type references.
    _egg_sort_to_type_ref: dict[str, JustTypeRef] = field(default_factory=dict)
    _type_ref_to_egg_sort: dict[JustTypeRef, str] = field(default_factory=dict)

    # Mapping from egg name (of sort or function) to command to create it.
    _cmds: dict[str, bindings._Command] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "!=" not in self._egg_fn_to_callable_refs:
            self.register_callable_ref(FunctionRef("!="), "!=")

    @classmethod
    def create(cls, *others: DeclerationsLike) -> Declarations:
        others = upcast_decleratioons(others)
        if not others:
            return Declarations()
        first, *rest = others
        new = first.copy()
        new.update(*rest)
        return new

    def copy(self) -> Declarations:
        return Declarations(
            _functions=self._functions.copy(),
            _classes=self._classes.copy(),
            _constants=self._constants.copy(),
            _egg_fn_to_callable_refs=defaultdict(set, {k: v.copy() for k, v in self._egg_fn_to_callable_refs.items()}),
            _callable_ref_to_egg_fn=self._callable_ref_to_egg_fn.copy(),
            _egg_sort_to_type_ref=self._egg_sort_to_type_ref.copy(),
            _type_ref_to_egg_sort=self._type_ref_to_egg_sort.copy(),
            _cmds=self._cmds.copy(),
        )

    def add_cmd(self, name: str, cmd: bindings._Command) -> None:
        self._cmds[name] = cmd

    def list_cmds(self) -> list[bindings._Command]:
        return list(self._cmds.values())

    def update(self, *others: DeclerationsLike) -> None:
        for other in others:
            self |= other

    def __or__(self, other: DeclerationsLike) -> Declarations:
        result = Declarations()
        result |= self
        result |= other
        return result

    def __ior__(self, other: DeclerationsLike) -> Self:
        if other is None:
            return self
        if isinstance(other, HasDeclerations):
            other = other.__egg_decls__
        # If cmds are == skip unioning for time savings
        # if set(self._cmds) == set(other._cmds) and self.record_cmds and other.record_cmds:
        #     return self

        self._functions |= other._functions
        self._classes |= other._classes
        self._constants |= other._constants
        self._egg_sort_to_type_ref |= other._egg_sort_to_type_ref
        self._type_ref_to_egg_sort |= other._type_ref_to_egg_sort
        self._cmds |= other._cmds
        self._callable_ref_to_egg_fn |= other._callable_ref_to_egg_fn
        for egg_fn, callable_refs in other._egg_fn_to_callable_refs.items():
            self._egg_fn_to_callable_refs[egg_fn] |= callable_refs
        return self

    def set_function_decl(self, ref: FunctionCallableRef, decl: FunctionDecl) -> None:
        """
        Sets a function declaration for the given callable reference.
        """
        match ref:
            case FunctionRef(name):
                if name in self._functions:
                    raise ValueError(f"Function {name} already registered")
                self._functions[name] = decl
            case MethodRef(class_name, method_name):
                if method_name in self._classes[class_name].methods:
                    raise ValueError(f"Method {class_name}.{method_name} already registered")
                self._classes[class_name].methods[method_name] = decl
            case ClassMethodRef(class_name, method_name):
                if method_name in self._classes[class_name].class_methods:
                    raise ValueError(f"Class method {class_name}.{method_name} already registered")
                self._classes[class_name].class_methods[method_name] = decl
            case PropertyRef(class_name, property_name):
                if property_name in self._classes[class_name].properties:
                    raise ValueError(f"Property {class_name}.{property_name} already registered")
                self._classes[class_name].properties[property_name] = decl
            case _:
                assert_never(ref)

    def set_constant_type(self, ref: ConstantCallableRef, tp: JustTypeRef) -> None:
        match ref:
            case ConstantRef(name):
                if name in self._constants:
                    raise ValueError(f"Constant {name} already registered")
                self._constants[name] = tp
            case ClassVariableRef(class_name, variable_name):
                if variable_name in self._classes[class_name].class_variables:
                    raise ValueError(f"Class variable {class_name}.{variable_name} already registered")
                self._classes[class_name].class_variables[variable_name] = tp
            case _:
                assert_never(ref)

    def register_callable_ref(self, ref: CallableRef, egg_name: str) -> None:
        """
        Registers a callable reference with the given egg name.

        The callable's function needs to be registered first.
        """
        if ref in self._callable_ref_to_egg_fn:
            raise ValueError(f"Callable ref {ref} already registered")
        self._callable_ref_to_egg_fn[ref] = egg_name
        self._egg_fn_to_callable_refs[egg_name].add(ref)

    def get_callable_refs(self, egg_name: str) -> Iterable[CallableRef]:
        return self._egg_fn_to_callable_refs[egg_name]

    def get_egg_fn(self, ref: CallableRef) -> str:
        return self._callable_ref_to_egg_fn[ref]

    def get_egg_sort(self, ref: JustTypeRef) -> str:
        return self._type_ref_to_egg_sort[ref]

    def op_mapping(self) -> dict[str, str]:
        """
        Create a mapping of egglog function name to Python function name, for use in the serialized format
        for better visualization.
        """
        return {k: str(next(iter(v))) for k, v in self._egg_fn_to_callable_refs.items() if len(v) == 1}

    def has_method(self, class_name: str, method_name: str) -> bool | None:
        """
        Returns whether the given class has the given method, or None if we cant find the class.
        """
        if class_name in self._classes:
            return method_name in self._classes[class_name].methods
        return None

    def get_function_decl(self, ref: CallableRef) -> FunctionDecl:
        match ref:
            case ConstantRef(name):
                return self._constants[name].to_constant_function_decl()
            case ClassVariableRef(class_name, variable_name):
                return self._classes[class_name].class_variables[variable_name].to_constant_function_decl()
            case FunctionRef(name):
                return self._functions[name]
            case MethodRef(class_name, method_name):
                return self._classes[class_name].methods[method_name]
            case ClassMethodRef(class_name, method_name):
                return self._classes[class_name].class_methods[method_name]
            case PropertyRef(class_name, property_name):
                return self._classes[class_name].properties[property_name]
        assert_never(ref)

    def get_class_decl(self, name: str) -> ClassDecl:
        return self._classes[name]

    def get_registered_class_args(self, cls_name: str) -> tuple[JustTypeRef, ...]:
        """
        Given a class name, returns the first typevar regsisted with args of that class.
        """
        for tp in self._type_ref_to_egg_sort:
            if tp.name == cls_name and tp.args:
                return tp.args
        return ()

    def register_class(self, name: str, n_type_vars: int, builtin: bool, egg_sort: str | None) -> None:
        # Register class first
        if name in self._classes:
            raise ValueError(f"Class {name} already registered")
        decl = ClassDecl(n_type_vars=n_type_vars)
        self._classes[name] = decl
        self.register_sort(JustTypeRef(name), builtin, egg_sort)

    def register_sort(self, ref: JustTypeRef, builtin: bool, egg_name: str | None = None) -> str:
        """
        Register a sort with the given name. If no name is given, one is generated.

        If this is a type called with generic args, register the generic args as well.
        """
        # If the sort is already registered, do nothing
        try:
            egg_sort = self.get_egg_sort(ref)
        except KeyError:
            pass
        else:
            return egg_sort
        egg_name = egg_name or ref.generate_egg_name()
        if egg_name in self._egg_sort_to_type_ref:
            raise ValueError(f"Sort {egg_name} is already registered.")
        self._egg_sort_to_type_ref[egg_name] = ref
        self._type_ref_to_egg_sort[ref] = egg_name

        if not builtin:
            self.add_cmd(
                egg_name,
                bindings.Sort(
                    egg_name,
                    (egg_sort, [bindings.Var(self.register_sort(arg, False)) for arg in ref.args])
                    if ref.args
                    else None,
                ),
            )

        return egg_name

    def register_function_callable(
        self,
        ref: FunctionCallableRef,
        fn_decl: FunctionDecl,
        egg_name: str | None,
        cost: int | None,
        default: ExprDecl | None,
        merge: ExprDecl | None,
        merge_action: list[bindings._Action],
        unextractable: bool,
        builtin: bool,
        is_relation: bool = False,
    ) -> None:
        """
        Registers a callable with the given egg name.

        The callable's function needs to be registered first.
        """
        egg_name = egg_name or ref.generate_egg_name()
        self.register_callable_ref(ref, egg_name)
        self.set_function_decl(ref, fn_decl)

        # Skip generating the cmds if we don't want to record them, like for the builtins
        if builtin:
            return

        if fn_decl.var_arg_type is not None:
            msg = "egglog does not support variable arguments yet."
            raise NotImplementedError(msg)
        # Remove all vars from the type refs, raising an errory if we find one,
        # since we cannot create egg functions with vars
        arg_sorts = [self.register_sort(a.to_just(), False) for a in fn_decl.arg_types]
        cmd: bindings._Command
        if is_relation:
            assert not default
            assert not merge
            assert not merge_action
            assert not cost
            cmd = bindings.Relation(egg_name, arg_sorts)
        else:
            egg_fn_decl = bindings.FunctionDecl(
                egg_name,
                bindings.Schema(arg_sorts, self.get_egg_sort(fn_decl.return_type.to_just())),
                default.to_egg(self) if default else None,
                merge.to_egg(self) if merge else None,
                merge_action,
                cost,
                unextractable,
            )
            cmd = bindings.Function(egg_fn_decl)
        self.add_cmd(egg_name, cmd)

    def register_constant_callable(self, ref: ConstantCallableRef, type_ref: JustTypeRef, egg_name: str | None) -> None:
        egg_name = egg_name or ref.generate_egg_name()
        self.register_callable_ref(ref, egg_name)
        self.set_constant_type(ref, type_ref)
        self.add_cmd(egg_name, bindings.Declare(egg_name, self.get_egg_sort(type_ref)))

    def register_preserved_method(self, class_: str, method: str, fn: Callable) -> None:
        self._classes[class_].preserved_methods[method] = fn


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
        return f"{self.name}_{args}"

    def to_var(self) -> TypeRefWithVars:
        return TypeRefWithVars(self.name, tuple(a.to_var() for a in self.args))

    def pretty(self) -> str:
        if not self.args:
            return self.name
        args = ", ".join(a.pretty() for a in self.args)
        return f"{self.name}[{args}]"

    def to_constant_function_decl(self) -> FunctionDecl:
        """
        Create a function declaration for a constant function.

        This is similar to how egglog compiles the `constant` command.
        """
        return FunctionDecl(
            arg_types=(),
            arg_names=(),
            arg_defaults=(),
            return_type=self.to_var(),
            mutates_first_arg=False,
            var_arg_type=None,
        )


@dataclass(frozen=True)
class ClassTypeVarRef:
    """
    A class type variable represents one of the types of the class, if it is a generic class.
    """

    index: int

    def to_just(self) -> JustTypeRef:
        msg = "egglog does not support generic classes yet."
        raise NotImplementedError(msg)


@dataclass(frozen=True)
class TypeRefWithVars:
    name: str
    args: tuple[TypeOrVarRef, ...] = ()

    def to_just(self) -> JustTypeRef:
        return JustTypeRef(self.name, tuple(a.to_just() for a in self.args))


TypeOrVarRef: TypeAlias = ClassTypeVarRef | TypeRefWithVars


@dataclass(frozen=True)
class FunctionRef:
    name: str

    def generate_egg_name(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


# Use this special character in place of the args, so that if the args are inlined
# in the viz, they will replace it
ARG = "·"


@dataclass(frozen=True)
class MethodRef:
    class_name: str
    method_name: str

    def generate_egg_name(self) -> str:
        return f"{self.class_name}_{self.method_name}"

    def __str__(self) -> str:  # noqa: PLR0911
        match self.method_name:
            case _ if self.method_name in UNARY_METHODS:
                return f"{UNARY_METHODS[self.method_name]}{ARG}"
            case _ if self.method_name in BINARY_METHODS:
                return f"({ARG} {BINARY_METHODS[self.method_name]} {ARG})"
            case "__getitem__":
                return f"{ARG}[{ARG}]"
            case "__call__":
                return f"{ARG}({ARG})"
            case "__delitem__":
                return f"del {ARG}[{ARG}]"
            case "__setitem__":
                return f"{ARG}[{ARG}] = {ARG}"
        return f"{ARG}.{self.method_name}"


@dataclass(frozen=True)
class ClassMethodRef:
    class_name: str
    method_name: str

    def generate_egg_name(self) -> str:
        return f"{self.class_name}_{self.method_name}"

    def __str__(self) -> str:
        if self.method_name == "__init__":
            return self.class_name
        return f"{self.class_name}.{self.method_name}"


@dataclass(frozen=True)
class ConstantRef:
    name: str

    def generate_egg_name(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class ClassVariableRef:
    class_name: str
    variable_name: str

    def generate_egg_name(self) -> str:
        return f"{self.class_name}_{self.variable_name}"

    def __str__(self) -> str:
        return f"{self.class_name}.{self.variable_name}"


@dataclass(frozen=True)
class PropertyRef:
    class_name: str
    property_name: str

    def generate_egg_name(self) -> str:
        return f"{self.class_name}_{self.property_name}"

    def __str__(self) -> str:
        return f"{ARG}.{self.property_name}"


ConstantCallableRef: TypeAlias = ConstantRef | ClassVariableRef
FunctionCallableRef: TypeAlias = FunctionRef | MethodRef | ClassMethodRef | PropertyRef
CallableRef: TypeAlias = ConstantCallableRef | FunctionCallableRef


@dataclass(frozen=True)
class FunctionDecl:
    arg_types: tuple[TypeOrVarRef, ...]
    # Is None for relation which doesn't have named args
    arg_names: tuple[str, ...] | None
    arg_defaults: tuple[ExprDecl | None, ...]
    return_type: TypeOrVarRef
    mutates_first_arg: bool
    var_arg_type: TypeOrVarRef | None = None

    def __post_init__(self) -> None:
        # If we mutate the first arg, then the first arg should be the same type as the return
        if self.mutates_first_arg:
            assert self.arg_types[0] == self.return_type

    def to_signature(self, transform_default: Callable[[TypedExprDecl], object]) -> Signature:
        arg_names = self.arg_names or tuple(f"__{i}" for i in range(len(self.arg_types)))
        parameters = [
            Parameter(
                n,
                Parameter.POSITIONAL_OR_KEYWORD,
                default=transform_default(TypedExprDecl(t.to_just(), d)) if d else Parameter.empty,
            )
            for n, d, t in zip(arg_names, self.arg_defaults, self.arg_types, strict=True)
        ]
        if self.var_arg_type is not None:
            parameters.append(Parameter("__rest", Parameter.VAR_POSITIONAL))
        return Signature(parameters)


@dataclass(frozen=True)
class VarDecl:
    name: str

    @classmethod
    def from_egg(cls, var: bindings.Var) -> TypedExprDecl:
        msg = "Cannot turn var into egg type because typing unknown."
        raise NotImplementedError(msg)

    def to_egg(self, _decls: Declarations) -> bindings.Var:
        return bindings.Var(self.name)

    def pretty(self, context: PrettyContext, **kwargs) -> str:
        return self.name


@dataclass(frozen=True)
class PyObjectDecl:
    value: object

    def __hash__(self) -> int:
        """Tries using the hash of the value, if unhashable use the ID."""
        try:
            return hash((type(self.value), self.value))
        except TypeError:
            return id(self.value)

    @classmethod
    def from_egg(cls, egraph: bindings.EGraph, call: bindings.Call) -> TypedExprDecl:
        return TypedExprDecl(JustTypeRef("PyObject"), cls(egraph.eval_py_object(call)))

    def to_egg(self, _decls: Declarations) -> bindings._Expr:
        return GLOBAL_PY_OBJECT_SORT.store(self.value)

    def pretty(self, context: PrettyContext, **kwargs) -> str:
        return repr(self.value)


LitType: TypeAlias = int | str | float | bool | None


@dataclass(frozen=True)
class LitDecl:
    value: LitType

    @classmethod
    def from_egg(cls, lit: bindings.Lit) -> TypedExprDecl:
        # TODO: Try rewriting with pattern matching once ctypes support __match_args__
        # https://peps.python.org/pep-0622/#the-match-protocol
        if isinstance(lit.value, bindings.Int):
            return TypedExprDecl(JustTypeRef("i64"), cls(lit.value.value))
        if isinstance(lit.value, bindings.String):
            return TypedExprDecl(JustTypeRef("String"), cls(lit.value.value))
        if isinstance(lit.value, bindings.F64):
            return TypedExprDecl(JustTypeRef("f64"), cls(lit.value.value))
        if isinstance(lit.value, bindings.Bool):
            return TypedExprDecl(JustTypeRef("Bool"), cls(lit.value.value))
        if isinstance(lit.value, bindings.Unit):
            return TypedExprDecl(JustTypeRef("Unit"), cls(None))
        assert_never(lit.value)

    def to_egg(self, _decls: Declarations) -> bindings.Lit:
        if self.value is None:
            return bindings.Lit(bindings.Unit())
        if isinstance(self.value, bool):
            return bindings.Lit(bindings.Bool(self.value))
        if isinstance(self.value, int):
            return bindings.Lit(bindings.Int(self.value))
        if isinstance(self.value, float):
            return bindings.Lit(bindings.F64(self.value))
        if isinstance(self.value, str):
            return bindings.Lit(bindings.String(self.value))
        assert_never(self.value)

    def pretty(self, context: PrettyContext, unwrap_lit: bool = True, **kwargs) -> str:
        """
        Returns a string representation of the literal.

        :param wrap_lit: If True, wraps the literal in a call to the literal constructor.
        """
        if self.value is None:
            return "Unit()"
        if isinstance(self.value, bool):
            return f"Bool({self.value})" if not unwrap_lit else str(self.value)
        if isinstance(self.value, int):
            return f"i64({self.value})" if not unwrap_lit else str(self.value)
        if isinstance(self.value, float):
            return f"f64({self.value})" if not unwrap_lit else str(self.value)
        if isinstance(self.value, str):
            return f"String({self.value!r})" if not unwrap_lit else repr(self.value)
        assert_never(self.value)


@dataclass(frozen=True)
class CallDecl:
    callable: CallableRef
    args: tuple[TypedExprDecl, ...] = ()
    # type parameters that were bound to the callable, if it is a classmethod
    # Used for pretty printing classmethod calls with type parameters
    bound_tp_params: tuple[JustTypeRef, ...] | None = None
    _cached_hash: int | None = None

    def __post_init__(self) -> None:
        if self.bound_tp_params and not isinstance(self.callable, ClassMethodRef):
            msg = "Cannot bind type parameters to a non-class method callable."
            raise ValueError(msg)

    def __hash__(self) -> int:
        # Modified hash which will cache result for performance
        if self._cached_hash is None:
            res = hash((self.callable, self.args, self.bound_tp_params))
            object.__setattr__(self, "_cached_hash", res)
            return res
        return self._cached_hash

    def __eq__(self, other: object) -> bool:
        # Override eq to use cached hash for perf
        if not isinstance(other, CallDecl):
            return False
        return hash(self) == hash(other)

    @classmethod
    def from_egg(cls, egraph: bindings.EGraph, decls: Declarations, call: bindings.Call) -> TypedExprDecl:
        """
        Convert an egg expression into a typed expression by using the declerations.

        For use in extract
        """
        from .type_constraint_solver import TypeConstraintSolver

        results = tuple(TypedExprDecl.from_egg(egraph, decls, a) for a in call.args)
        arg_types = tuple(r.tp for r in results)

        # Find the first callable ref that matches the call
        for callable_ref in decls.get_callable_refs(call.name):
            # If this is a classmethod, we might need the type params that were bound for this type
            # egglog currently only allows one instantiated type of any generic sort to be used in any program
            # So we just lookup what args were registered for this sort
            if isinstance(callable_ref, ClassMethodRef):
                cls_args = decls.get_registered_class_args(callable_ref.class_name)
                tcs = TypeConstraintSolver.from_type_parameters(cls_args)
            else:
                tcs = TypeConstraintSolver()
            fn_decl = decls.get_function_decl(callable_ref)
            return_tp = tcs.infer_return_type(fn_decl.arg_types, fn_decl.return_type, fn_decl.var_arg_type, arg_types)
            return TypedExprDecl(return_tp, cls(callable_ref, tuple(results)))
        raise ValueError(f"Could not find callable ref for call {call}")

    def to_egg(self, decls: Declarations) -> bindings._Expr:
        """Convert a Call to an egg Call."""
        # If this is a constant, then emit it just as a var, not as a call
        egg_fn = decls.get_egg_fn(self.callable)
        if isinstance(self.callable, ConstantRef | ClassVariableRef):
            decls.get_egg_fn
            return bindings.Var(egg_fn)
        return bindings.Call(egg_fn, [a.to_egg(decls) for a in self.args])

    def pretty(self, context: PrettyContext, parens: bool = True, **kwargs) -> str:  # noqa: C901
        """
        Pretty print the call.

        :param parens: If true, wrap the call in parens if it is a binary method call.
        """
        if self in context.names:
            return context.names[self]
        ref, args = self.callable, [a.expr for a in self.args]
        # Special case !=
        if ref == FunctionRef("!="):
            return f"ne({args[0].pretty(context, parens=False)}).to({args[1].pretty(context, parens=False)})"
        function_decl = context.decls.get_function_decl(ref)
        # Determine how many of the last arguments are defaults, by iterating from the end and comparing the arg with the default
        n_defaults = 0
        for arg, default in zip(
            reversed(args), reversed(function_decl.arg_defaults), strict=not function_decl.var_arg_type
        ):
            if arg != default:
                break
            n_defaults += 1
        if n_defaults:
            args = args[:-n_defaults]
        if function_decl.mutates_first_arg:
            first_arg = args[0]
            expr_str = first_arg.pretty(context, parens=False)
            # copy an identifer expression iff it has multiple parents (b/c then we can't mutate it directly)
            has_multiple_parents = context.parents[first_arg] > 1
            expr_name = context.name_expr(function_decl.arg_types[0], expr_str, copy_identifier=has_multiple_parents)
            # Set the first arg to be the name of the mutated arg and return the name
            args[0] = VarDecl(expr_name)
        else:
            expr_name = None
        match ref:
            case FunctionRef(name):
                expr = _pretty_call(context, name, args)
            case ClassMethodRef(class_name, method_name):
                tp_ref = JustTypeRef(class_name, self.bound_tp_params or ())
                fn_str = tp_ref.pretty() if method_name == "__init__" else f"{tp_ref.pretty()}.{method_name}"
                expr = _pretty_call(context, fn_str, args)
            case MethodRef(_class_name, method_name):
                slf, *args = args
                slf = slf.pretty(context, unwrap_lit=False)
                match method_name:
                    case _ if method_name in UNARY_METHODS:
                        expr = f"{UNARY_METHODS[method_name]}{slf}"
                    case _ if method_name in BINARY_METHODS:
                        assert len(args) == 1
                        expr = f"{slf} {BINARY_METHODS[method_name]} {args[0].pretty(context)}"
                        if parens:
                            expr = f"({expr})"
                    case "__getitem__":
                        assert len(args) == 1
                        expr = f"{slf}[{args[0].pretty(context, parens=False)}]"
                    case "__call__":
                        expr = _pretty_call(context, slf, args)
                    case "__delitem__":
                        assert len(args) == 1
                        expr = f"del {slf}[{args[0].pretty(context, parens=False)}]"
                    case "__setitem__":
                        assert len(args) == 2
                        expr = (
                            f"{slf}[{args[0].pretty(context, parens=False)}] = {args[1].pretty(context, parens=False)}"
                        )
                    case _:
                        expr = _pretty_call(context, f"{slf}.{method_name}", args)
            case ConstantRef(name):
                expr = name
            case ClassVariableRef(class_name, variable_name):
                expr = f"{class_name}.{variable_name}"
            case PropertyRef(_class_name, property_name):
                expr = f"{args[0].pretty(context)}.{property_name}"
            case _:
                assert_never(ref)
        # If we have a name, then we mutated
        if expr_name:
            context.statements.append(expr)
            context.names[self] = expr_name
            return expr_name

        # We use a heuristic to decide whether to name this sub-expression as a variable
        # The rough goal is to reduce the number of newlines, given our line length of ~180
        # We determine it's worth making a new line for this expression if the total characters
        # it would take up is > than some constant (~ line length).
        n_parents = context.parents[self]
        line_diff: int = len(expr) - LINE_DIFFERENCE
        if n_parents > 1 and n_parents * line_diff > MAX_LINE_LENGTH:
            expr_name = context.name_expr(function_decl.return_type, expr, copy_identifier=False)
            context.names[self] = expr_name
            return expr_name
        return expr


MAX_LINE_LENGTH = 110
LINE_DIFFERENCE = 10


def _plot_line_length(expr: object):
    """
    Plots the number of line lengths based on different max lengths
    """
    global MAX_LINE_LENGTH, LINE_DIFFERENCE
    import altair as alt
    import pandas as pd

    sizes = []
    for line_length in range(40, 180, 10):
        MAX_LINE_LENGTH = line_length
        for diff in range(0, 40, 5):
            LINE_DIFFERENCE = diff
            new_l = len(str(expr).split())
            sizes.append((line_length, diff, new_l))

    df = pd.DataFrame(sizes, columns=["MAX_LINE_LENGTH", "LENGTH_DIFFERENCE", "n"])  # noqa: PD901

    return alt.Chart(df).mark_rect().encode(x="MAX_LINE_LENGTH:O", y="LENGTH_DIFFERENCE:O", color="n:Q")


def _pretty_call(context: PrettyContext, fn: str, args: Iterable[ExprDecl]) -> str:
    return f"{fn}({', '.join(a.pretty(context, parens=False) for a in args)})"


@dataclass
class PrettyContext:
    decls: Declarations
    # List of statements of "context" setting variable for the expr
    statements: list[str] = field(default_factory=list)

    names: dict[ExprDecl, str] = field(default_factory=dict)
    parents: dict[ExprDecl, int] = field(default_factory=lambda: defaultdict(lambda: 0))
    _traversed_exprs: set[ExprDecl] = field(default_factory=set)

    # Mapping of type to the number of times we have generated a name for that type, used to generate unique names
    _gen_name_types: dict[str, int] = field(default_factory=lambda: defaultdict(lambda: 0))

    def generate_name(self, typ: str) -> str:
        self._gen_name_types[typ] += 1
        return f"_{typ}_{self._gen_name_types[typ]}"

    def name_expr(self, expr_type: TypeOrVarRef, expr_str: str, copy_identifier: bool) -> str:
        tp_name = expr_type.to_just().name
        # If the thing we are naming is already a variable, we don't need to name it
        if expr_str.isidentifier():
            if copy_identifier:
                name = self.generate_name(tp_name)
                self.statements.append(f"{name} = copy({expr_str})")
            else:
                name = expr_str
        else:
            name = self.generate_name(tp_name)
            self.statements.append(f"{name} = {expr_str}")
        return name

    def render(self, expr: str) -> str:
        return "\n".join([*self.statements, expr])

    def traverse_for_parents(self, expr: ExprDecl) -> None:
        if expr in self._traversed_exprs:
            return
        self._traversed_exprs.add(expr)
        if isinstance(expr, CallDecl):
            for arg in set(expr.args):
                self.parents[arg.expr] += 1
                self.traverse_for_parents(arg.expr)


# def test_expr_pretty():
#     context = PrettyContext(ModuleDeclarations(Declarations()))
#     assert VarDecl("x").pretty(context) == "x"
#     assert LitDecl(42).pretty(context) == "i64(42)"
#     assert LitDecl("foo").pretty(context) == 'String("foo")'
#     assert LitDecl(None).pretty(context) == "unit()"

#     def v(x: str) -> TypedExprDecl:
#         return TypedExprDecl(JustTypeRef(""), VarDecl(x))

#     assert CallDecl(FunctionRef("foo"), (v("x"),)).pretty(context) == "foo(x)"
#     assert CallDecl(FunctionRef("foo"), (v("x"), v("y"), v("z"))).pretty(context) == "foo(x, y, z)"
#     assert CallDecl(MethodRef("foo", "__add__"), (v("x"), v("y"))).pretty(context) == "x + y"
#     assert CallDecl(MethodRef("foo", "__getitem__"), (v("x"), v("y"))).pretty(context) == "x[y]"
#     assert CallDecl(ClassMethodRef("foo", "__init__"), (v("x"), v("y"))).pretty(context) == "foo(x, y)"
#     assert CallDecl(ClassMethodRef("foo", "bar"), (v("x"), v("y"))).pretty(context) == "foo.bar(x, y)"
#     assert CallDecl(MethodRef("foo", "__call__"), (v("x"), v("y"))).pretty(context) == "x(y)"
#     assert (
#         CallDecl(
#             ClassMethodRef("Map", "__init__"),
#             (),
#             (JustTypeRef("i64"), JustTypeRef("Unit")),
#         ).pretty(context)
#         == "Map[i64, Unit]()"
#     )


# def test_setitem_pretty():
#     context = PrettyContext(ModuleDeclarations(Declarations()))

#     def v(x: str) -> TypedExprDecl:
#         return TypedExprDecl(JustTypeRef("typ"), VarDecl(x))

#     final_expr = CallDecl(MethodRef("foo", "__setitem__"), (v("x"), v("y"), v("z"))).pretty(context)
#     assert context.render(final_expr) == "_typ_1 = x\n_typ_1[y] = z\n_typ_1"


# def test_delitem_pretty():
#     context = PrettyContext(ModuleDeclarations(Declarations()))

#     def v(x: str) -> TypedExprDecl:
#         return TypedExprDecl(JustTypeRef("typ"), VarDecl(x))

#     final_expr = CallDecl(MethodRef("foo", "__delitem__"), (v("x"), v("y"))).pretty(context)
#     assert context.render(final_expr) == "_typ_1 = x\ndel _typ_1[y]\n_typ_1"


# TODO: Multiple mutations,

ExprDecl: TypeAlias = VarDecl | LitDecl | CallDecl | PyObjectDecl


@dataclass(frozen=True)
class TypedExprDecl:
    tp: JustTypeRef
    expr: ExprDecl

    @classmethod
    def from_egg(cls, egraph: bindings.EGraph, decls: Declarations, expr: bindings._Expr) -> TypedExprDecl:
        if isinstance(expr, bindings.Var):
            return VarDecl.from_egg(expr)
        if isinstance(expr, bindings.Lit):
            return LitDecl.from_egg(expr)
        if isinstance(expr, bindings.Call):
            if expr.name == "py-object":
                return PyObjectDecl.from_egg(egraph, expr)
            return CallDecl.from_egg(egraph, decls, expr)
        assert_never(expr)

    def to_egg(self, decls: Declarations) -> bindings._Expr:
        return self.expr.to_egg(decls)


@dataclass
class ClassDecl:
    methods: dict[str, FunctionDecl] = field(default_factory=dict)
    class_methods: dict[str, FunctionDecl] = field(default_factory=dict)
    class_variables: dict[str, JustTypeRef] = field(default_factory=dict)
    properties: dict[str, FunctionDecl] = field(default_factory=dict)
    preserved_methods: dict[str, Callable] = field(default_factory=dict)
    n_type_vars: int = 0
