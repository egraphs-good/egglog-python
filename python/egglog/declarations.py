"""
Data only descriptions of the components of an egraph and the expressions.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from inspect import Parameter, Signature
from typing import Callable, Iterable, Optional, Union

from typing_extensions import assert_never

from . import bindings

__all__ = [
    "Declarations",
    "ModuleDeclarations",
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
    "LitDecl",
    "CallDecl",
    "ExprDecl",
    "TypedExprDecl",
    "ClassDecl",
    "PrettyContext",
    "register_down_converter",
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

    def set_function_decl(self, ref: FunctionCallableRef, decl: FunctionDecl) -> None:
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
        elif isinstance(ref, PropertyRef):
            if ref.property_name in self._classes[ref.class_name].properties:
                raise ValueError(f"Property {ref.class_name}.{ref.property_name} already registered")
            self._classes[ref.class_name].properties[ref.property_name] = decl
        else:
            assert_never(ref)

    def set_constant_type(self, ref: ConstantCallableRef, tp: JustTypeRef) -> None:
        if isinstance(ref, ConstantRef):
            if ref.name in self._constants:
                raise ValueError(f"Constant {ref.name} already registered")
            self._constants[ref.name] = tp
        elif isinstance(ref, ClassVariableRef):
            if ref.variable_name in self._classes[ref.class_name].class_variables:
                raise ValueError(f"Class variable {ref.class_name}.{ref.variable_name} already registered")
            self._classes[ref.class_name].class_variables[ref.variable_name] = tp
        else:
            assert_never(ref)

    def register_callable_ref(self, ref: CallableRef, egg_name: str) -> None:
        """
        Registers a callable reference with the given egg name. The callable's function needs to be registered
        first.
        """
        if ref in self._callable_ref_to_egg_fn:
            raise ValueError(f"Callable ref {ref} already registered")
        self._callable_ref_to_egg_fn[ref] = egg_name
        self._egg_fn_to_callable_refs[egg_name].add(ref)

    def get_function_decl(self, ref: FunctionCallableRef) -> FunctionDecl:
        if isinstance(ref, FunctionRef):
            return self._functions[ref.name]
        elif isinstance(ref, MethodRef):
            return self._classes[ref.class_name].methods[ref.method_name]
        elif isinstance(ref, ClassMethodRef):
            return self._classes[ref.class_name].class_methods[ref.method_name]
        elif isinstance(ref, PropertyRef):
            return self._classes[ref.class_name].properties[ref.property_name]
        assert_never(ref)

    def get_constant_type(self, ref: ConstantCallableRef) -> JustTypeRef:
        if isinstance(ref, ConstantRef):
            return self._constants[ref.name]
        elif isinstance(ref, ClassVariableRef):
            return self._classes[ref.class_name].class_variables[ref.variable_name]
        assert_never(ref)

    def get_callable_refs(self, egg_name: str) -> Iterable[CallableRef]:
        return self._egg_fn_to_callable_refs[egg_name]

    def get_egg_fn(self, ref: CallableRef) -> str:
        return self._callable_ref_to_egg_fn[ref]

    def get_egg_sort(self, ref: JustTypeRef) -> str:
        return self._type_ref_to_egg_sort[ref]


@dataclass
class ModuleDeclarations:
    """
    A set of working declerations for a module.
    """

    # The modules declarations we have, which we can edit
    _decl: Declarations
    # A list of other declarations we can use, but not edit
    _included_decls: list[Declarations] = field(default_factory=list, repr=False)

    @property
    def all_decls(self) -> Iterable[Declarations]:
        return itertools.chain([self._decl], self._included_decls)

    def get_function_decl(self, ref: CallableRef) -> FunctionDecl:
        if isinstance(ref, (ClassVariableRef, ConstantRef)):
            for decls in self.all_decls:
                try:
                    return decls.get_constant_type(ref).to_constant_function_decl()
                except KeyError:
                    pass
            raise KeyError(f"Constant {ref} not found")
        elif isinstance(ref, (FunctionRef, MethodRef, ClassMethodRef, PropertyRef)):
            for decls in self.all_decls:
                try:
                    return decls.get_function_decl(ref)
                except KeyError:
                    pass
            raise KeyError(f"Function {ref} not found")
        else:
            assert_never(ref)

    def get_callable_refs(self, egg_name: str) -> Iterable[CallableRef]:
        return itertools.chain.from_iterable(decls.get_callable_refs(egg_name) for decls in self.all_decls)

    def get_egg_fn(self, ref: CallableRef) -> str:
        for decls in self.all_decls:
            try:
                return decls.get_egg_fn(ref)
            except KeyError:
                pass
        raise KeyError(f"Callable ref {ref} not found")

    def get_egg_sort(self, ref: JustTypeRef) -> str:
        for decls in self.all_decls:
            try:
                return decls.get_egg_sort(ref)
            except KeyError:
                pass
        raise KeyError(f"Type {ref} not found")

    def get_class_decl(self, name: str) -> ClassDecl:
        for decls in self.all_decls:
            try:
                return decls._classes[name]
            except KeyError:
                pass
        raise KeyError(f"Class {name} not found")

    def get_registered_class_args(self, cls_name: str) -> tuple[JustTypeRef, ...]:
        """
        Given a class name, returns the first typevar regsisted with args of that class.
        """
        for decl in self.all_decls:
            for tp in decl._type_ref_to_egg_sort.keys():
                if tp.name == cls_name and tp.args:
                    return tp.args
        return ()

    def register_class(self, name: str, n_type_vars: int, egg_sort: Optional[str]) -> Iterable[bindings._Command]:
        # Register class first
        if name in self._decl._classes:
            raise ValueError(f"Class {name} already registered")
        decl = ClassDecl(n_type_vars=n_type_vars)
        self._decl._classes[name] = decl
        _egg_sort, cmds = self.register_sort(JustTypeRef(name), egg_sort)
        return cmds

    def register_sort(
        self, ref: JustTypeRef, egg_name: Optional[str] = None
    ) -> tuple[str, Iterable[bindings._Command]]:
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
            return (egg_sort, [])
        egg_name = egg_name or ref.generate_egg_name()
        if egg_name in self._decl._egg_sort_to_type_ref:
            raise ValueError(f"Sort {egg_name} is already registered.")
        self._decl._egg_sort_to_type_ref[egg_name] = ref
        self._decl._type_ref_to_egg_sort[ref] = egg_name
        return egg_name, ref.to_commands(self)

    def register_function_callable(
        self,
        ref: FunctionCallableRef,
        fn_decl: FunctionDecl,
        egg_name: Optional[str],
        cost: Optional[int],
        default: Optional[ExprDecl],
        merge: Optional[ExprDecl],
        merge_action: list[bindings._Action],
    ) -> Iterable[bindings._Command]:
        """
        Registers a callable with the given egg name. The callable's function needs to be registered
        first.
        """
        egg_name = egg_name or ref.generate_egg_name()
        self._decl.register_callable_ref(ref, egg_name)
        self._decl.set_function_decl(ref, fn_decl)
        return fn_decl.to_commands(self, egg_name, cost, default, merge, merge_action)

    def register_constant_callable(
        self, ref: ConstantCallableRef, type_ref: JustTypeRef, egg_name: Optional[str]
    ) -> Iterable[bindings._Command]:
        egg_function = ref.generate_egg_name()
        self._decl.register_callable_ref(ref, egg_function)
        self._decl.set_constant_type(ref, type_ref)
        # Create a function decleartion for a constant function. This is similar to how egglog compiles
        # the `declare` command.
        return FunctionDecl((), (), (), type_ref.to_var(), False).to_commands(self, egg_name or ref.generate_egg_name())

    def register_preserved_method(self, class_: str, method: str, fn: Callable) -> None:
        self._decl._classes[class_].preserved_methods[method] = fn


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
        args = ", ".join(a.generate_egg_name() for a in self.args)
        return f"{self.name}[{args}]"

    def to_commands(self, mod_decls: ModuleDeclarations) -> Iterable[bindings._Command]:
        """
        Returns commands to register this as a sort, as well as for any of its arguments.
        """
        egg_name = mod_decls.get_egg_sort(self)
        arg_sorts: list[bindings._Expr] = []
        for arg in self.args:
            egg_sort, cmds = mod_decls.register_sort(arg)
            arg_sorts.append(bindings.Var(egg_sort))
            yield from cmds
        yield bindings.Sort(egg_name, (self.name, arg_sorts) if arg_sorts else None)

    def to_var(self) -> TypeRefWithVars:
        return TypeRefWithVars(self.name, tuple(a.to_var() for a in self.args))

    def pretty(self) -> str:
        if not self.args:
            return self.name
        args = ", ".join(a.pretty() for a in self.args)
        return f"{self.name}[{args}]"

    def to_constant_function_decl(self) -> FunctionDecl:
        """
        Create a function declaration for a constant function. This is similar to how egglog compiles
        the `constant` command.
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
    A class type variable represents one of the types of the class, if it is a generic
    class.
    """

    index: int

    def to_just(self) -> JustTypeRef:
        raise NotImplementedError("egglog does not support generic classes yet.")


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
        return f"{self.class_name}.{self.method_name}"


@dataclass(frozen=True)
class ClassMethodRef:
    class_name: str
    method_name: str

    def to_egg(self, decls: Declarations) -> str:
        return decls.get_egg_fn(self)

    def generate_egg_name(self) -> str:
        return f"{self.class_name}.{self.method_name}"


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
        return f"{self.class_name}.{self.variable_name}"


@dataclass(frozen=True)
class PropertyRef:
    class_name: str
    property_name: str

    def generate_egg_name(self) -> str:
        return f"{self.class_name}.{self.property_name}"


ConstantCallableRef = Union[ConstantRef, ClassVariableRef]
FunctionCallableRef = Union[FunctionRef, MethodRef, ClassMethodRef, PropertyRef]
CallableRef = Union[ConstantCallableRef, FunctionCallableRef]


@dataclass(frozen=True)
class FunctionDecl:
    arg_types: tuple[TypeOrVarRef, ...]
    # Is None for relation which doesn't have named args
    arg_names: Optional[tuple[str, ...]]
    arg_defaults: tuple[Optional[ExprDecl], ...]
    return_type: TypeOrVarRef
    mutates_first_arg: bool
    var_arg_type: Optional[TypeOrVarRef] = None

    def __post_init__(self):
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
            for n, d, t in zip(arg_names, self.arg_defaults, self.arg_types)
        ]
        if self.var_arg_type is not None:
            parameters.append(Parameter("__rest", Parameter.VAR_POSITIONAL))
        return Signature(parameters)

    def to_commands(
        self,
        mod_decls: ModuleDeclarations,
        egg_name: str,
        cost: Optional[int] = None,
        default: Optional[ExprDecl] = None,
        merge: Optional[ExprDecl] = None,
        merge_action: list[bindings._Action] = [],
    ) -> Iterable[bindings._Command]:
        if self.var_arg_type is not None:
            raise NotImplementedError("egglog does not support variable arguments yet.")
        arg_sorts: list[str] = []
        for a in self.arg_types:
            # Remove all vars from the type refs, raising an errory if we find one,
            # since we cannot create egg functions with vars
            arg_sort, cmds = mod_decls.register_sort(a.to_just())
            yield from cmds
            arg_sorts.append(arg_sort)
        return_sort, cmds = mod_decls.register_sort(self.return_type.to_just())
        yield from cmds

        egg_fn_decl = bindings.FunctionDecl(
            egg_name,
            bindings.Schema(arg_sorts, return_sort),
            default.to_egg(mod_decls) if default else None,
            merge.to_egg(mod_decls) if merge else None,
            merge_action,
            cost,
        )
        yield bindings.Function(egg_fn_decl)


@dataclass(frozen=True)
class VarDecl:
    name: str

    @classmethod
    def from_egg(cls, var: bindings.Var) -> TypedExprDecl:
        raise NotImplementedError("Cannot turn var into egg type because typing unknown.")

    def to_egg(self, _decls: ModuleDeclarations) -> bindings.Var:
        return bindings.Var(self.name)

    def pretty(self, context: PrettyContext, **kwargs) -> str:
        return self.name


LitType = Union[int, str, float, None]


@dataclass(frozen=True)
class LitDecl:
    value: LitType

    @classmethod
    def from_egg(cls, lit: bindings.Lit) -> TypedExprDecl:
        if isinstance(lit.value, bindings.Int):
            return TypedExprDecl(JustTypeRef("i64"), cls(lit.value.value))
        if isinstance(lit.value, bindings.String):
            return TypedExprDecl(JustTypeRef("String"), cls(lit.value.value))
        if isinstance(lit.value, bindings.F64):
            return TypedExprDecl(JustTypeRef("f64"), cls(lit.value.value))
        elif isinstance(lit.value, bindings.Unit):
            return TypedExprDecl(JustTypeRef("Unit"), cls(None))
        assert_never(lit.value)

    def to_egg(self, _decls: ModuleDeclarations) -> bindings.Lit:
        if self.value is None:
            return bindings.Lit(bindings.Unit())
        if isinstance(self.value, int):
            return bindings.Lit(bindings.Int(self.value))
        if isinstance(self.value, float):
            return bindings.Lit(bindings.F64(self.value))
        if isinstance(self.value, str):
            return bindings.Lit(bindings.String(self.value))
        assert_never(self.value)

    def pretty(self, context: PrettyContext, down_convert=True, **kwargs) -> str:
        """
        Returns a string representation of the literal.

        :param wrap_lit: If True, wraps the literal in a call to the literal constructor.
        """
        if self.value is None:
            return "Unit()"
        if isinstance(self.value, int):
            return f"i64({self.value})" if not down_convert else str(self.value)
        if isinstance(self.value, float):
            return f"f64({self.value})" if not down_convert else str(self.value)
        if isinstance(self.value, str):
            return f"String({repr(self.value)})" if not down_convert else repr(self.value)
        assert_never(self.value)


# Partial mapping of callable refs to functions which down convert them based on their args
# The args to the down converter
# TODO: Do we extend lit with other types to represent them?
# TODO: Do we allow vars?
# Can we convert args first? Have to turn them back into Runtime objects?
# Oh gosh but with context... it doesn't work if its say in a tuple. We need that context to be a global.
# b/c you could have a python tuple with expressions in it, no?

DOWN_CONVERTERS: dict[CallableRef, Callable[..., object]] = {}


def register_down_converter(ref: CallableRef, down_convert: Callable[..., object]) -> None:
    if ref in DOWN_CONVERTERS:
        raise ValueError("{} already has a down converter", ref)
    DOWN_CONVERTERS[ref] = down_convert


@dataclass(frozen=True)
class CallDecl:
    callable: CallableRef
    args: tuple[TypedExprDecl, ...] = ()
    # type parameters that were bound to the callable, if it is a classmethod
    # Used for pretty printing classmethod calls with type parameters
    bound_tp_params: Optional[tuple[JustTypeRef, ...]] = None

    def __post_init__(self):
        if self.bound_tp_params and not isinstance(self.callable, ClassMethodRef):
            raise ValueError("Cannot bind type parameters to a non-class method callable.")

    @classmethod
    def from_egg(cls, mod_decls: ModuleDeclarations, call: bindings.Call) -> TypedExprDecl:
        from .type_constraint_solver import TypeConstraintSolver

        results = tuple(TypedExprDecl.from_egg(mod_decls, a) for a in call.args)
        arg_types = tuple(r.tp for r in results)

        # Find the first callable ref that matches the call
        for callable_ref in mod_decls.get_callable_refs(call.name):
            # If this is a classmethod, we might need the type params that were bound for this type
            # egglog currently only allows one instantiated type of any generic sort to be used in any program
            # So we just lookup what args were registered for this sort
            if isinstance(callable_ref, ClassMethodRef):
                cls_args = mod_decls.get_registered_class_args(callable_ref.class_name)
                tcs = TypeConstraintSolver.from_type_parameters(cls_args)
            else:
                tcs = TypeConstraintSolver()
            fn_decl = mod_decls.get_function_decl(callable_ref)
            return_tp = tcs.infer_return_type(fn_decl.arg_types, fn_decl.return_type, fn_decl.var_arg_type, arg_types)
            return TypedExprDecl(return_tp, cls(callable_ref, tuple(results)))
        raise ValueError(f"Could not find callable ref for call {call}")

    def to_egg(self, mod_decls: ModuleDeclarations) -> bindings.Call:
        """Convert a Call to an egg Call."""
        egg_fn = mod_decls.get_egg_fn(self.callable)
        return bindings.Call(egg_fn, [a.to_egg(mod_decls) for a in self.args])

    def pretty(self, context: PrettyContext, parens=True, down_convert=False) -> str:
        """
        Pretty print the call.

        :param parens: If true, wrap the call in parens if it is a binary method call.
        :param down_convert: If True, try to down convert the call to a simpler value that could be converted back to the call.
        """
        #
        # if down_convert:
        # TODO: Why aren't default args ignored here for slice?
        ref, args = self.callable, [a.expr for a in self.args]
        # Special case != since it doesn't have a decl
        if isinstance(ref, MethodRef) and ref.method_name == "__ne__":
            return f"{args[0].pretty(context)} != {args[1].pretty(context)}"
        function_decl = context.mod_decls.get_function_decl(ref)
        defaults = function_decl.arg_defaults
        if function_decl.mutates_first_arg:
            mutated_arg_type = function_decl.arg_types[0].to_just().name
        else:
            mutated_arg_type = None
        if isinstance(ref, FunctionRef):
            fn_str = ref.name
        elif isinstance(ref, ClassMethodRef):
            tp_ref = JustTypeRef(ref.class_name, self.bound_tp_params or ())
            if ref.method_name == "__init__":
                fn_str = tp_ref.pretty()
            else:
                fn_str = f"{tp_ref.pretty( )}.{ref.method_name}"
        elif isinstance(ref, MethodRef):
            name = ref.method_name
            slf, *args = args
            defaults = defaults[1:]
            if name in UNARY_METHODS:
                return f"{UNARY_METHODS[name]}{slf.pretty(context)}"
            elif name in BINARY_METHODS:
                assert len(args) == 1
                expr = f"{slf.pretty(context)} {BINARY_METHODS[name]} {args[0].pretty(context, down_convert=True)}"
                return expr if not parens else f"({expr})"
            elif name == "__getitem__":
                assert len(args) == 1
                return f"{slf.pretty(context)}[{args[0].pretty(context, down_convert=True, parens=False)}]"
            elif name == "__call__":
                return f"{slf.pretty(context)}({', '.join(a.pretty(context, down_convert=True, parens=False) for a in args)})"
            elif name == "__delitem__":
                assert len(args) == 1
                assert mutated_arg_type
                name = context.name_expr(mutated_arg_type, slf)
                context.statements.append(f"del {name}[{args[0].pretty(context, parens=False, down_convert=True)}]")
                return name
            elif name == "__setitem__":
                assert len(args) == 2
                assert mutated_arg_type
                name = context.name_expr(mutated_arg_type, slf)
                context.statements.append(
                    f"{name}[{args[0].pretty(context, parens=False, down_convert=True)}] = {args[1].pretty(context, parens=False, down_convert=True)}"
                )
                return name
            fn_str = f"{slf.pretty(context)}.{name}"
        elif isinstance(ref, ConstantRef):
            return ref.name
        elif isinstance(ref, ClassVariableRef):
            return f"{ref.class_name}.{ref.variable_name}"
        elif isinstance(ref, PropertyRef):
            return f"{args[0].pretty(context)}.{ref.property_name}"
        else:
            assert_never(ref)
        # Determine how many of the last arguments are defaults, by iterating from the end and comparing the arg with the default
        n_defaults = 0
        for arg, default in zip(reversed(args), reversed(defaults)):
            if arg != default:
                break
            n_defaults += 1
        if n_defaults:
            args = args[:-n_defaults]
        if mutated_arg_type:
            name = context.name_expr(mutated_arg_type, args[0])
            context.statements.append(
                f"{fn_str}({', '.join({name}, *(a.pretty(context, down_convert=True, parens=False) for a in args[1:]))})"
            )
            return name
        return f"{fn_str}({', '.join(a.pretty(context, down_convert=True, parens=False) for a in args)})"


@dataclass
class PrettyContext:
    mod_decls: ModuleDeclarations
    # List of statements of "context" setting variable for the expr
    statements: list[str] = field(default_factory=list)

    _gen_name_types: dict[str, int] = field(default_factory=lambda: defaultdict(lambda: 0))

    def generate_name(self, typ: str) -> str:
        self._gen_name_types[typ] += 1
        return f"_{typ}_{self._gen_name_types[typ]}"

    def name_expr(self, expr_type: str, expr: ExprDecl) -> str:
        name = self.generate_name(expr_type)
        self.statements.append(f"{name} = copy({expr.pretty(self, parens=False)})")
        return name

    def render(self, expr: str) -> str:
        return "\n".join(self.statements + [expr])


def test_expr_pretty():
    context = PrettyContext(ModuleDeclarations(Declarations()))
    assert VarDecl("x").pretty(context) == "x"
    assert LitDecl(42).pretty(context) == "i64(42)"
    assert LitDecl("foo").pretty(context) == 'String("foo")'
    assert LitDecl(None).pretty(context) == "unit()"

    def v(x: str) -> TypedExprDecl:
        return TypedExprDecl(JustTypeRef(""), VarDecl(x))

    assert CallDecl(FunctionRef("foo"), (v("x"),)).pretty(context) == "foo(x)"
    assert CallDecl(FunctionRef("foo"), (v("x"), v("y"), v("z"))).pretty(context) == "foo(x, y, z)"
    assert CallDecl(MethodRef("foo", "__add__"), (v("x"), v("y"))).pretty(context) == "x + y"
    assert CallDecl(MethodRef("foo", "__getitem__"), (v("x"), v("y"))).pretty(context) == "x[y]"
    assert CallDecl(ClassMethodRef("foo", "__init__"), (v("x"), v("y"))).pretty(context) == "foo(x, y)"
    assert CallDecl(ClassMethodRef("foo", "bar"), (v("x"), v("y"))).pretty(context) == "foo.bar(x, y)"
    assert CallDecl(MethodRef("foo", "__call__"), (v("x"), v("y"))).pretty(context) == "x(y)"
    assert (
        CallDecl(
            ClassMethodRef("Map", "__init__"),
            (),
            (JustTypeRef("i64"), JustTypeRef("Unit")),
        ).pretty(context)
        == "Map[i64, Unit]()"
    )


def test_setitem_pretty():
    context = PrettyContext(ModuleDeclarations(Declarations()))

    def v(x: str) -> TypedExprDecl:
        return TypedExprDecl(JustTypeRef("typ"), VarDecl(x))

    final_expr = CallDecl(MethodRef("foo", "__setitem__"), (v("x"), v("y"), v("z"))).pretty(context)
    assert context.render(final_expr) == "_typ_1 = x\n_typ_1[y] = z\n_typ_1"


def test_delitem_pretty():
    context = PrettyContext(ModuleDeclarations(Declarations()))

    def v(x: str) -> TypedExprDecl:
        return TypedExprDecl(JustTypeRef("typ"), VarDecl(x))

    final_expr = CallDecl(MethodRef("foo", "__delitem__"), (v("x"), v("y"))).pretty(context)
    assert context.render(final_expr) == "_typ_1 = x\ndel _typ_1[y]\n_typ_1"


# TODO: Multiple mutations,

ExprDecl = Union[VarDecl, LitDecl, CallDecl]


@dataclass(frozen=True)
class TypedExprDecl:
    tp: JustTypeRef
    expr: ExprDecl

    @classmethod
    def from_egg(cls, mod_decls: ModuleDeclarations, expr: bindings._Expr) -> TypedExprDecl:
        if isinstance(expr, bindings.Var):
            return VarDecl.from_egg(expr)
        if isinstance(expr, bindings.Lit):
            return LitDecl.from_egg(expr)
        if isinstance(expr, bindings.Call):
            return CallDecl.from_egg(mod_decls, expr)
        assert_never(expr)

    def to_egg(self, decls: ModuleDeclarations) -> bindings._Expr:
        return self.expr.to_egg(decls)


@dataclass
class ClassDecl:
    methods: dict[str, FunctionDecl] = field(default_factory=dict)
    class_methods: dict[str, FunctionDecl] = field(default_factory=dict)
    class_variables: dict[str, JustTypeRef] = field(default_factory=dict)
    properties: dict[str, FunctionDecl] = field(default_factory=dict)
    preserved_methods: dict[str, Callable] = field(default_factory=dict)
    n_type_vars: int = 0


# def down_converter(x: Callable[..., Iterable[tuple["Expr", object]]]) -> None:
#     """
#     Register a number of down converters, which turn expressions into
#     """
#     pass
