"""
Holds a number of types which are only used at runtime to emulate Python objects.

Users will not import anything from this module, and statically they won't know these are the types they are using.

But at runtime they will be exposed.

Note that all their internal fields are prefixed with __egg_ to avoid name collisions with user code, but will end in __
so they are not mangled by Python and can be accessed by the user.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, replace
from inspect import Parameter, Signature
from itertools import zip_longest
from typing import TYPE_CHECKING, TypeVar, Union, cast, get_args, get_origin

from .declarations import *
from .pretty import *
from .thunk import Thunk
from .type_constraint_solver import *
from .version_compat import *

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .egraph import Fact


__all__ = [
    "LIT_CLASS_NAMES",
    "REFLECTED_BINARY_METHODS",
    "RuntimeClass",
    "RuntimeExpr",
    "RuntimeFunction",
    "resolve_callable",
    "resolve_type_annotation",
    "resolve_type_annotation_mutate",
]


UNIT_CLASS_NAME = "Unit"
UNARY_LIT_CLASS_NAMES = {"i64", "f64", "Bool", "String"}
LIT_CLASS_NAMES = UNARY_LIT_CLASS_NAMES | {UNIT_CLASS_NAME, "PyObject"}

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

# Methods that need to return real Python values not expressions
PRESERVED_METHODS = [
    "__bool__",
    "__len__",
    "__complex__",
    "__int__",
    "__float__",
    "__iter__",
    "__index__",
    "__float__",
    "__int__",
]

# Set this globally so we can get access to PyObject when we have a type annotation of just object.
# This is the only time a type annotation doesn't need to include the egglog type b/c object is top so that would be redundant statically.
_PY_OBJECT_CLASS: RuntimeClass | None = None
# Same for functions
_UNSTABLE_FN_CLASS: RuntimeClass | None = None

T = TypeVar("T")


def resolve_type_annotation_mutate(decls: Declarations, tp: object) -> TypeOrVarRef:
    """
    Wrap resolve_type_annotation to mutate decls, as a helper for internal use in sitations where that is more ergonomic.
    """
    new_decls, tp = resolve_type_annotation(tp)
    decls |= new_decls
    return tp


def resolve_type_annotation(tp: object) -> tuple[DeclerationsLike, TypeOrVarRef]:
    """
    Resolves a type object into a type reference.

    Any runtime type object decls will be returned as well. We do this so we can use this without having to
    resolve the decls if need be.
    """
    if isinstance(tp, TypeVar):
        return None, ClassTypeVarRef.from_type_var(tp)
    # If there is a union, then we assume the first item is the type we want, and the others are types that can be converted to that type.
    if get_origin(tp) == Union:
        first, *_rest = get_args(tp)
        return resolve_type_annotation(first)

    # If the type is `object` then this is assumed to be a PyObjectLike, i.e. converted into a PyObject
    if tp is object:
        assert _PY_OBJECT_CLASS
        return resolve_type_annotation(_PY_OBJECT_CLASS)
    # If the type is a `Callable` then convert it into a UnstableFn
    if get_origin(tp) == Callable:
        assert _UNSTABLE_FN_CLASS
        args, ret = get_args(tp)
        return resolve_type_annotation(_UNSTABLE_FN_CLASS[(ret, *args)])
    if isinstance(tp, RuntimeClass):
        return tp, tp.__egg_tp__
    raise TypeError(f"Unexpected type annotation {tp}")


def inverse_resolve_type_annotation(decls_thunk: Callable[[], Declarations], tp: TypeOrVarRef) -> object:
    """
    Inverse of resolve_type_annotation
    """
    if isinstance(tp, ClassTypeVarRef):
        return tp.to_type_var()
    return RuntimeClass(decls_thunk, tp)


##
# Runtime objects
##


@dataclass
class RuntimeClass(DelayedDeclerations):
    __egg_tp__: TypeRefWithVars

    def __post_init__(self) -> None:
        global _PY_OBJECT_CLASS, _UNSTABLE_FN_CLASS
        if (name := self.__egg_tp__.name) == "PyObject":
            _PY_OBJECT_CLASS = self
        elif name == "UnstableFn" and not self.__egg_tp__.args:
            _UNSTABLE_FN_CLASS = self

    def verify(self) -> None:
        if not self.__egg_tp__.args:
            return

        # Raise error if we have args, but they are the wrong number
        desired_args = self.__egg_decls__.get_class_decl(self.__egg_tp__.name).type_vars
        if len(self.__egg_tp__.args) != len(desired_args):
            raise ValueError(f"Expected {desired_args} type args, got {len(self.__egg_tp__.args)}")

    def __call__(self, *args: object, **kwargs: object) -> RuntimeExpr | None:
        """
        Create an instance of this kind by calling the __init__ classmethod
        """
        # If this is a literal type, initializing it with a literal should return a literal
        if (name := self.__egg_tp__.name) == "PyObject":
            assert len(args) == 1
            return RuntimeExpr(
                self.__egg_decls_thunk__, Thunk.value(TypedExprDecl(self.__egg_tp__.to_just(), PyObjectDecl(args[0])))
            )
        if name == "UnstableFn":
            assert not kwargs
            fn_arg, *partial_args = args
            del args
            # Assumes we don't have types set for UnstableFn w/ generics, that they have to be inferred

            # 1. Call it with the partial args, and use untyped vars for the rest of the args
            res = cast("Callable", fn_arg)(*partial_args, _egg_partial_function=True)
            assert res is not None, "Mutable partial functions not supported"
            # 2. Use the inferred return type and inferred rest arg types as the types of the function, and
            #    the partially applied args as the args.
            call = (res_typed_expr := res.__egg_typed_expr__).expr
            return_tp = res_typed_expr.tp
            assert isinstance(call, CallDecl), "partial function must be a call"
            n_args = len(partial_args)
            value = PartialCallDecl(replace(call, args=call.args[:n_args]))
            remaining_arg_types = [a.tp for a in call.args[n_args:]]
            type_ref = JustTypeRef("UnstableFn", (return_tp, *remaining_arg_types))
            return RuntimeExpr.__from_values__(Declarations.create(self, res), TypedExprDecl(type_ref, value))

        if name in UNARY_LIT_CLASS_NAMES:
            assert len(args) == 1
            assert isinstance(args[0], int | float | str | bool)
            return RuntimeExpr(
                self.__egg_decls_thunk__, Thunk.value(TypedExprDecl(self.__egg_tp__.to_just(), LitDecl(args[0])))
            )
        if name == UNIT_CLASS_NAME:
            assert len(args) == 0
            return RuntimeExpr(
                self.__egg_decls_thunk__, Thunk.value(TypedExprDecl(self.__egg_tp__.to_just(), LitDecl(None)))
            )
        fn = RuntimeFunction(self.__egg_decls_thunk__, Thunk.value(InitRef(name)), self.__egg_tp__.to_just())
        return fn(*args, **kwargs)  # type: ignore[arg-type]

    def __dir__(self) -> list[str]:
        cls_decl = self.__egg_decls__.get_class_decl(self.__egg_tp__.name)
        possible_methods = (
            list(cls_decl.class_methods) + list(cls_decl.class_variables) + list(cls_decl.preserved_methods)
        )
        if "__init__" in possible_methods:
            possible_methods.remove("__init__")
            possible_methods.append("__call__")
        return possible_methods

    def __getitem__(self, args: object) -> RuntimeClass:
        if not isinstance(args, tuple):
            args = (args,)
        # defer resolving decls so that we can do generic instantiation for converters before all
        # method types are defined.
        decls_like, new_args = cast(
            "tuple[tuple[DeclerationsLike, ...], tuple[TypeOrVarRef, ...]]",
            zip(*(resolve_type_annotation(arg) for arg in args), strict=False),
        )
        # if we already have some args bound and some not, then we shold replace all existing args of typevars with new
        # args
        if old_args := self.__egg_tp__.args:
            is_typevar = [isinstance(arg, ClassTypeVarRef) for arg in old_args]
            if sum(is_typevar) != len(new_args):
                raise TypeError(f"Expected {sum(is_typevar)} typevars, got {len(new_args)}")
            new_args_list = list(new_args)
            final_args = tuple(new_args_list.pop(0) if is_typevar[i] else old_args[i] for i in range(len(old_args)))
        else:
            final_args = new_args
        tp = TypeRefWithVars(self.__egg_tp__.name, final_args)
        return RuntimeClass(Thunk.fn(Declarations.create, self, *decls_like), tp)

    def __getattr__(self, name: str) -> RuntimeFunction | RuntimeExpr | Callable:
        if name == "__origin__" and self.__egg_tp__.args:
            return RuntimeClass(self.__egg_decls_thunk__, TypeRefWithVars(self.__egg_tp__.name))

        # Special case some names that don't exist so we can exit early without resolving decls
        # Important so if we take union of RuntimeClass it won't try to resolve decls
        if name in {
            "__typing_subst__",
            "__parameters__",
            # Origin is used in get_type_hints which is used when resolving the class itself
            "__origin__",
            "__typing_unpacked_tuple_args__",
            "__typing_is_unpacked_typevartuple__",
        }:
            raise AttributeError

        try:
            cls_decl = self.__egg_decls__._classes[self.__egg_tp__.name]
        except Exception as e:
            raise add_note(f"Error processing class {self.__egg_tp__.name}", e) from None

        preserved_methods = cls_decl.preserved_methods
        if name in preserved_methods:
            return preserved_methods[name].__get__(self)

        # if this is a class variable, return an expr for it, otherwise, assume it's a method
        if name in cls_decl.class_variables:
            return_tp = cls_decl.class_variables[name]
            return RuntimeExpr(
                self.__egg_decls_thunk__,
                Thunk.value(TypedExprDecl(return_tp.type_ref, CallDecl(ClassVariableRef(self.__egg_tp__.name, name)))),
            )
        if name in cls_decl.class_methods:
            return RuntimeFunction(
                self.__egg_decls_thunk__,
                Thunk.value(ClassMethodRef(self.__egg_tp__.name, name)),
                self.__egg_tp__.to_just(),
            )
        # allow referencing properties and methods as class variables as well
        if name in cls_decl.properties:
            return RuntimeFunction(self.__egg_decls_thunk__, Thunk.value(PropertyRef(self.__egg_tp__.name, name)))
        if name in cls_decl.methods:
            return RuntimeFunction(self.__egg_decls_thunk__, Thunk.value(MethodRef(self.__egg_tp__.name, name)))

        msg = f"Class {self.__egg_tp__.name} has no method {name}"
        raise AttributeError(msg) from None

    def __str__(self) -> str:
        return str(self.__egg_tp__)

    def __repr__(self) -> str:
        return str(self)

    # Make hashable so can go in Union
    def __hash__(self) -> int:
        return hash((id(self.__egg_decls_thunk__), self.__egg_tp__))

    # Support unioning like types
    def __or__(self, value: type) -> object:
        return Union[self, value]  # noqa: UP007

    @property
    def __parameters__(self) -> tuple[object, ...]:
        """
        Emit a number of typevar params so that when using generic type aliases, we know how to resolve these properly.
        """
        return tuple(inverse_resolve_type_annotation(self.__egg_decls_thunk__, tp) for tp in self.__egg_tp__.args)


@dataclass
class RuntimeFunction(DelayedDeclerations):
    __egg_ref_thunk__: Callable[[], CallableRef]
    # bound methods need to store RuntimeExpr not just TypedExprDecl, so they can mutate the expr if required on self
    __egg_bound__: JustTypeRef | RuntimeExpr | None = None

    @property
    def __egg_ref__(self) -> CallableRef:
        return self.__egg_ref_thunk__()

    def __call__(self, *args: object, _egg_partial_function: bool = False, **kwargs: object) -> RuntimeExpr | None:
        from .conversion import resolve_literal

        if isinstance(self.__egg_bound__, RuntimeExpr):
            args = (self.__egg_bound__, *args)
        try:
            signature = self.__egg_decls__.get_callable_decl(self.__egg_ref__).signature
        except Exception as e:
            raise add_note(f"Failed to find callable {self}", e)  # noqa: B904
        decls = self.__egg_decls__.copy()
        # Special case function application bc we dont support variadic generics yet generally
        if signature == "fn-app":
            fn, *rest_args = args
            args = tuple(rest_args)
            assert not kwargs
            assert isinstance(fn, RuntimeExpr)
            decls.update(fn)
            function_value = fn.__egg_typed_expr__
            fn_tp = function_value.tp
            assert fn_tp.name == "UnstableFn"
            fn_return_tp, *fn_arg_tps = fn_tp.args
            signature = FunctionSignature(
                tuple(tp.to_var() for tp in fn_arg_tps),
                tuple(f"_{i}" for i in range(len(fn_arg_tps))),
                (None,) * len(fn_arg_tps),
                fn_return_tp.to_var(),
            )
        else:
            function_value = None
            assert isinstance(signature, FunctionSignature)

        # Turn all keyword args into positional args
        py_signature = to_py_signature(signature, self.__egg_decls__, _egg_partial_function)
        try:
            bound = py_signature.bind(*args, **kwargs)
        except TypeError as err:
            raise TypeError(f"Failed to call {self} with args {args} and kwargs {kwargs}") from err
        del kwargs
        bound.apply_defaults()
        assert not bound.kwargs
        args = bound.args

        tcs = TypeConstraintSolver(decls)
        bound_tp = (
            None
            if self.__egg_bound__ is None
            else self.__egg_bound__.__egg_typed_expr__.tp
            if isinstance(self.__egg_bound__, RuntimeExpr)
            else self.__egg_bound__
        )
        if (
            bound_tp
            and bound_tp.args
            # Don't  bind class if we have a first class function arg, b/c we don't support that yet
            and not function_value
        ):
            tcs.bind_class(bound_tp)
        assert (operator.ge if signature.var_arg_type else operator.eq)(len(args), len(signature.arg_types))
        cls_name = bound_tp.name if bound_tp else None
        upcasted_args = [
            resolve_literal(cast("TypeOrVarRef", tp), arg, Thunk.value(decls), tcs=tcs, cls_name=cls_name)
            for arg, tp in zip_longest(args, signature.arg_types, fillvalue=signature.var_arg_type)
        ]
        decls.update(*upcasted_args)
        arg_exprs = tuple(arg.__egg_typed_expr__ for arg in upcasted_args)
        return_tp = tcs.substitute_typevars(signature.semantic_return_type, cls_name)
        bound_params = (
            cast("JustTypeRef", bound_tp).args if isinstance(self.__egg_ref__, ClassMethodRef | InitRef) else None
        )
        # If we were using unstable-app to call a funciton, add that function back as the first arg.
        if function_value:
            arg_exprs = (function_value, *arg_exprs)
        expr_decl = CallDecl(self.__egg_ref__, arg_exprs, bound_params)
        typed_expr_decl = TypedExprDecl(return_tp, expr_decl)
        # If there is not return type, we are mutating the first arg
        if not signature.return_type:
            first_arg = upcasted_args[0]
            first_arg.__egg_decls_thunk__ = Thunk.value(decls)
            first_arg.__egg_typed_expr_thunk__ = Thunk.value(typed_expr_decl)
            return None
        return RuntimeExpr.__from_values__(decls, typed_expr_decl)

    def __str__(self) -> str:
        first_arg, bound_tp_params = None, None
        match self.__egg_bound__:
            case RuntimeExpr(_):
                first_arg = self.__egg_bound__.__egg_typed_expr__.expr
            case JustTypeRef(_, args):
                bound_tp_params = args
        return pretty_callable_ref(self.__egg_decls__, self.__egg_ref__, first_arg, bound_tp_params)


def to_py_signature(sig: FunctionSignature, decls: Declarations, optional_args: bool) -> Signature:
    """
    Convert to a Python signature.

    If optional_args is true, then all args will be treated as optional, as if a default was provided that makes them
    a var with that arg name as the value.

    Used for partial application to try binding a function with only some of its args.
    """
    parameters = [
        Parameter(
            n,
            Parameter.POSITIONAL_OR_KEYWORD,
            default=RuntimeExpr.__from_values__(
                decls, TypedExprDecl(t.to_just(), d if d is not None else VarDecl(n, True))
            )
            if d is not None or optional_args
            else Parameter.empty,
        )
        for n, d, t in zip(sig.arg_names, sig.arg_defaults, sig.arg_types, strict=True)
    ]
    if isinstance(sig, FunctionSignature) and sig.var_arg_type is not None:
        parameters.append(Parameter("__rest", Parameter.VAR_POSITIONAL))
    return Signature(parameters)


# All methods which should return NotImplemented if they fail to resolve
# From https://docs.python.org/3/reference/datamodel.html
PARTIAL_METHODS = {
    "__lt__",
    "__le__",
    "__eq__",
    "__ne__",
    "__gt__",
    "__ge__",
    "__add__",
    "__sub__",
    "__mul__",
    "__matmul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__divmod__",
    "__pow__",
    "__lshift__",
    "__rshift__",
    "__and__",
    "__xor__",
    "__or__",
}


@dataclass
class RuntimeExpr(DelayedDeclerations):
    __egg_typed_expr_thunk__: Callable[[], TypedExprDecl]

    @classmethod
    def __from_values__(cls, d: Declarations, e: TypedExprDecl) -> RuntimeExpr:
        return cls(Thunk.value(d), Thunk.value(e))

    def __with_expr__(self, e: TypedExprDecl) -> RuntimeExpr:
        return RuntimeExpr(self.__egg_decls_thunk__, Thunk.value(e))

    @property
    def __egg_typed_expr__(self) -> TypedExprDecl:
        return self.__egg_typed_expr_thunk__()

    def __getattr__(self, name: str) -> RuntimeFunction | RuntimeExpr | Callable | None:
        cls_name = self.__egg_class_name__
        class_decl = self.__egg_class_decl__

        if name in (preserved_methods := class_decl.preserved_methods):
            return preserved_methods[name].__get__(self)

        if name in class_decl.methods:
            return RuntimeFunction(self.__egg_decls_thunk__, Thunk.value(MethodRef(cls_name, name)), self)
        if name in class_decl.properties:
            return RuntimeFunction(self.__egg_decls_thunk__, Thunk.value(PropertyRef(cls_name, name)), self)()
        raise AttributeError(f"{cls_name} has no method {name}") from None

    def __repr__(self) -> str:
        """
        The repr of the expr is the pretty printed version of the expr.
        """
        return str(self)

    def __str__(self) -> str:
        return self.__egg_pretty__(None)

    def __egg_pretty__(self, wrapping_fn: str | None) -> str:
        return pretty_decl(self.__egg_decls__, self.__egg_typed_expr__.expr, wrapping_fn=wrapping_fn)

    def _ipython_display_(self) -> None:
        from IPython.display import Code, display

        display(Code(str(self), language="python"))

    def __dir__(self) -> Iterable[str]:
        class_decl = self.__egg_class_decl__
        return list(class_decl.methods) + list(class_decl.properties) + list(class_decl.preserved_methods)

    @property
    def __egg_class_name__(self) -> str:
        return self.__egg_typed_expr__.tp.name

    @property
    def __egg_class_decl__(self) -> ClassDecl:
        return self.__egg_decls__.get_class_decl(self.__egg_class_name__)

    # These both will be overriden below in the special methods section, but add these here for type hinting purposes
    def __eq__(self, other: object) -> Fact:  # type: ignore[override, empty-body]
        ...

    def __ne__(self, other: object) -> RuntimeExpr:  # type: ignore[override, empty-body]
        ...

    # Implement these so that copy() works on this object
    # otherwise copy will try to call `__getstate__` before object is initialized with properties which will cause inifinite recursion

    def __getstate__(self) -> tuple[Declarations, TypedExprDecl]:
        return self.__egg_decls__, self.__egg_typed_expr__

    def __setstate__(self, d: tuple[Declarations, TypedExprDecl]) -> None:
        self.__egg_decls_thunk__ = Thunk.value(d[0])
        self.__egg_typed_expr_thunk__ = Thunk.value(d[1])

    def __hash__(self) -> int:
        return hash(self.__egg_typed_expr__)


# Define each of the special methods, since we have already declared them for pretty printing
for name in list(BINARY_METHODS) + list(UNARY_METHODS) + ["__getitem__", "__call__", "__setitem__", "__delitem__"]:

    def _special_method(
        self: RuntimeExpr,
        *args: object,
        __name: str = name,
        **kwargs: object,
    ) -> RuntimeExpr | Fact | None:
        from .conversion import ConvertError

        class_name = self.__egg_class_name__
        class_decl = self.__egg_class_decl__
        # First, try to resolve as preserved method
        try:
            method = class_decl.preserved_methods[__name]
        except KeyError:
            pass
        else:
            return method(self, *args, **kwargs)
        # If this is a "partial" method meaning that it can return NotImplemented,
        # we want to find the "best" superparent (lowest cost) of the arg types to call with it, instead of just
        # using the arg type of the self arg.
        # This is neccesary so if we add like an int to a ndarray, it will upcast the int to an ndarray, instead of vice versa.
        if __name in PARTIAL_METHODS:
            try:
                return call_method_min_conversion(self, args[0], __name)
            except ConvertError:
                # Defer raising not imeplemented in case the dunder method is not symmetrical, then
                # we use the standard process
                pass
        if __name in class_decl.methods:
            fn = RuntimeFunction(self.__egg_decls_thunk__, Thunk.value(MethodRef(class_name, __name)), self)
            return fn(*args, **kwargs)  # type: ignore[arg-type]
        # Handle == and != fallbacks to eq and ne helpers if the methods aren't defined on the class explicitly.
        if __name == "__eq__":
            from .egraph import BaseExpr, eq

            return eq(cast("BaseExpr", self)).to(cast("BaseExpr", args[0]))
        if __name == "__ne__":
            from .egraph import BaseExpr, ne

            return cast("RuntimeExpr", ne(cast("BaseExpr", self)).to(cast("BaseExpr", args[0])))

        if __name in PARTIAL_METHODS:
            return NotImplemented
        raise TypeError(f"{class_name!r} object does not support {__name}")

    setattr(RuntimeExpr, name, _special_method)

# For each of the reflected binary methods, translate to the corresponding non-reflected method
for reflected, non_reflected in REFLECTED_BINARY_METHODS.items():

    def _reflected_method(self: RuntimeExpr, other: object, __non_reflected: str = non_reflected) -> RuntimeExpr | None:
        # All binary methods are also "partial" meaning we should try to upcast first.
        return call_method_min_conversion(other, self, __non_reflected)

    setattr(RuntimeExpr, reflected, _reflected_method)


def call_method_min_conversion(slf: object, other: object, name: str) -> RuntimeExpr | None:
    from .conversion import min_convertable_tp, resolve_literal

    # find a minimum type that both can be converted to
    # This is so so that calls like `-0.1 * Int("x")` work by upcasting both to floats.
    min_tp = min_convertable_tp(slf, other, name).to_var()
    slf = resolve_literal(min_tp, slf)
    other = resolve_literal(min_tp, other)
    method = RuntimeFunction(slf.__egg_decls_thunk__, Thunk.value(MethodRef(slf.__egg_class_name__, name)), slf)
    return method(other)


for name in PRESERVED_METHODS:

    def _preserved_method(self: RuntimeExpr, __name: str = name):
        try:
            method = self.__egg_decls__.get_class_decl(self.__egg_typed_expr__.tp.name).preserved_methods[__name]
        except KeyError as e:
            raise TypeError(f"{self.__egg_typed_expr__.tp.name} has no method {__name}") from e
        return method(self)

    setattr(RuntimeExpr, name, _preserved_method)


def resolve_callable(callable: object) -> tuple[CallableRef, Declarations]:
    """
    Resolves a runtime callable into a ref
    """
    match callable:
        case RuntimeFunction(decls, ref, _):
            return ref(), decls()
        case RuntimeClass(thunk, tp):
            return InitRef(tp.name), thunk()
        case RuntimeExpr(decl_thunk, expr_thunk):
            if not isinstance((expr := expr_thunk().expr), CallDecl) or not isinstance(
                expr.callable, ConstantRef | ClassVariableRef
            ):
                raise NotImplementedError(f"Can only turn constants or classvars into callable refs, not {expr}")
            return expr.callable, decl_thunk()
        case _:
            raise NotImplementedError(f"Cannot turn {callable} of type {type(callable)} into a callable ref")
