from __future__ import annotations

from collections.abc import Callable
from functools import partial
from inspect import Parameter, signature
from typing import Any, TypeVar, cast

__all__ = ["functionalize"]


T = TypeVar("T", bound=Callable)


# TODO: Add `to_lift` param so that we only transform those with vars in them to args


def functionalize(f: T, get_annotation: Callable[[object], type | None]) -> T:
    """
    Takes a function and returns a new function with all names (co_names) and free variables (co_freevars) added as arguments
    and then partially applied with their values. The second arg, get_annotation, will be applied to all values
    to get their type annotation. If it is None, that arg will not be added as a parameter.

    For example if you have:

        def get_annotation(x): return int if x <= 10 else None

        g = 10
        def f(a, a2):
            def h(b: Z):
                return a + a2 + b + g

            return functionalize(h, get_annotation)
        res = f(9, 11)

    It should be equivalent to (according to body, signature, and annotations) (Note that the new arguments will be positional only):

        def h(a: get_annotation(a), g: get_annotation(g), b: Z):
            return a + b + g
        res = partial(h, a, g)
    """
    code = f.__code__
    names = tuple(n for n in code.co_names if n in f.__globals__)
    free_vars = code.co_freevars

    global_values: list[Any] = [f.__globals__[name] for name in names]
    free_var_values = [cell.cell_contents for cell in f.__closure__] if f.__closure__ else []
    assert len(free_var_values) == len(free_vars), "Free vars and their values do not match"
    global_values_filtered = [
        (i, name, value, annotation)
        for i, (name, value) in enumerate(zip(names, global_values, strict=True))
        if (annotation := get_annotation(value)) is not None
    ]
    free_var_values_filtered = [
        (i, name, value, annotation)
        for i, (name, value) in enumerate(zip(free_vars, free_var_values, strict=True))
        if (annotation := get_annotation(value)) is not None
    ]
    additional_arg_filtered = global_values_filtered + free_var_values_filtered

    # Create a wrapper function
    def wrapper(*args):
        # Split args into names, free vars and other args
        name_args, free_var_args, rest_args = (
            args[: (n_names := len(global_values_filtered))],
            args[n_names : (n_args := len(additional_arg_filtered))],
            args[n_args:],
        )
        # Update globals with names
        f.__globals__.update({
            name: arg for (_, name, _, _), arg in zip(global_values_filtered, name_args, strict=False)
        })
        # update function free vars with free var args
        for (i, _, _, _), value in zip(free_var_values_filtered, free_var_args, strict=True):
            assert f.__closure__, "Function does not have closure"
            f.__closure__[i].cell_contents = value
        return f(*rest_args)

    # Set the signature of the new function to a signature with the free vars and names added as arguments
    orig_signature = signature(f)
    wrapper.__signature__ = orig_signature.replace(  # type: ignore[attr-defined]
        parameters=[
            *[Parameter(n, Parameter.POSITIONAL_OR_KEYWORD) for _, n, _, _ in additional_arg_filtered],
            *orig_signature.parameters.values(),
        ]
    )
    # Set the annotations of the new function to the annotations of the original function + annotations of passed in values
    wrapper.__annotations__ = f.__annotations__ | {n: a for _, n, _, a in additional_arg_filtered}
    wrapper.__name__ = f.__name__

    # Partially apply the wrapper function with the current values of the free vars
    return cast("T", partial(wrapper, *(v for _, _, v, _ in additional_arg_filtered)))
