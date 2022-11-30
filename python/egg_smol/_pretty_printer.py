from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import black
import egg_smol.bindings as py

BLACK_MODE = black.Mode(line_length=120)


@dataclass
class _PrettyPrinter:
    # Mapping of egg function name to a function which mapps the string args to the result
    functions: dict[str, Callable[[list[str]], str]] = field(default_factory=dict)

    def __call__(self, expr: py._Expr) -> str:
        """
        Pretty print an expression.

        This should give a string which can be used to recreate the expression.
        """
        # Run black on only the most outer expression, to remove any redundant parantheses
        # and wrap it if it is too long
        return blacken_python_expression(self._expr(expr))

    def _expr(self, expr: py._Expr) -> str:
        if isinstance(expr, py.Lit):
            return self._literal(expr.value)
        elif isinstance(expr, py.Var):
            return expr.name
        elif isinstance(expr, py.Call):
            return self._call(expr.name, expr.args)
        else:
            raise NotImplementedError

    def _literal(self, literal: py._Literal) -> str:
        if isinstance(literal, py.Int):
            return f"i64({literal.value})"
        elif isinstance(literal, py.String):
            return f'String("{literal.value}")'
        elif isinstance(literal, py.Unit):
            return "Unit()"
        else:
            raise NotImplementedError

    def _call(self, name: str, args: list[py._Expr]) -> str:
        if name in self.functions:
            return self.functions[name]([self._expr(arg) for arg in args])
        else:
            raise NotImplementedError


def blacken_python_expression(expr: str) -> str:
    """
    Runs black on a Python expression.
    """
    return black.format_str("x = " + expr, mode=BLACK_MODE)[4:-1]


def test_pretty_print_lit():
    pp = _PrettyPrinter()
    assert pp(py.Lit(py.Int(1))) == "i64(1)"
    assert pp(py.Lit(py.Unit())) == "Unit()"
    assert pp(py.Lit(py.String("hello"))) == 'String("hello")'


def test_pretty_print_var():
    pp = _PrettyPrinter()
    assert pp(py.Var("x")) == "x"


def test_pretty_print_function():
    pp = _PrettyPrinter()
    pp.functions["add"] = lambda args: f"({args[0]} + {args[1]})"
    assert (
        pp(py.Call("add", [py.Lit(py.Int(1)), py.Lit(py.Int(2))])) == "i64(1) + i64(2)"
    )
    # Test nested call keeps the outer parentheses
    assert (
        pp(
            py.Call(
                "add",
                [
                    py.Lit(py.Int(1)),
                    py.Call("add", [py.Lit(py.Int(2)), py.Lit(py.Int(3))]),
                ],
            )
        )
        == "i64(1) + (i64(2) + i64(3))"
    )
