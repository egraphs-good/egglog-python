"""
Package for creating e-graphs in Python.
"""

from . import config, ipython_magic  # noqa: F401
from .bindings import EggSmolError  # noqa: F401
from .builtins import *  # noqa: UP029
from .conversion import *
from .deconstruct import *
from .egraph import *
from .runtime import define_expr_method as define_expr_method

del ipython_magic
