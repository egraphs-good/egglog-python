"""
Package for creating e-graphs in Python.
"""

from . import config, ipython_magic  # noqa: F401
from .bindings import EggSmolError, StageInfo, TimeOnly, WithPlan  # noqa: F401
from .builtins import *
from .conversion import *
from .deconstruct import *
from .egraph import *
from .egraph import ActionLike as ActionLike
from .runtime import define_expr_method as define_expr_method

del ipython_magic
