"""
Package for creating e-graphs in Python.
"""

from . import config, ipython_magic  # noqa: F401
from .builtins import *  # noqa: UP029
from .egraph import *
from .runtime import convert, converter  # noqa: F401

del ipython_magic
