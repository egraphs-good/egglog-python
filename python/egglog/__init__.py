"""
Package for creating e-graphs in Python.
"""

from . import config, ipython_magic  # noqa: F401
from .builtins import *  # noqa: UP029
from .conversion import convert, converter  # noqa: F401
from .egraph import *

del ipython_magic
