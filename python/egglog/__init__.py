from . import config  # noqa: F401
from .builtins import *  # noqa: F401
from .egraph import *  # noqa: F401
from .ipython_magic import load_ipython_extension

load_ipython_extension()
