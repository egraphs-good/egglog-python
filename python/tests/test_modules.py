import pytest
from egglog.declarations import ModuleDeclarations
from egglog.egraph import *
from egglog.egraph import _BUILTIN_DECLS, BUILTINS


def test_tree_modules():
    """
      BUILTINS
      /  |  \
      A  B  C
      |  /
      D
    """
    assert _BUILTIN_DECLS
    assert BUILTINS._mod_decls == ModuleDeclarations(_BUILTIN_DECLS, [])

    A, B, C = Module(), Module(), Module()
    assert list(A._mod_decls._included_decls) == [_BUILTIN_DECLS]

    a = A.relation("a")
    b = B.relation("b")
    c = C.relation("c")
    A.register(a())
    B.register(b())
    C.register(c())

    D = Module([A, B])
    d = D.relation("d")
    D.register(d())

    assert D._flatted_deps == [A, B]

    egraph = EGraph([D, B])
    assert egraph._flatted_deps == [A, B, D]
    egraph.check(a(), b(), d())
    with pytest.raises(Exception):
        egraph.check(c())
