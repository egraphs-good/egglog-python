from __future__ import annotations

import egglog as el
from egglog import f64Like


def test_no_import_star():
    """
    https://github.com/egraphs-good/egglog-python/issues/210
    """

    class Num(el.Expr):
        def __init__(self, value: el.i64Like) -> None: ...

    Num(1)


def test_f64_import():
    """
    For some reason this wasn't working until we moved the union definition below the class
    """

    class Num(el.Expr):
        def __init__(self, value: f64Like) -> None: ...

    Num(1.0)


def test_no_import_star_rulesset():
    """
    https://github.com/egraphs-good/egglog-python/issues/283
    """

    @el.ruleset
    def _rules(_: el.i64Like):
        return []

    el.EGraph().run(_rules)
