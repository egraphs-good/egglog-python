from __future__ import annotations

import egglog as el


def test_no_import_star():
    """
    https://github.com/egraphs-good/egglog-python/issues/210
    """

    class Num(el.Expr):
        def __init__(self, value: el.i64Like) -> None: ...

    Num(1)  # gets an error "NameError: name 'i64' is not defined"


def test_no_import_star_rulesset():
    """
    https://github.com/egraphs-good/egglog-python/issues/283
    """

    @el.ruleset
    def _rules(_: el.i64Like):
        return []

    el.EGraph().run(_rules)
