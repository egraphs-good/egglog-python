import subprocess
import sys

from egglog import declarations
from egglog.declarations import Ident


def test_declarations_has_no_legacy_semantic_model() -> None:
    assert declarations.__all__ == ["Ident", "RuleEvalMode"]
    assert {
        name
        for name, value in vars(declarations).items()
        if isinstance(value, type) and value.__module__ == declarations.__name__
    } == {"Ident"}


def test_ident_preserves_exact_name_and_module() -> None:
    ident = Ident("node", "pkg.one")
    assert ident == Ident("node", "pkg.one")
    assert len({ident, Ident("node", "pkg.two"), Ident("node"), Ident("node", "")}) == 4
    assert str(ident) == "pkg.one.node"
    assert str(Ident("node")) == "node"


def test_call_decl_interning_allows_reentrant_literal_hashing() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from egglog import i64

class ReentrantInt(int):
    def __hash__(self):
        i64(1) + 2
        return super().__hash__()

assert str(i64(ReentrantInt(3)) + 4) == "i64(3) + 4"
""",
        ],
        check=True,
        timeout=10,
    )
