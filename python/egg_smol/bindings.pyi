from typing import Optional

from typing_extensions import final

from .bindings_py import Expr, FunctionDecl, Variant

@final
class EGraph:
    def parse_and_run_program(self, input: str) -> list[str]: ...
    def declare_constructor(self, variant: Variant, sort: str) -> None: ...
    def declare_sort(self, name: str) -> None: ...
    def declare_function(self, decl: FunctionDecl) -> None: ...
    def define(self, name: str, expr: Expr, cost: Optional[int] = None) -> None: ...

@final
class EggSmolError(Exception):
    context: str
