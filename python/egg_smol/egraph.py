from __future__ import annotations
from dataclasses import dataclass, field


from . import bindings

from .new import Fact, Registry

__all__ = ["EGraph"]


@dataclass
class EGraph:
    egraph: bindings.EGraph = field(default_factory=bindings.EGraph)
    register: Registry = field(default_factory=Registry)

    def run(self, iterations: int) -> None:
        self.egraph.run_rules(iterations)

    def check(self, fact: Fact) -> None:
        return self.egraph.check_fact(eq.egg_eq)
