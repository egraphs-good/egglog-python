from __future__ import annotations
from dataclasses import dataclass, field

from .egg_mapper import EggMapper

from . import bindings, high_level


@dataclass
class EGraph:
    egraph: bindings.EGraph = field(default_factory=bindings.EGraph)
    mapper: EggMapper = field(default_factory=EggMapper)

    def declare_function(self, function: high_level.Function) -> None:
        """
        Declare a function in the egraph.
        """
        self.mapper.namespace.add_function(function)

    def _function_to_function_decl(self, function: high_level.Function, name: str) -> bindings.FunctionDecl:
        """
        Convert a python function to an egg function declaration.
        """
        merge = self.mapper.to_egg(function.merge.value) if function.merge else None
        # TODO: handle default
        default = None

        schema = bindings.Schema(
        )
        return bindings.FunctionDecl(
            name=name, schema=schema, default=default, merge=merge, cost=function.cost
        )
        name = self.mapper.namespace.get_function_name(function)
        return bindings.FunctionDecl(name, function.arity)

    def _to_egg_sort_name(self, type: high_level.Type) -> str:
        if type.args:

        else:
            return
