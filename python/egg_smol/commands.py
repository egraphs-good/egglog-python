from __future__ import annotations

from typing_extensions import assert_never

from . import bindings


def run_command(egraph: bindings.EGraph, command: bindings._Command) -> None:
    """Run a command on an EGraph."""
    if isinstance(command, bindings.Datatype):
        egraph.declare_sort(command.name, None)
        for variant in command.variants:
            egraph.declare_constructor(variant, command.name)
    elif isinstance(command, bindings.Sort):
        egraph.declare_sort(command.name, (command.presort, command.args))
    elif isinstance(command, bindings.Function):
        egraph.declare_function(command.decl)
    elif isinstance(command, bindings.Define):
        egraph.define(command.name, command.expr, command.cost)
    elif isinstance(command, bindings.RuleCommand):
        egraph.add_rule(command.rule)
    elif isinstance(command, bindings.RewriteCommand):
        egraph.add_rewrite(command.rewrite)
    elif isinstance(command, bindings.ActionCommand):
        egraph.eval_actions(command.action)
    elif isinstance(command, bindings.Run):
        egraph.run_rules(command.length)
    elif isinstance(command, bindings.Extract):
        egraph.extract_expr(command.expr, command.variants)
    elif isinstance(command, bindings.Check):
        egraph.check_fact(command.fact)
    elif isinstance(command, bindings.ClearRules):
        egraph.clear_rules()
    elif isinstance(command, bindings.Clear):
        egraph.clear()
    elif isinstance(command, bindings.Print):
        egraph.print_function(command.name, command.length)
    elif isinstance(command, bindings.PrintSize):
        egraph.print_size(command.name)
    elif isinstance(command, bindings.Input):
        raise NotImplementedError("Input command not implemented")
    elif isinstance(command, bindings.Query):
        raise NotImplementedError("Query command not implemented")
    elif isinstance(command, bindings.Push):
        for _ in range(command.length):
            egraph.push()
    elif isinstance(command, bindings.Pop):
        for _ in range(command.length):
            egraph.pop()
    else:
        assert_never(command)


def parse_and_run(input: str) -> None:
    """Parse a string and run the commands."""
    egraph = bindings.EGraph()
    for command in bindings.parse(input):
        run_command(egraph, command)
