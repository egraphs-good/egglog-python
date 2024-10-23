from .bindings import EGraph

EGRAPH_VAR = "_MAGIC_EGRAPH"

try:
    get_ipython()  # type: ignore[name-defined]
    IN_IPYTHON = True
except NameError:
    IN_IPYTHON = False

if IN_IPYTHON:
    import graphviz
    from IPython.core.magic import needs_local_scope, register_cell_magic

    @needs_local_scope
    @register_cell_magic
    def egglog(line, cell, local_ns):
        """
        Run an egglog program.

        Usage:

            %%egglog [output] [continue] [graph]
            (egglog program)

        If `output` is specified, the output of the program will be printed.
        If `continue` is specified, the program will be run in the same EGraph as the previous cell.
        If `graph` is specified, the EGraph will be displayed as a graph.
        """
        if EGRAPH_VAR in local_ns and "continue" in line:
            e = local_ns[EGRAPH_VAR]
        else:
            e = EGraph()
            local_ns[EGRAPH_VAR] = e
        cmds = e.parse_program(cell)
        res = e.run_program(*cmds)
        if "output" in line:
            print("\n".join(res))
        if "graph" in line:
            return graphviz.Source(e.to_graphviz_string())
        return None
