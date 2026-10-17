try:
    get_ipython()  # type: ignore[name-defined]
    IN_IPYTHON = True
except NameError:
    IN_IPYTHON = False

if IN_IPYTHON:
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
        msg = "Notebook Egglog text requires the shared source-to-protobuf producer, which is not exposed to Python"
        raise NotImplementedError(msg)
