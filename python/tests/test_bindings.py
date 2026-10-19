# mypy: allow-empty-bodies
# Ellipsis bodies are DSL declarations or annotation-introspection fixtures.
"""
Public replacements for the meaningful native binding behavior.

The native text corpus and its internal-uf/hidden-table controls belong to
Experimental's Source-to-protobuf corpus harness, not a second Python AST API.
"""

import concurrent.futures
import subprocess
import sys
import sysconfig
import threading

import pytest

from egglog import BaseExpr, DagCostModel, EGraph, Expr, eq, get_callable_args, i64, i64Like, method, union


class TestEGraph:
    def test_extract(self):
        class Num(Expr):
            @method(cost=5)
            def __init__(self, value: i64Like) -> None: ...

        assert EGraph().extract(Num(1), include_cost=True) == (Num(1), 6)

    def test_extract_value_reports_extraction_failure(self):
        class Hidden(Expr):
            @method(unextractable=True)
            @classmethod
            def make(cls, value: i64Like) -> "Hidden": ...

        with pytest.raises(ValueError, match="Unextractable root"):
            EGraph().extract(Hidden.make(42))

    def test_tree_extractor_extract_variants(self):
        class Num(Expr):
            def __init__(self, value: i64Like) -> None: ...

        graph = EGraph(union(Num(1)).with_(Num(2)), union(Num(1)).with_(Num(3)))

        def model(egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            return 1 + sum(children_costs)

        variants = graph.extract_multiple(Num(1), 2, cost_model=model, include_cost=True)
        assert len(variants) == 2
        assert all(term in (Num(1), Num(2), Num(3)) for term, _cost in variants)
        assert all(cost == 2 for _term, cost in variants)

    @pytest.mark.parametrize("extractor", ["tree", "greedy-dag"])
    def test_dag_cost_model_batch_extraction(self, extractor):
        class Num(Expr):
            def __init__(self, value: i64Like) -> None: ...

        graph = EGraph(union(Num(1)).with_(Num(2)))
        model = DagCostModel(lambda _graph, _node: 1, 0)
        result, cost = graph.extract(Num(1), include_cost=True, cost_model=model, extractor=extractor)
        assert result in (Num(1), Num(2))
        assert cost == 2

    def test_tree_extractor_observes_post_construction_mutation(self):
        class Num(Expr):
            def __init__(self, value: i64Like) -> None: ...

        graph = EGraph(Num(1))
        callback_count = 0

        def model(egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            nonlocal callback_count
            if not isinstance(expr, Num):
                return 0
            callback_count += 1
            args = get_callable_args(expr)
            assert args is not None
            assert isinstance(args[0], i64)
            return (0 if args[0].value == 2 else 1) + sum(children_costs)

        assert callback_count == 0
        assert graph.extract(Num(1), include_cost=True, cost_model=model) == (Num(1), 1)
        first_callback_count = callback_count
        graph.register(union(Num(1)).with_(Num(2)))
        assert graph.extract(Num(1), include_cost=True, cost_model=model) == (Num(2), 0)
        assert callback_count > first_callback_count

    @pytest.mark.parametrize("extractor", ["tree", "greedy-dag"])
    def test_extract_multiple_costs_preserve_batches_and_model_domain(self, extractor):
        class Num(Expr):
            def __init__(self, value: i64Like) -> None: ...

            @method(unextractable=True)
            def opaque(self) -> "Num": ...

        graph = EGraph(union(Num(1)).with_(Num(2)))
        opaque = graph.let("opaque", Num(3).opaque())

        def model(egraph: EGraph, expr: BaseExpr, children_costs: list[bool]) -> bool:
            return isinstance(expr, Num) or any(children_costs)

        batches = graph.extract_multiple([i64(3), opaque, Num(1)], 2, cost_model=model, include_cost=True)
        assert batches[0] == [(i64(3), False)]
        assert batches[1] == []
        assert len(batches[2]) == 2
        assert all(term in (Num(1), Num(2)) and cost is True for term, cost in batches[2])
        assert type(batches[0][0][1]) is bool
        terms = graph.extract_multiple(Num(1), 2, cost_model=model)
        assert len(terms) == 2
        assert all(term in (Num(1), Num(2)) for term in terms)
        defaults = graph.extract_multiple(Num(1), 2, include_cost=True)
        assert len(defaults) == 2
        assert all(cost == 2 and type(cost) is int for _term, cost in defaults)
        dag_model = DagCostModel(lambda _graph, _node: 1, 0)
        best = graph.extract_multiple([Num(1)], 1, include_cost=True, cost_model=dag_model, extractor=extractor)
        assert len(best) == len(best[0]) == 1
        assert best[0][0][0] in (Num(1), Num(2))
        assert best[0][0][1] == 2
        with pytest.raises(TypeError, match="requires a DagCostModel"):
            graph.extract_multiple(Num(1), 1, cost_model=model, extractor="greedy-dag")


class TestThreads:
    """
    Independent EGraphs can run concurrently, including Python cost callbacks.

    Access to the same EGraph during mutation requires caller serialization.
    """

    @pytest.mark.skipif(sysconfig.get_config_var("Py_GIL_DISABLED") != 1, reason="requires free-threaded CPython")
    def test_import_keeps_gil_disabled(self):
        # Isolated mode ignores PYTHON_GIL, so forcing it off in the test runner
        # cannot conceal an extension that enables the GIL during import.
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "import sys; assert not sys._is_gil_enabled(); import egglog; assert not sys._is_gil_enabled()",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_egraph_sequential_handoff(self):
        class Num(Expr):
            def __init__(self, value: i64Like) -> None: ...

        graph = EGraph()
        root = graph.let("root", Num(1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(graph.register, union(root).with_(Num(2))).result(timeout=30)
        graph.check(eq(root).to(Num(2)))

    def test_independent_egraphs_and_cost_callbacks_run_concurrently(self):
        class Num(Expr):
            def __init__(self, value: i64Like) -> None: ...

        workers = 4
        barrier = threading.Barrier(workers, timeout=30)
        roots = [Num(number) for number in range(workers)]
        graphs = [EGraph(root) for root in roots]

        def model(egraph: EGraph, expr: BaseExpr, children_costs: list[int]) -> int:
            if isinstance(expr, Num):
                # All four extractions must overlap inside their cost callbacks.
                barrier.wait()
            return 1 + sum(children_costs)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(graph.extract, root, include_cost=True, cost_model=model)
                for graph, root in zip(graphs, roots, strict=True)
            ]
            results = [future.result(timeout=30) for future in futures]

        assert results == [(Num(number), 2) for number in range(workers)]
