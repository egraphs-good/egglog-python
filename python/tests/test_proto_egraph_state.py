"""Record/bytes plumbing tests; the inert responder is not engine conformance."""

from operator import index
from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import EGraph, Expr, bindings, constant, eq, function, i64, method, set_
from egglog._catalog import builtin_catalog
from egglog._program import Builder, pack
from egglog.runtime import RuntimeFunction, create_callable, resolve_callable


@pytest.fixture
def table_transport(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Table inspection used the legacy engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
    graph = EGraph()
    transport = graph._state.transport
    requests: list[pb.RunProgramRequest] = []
    responses: list[pb.RunProgramResponse] = []

    class AuditedTransport:
        destroy = transport.destroy

        def run(self, encoded):
            assert isinstance(encoded, bytes)
            position = len(requests)
            (tmp_path / f"request-{position}.pb").write_bytes(encoded)
            requests.append(pb.RunProgramRequest.from_binary(encoded))
            returned = transport.run(encoded)
            assert isinstance(returned, bytes)
            (tmp_path / f"response-{position}.pb").write_bytes(returned)
            responses.append(pb.RunProgramResponse.from_binary(returned))
            return returned

    monkeypatch.setattr(graph._state, "transport", AuditedTransport())
    yield graph, requests, responses
    graph._state.transport = transport
    graph._state.destroy()


def test_function_sizes_native_bytes_register_and_filter_unmapped(table_transport):
    graph, requests, responses = table_transport

    @function(egg_fn="z-known")
    def known(value: i64) -> i64: ...  # type: ignore[empty-body]

    @function(egg_fn="a-unmapped")
    def unmapped(value: i64) -> i64: ...  # type: ignore[empty-body]

    assert graph.function_size(known) == 0
    assert len(requests) == 1
    assert any(decl.kind.value.name == "z-known" for decl in requests[0].program.declarations)
    assert requests[0].program.commands[0].kind == Oneof("print_size", pb.PrintSize(tables=["z-known"]))
    assert responses[0].outputs[0].location == pb.CommandLocation(path=[0])
    assert responses[0].outputs[0].kind == Oneof("sizes", pb.TableSizes(sizes=[pb.TableSize(table="z-known", rows=0)]))
    graph.register(set_(known(i64(1))).to(i64(2)), set_(unmapped(i64(1))).to(i64(3)))
    # The legacy public API filtered native tables lacking a Python mapping.
    # Keep the native row, but remove only its canonical language lookup entry.
    del graph._state.definitions["callable", "a-unmapped"]
    assert graph.all_function_sizes() == [(known, 1)]
    assert requests[-1].program.declarations == []
    assert requests[-1].program.commands[0].kind == Oneof("print_size", pb.PrintSize())
    assert [entry.table for entry in responses[-1].outputs[0].kind.value.sizes] == ["a-unmapped", "z-known"]


def test_function_sizes_include_unavailable_and_subsumed_native_bytes(table_transport):
    graph, _, _ = table_transport
    program = pb.Program(
        ir_version=1,
        sorts=[
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))),
            pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "CountedHidden")),
        ],
        declarations=[
            pb.Declaration(kind=Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="CountedHidden"))),
            pb.Declaration(
                kind=Oneof[Literal["constructor"], pb.Constructor](
                    "constructor",
                    pb.Constructor(name="hidden-count", inputs=[pb.Arg(sort=0)], output=1, unextractable=True),
                )
            ),
        ],
        nodes=[
            pb.Node(
                sort_id=0,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", n))
                ),
            )
            for n in (1, 2)
        ]
        + [
            pb.Node(sort_id=1, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="hidden-count", args=[n])))
            for n in (0, 1)
        ],
        commands=[
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action", pb.Action(kind=Oneof[Literal["term"], int]("term", n))
                )
            )
            for n in (2, 3)
        ]
        + [
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action",
                    pb.Action(
                        kind=Oneof[Literal["subsume"], pb.Call]("subsume", pb.Call(func="hidden-count", args=[0]))
                    ),
                )
            )
        ],
    )
    owner = Builder.from_program(program).publish()
    graph._state.run_program([owner.ref("commands", n) for n in range(3)])
    target = RuntimeFunction(owner.ref("declarations", 1))
    assert graph.function_size(target) == 2
    assert graph.all_function_sizes() == [(target, 2)]
    with pytest.raises(ValueError, match="has no extractable representative"):
        graph.function_values(target)


def test_table_callable_projection_keeps_first_view_without_invocation(monkeypatch):
    owner = Builder.from_program(
        pb.Program(
            ir_version=1,
            sorts=[pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))],
            declarations=[
                pb.Declaration(
                    kind=Oneof[Literal["function"], pb.Function](
                        "function", pb.Function(name="two-views", inputs=[pb.Arg(sort=0)], output=0)
                    ),
                    bindings=pb.CallableBindings(
                        python=pb.PythonBindings(
                            views=[
                                pb.PythonCallable(
                                    kind=pb.PythonCallKind.FUNCTION,
                                    path=[name],
                                    params=[pb.PythonParameter(name="value", core_input=0)],
                                )
                                for name in ("first", "second")
                            ]
                        )
                    ),
                )
            ],
        )
    ).publish()

    def forbidden(*args, **kwargs):
        pytest.fail("Callable projection invoked a Python callable")

    monkeypatch.setattr(RuntimeFunction, "__call__", forbidden)
    reference = owner.ref("declarations", 0)
    projected = create_callable(reference)
    assert projected == RuntimeFunction(reference, 0)
    assert hash(projected) == hash(RuntimeFunction(reference, 0))
    assert projected != RuntimeFunction(reference, 1)


def test_table_constant_projection_does_not_invoke_callable(monkeypatch):
    current = constant("projected-current", i64)
    reference = resolve_callable(current)

    def forbidden(*args, **kwargs):
        pytest.fail("Constant projection invoked a Python callable")

    monkeypatch.setattr(RuntimeFunction, "__call__", forbidden)
    projected = create_callable(reference)
    assert projected == current
    assert hash(projected) == hash(current)


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("corruption", ["outputs", "kind", "location", "run report", "table", "duplicate"])
def test_function_sizes_reject_malformed_response_without_engine(monkeypatch, selected, corruption):  # noqa: C901
    @function(egg_fn="size-table")
    def table(value: i64) -> i64: ...  # type: ignore[empty-body]

    sizes = pb.TableSizes(sizes=[pb.TableSize(table="size-table", rows=1)])
    output = pb.CommandOutput(
        kind=Oneof[Literal["sizes"], pb.TableSizes]("sizes", sizes),
        location=pb.CommandLocation(path=[0]),
    )
    response = pb.RunProgramResponse(outputs=[output], ruleset_publication=pb.RulesetPublication())
    match corruption:
        case "outputs":
            response.outputs.clear()
        case "kind":
            output.kind = Oneof[Literal["printed_function"], pb.PrintedFunction](
                "printed_function", pb.PrintedFunction()
            )
        case "location":
            output.location = pb.CommandLocation(path=[1])
        case "run report":
            output.run_report = pb.NativeRunReport()
        case "table":
            sizes.sizes[0].table = ""
        case "duplicate":
            sizes.sizes.append(pb.TableSize(table="size-table", rows=2))

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            assert request.program is not None
            assert request.program.commands[0].kind is not None
            assert request.program.commands[0].kind.field == "print_size"
            return response.to_binary()

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    with pytest.raises(ValueError, match="table size"):
        graph.function_size(table) if selected else graph.all_function_sizes()


def test_function_values_native_bytes_and_shared_terms(table_transport):
    graph, requests, responses = table_transport

    @function
    def repeated(left: i64, right: i64) -> i64: ...  # type: ignore[empty-body]

    graph.register(set_(repeated(i64(1), i64(1))).to(i64(1)), set_(repeated(i64(2), i64(3))).to(i64(7)))
    rows = graph.function_values(repeated)
    assert len(requests) == 2
    command = requests[-1].program.commands[0].kind
    assert command.field == "print_function"
    assert command.value.table == resolve_callable(repeated).read().kind.value.name
    assert not command.value.has_field("max_rows")
    output = responses[-1].outputs[0]
    assert output.location == pb.CommandLocation(path=[0])
    assert output.kind.field == "printed_function"
    row = output.kind.value.rows[0]
    assert row.args[0] == row.args[1] == row.output == pb.TableCell(kind=Oneof[Literal["term"], int]("term", 0))
    assert responses[-1].nodes[0].kind.value.value == Oneof("i64", 1)
    key, result = next(iter(rows.items()))
    args = key.__egg_ref__.read().kind.value.args
    assert key.__egg_ref__.owner.ref("nodes", args[0]) == key.__egg_ref__.owner.ref("nodes", args[1])
    assert graph.function_values(repeated, 0) == {}
    assert requests[-2].program.commands == []
    assert requests[-1].program.commands[0].kind.value.has_field("max_rows")
    assert requests[-1].program.commands[0].kind.value.max_rows == 0
    graph._state.destroy()
    assert rows == {repeated(i64(1), i64(1)): i64(1), repeated(i64(2), i64(3)): i64(7)}
    assert result.value == 1
    assert str(key) == "repeated(1, 1)"
    assert len(requests) == 4


@pytest.mark.parametrize("value", [-1, 1])
def test_function_values_installs_before_index_conversion_native_bytes(table_transport, value):
    graph, requests, responses = table_transport

    @function
    def untouched(value: i64) -> i64: ...  # type: ignore[empty-body]

    calls = []

    class IndexedLength:
        def __index__(self) -> int:
            calls.append(len(requests))
            assert len(requests) == 1
            assert requests[0].program.commands == []
            assert any(decl.kind.field == "function" for decl in requests[0].program.declarations)
            assert responses[0].error is None
            return value

    if value < 0:
        with pytest.raises(OverflowError, match="can't convert negative int to unsigned") as caught:
            graph.function_values(untouched, IndexedLength())
        assert caught.value.__notes__ == ["while processing 'length'"]
        assert len(requests) == 1
    else:
        assert graph.function_values(untouched, IndexedLength()) == {}
        assert len(requests) == 2
    assert calls == [1]
    assert graph.function_values(untouched) == {}


@pytest.mark.parametrize(
    ("length", "error", "message"),
    [
        (-1, OverflowError, "can't convert negative int to unsigned"),
        (1.0, TypeError, "'float' object cannot be interpreted as an integer"),
        (2**64, OverflowError, "int too big to convert"),
    ],
)
def test_function_values_bad_length_keeps_installed_table_native_bytes(table_transport, length, error, message):
    graph, requests, responses = table_transport

    @function
    def untouched(value: i64) -> i64: ...  # type: ignore[empty-body]

    with pytest.raises(error, match=message) as caught:
        graph.function_values(untouched, length)
    assert caught.value.__notes__ == ["while processing 'length'"]
    assert len(requests) == 1
    assert requests[0].program.commands == []
    assert responses[0].error is None
    assert graph.function_values(untouched) == {}


@pytest.mark.parametrize("unavailable", ["argument 1", "output"])
def test_function_values_unavailable_cells_raise_from_native_bytes(table_transport, unavailable):
    graph, requests, responses = table_transport
    # Canonical backend fixture: public unextractable authoring stays guarded.
    sorts = [
        pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))),
        pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Hidden")),
    ]
    nodes = [
        pb.Node(
            sort_id=0,
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", n))
            ),
        )
        for n in (1, 2)
    ]
    nodes.extend([
        pb.Node(sort_id=1, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="Hidden", args=[0]))),
        pb.Node(sort_id=1, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="Visible"))),
    ])
    inputs = [pb.Arg(sort=0), pb.Arg(sort=1)] if unavailable == "argument 1" else [pb.Arg(sort=0)]
    program = pb.Program(
        ir_version=1,
        sorts=sorts,
        nodes=nodes,
        declarations=[
            pb.Declaration(kind=Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="Hidden"))),
            pb.Declaration(
                kind=Oneof[Literal["constructor"], pb.Constructor](
                    "constructor", pb.Constructor(name="Hidden", inputs=[pb.Arg(sort=0)], output=1, unextractable=True)
                )
            ),
            pb.Declaration(
                kind=Oneof[Literal["constructor"], pb.Constructor](
                    "constructor", pb.Constructor(name="Visible", output=1)
                )
            ),
            pb.Declaration(
                kind=Oneof[Literal["function"], pb.Function](
                    "function",
                    pb.Function(name="inspected", inputs=inputs, output=0 if unavailable == "argument 1" else 1),
                )
            ),
        ],
        commands=[
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action",
                    pb.Action(
                        kind=Oneof[Literal["set"], pb.Set](
                            "set",
                            pb.Set(
                                target=pb.Call(
                                    func="inspected", args=[number, value] if unavailable == "argument 1" else [number]
                                ),
                                value=number if unavailable == "argument 1" else value,
                            ),
                        )
                    ),
                )
            )
            for number, value in ((0, 3), (1, 2))
        ],
    )
    owner = Builder.from_program(program).publish()
    graph._state.run_program([owner.ref("commands", n) for n in range(2)])
    target = RuntimeFunction(owner.ref("declarations", 3))
    with pytest.raises(ValueError, match=f"Table 'inspected' row 1 {unavailable} has no extractable representative"):
        graph.function_values(target)
    table = responses[-1].outputs[0].kind.value
    assert responses[-1].error is None
    assert len(table.rows) == 2
    cell = table.rows[1].args[1] if unavailable == "argument 1" else table.rows[1].output
    assert cell.kind.field == "unavailable_sort"
    assert responses[-1].sorts[cell.kind.value].kind == Oneof("eq", "Hidden")
    assert requests[-1].program.commands[0].kind.field == "print_function"
    assert graph.function_values(target, 0) == {}


@pytest.mark.parametrize(
    "corruption",
    [
        "outputs",
        "kind",
        "location",
        "table",
        "arity",
        "output",
        "cell",
        "index",
        "node",
        "argument sort",
        "output sort",
    ],
)
def test_function_values_rejects_malformed_response_without_engine(monkeypatch, corruption):  # noqa: C901
    @function(egg_fn="table")
    def table(value: i64) -> i64: ...  # type: ignore[empty-body]

    available = pb.TableCell(kind=Oneof[Literal["term"], int]("term", 0))
    row = pb.FunctionRow(args=[available], output=available)
    output = pb.CommandOutput(
        kind=Oneof[Literal["printed_function"], pb.PrintedFunction](
            "printed_function", pb.PrintedFunction(table="table", rows=[row])
        ),
        location=pb.CommandLocation(path=[0]),
    )
    response = pb.RunProgramResponse(
        sorts=[
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name=name)))
            for name in ("i64", "Bool")
        ],
        nodes=[
            pb.Node(
                sort_id=0,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 1))
                ),
            ),
            pb.Node(
                sort_id=1,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["bool"], bool]("bool", True))
                ),
            ),
        ],
        outputs=[output],
        ruleset_publication=pb.RulesetPublication(),
    )
    # Generated messages retain the assigned child messages in this inert fixture.
    match corruption:
        case "outputs":
            response.outputs.clear()
        case "kind":
            output.kind = Oneof[Literal["extraction"], pb.ExtractResult]("extraction", pb.ExtractResult())
        case "location":
            output.location = pb.CommandLocation(path=[1])
        case "table":
            assert output.kind is not None
            assert output.kind.field == "printed_function"
            output.kind.value.table = "wrong"
        case "arity":
            row.args.clear()
        case "output":
            row.output = None
        case "cell":
            row.args[0] = pb.TableCell()
        case "index":
            row.args[0] = pb.TableCell(kind=Oneof[Literal["term"], int]("term", 99))
        case "node":
            response.nodes[0].kind = None
        case "argument sort":
            row.args[0] = pb.TableCell(kind=Oneof[Literal["term"], int]("term", 1))
        case "output sort":
            row.output = pb.TableCell(kind=Oneof[Literal["term"], int]("term", 1))

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            assert request.program is not None
            assert request.program.commands[0].kind is not None
            assert request.program.commands[0].kind.field == "print_function"
            return response.to_binary()

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    with pytest.raises(ValueError, match=r"table|Table"):
        EGraph().function_values(table)


def test_function_values_rejects_primitives_without_table_request(table_transport):
    graph, requests, _ = table_transport
    owner = Builder.from_program(
        pb.Program(
            ir_version=1,
            sorts=[pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))],
            nodes=[
                pb.Node(
                    sort_id=0,
                    kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                        "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 1))
                    ),
                )
            ],
            declarations=[
                pb.Declaration(
                    kind=Oneof[Literal["primitive"], pb.Primitive](
                        "primitive", pb.Primitive(name="eager", output=0, body=0)
                    )
                )
            ],
        )
    ).publish()
    for target in (i64.__add__, RuntimeFunction(owner.ref("declarations", 0))):
        with pytest.raises(ValueError, match="table-backed"):
            graph.function_values(target)
        with pytest.raises(ValueError, match="table-backed"):
            graph.function_size(target)
    assert requests == []


def test_scope_return_consumption_and_rejection_without_engine():
    from egglog.egraph_state import EGraphState  # noqa: PLC0415

    calls = []

    class InertTransport:
        reject = True

        def return_to(self, encoded):
            request = pb.ReturnToEGraphRequest.from_binary(encoded)
            calls.append((request.active_id, request.saved_id))
            if self.reject:
                msg = "Return rejection"
                raise ValueError(msg)
            return pb.ReturnToEGraphResponse().to_binary()

    active, saved = object.__new__(EGraphState), object.__new__(EGraphState)
    active.handle, saved.handle = 2, 1
    active.transport = saved.transport = InertTransport()
    with pytest.raises(ValueError, match="Return rejection"):
        active.return_to(saved)
    assert (active.handle, saved.handle) == (2, 1)
    active.transport.reject = False
    active.return_to(saved)
    assert (active.handle, saved.handle) == (None, 1)
    with pytest.raises(RuntimeError, match="live"):
        active.return_to(saved)
    assert calls == [(2, 1), (2, 1)]
    active.handle = 2
    original = active.transport
    active.transport = InertTransport()
    with pytest.raises(ValueError, match="same transport"):
        active.return_to(saved)
    active.transport = original
    active.handle = 1
    with pytest.raises(ValueError, match="distinct"):
        active.return_to(saved)
    active.handle = None
    saved.handle = None


def test_scope_return_corrupt_success_consumes_active_without_selecting_saved(monkeypatch):
    destroyed = []

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def clone_graph(self, encoded):
            return pb.CloneEGraphResponse(egraph_id=2).to_binary()

        def return_to(self, encoded):
            request = pb.ReturnToEGraphRequest.from_binary(encoded)
            assert (request.active_id, request.saved_id) == (2, 1)
            return b"\x80"  # Successful transport, then a truncated acknowledgement.

        def destroy(self, encoded):
            destroyed.append(pb.DestroyEGraphRequest.from_binary(encoded).egraph_id)
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    saved = graph._state
    graph.push()
    active = graph._state
    with pytest.raises(ValueError, match="invalid varint"):
        graph.pop()
    assert graph._state is active
    assert graph._state_stack == [saved]
    assert active.handle is None
    assert saved.handle == 1
    active.destroy()
    assert destroyed == []  # Never Destroy a handle already consumed by Return.


def test_schedule_publication_and_report_requests_are_pure_and_ordered(monkeypatch):
    from egglog import rule, ruleset, run, seq  # noqa: PLC0415
    from egglog.declarations import Ident  # noqa: PLC0415
    from egglog.egraph import set_current_ruleset  # noqa: PLC0415

    requests = []

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            requests.append(request)
            program = request.program
            assert program is not None
            outputs = []
            if program.commands:
                assert len(program.commands) == 1
                command = program.commands[0].kind
                report = pb.NativeRunReport(can_stop=True)
                if command.field == "stats":
                    assert not request.include_run_reports
                    outputs = [
                        pb.CommandOutput(
                            kind=Oneof[Literal["statistics"], pb.NativeRunReport]("statistics", report),
                            location=pb.CommandLocation(path=[0]),
                        )
                    ]
                else:
                    assert request.include_run_reports
                    outputs = [
                        pb.CommandOutput(
                            kind=Oneof("run", pb.RunOutcome(can_stop=True))
                            if command.field == "run"
                            else Oneof(
                                "loop", pb.LoopOutcome(iterations=1, termination=pb.TerminationReason.COUNT_REACHED)
                            ),
                            run_report=report,
                            location=pb.CommandLocation(path=[0]),
                        )
                    ]
            return pb.RunProgramResponse(
                ruleset_publication=pb.RulesetPublication(names=[r.name for r in program.rulesets]),
                outputs=outputs,
            ).to_binary()

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

        def configure_resources(self, encoded):
            request = pb.ConfigureEGraphResourcesRequest.from_binary(encoded)
            requests.append(request)
            return pb.ConfigureEGraphResourcesResponse().to_binary()

    def forbidden(*args, **kwargs):
        pytest.fail("Schedule request assembly accessed a native engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    first = ruleset(rule(name="first").then(), name="same")
    last = ruleset(rule(name="last").then(), name="same")
    a, b = run(first), run(last)
    schedule = seq(a, b, a)
    assert "first" in str(schedule)
    assert graph.run(schedule).can_stop
    installed = graph._state.rulesets[first.__egg_ident__]
    assert [ref.read().name for ref in graph._state._ruleset_members(installed)] == ["first"]
    execution = requests[-1]
    assert execution.program.commands[0].kind.field == "repeat"
    assert len(execution.program.commands[0].kind.value.body) == 3
    assert not any(request.program.commands for request in requests[:-1])
    before = len(requests)
    graph.run(schedule)
    assert len(requests) == before + 1  # Same members/cursors, no fresh publication.
    first.append(rule(name="append").then())
    graph.run(schedule)
    assert [ref.read().name for ref in graph._state._ruleset_members(graph._state.rulesets[first.__egg_ident__])] == [
        "first",
        "append",
    ]
    graph.run(seq(run()))
    assert Ident("") in graph._state.rulesets
    assert graph.stats().can_stop
    assert requests[-1].program.commands[0].kind.field == "stats"
    for length in (-1, 2**64):
        before = len(requests)
        with pytest.raises(OverflowError):
            graph.run(run() * length)
        assert len(requests) == before  # No publication precedes unsigned-count rejection.
    other = ruleset(rule(name="other").then(), name="other")
    combined = first | other
    graph.run(combined)
    first.append(rule(name="deferred-append").then())
    graph.run(combined)
    installed = graph._state.rulesets[combined.__egg_ident__]
    assert [ref.read().name for ref in graph._state._ruleset_members(installed)] == ["first", "append", "other"]
    graph.run(first)
    graph.run(combined)
    installed = graph._state.rulesets[combined.__egg_ident__]
    assert [ref.read().name for ref in graph._state._ruleset_members(installed)] == [
        "first",
        "append",
        "deferred-append",
        "other",
    ]
    with set_current_ruleset(other):
        graph.run(run())
    selected = requests[-1].program.commands[0].kind.value.ruleset.kind.value
    assert requests[-1].program.rulesets[selected].name == graph._state.rulesets[Ident("")].read().name
    for level, expected in (
        (bindings.TimeOnly(), pb.ReportLevel.TIME_ONLY),
        (bindings.WithPlan(), pb.ReportLevel.WITH_PLAN),
        (bindings.StageInfo(), pb.ReportLevel.STAGE_INFO),
    ):
        graph.set_report_level(level)
        assert requests[-1].operation.value == expected


@pytest.mark.parametrize("operation", ["run", "stats"])
@pytest.mark.parametrize("malformed", ["location", "kind", "missing", "flags"])
def test_public_reports_require_the_requested_top_level_output(monkeypatch, operation, malformed):
    from egglog import seq  # noqa: PLC0415

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            report = pb.NativeRunReport(can_stop=True)
            output = pb.CommandOutput(
                location=pb.CommandLocation(path=[1] if malformed == "location" else [0]),
                kind=Oneof("statistics", report)
                if operation == "stats"
                else Oneof("loop", pb.LoopOutcome(termination=pb.TerminationReason.COUNT_REACHED)),
                run_report=report if operation == "run" else None,
            )
            if malformed == "kind":
                output.kind = Oneof("run", pb.RunOutcome(can_stop=True))
            elif malformed == "missing":
                output.kind = None
            elif malformed == "flags":
                if operation == "run":
                    output.kind.value.updated = True
                else:
                    output.run_report = report
            return pb.RunProgramResponse(ruleset_publication=pb.RulesetPublication(), outputs=[output]).to_binary()

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    with pytest.raises(ValueError, match=r"report|statistics"):
        graph.run(seq()) if operation == "run" else graph.stats()


def _run_ruleset_step(graph, selected=None):
    """Freeze a live target, execute canonical Run bytes, and read raw matches."""
    from egglog._program import Builder  # noqa: PLC0415
    from egglog.declarations import Ident  # noqa: PLC0415

    if selected is None:
        published = graph._state.rulesets[Ident("")]
    else:
        identifiers = {}
        snapshot = selected._snapshot(identifiers)
        published = graph._state.publish_ruleset(snapshot, identifiers)
    builder = Builder()
    ruleset_index = builder.import_ref(published)
    command = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["run"], pb.Run](
                "run", pb.Run(ruleset=pb.RulesetRef(kind=Oneof[Literal["index"], int]("index", ruleset_index)))
            )
        ),
    )
    response, _ = graph._state.run_program([builder.publish().ref("commands", command)])
    assert response.error is None
    assert response.profile is not None
    assert len(response.profile.runs) == 1
    labels = {index: attribution.name for index, attribution in enumerate(response.rules)}
    matches = {}
    for summary in response.profile.runs[0].rules:
        assert summary.matches is not None
        label = labels[summary.rule]
        matches[label] = matches.get(label, 0) + summary.matches
    return matches


@pytest.mark.parametrize("completed", [False, True])
def test_publication_adopts_exact_receipt_before_install_error(monkeypatch, completed):
    from egglog import rule, ruleset  # noqa: PLC0415

    requests = []
    fail = True

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            requests.append(request)
            names = [record.name for record in request.program.rulesets if record.has_field("name")]
            return pb.RunProgramResponse(
                ruleset_publication=pb.RulesetPublication(names=names if completed or not fail else []),
                error=pb.Error(code=pb.ErrorCode.EVALUATION_FAILED, message="injected install boundary")
                if fail
                else None,
            ).to_binary()

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    def forbidden(*args, **kwargs):
        pytest.fail("Publication bookkeeping used the legacy engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    selected = ruleset(rule(name="first").then(), name="receipt")
    identifiers = {}
    snapshot = selected._snapshot(identifiers)
    with pytest.raises(bindings.EggSmolError, match="injected install boundary"):
        graph._state.publish_ruleset(snapshot, identifiers)
    first_name = requests[0].program.rulesets[0].name
    assert (("ruleset", first_name) in graph._state.definitions) is completed
    assert (selected.__egg_ident__ in graph._state.rulesets) is completed
    if completed:
        confirmed = graph._state.rulesets[selected.__egg_ident__]
        assert graph._state._ruleset_identifiers[confirmed] == selected.__egg_ident__
    fail = False
    version = graph._state.publish_ruleset(snapshot, identifiers)
    assert version.read().name == first_name
    if completed:
        assert len(requests) == 1  # Retry must not reset the completed occurrence.
    else:
        assert len(requests) == 2
        assert requests[1].program == requests[0].program
    assert graph._state._ruleset_members(version)[0] != snapshot.owner.ref("rules", snapshot.read().kind.value.rules[0])


@pytest.mark.parametrize("receipt", [None, ["foreign"], ["requested", "requested"], []])
def test_invalid_publication_receipt_does_not_adopt_roots(monkeypatch, receipt):
    from egglog._program import Builder  # noqa: PLC0415

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            return pb.RunProgramResponse(
                ruleset_publication=None if receipt is None else pb.RulesetPublication(names=receipt)
            ).to_binary()

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    builder = Builder()
    index = builder.add(
        "rulesets", pb.Ruleset(name="requested", kind=Oneof[Literal["rules"], pb.RuleList]("rules", pb.RuleList()))
    )
    before = graph._state.definitions.copy()
    with pytest.raises(ValueError, match=r"receipt|confirmation"):
        graph._state.run_program([], roots=[builder.publish().ref("rulesets", index)])
    assert graph._state.definitions == before


@pytest.mark.parametrize("across_actions", [False, True])
def test_mixed_registration_cut_distinguishes_query_from_action_identity(monkeypatch, across_actions):
    from egglog import Action, Fact, expr_action, rule  # noqa: PLC0415
    from egglog._program import Builder  # noqa: PLC0415

    requests = []

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            requests.append(request)
            return pb.RunProgramResponse(
                ruleset_publication=pb.RulesetPublication(
                    names=[root.name for root in request.program.rulesets if root.has_field("name")]
                )
            ).to_binary()

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)

    class Box(Expr):
        def __init__(self) -> None: ...

    builder = Builder()
    union = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(Box.__egg_sort__),
            kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[builder.import_ref(Box().__egg_ref__)])),
        ),
    )
    command = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["action"], pb.Action](
                "action", pb.Action(kind=Oneof[Literal["term"], int]("term", union))
            )
        ),
    )
    owner = builder.publish()
    action = Action(owner.ref("commands", command))
    query = Fact(owner.ref("nodes", union))
    graph = EGraph()
    if across_actions:
        with pytest.raises(NotImplementedError, match="shared allocating Union"):
            graph.register(action, rule(query).then(), action)
        assert not requests
        assert not graph._state.rulesets
    else:
        graph.register(action, rule(query).then(), expr_action(i64(1)))
        assert [len(request.program.commands) for request in requests] == [1, 0, 1]
        assert len(requests[1].program.rules) == 1


@pytest.mark.parametrize("reuse", ["separate", "same", "copy"])
def test_native_public_union_occurrence_boundary(tmp_path, reuse):
    from copy import copy  # noqa: PLC0415

    from egglog import union  # noqa: PLC0415

    class UnionNode(Expr):
        def __init__(self, value: i64) -> None: ...

    graph = EGraph()
    lhs, rhs = UnionNode(1), UnionNode(2)
    graph.register(lhs, rhs)
    first = union(lhs).with_(rhs)
    second = union(lhs).with_(rhs) if reuse == "separate" else first if reuse == "same" else copy(first)
    transport = graph._state.transport
    requests = []
    responses = []

    class AuditedTransport:
        def run(self, encoded):
            index = len(requests)
            (tmp_path / f"request-{index}.pb").write_bytes(encoded)
            requests.append(pb.RunProgramRequest.from_binary(encoded))
            returned = transport.run(encoded)
            (tmp_path / f"response-{index}.pb").write_bytes(returned)
            responses.append(pb.RunProgramResponse.from_binary(returned))
            return returned

        def destroy(self, encoded):
            return transport.destroy(encoded)

    graph._state.transport = AuditedTransport()
    if reuse == "separate":
        graph.register(first, second)
        assert responses[0].error is None
        graph.check(eq(lhs).to(rhs))
    else:
        with pytest.raises(bindings.EggSmolError, match="repeated action Union occurrence"):
            graph.register(first, second)
        assert responses[0].error.code == pb.ErrorCode.INVALID_PROGRAM
        assert responses[0].error.location is None
        graph.check_fail(eq(lhs).to(rhs))
    assert len(requests[0].program.commands) == 2
    roots = [command.kind.value.kind.value for command in requests[0].program.commands]
    assert (roots[0] == roots[1]) is (reuse != "separate")
    assert all(requests[0].program.nodes[index].kind.field == "union" for index in roots)


def test_ruleset_append_preserves_native_cursor(tmp_path):
    """Real encoded execution; raw summaries are not a public RunReport."""
    from egglog import rule, ruleset, var  # noqa: PLC0415

    class Input(Expr):
        def __init__(self, value: i64) -> None: ...

    class Old(Expr):
        def __init__(self, value: i64) -> None: ...

    class New(Expr):
        def __init__(self, value: i64) -> None: ...

    graph = EGraph()
    transport = graph._state.transport
    responses = []

    class AuditedTransport:
        def run(self, encoded):
            index = len(responses)
            (tmp_path / f"request-{index}.pb").write_bytes(encoded)
            decoded = pb.RunProgramRequest.from_binary(encoded)
            assert pb.RunProgramRequest.from_binary(decoded.to_binary()) == decoded
            returned = transport.run(encoded)
            (tmp_path / f"response-{index}.pb").write_bytes(returned)
            response = pb.RunProgramResponse.from_binary(returned)
            assert response.ruleset_publication is not None
            responses.append(response)
            return returned

        def destroy(self, encoded):
            return transport.destroy(encoded)

    graph._state.transport = AuditedTransport()
    graph.register(Input(1))
    x = var("x", i64)
    selected = ruleset(rule(Input(x), name="old").then(Old(x)), name="append_cursor")

    assert _run_ruleset_step(graph, selected) == {"old": 1}
    graph.check(Old(1))
    selected.append(rule(Input(x), name="new").then(New(x)))
    assert _run_ruleset_step(graph, selected) == {"new": 1}
    graph.check(Old(1), New(1))
    assert _run_ruleset_step(graph, selected) == {}
    assert any(response.ruleset_publication.names for response in responses)


@pytest.mark.parametrize("seeded", [False, True])
def test_native_class_default_rewrite_and_unregistered_closure(tmp_path, seeded):
    """Default authoring reaches real bytes; public report projection is separate."""
    from types import SimpleNamespace  # noqa: PLC0415

    from egglog import ruleset  # noqa: PLC0415

    class Result(Expr, egg_sort=f"NativeDefaultResult_{seeded}"):
        def __init__(self, value: i64) -> None: ...

    selected = ruleset()
    callbacks = []

    class Source(Expr, ruleset=selected):
        def __init__(self, value: i64) -> None: ...

        def wrapped(self, value: i64) -> Result:
            callbacks.append(1)
            return Result(value)

    expression = Source(1).wrapped(2)
    assert callbacks == [1]
    graph = EGraph()
    transport = graph._state.transport
    requests, responses = [], []

    def audited_run(encoded):
        index = len(requests)
        request = pb.RunProgramRequest.from_binary(encoded)
        assert request.to_binary() == encoded
        requests.append(request)
        (tmp_path / f"request-{index}.pb").write_bytes(encoded)
        returned = transport.run(encoded)
        (tmp_path / f"response-{index}.pb").write_bytes(returned)
        response = pb.RunProgramResponse.from_binary(returned)
        assert response.ruleset_publication is not None
        responses.append(response)
        return returned

    graph._state.transport = SimpleNamespace(run=audited_run, destroy=transport.destroy)
    if seeded:
        graph.register(expression)
    matches = _run_ruleset_step(graph, selected)
    assert matches == ({"": 1} if seeded else {})
    if seeded:
        graph.check(eq(expression).to(Result(2)))
    else:
        assert _run_ruleset_step(graph, selected) == {}
    assert callbacks == [1]
    assert any(response.ruleset_publication.names for response in responses)
    declared = {declaration.kind.value.name for request in requests for declaration in request.program.declarations}
    assert resolve_callable(Source.wrapped).read().kind.value.name in declared
    assert resolve_callable(Result).read().kind.value.name in declared


@pytest.fixture
def native_rule_program():
    from egglog import rule, var  # noqa: PLC0415

    class Seed(Expr):
        def __init__(self, value: i64) -> None: ...

    class First(Expr):
        def __init__(self, value: i64) -> None: ...

    class Second(Expr):
        def __init__(self, value: i64) -> None: ...

    x = var("x", i64)
    first = rule(Seed(x), name="first").then(First(x))
    second = rule(Seed(x), name="second").then(Second(x))
    return Seed, First, Second, first, second


def test_native_same_ident_retains_all_completed_memberships(native_rule_program):
    from egglog import ruleset  # noqa: PLC0415

    Seed, First, Second, first, second = native_rule_program
    r1, r2 = ruleset(first, name="same"), ruleset(second, name="same")
    graph = EGraph()
    graph.register(Seed(1))
    assert _run_ruleset_step(graph, r1) == {"first": 1}
    assert _run_ruleset_step(graph, r2) == {"second": 1}
    assert _run_ruleset_step(graph, r1) == {}
    graph.register(Seed(2))
    assert _run_ruleset_step(graph, r1) == {"first": 1, "second": 1}
    graph.check(First(1), Second(1), First(2), Second(2))


@pytest.mark.parametrize("reverse", [False, True])
def test_native_combined_same_ident_selects_rightmost_before_install(native_rule_program, reverse):
    from egglog import ruleset  # noqa: PLC0415

    Seed, First, Second, first, second = native_rule_program
    r1, r2 = ruleset(first, name="same"), ruleset(second, name="same")
    graph = EGraph()
    graph.register(Seed(1))
    assert _run_ruleset_step(graph, r2 | r1 if reverse else r1 | r2) == {"first" if reverse else "second": 1}
    assert graph.check_bool(First(1)) is reverse
    assert graph.check_bool(Second(1)) is not reverse


def test_native_combined_uses_only_completed_child_updates(native_rule_program):
    from egglog import ruleset  # noqa: PLC0415

    Seed, First, Second, first, second = native_rule_program
    left = ruleset(first, name="left")
    combined = left | ruleset(name="empty")
    graph = EGraph()
    graph.register(Seed(1))
    assert _run_ruleset_step(graph, combined) == {"first": 1}
    left.append(second)
    assert _run_ruleset_step(graph, combined) == {}
    graph.check_fail(Second(1))
    assert _run_ruleset_step(graph, left) == {"second": 1}
    graph.register(Seed(2))
    assert _run_ruleset_step(graph, combined) == {"first": 1, "second": 1}
    graph.check(First(2), Second(2))


def test_native_distinct_ident_memberships_do_not_share_cursors(native_rule_program):
    from egglog import ruleset  # noqa: PLC0415

    Seed, _, _, first, _ = native_rule_program
    graph = EGraph()
    graph.register(Seed(1))
    r1 = ruleset(first, first, name="independent_one")
    r2 = ruleset(first, name="independent_two")
    assert _run_ruleset_step(graph, r1) == {"first": 1}
    assert _run_ruleset_step(graph, r2) == {"first": 1}
    assert _run_ruleset_step(graph, r1) == {}
    roots = [graph._state.rulesets[r.__egg_ident__] for r in (r1, r2)]
    occurrences = [graph._state._ruleset_members(root) for root in roots]
    assert len(occurrences[0]) == len(occurrences[1]) == 1
    assert occurrences[0][0] != occurrences[1][0]


@pytest.mark.parametrize("prefix_rule", [False, True])
def test_native_mixed_registration_keeps_failure_prefix(native_rule_program, prefix_rule):
    from egglog import rule, var  # noqa: PLC0415

    Seed, First, Second, first, _ = native_rule_program
    bad = rule(Seed(var("x", i64)), name="unbound-head").then(Second(var("unbound_y", i64)))
    graph = EGraph()
    if prefix_rule:
        graph.register(Seed(1))
    with pytest.raises(bindings.EggSmolError):
        graph.register(first if prefix_rule else Seed(1), bad, Second(2))
    graph.check(Seed(1))
    graph.check_fail(First(1))
    graph.check_fail(Second(2))
    if prefix_rule:
        assert _run_ruleset_step(graph) == {"first": 1}
        graph.check(First(1))
    graph.check_fail(Second(2))


def test_native_valid_mixed_registration_and_scope_memberships(native_rule_program):
    from egglog import ruleset  # noqa: PLC0415

    Seed, First, Second, first, second = native_rule_program
    graph = EGraph()
    graph.register(Seed(1), first, Second(1))
    assert _run_ruleset_step(graph) == {"first": 1}
    graph.check(First(1), Second(1))
    original_direct = graph._state._direct_rules
    graph.push()
    assert graph._state._direct_rules == original_direct
    assert graph._state._direct_rules is not original_direct
    graph.register(second)
    assert len(graph._state._direct_rules) == 2
    assert len(original_direct) == 1
    assert _run_ruleset_step(graph) == {"second": 1}
    graph.pop()
    assert graph._state._direct_rules is original_direct
    graph.register(Seed(2))
    assert _run_ruleset_step(graph) == {"first": 1}
    graph.check(First(2))
    graph.check_fail(Second(2))
    explicit = ruleset(second, name="explicit")
    assert _run_ruleset_step(graph, explicit) == {"second": 2}


def test_native_direct_unnamed_registration_creates_fresh_occurrences(native_rule_program):
    from egglog import rule, var  # noqa: PLC0415
    from egglog.declarations import Ident  # noqa: PLC0415

    Seed, First, _, _, _ = native_rule_program
    x = var("x", i64)
    authored = rule(Seed(x)).then(First(x))
    graph = EGraph()
    graph.register(Seed(1))
    graph.register(authored, authored)
    assert _run_ruleset_step(graph) == {"": 2}
    graph.register(authored)
    assert _run_ruleset_step(graph) == {"": 1}
    assert len(graph._state._ruleset_members(graph._state.rulesets[Ident("")])) == 3
    graph.check(First(1))


def test_native_direct_named_collision_keeps_prefix_and_error_shape(native_rule_program):
    Seed, First, Second, first, _ = native_rule_program
    graph = EGraph()
    graph.register(Seed(1))
    for prefix in [True, False]:
        with pytest.raises(bindings.EggSmolError) as raised:
            graph.register(first, first, Second(1)) if prefix else graph.register(first, Second(1))
        message = "At 0:0 of \nRule already exists: first"
        assert type(raised.value) is bindings.EggSmolError
        assert str(raised.value) == message
        assert raised.value.args == (message, False)
        assert raised.value.replayable_by_fail is False
        assert "_egglog_proto_error" not in raised.value.__dict__  # Frontend compatibility validation.
        assert _run_ruleset_step(graph) == ({"first": 1} if prefix else {})
        graph.check(First(1))
        graph.check_fail(Second(1))


@pytest.mark.parametrize("default", [False, True])
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("direct_first", [False, True])
def test_native_direct_and_materialized_memberships(native_rule_program, default, named, direct_first):
    from egglog import rule, ruleset, var  # noqa: PLC0415

    Seed, First, _, _, _ = native_rule_program
    selected = ruleset(name="" if default else "explicit")
    graph = EGraph()
    graph.register(Seed(1))
    assert _run_ruleset_step(graph, selected) == {}
    authored = rule(Seed(var("x", i64)), name="first" if named else None, ruleset=selected).then(First(var("x", i64)))
    label = "first" if named else ""
    if direct_first:
        graph.register(authored)
        if named and default:
            with pytest.raises(bindings.EggSmolError, match="Rule already exists: first"):
                _run_ruleset_step(graph, selected)
            assert _run_ruleset_step(graph) == {label: 1}
        else:
            assert _run_ruleset_step(graph, selected) == {label: 2 if default else 1}
    else:
        assert _run_ruleset_step(graph, selected) == {label: 1}
        if named and default:
            with pytest.raises(bindings.EggSmolError, match="Rule already exists: first"):
                graph.register(authored)
        else:
            graph.register(authored)
            assert _run_ruleset_step(graph, selected) == ({label: 1} if default else {})
    graph.check(First(1))


def test_native_publication_clone_before_and_after_append(native_rule_program):
    from egglog import ruleset  # noqa: PLC0415

    Seed, First, Second, first, second = native_rule_program
    selected = ruleset(first, name="cloned")
    graph = EGraph()
    graph.register(Seed(1))
    assert _run_ruleset_step(graph, selected) == {"first": 1}
    graph.push()
    selected.append(second)
    assert _run_ruleset_step(graph, selected) == {"second": 1}
    graph.pop()
    graph.check(First(1))
    graph.check_fail(Second(1))
    assert _run_ruleset_step(graph, selected) == {"second": 1}
    graph.push()
    assert _run_ruleset_step(graph, selected) == {}
    graph.pop()
    assert _run_ruleset_step(graph, selected) == {}


def test_native_publication_keeps_independent_compatible_declaration_owners():
    from egglog import rule, ruleset  # noqa: PLC0415

    left = constant("publication_owned_target", i64)
    right = constant("publication_owned_target", i64)
    assert left.__egg_ref__.owner is not right.__egg_ref__.owner
    selected = ruleset(rule(name="first").then(set_(left).to(1)), name="owners")
    graph = EGraph()
    assert _run_ruleset_step(graph, selected) == {"first": 1}
    selected.append(rule(name="second").then(set_(right).to(1)))
    assert _run_ruleset_step(graph, selected) == {"second": 1}
    # Packing has an explicit, pre-existing same-Program reconciliation guard.
    with pytest.raises(ValueError, match="Ambiguous same-name definitions"):
        graph.check(eq(left).to(1), eq(right).to(1))
    graph.check(eq(left).to(1))
    graph.check(eq(right).to(1))


def test_native_registration_materializes_generator_before_execution(native_rule_program):
    from egglog import rule  # noqa: PLC0415

    Seed, First, _, _, _ = native_rule_program
    graph = EGraph()
    graph.register(Seed(1))

    def broken(x: i64):
        yield rule(Seed(x)).then(First(x))
        msg = "authoring failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="authoring failed"):
        graph.register(broken)
    assert not graph._state.rulesets
    graph.check_fail(First(1))


def test_check_fail_authors_one_canonical_nested_command(monkeypatch):
    requests = []

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            requests.append(pb.RunProgramRequest.from_binary(encoded))
            return pb.RunProgramResponse(ruleset_publication=pb.RulesetPublication()).to_binary()

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    graph.check_fail(eq(i64(1)).to(i64(2)))
    program = requests[0].program
    assert len(program.commands) == 1
    assert program.commands[0].kind.field == "fail"
    check = program.commands[0].kind.value
    assert check.kind.field == "check"
    assert len(check.kind.value.facts) == 1
    fact = program.nodes[check.kind.value.facts[0]]
    assert fact.kind.field == "union"
    assert [program.nodes[index].kind.value.value.value for index in fact.kind.value.members] == [1, 2]
    graph.check_fail()
    assert requests[1].program.commands[0].kind.value.kind.value.facts == []


@pytest.mark.parametrize("operation", ["check", "check_bool", "check_fail"])
@pytest.mark.parametrize(
    ("code", "location"),
    [
        (pb.ErrorCode.CHECK_FAILED, pb.CommandLocation(path=[0])),
        (pb.ErrorCode.CHECK_FAILED, None),
        (pb.ErrorCode.CHECK_FAILED, pb.CommandLocation(path=[1])),
        (pb.ErrorCode.EVALUATION_FAILED, pb.CommandLocation(path=[0])),
        (pb.ErrorCode.PROOF_FAILED, pb.CommandLocation(path=[0])),
        (pb.ErrorCode.INVALID_PROGRAM, None),
    ],
)
def test_check_errors_use_codes_and_locations_not_messages(monkeypatch, operation, code, location):
    record = pb.Error(code=code, message="Check failed: same diagnostic", location=location)
    if code == pb.ErrorCode.PROOF_FAILED:
        record.proof_failure = pb.ProofFailure(reason=pb.ProofFailureReason.PROOFS_NOT_ENABLED)

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            return pb.RunProgramResponse(error=record, ruleset_publication=pb.RulesetPublication()).to_binary()

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    method = getattr(graph, operation)
    if (
        operation == "check_bool"
        and code == pb.ErrorCode.CHECK_FAILED
        and location is not None
        and location.path == [0]
    ):
        assert method(eq(i64(1)).to(i64(1))) is False
    else:
        with pytest.raises(bindings.EggSmolError) as caught:
            method(eq(i64(1)).to(i64(1)))
        assert type(caught.value) is bindings.EggSmolError
        assert caught.value.__dict__["_egglog_proto_error"] == record
        assert str(caught.value) == record.message
        assert caught.value.args == (record.message,)
        if operation == "check_bool":
            traceback = caught.value.__traceback__
            frames = []
            while traceback is not None:
                frames.append(traceback.tb_frame.f_code.co_name)
                traceback = traceback.tb_next
            assert frames.count("check_bool") == 1
            assert frames.count("run_program") == 1


@pytest.mark.parametrize("located", [False, True])
def test_execution_error_retains_only_confirmed_declaration_refs(monkeypatch, located):
    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            return pb.RunProgramResponse(
                error=pb.Error(
                    code=pb.ErrorCode.EVALUATION_FAILED,
                    message="installation or command failed",
                    location=pb.CommandLocation(path=[0]) if located else None,
                ),
                ruleset_publication=pb.RulesetPublication(),
            ).to_binary()

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    current = constant("failed_request_definition", i64)
    graph = EGraph()
    action = set_(current).to(i64(1))
    from egglog._program import _definition_key  # noqa: PLC0415

    expected = {
        _definition_key(reference.read()): reference
        for reference in pack(commands=[action.__egg_ref__]).definitions
        if reference.role == "declarations"
    }
    before = graph._state.definitions.copy()
    with pytest.raises(bindings.EggSmolError):
        graph.register(action)
    assert graph._state.definitions == (before | expected if located else before)


@pytest.mark.parametrize("error_type", [ValueError, bindings.EggSmolError])
@pytest.mark.parametrize("misleading_record", [False, True])
def test_check_bool_preserves_transport_exception_and_traceback(monkeypatch, error_type, misleading_record):
    sentinel = error_type("Check failed is only text")
    if misleading_record:
        sentinel.__dict__["_egglog_proto_error"] = {"code": pb.ErrorCode.CHECK_FAILED}

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            raise sentinel

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport)
    graph = EGraph()
    with pytest.raises(error_type) as caught:
        graph.check_bool(eq(i64(1)).to(i64(1)))
    assert caught.value is sentinel
    traceback = caught.value.__traceback__
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    assert traceback.tb_frame.f_code is InertTransport.run.__code__


def test_constant_actions_submit_one_ordered_binary_program(monkeypatch):
    requests = []

    class InertTransport:
        def create(self, encoded):
            pb.CreateEGraphRequest.from_binary(encoded)
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            requests.append(pb.RunProgramRequest.from_binary(encoded))
            return pb.RunProgramResponse(ruleset_publication=pb.RulesetPublication()).to_binary()

        def destroy(self, encoded):
            assert pb.DestroyEGraphRequest.from_binary(encoded).egraph_id == 1
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport, raising=False)
    current = constant("current", i64)
    later = constant("later", i64)
    graph = EGraph()
    graph.register(set_(current).to(i64(1)), current, set_(later).to(current + 2), later)
    assert len(requests) == 1
    program = requests[0].program
    assert [command.kind.value.kind.field for command in program.commands] == ["set", "term", "set", "term"]
    first = program.commands[0].kind.value.kind.value
    second = program.commands[2].kind.value.kind.value
    assert program.nodes[program.commands[1].kind.value.kind.value].kind.value.func == first.target.func
    assert program.nodes[program.commands[3].kind.value.kind.value].kind.value.func == second.target.func
    definitions = {declaration.kind.value.name: declaration for declaration in program.declarations}
    assert definitions[first.target.func].kind.field == "function"
    assert definitions[second.target.func].bindings.python.views[0].kind == pb.PythonCallKind.CONSTANT
    rhs = program.nodes[second.value].kind.value
    assert rhs.func == "egglog.core.i64.add"
    assert program.nodes[rhs.args[0]].kind.value.func == first.target.func
    with pytest.raises(NotImplementedError, match="Fact-to-action"):
        graph.register(set_(current).to(i64(1)), eq(i64(1)).to(i64(1)))
    assert len(requests) == 1
    graph._state.destroy()


def test_response_owner_keeps_nominal_and_ambient_family_contexts(monkeypatch):
    family = builtin_catalog().definitions["host_sort_family", "Pair"]

    class InertTransport:
        def create(self, encoded):
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded):
            request = pb.RunProgramRequest.from_binary(encoded)
            assert any(d.kind.field == "eq_sort" and d.kind.value.name == "Pair" for d in request.program.declarations)
            return pb.RunProgramResponse(
                ruleset_publication=pb.RulesetPublication(),
                sorts=[
                    pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Pair")),
                    pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))),
                ],
                nodes=[
                    pb.Node(sort_id=0, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="P"))),
                    pb.Node(
                        sort_id=1,
                        kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                            "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 1))
                        ),
                    ),
                ],
                outputs=[
                    pb.CommandOutput(
                        kind=Oneof[Literal["extraction"], pb.ExtractResult](
                            "extraction",
                            pb.ExtractResult(roots=[pb.ExtractedRoot(variants=[pb.ExtractedTerm(term=0, cost=1)])]),
                        ),
                        location=pb.CommandLocation(path=[0]),
                    )
                ],
            ).to_binary()

        def destroy(self, encoded):
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport, raising=False)

    class NominalPair(Expr, egg_sort="Pair"):
        @method(egg_fn="P")
        def __init__(self) -> None: ...

        @method(preserve=True)
        def tag(self):
            return "nominal hook"

    graph = EGraph()
    result = graph.extract(NominalPair())
    assert graph._state.definitions["host_sort_family", "Pair"] == family
    assert graph._state.definitions["eq_sort", "Pair"] == NominalPair.__egg_definition__
    assert result.tag() == "nominal hook"
    assert str(result) == "NominalPair()"
    graph._state.destroy()
    assert result.tag() == "nominal hook"


def test_public_bytes_requests_and_response_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[pb.RunProgramRequest] = []
    destroyed: list[int] = []

    class InertTransport:
        def create(self, encoded: bytes) -> bytes:
            assert isinstance(encoded, bytes)
            request = pb.CreateEGraphRequest.from_binary(encoded)
            assert request.has_field("threads")
            assert request.threads == 1
            assert not request.declarations
            assert request.options is not None
            assert request.sorts[request.options.cost_sort].kind.value.name == "i64"
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def run(self, encoded: bytes) -> bytes:
            assert isinstance(encoded, bytes)
            request = pb.RunProgramRequest.from_binary(encoded)
            requests.append(request)
            # Fixed inert reply: no interpretation or computation of the request.
            return pb.RunProgramResponse(
                ruleset_publication=pb.RulesetPublication(),
                sorts=[pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))],
                nodes=[
                    pb.Node(
                        kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                            "primitive_value",
                            pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 31)),
                        )
                    )
                ],
                outputs=[
                    pb.CommandOutput(
                        kind=Oneof[Literal["extraction"], pb.ExtractResult](
                            "extraction",
                            pb.ExtractResult(roots=[pb.ExtractedRoot(variants=[pb.ExtractedTerm(term=0, cost=0)])]),
                        ),
                        location=pb.CommandLocation(path=[0]),
                    )
                ],
            ).to_binary()

        def clone_graph(self, encoded: bytes) -> bytes:
            assert pb.CloneEGraphRequest.from_binary(encoded).egraph_id == 1
            return pb.CloneEGraphResponse(egraph_id=2).to_binary()

        def return_to(self, encoded: bytes) -> bytes:
            request = pb.ReturnToEGraphRequest.from_binary(encoded)
            assert (request.active_id, request.saved_id) == (2, 1)
            return pb.ReturnToEGraphResponse().to_binary()

        def destroy(self, encoded: bytes) -> bytes:
            destroyed.append(pb.DestroyEGraphRequest.from_binary(encoded).egraph_id)
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", InertTransport, raising=False)
    graph = EGraph()
    result, cost = graph.extract(i64(1) + 2, include_cost=True)
    assert result.value == cost == 31
    request = requests[0]
    assert request.program is not None
    program = request.program
    assert program.ir_version == 1
    command = program.commands[0].kind
    assert command is not None
    call = program.nodes[command.value.roots[0]].kind
    assert call is not None
    assert call.field == "call"
    assert call.value.func == "egglog.core.i64.add"
    assert [program.nodes[index].kind.value.value.value for index in call.value.args] == [1, 2]
    add = next(declaration for declaration in program.declarations if declaration.kind.value.name == call.value.func)
    assert add.bindings is not None
    assert add.bindings.python is not None
    assert add.bindings.python.views[0].path == ["__add__"]
    family = next(declaration for declaration in program.declarations if declaration.kind.value.name == "i64")
    assert family.kind.value.bindings.python.path == ["egglog", "builtins", "i64"]
    # Re-authoring from a returned value retains the original catalog refs,
    # not a second declaration copy reconstructed from the response.
    repacked = pack([(result + 4).__egg_ref__])
    assert builtin_catalog().definitions["callable", "egglog.core.i64.add"] in repacked.definitions
    with graph:
        assert graph._state.handle == 2
    assert graph._state.handle == 1
    assert destroyed == []
    graph.close()
    assert graph._state.handle == 1
    graph._state.destroy()
    graph._state.destroy()
    assert destroyed == [1]


@pytest.mark.parametrize("body_raises", [False, True])
def test_context_keeps_active_and_saved_state_when_return_rejected(
    monkeypatch: pytest.MonkeyPatch, body_raises: bool
) -> None:
    class FaultTransport:
        def create(self, encoded: bytes) -> bytes:
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def clone_graph(self, encoded: bytes) -> bytes:
            return pb.CloneEGraphResponse(egraph_id=2).to_binary()

        def return_to(self, encoded: bytes) -> bytes:
            request = pb.ReturnToEGraphRequest.from_binary(encoded)
            assert (request.active_id, request.saved_id) == (2, 1)
            msg = "cleanup sentinel"
            raise RuntimeError(msg)

        def destroy(self, encoded: bytes) -> bytes:
            return pb.DestroyEGraphResponse().to_binary()

    monkeypatch.setattr(bindings, "_ProtoEngine", FaultTransport, raising=False)
    graph = EGraph()
    saved = graph._state
    with (  # noqa: PT012 -- exercise body and disposal together
        pytest.raises(
            ValueError if body_raises else RuntimeError, match="body sentinel" if body_raises else "cleanup sentinel"
        ),
        graph,
    ):
        assert graph._state is not saved
        if body_raises:
            msg = "body sentinel"
            raise ValueError(msg)
    assert graph._state is not saved
    assert graph._state.handle == 2
    assert saved.handle == 1
    assert graph._state_stack == [saved]


class ResourceTransport:
    """An inert bytes receiver, not a simulation of native resource allocation."""

    def __init__(self) -> None:
        self.creations: list[pb.CreateEGraphRequest] = []
        self.requests: list[pb.ConfigureEGraphResourcesRequest] = []
        self.threads: dict[int, int] = {}
        self.error: ValueError | None = None

    def create(self, encoded: bytes) -> bytes:
        request = pb.CreateEGraphRequest.from_binary(encoded)
        self.creations.append(request)
        handle = len(self.creations)
        self.threads[handle] = request.threads or 7
        return pb.CreateEGraphResponse(egraph_id=handle).to_binary()

    def configure_resources(self, encoded: bytes) -> bytes:
        assert isinstance(encoded, bytes)
        request = pb.ConfigureEGraphResourcesRequest.from_binary(encoded)
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        assert request.operation is not None
        if request.operation.field == "threads":
            self.threads[request.egraph_id] = request.operation.value or 7
        else:
            assert request.operation.field == "query"
            assert isinstance(request.operation.value, pb.Unit)
        return pb.ConfigureEGraphResourcesResponse(threads=self.threads[request.egraph_id]).to_binary()

    def clone_graph(self, encoded: bytes) -> bytes:
        request = pb.CloneEGraphRequest.from_binary(encoded)
        handle = max(self.threads) + 1
        self.threads[handle] = self.threads[request.egraph_id]
        return pb.CloneEGraphResponse(egraph_id=handle).to_binary()

    def destroy(self, encoded: bytes) -> bytes:
        del self.threads[pb.DestroyEGraphRequest.from_binary(encoded).egraph_id]
        return pb.DestroyEGraphResponse().to_binary()

    def return_to(self, encoded: bytes) -> bytes:
        request = pb.ReturnToEGraphRequest.from_binary(encoded)
        del self.threads[request.active_id]
        assert request.saved_id in self.threads
        return pb.ReturnToEGraphResponse().to_binary()


@pytest.fixture
def resource_transport(monkeypatch: pytest.MonkeyPatch) -> ResourceTransport:
    transport = ResourceTransport()
    monkeypatch.setattr(bindings, "_ProtoEngine", lambda: transport, raising=False)
    return transport


@pytest.mark.parametrize("use_setter", [False, True], ids=["constructor", "setter"])
def test_zero_threads_bytes_keep_presence_and_read_actual_count(
    resource_transport: ResourceTransport, *, use_setter: bool
) -> None:
    graph = EGraph(num_threads=1 if use_setter else 0)
    if use_setter:
        assert graph.set_num_threads(0) is None
        assert resource_transport.requests[0].operation == Oneof[Literal["threads"], int]("threads", 0)
    else:
        assert resource_transport.creations[0].has_field("threads")
        assert resource_transport.creations[0].threads == 0
    assert graph.num_threads() == 7
    assert resource_transport.requests[-1].operation == Oneof[Literal["query"], pb.Unit]("query", pb.Unit())


def test_resource_queries_are_uncached_uint64_and_updates_are_handle_local(
    resource_transport: ResourceTransport,
) -> None:
    graph = EGraph(num_threads=2)
    assert graph.num_threads() == 2
    with graph:
        assert graph.num_threads() == 2
        assert graph.set_num_threads(1) is None
        assert graph.num_threads() == 1
        assert resource_transport.requests[-1].egraph_id == 2
    assert graph.num_threads() == 2
    assert resource_transport.requests[-1].egraph_id == 1
    resource_transport.threads[1] = (1 << 32) + 9
    assert graph.num_threads() == (1 << 32) + 9
    graph._state.destroy()
    calls = len(resource_transport.requests)
    with pytest.raises(RuntimeError, match="destroyed"):
        graph.num_threads()
    with pytest.raises(RuntimeError, match="destroyed"):
        graph.set_num_threads(1)
    assert len(resource_transport.requests) == calls


class IndexedThreadCount:
    def __index__(self) -> int:
        return 2


class IntOnlyThreadCount:
    def __int__(self) -> int:
        return 2


@pytest.mark.parametrize("value", [IndexedThreadCount(), True, False, (1 << 32) - 1])
@pytest.mark.parametrize("use_setter", [False, True], ids=["constructor", "setter"])
def test_thread_count_accepts_index_protocol(
    resource_transport: ResourceTransport, value: object, *, use_setter: bool
) -> None:
    graph = EGraph(num_threads=1 if use_setter else value)
    if use_setter:
        graph.set_num_threads(value)
        assert resource_transport.requests[0].operation.value == index(value)
    else:
        assert resource_transport.creations[0].threads == index(value)
    assert graph.num_threads() == (index(value) or 7)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (-1, OverflowError),
        (1 << 32, OverflowError),
        (1 << 64, OverflowError),
        (None, TypeError),
        (1.0, TypeError),
        ("1", TypeError),
        (IntOnlyThreadCount(), TypeError),
    ],
)
@pytest.mark.parametrize("use_setter", [False, True], ids=["constructor", "setter"])
def test_invalid_thread_count_has_no_transport_effect(
    resource_transport: ResourceTransport, value: object, error: type[Exception], *, use_setter: bool
) -> None:
    graph = EGraph(num_threads=2) if use_setter else None
    creations = len(resource_transport.creations)
    if graph is None:
        with pytest.raises(error):
            EGraph(num_threads=value)
    else:
        with pytest.raises(error):
            graph.set_num_threads(value)
    assert len(resource_transport.creations) == creations
    assert not resource_transport.requests
    if graph is not None:
        assert graph.num_threads() == 2


@pytest.mark.parametrize("use_setter", [False, True], ids=["query", "setter"])
def test_resource_transport_failure_does_not_manufacture_success(
    resource_transport: ResourceTransport, *, use_setter: bool
) -> None:
    graph = EGraph(num_threads=2)
    resource_transport.error = ValueError("resource sentinel")
    if use_setter:
        with pytest.raises(ValueError, match="resource sentinel"):
            graph.set_num_threads(1)
    else:
        with pytest.raises(ValueError, match="resource sentinel"):
            graph.num_threads()
    resource_transport.error = None
    assert graph.num_threads() == 2


def test_resource_response_requires_positive_actual_count(resource_transport: ResourceTransport) -> None:
    graph = EGraph()
    resource_transport.threads[1] = 0
    with pytest.raises(ValueError, match="thread count"):
        graph.num_threads()
