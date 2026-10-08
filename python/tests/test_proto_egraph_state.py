"""Record/bytes plumbing tests; the inert responder is not engine conformance."""

from operator import index
from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import EGraph, bindings, i64
from egglog._catalog import builtin_catalog
from egglog._program import pack


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
    assert destroyed == [2]
    graph.close()
    assert graph._state.handle == 1
    graph._state.destroy()
    graph._state.destroy()
    assert destroyed == [2, 1]


@pytest.mark.parametrize("body_raises", [False, True])
def test_context_restores_saved_state_when_clone_disposal_fails(
    monkeypatch: pytest.MonkeyPatch, body_raises: bool
) -> None:
    class FaultTransport:
        def create(self, encoded: bytes) -> bytes:
            return pb.CreateEGraphResponse(egraph_id=1).to_binary()

        def clone_graph(self, encoded: bytes) -> bytes:
            return pb.CloneEGraphResponse(egraph_id=2).to_binary()

        def destroy(self, encoded: bytes) -> bytes:
            if pb.DestroyEGraphRequest.from_binary(encoded).egraph_id == 2:
                msg = "cleanup sentinel"
                raise RuntimeError(msg)
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
    assert graph._state is saved
    assert not graph._state_stack


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
