"""
Bytes-only EGraph transport and response ownership.

No expression, declaration, or command is reconstructed from Python semantic
dataclasses here. Language lookup indexes retain canonical declaration Refs;
installation compatibility and idempotence belong to the shared adapter.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress
from operator import index
from typing import Literal, SupportsIndex, cast

from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from . import bindings
from ._catalog import builtin_catalog
from ._program import Builder, Owner, Ref, _definition_key, pack

__all__ = ["EGraphState"]

_QUERY_RESOURCES = object()


def _thread_count(value: object) -> int:
    """Keep Python's integer-index acceptance within the protocol's uint32 domain."""
    count = index(cast("SupportsIndex", value))
    if count < 0:
        msg = "can't convert negative int to unsigned"
        raise OverflowError(msg)
    if count > (1 << 32) - 1:
        msg = "thread count exceeds the protocol's uint32 range"
        raise OverflowError(msg)
    return count


class EGraphState:
    """Own one native handle and the declaration refs needed to read responses."""

    def __init__(self, *, num_threads: int = 1) -> None:
        num_threads = _thread_count(num_threads)
        self.transport = bindings._ProtoEngine()
        # The public default cost sort is an ambient engine builtin. Creation
        # references it directly; authored Programs still send full declarations.
        request = pb.CreateEGraphRequest(
            sorts=[pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))],
            options=pb.EGraphOptions(cost_sort=0),
            threads=num_threads,
        )
        response = pb.CreateEGraphResponse.from_binary(self.transport.create(request.to_binary()))
        self.handle: int | None = response.egraph_id
        self.definitions = dict(builtin_catalog().definitions)

    def configure_resources(self, num_threads: object = _QUERY_RESOURCES) -> int:
        """Query/update this handle over bytes and return the receiver's actual count."""
        if self.handle is None:
            msg = "EGraph handle has been destroyed"
            raise RuntimeError(msg)
        operation = (
            Oneof[Literal["query"], pb.Unit]("query", pb.Unit())
            if num_threads is _QUERY_RESOURCES
            else Oneof[Literal["threads"], int]("threads", _thread_count(num_threads))
        )
        request = pb.ConfigureEGraphResourcesRequest(egraph_id=self.handle, operation=operation)
        response = pb.ConfigureEGraphResourcesResponse.from_binary(
            self.transport.configure_resources(request.to_binary())
        )
        if response.threads < 1:
            msg = "Resource response must contain a positive actual thread count"
            raise ValueError(msg)
        return response.threads

    def run_program(self, commands: Iterable[Ref]) -> tuple[pb.RunProgramResponse, Owner]:
        """Pack canonical command closure once and adopt the returned records."""
        if self.handle is None:
            msg = "EGraph handle has been destroyed"
            raise RuntimeError(msg)
        packed = pack(commands=commands)
        request = pb.RunProgramRequest(egraph_id=self.handle, program=packed.program, profile=True)
        response = pb.RunProgramResponse.from_binary(self.transport.run(request.to_binary()))
        if response.error is not None:
            raise bindings.EggSmolError(response.error.message)
        for reference in packed.definitions:
            key = _definition_key(reference.read())
            if key is not None and reference.role == "declarations":
                self.definitions[key] = reference
        builder = Builder.from_program(
            pb.Program(
                ir_version=1,
                nodes=response.nodes,
                sorts=response.sorts,
                files=response.files,
            )
        )
        for reference in self.definitions.values():
            builder.import_ref(reference)
        return response, builder.publish()

    def copy(self) -> EGraphState:
        """Clone the native handle while sharing immutable authored definitions."""
        if self.handle is None:
            msg = "EGraph handle has been destroyed"
            raise RuntimeError(msg)
        response = pb.CloneEGraphResponse.from_binary(
            self.transport.clone_graph(pb.CloneEGraphRequest(egraph_id=self.handle).to_binary())
        )
        result = object.__new__(EGraphState)
        result.transport = self.transport
        result.handle = response.egraph_id
        result.definitions = self.definitions.copy()
        return result

    def destroy(self) -> None:
        """Release this handle once without affecting cloned graph handles."""
        if self.handle is not None:
            pb.DestroyEGraphResponse.from_binary(
                self.transport.destroy(pb.DestroyEGraphRequest(egraph_id=self.handle).to_binary())
            )
            self.handle = None

    def __del__(self) -> None:
        # A failed constructor may have no handle. Never let native cleanup
        # replace an exception already propagating during Python disposal.
        if getattr(self, "handle", None) is not None:
            with suppress(Exception):
                self.destroy()
