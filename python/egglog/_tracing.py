from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from opentelemetry import trace

_R = TypeVar("_R")


def call_with_current_trace(fn: Callable[..., _R], /, *args: Any, **kwargs: Any) -> _R:
    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return fn(*args, **kwargs)

    trace_kwargs = {
        "traceparent": (
            f"00-{span_context.trace_id:032x}-{span_context.span_id:016x}-{int(span_context.trace_flags):02x}"
        )
    }
    tracestate = span_context.trace_state.to_header()
    if tracestate:
        trace_kwargs["tracestate"] = tracestate
    return fn(*args, **kwargs, **trace_kwargs)
