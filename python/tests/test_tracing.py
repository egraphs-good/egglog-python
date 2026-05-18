from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from unittest.mock import patch

from egglog_pytest_otel import DEFAULT_OTLP_ENDPOINT, configure_pytest_otel, get_pytest_otel_config


def _console_tracing_script(body: str, *, extra_imports: str = "") -> str:
    return "\n\n".join(
        part
        for part in (
            textwrap.dedent(
                """
                from opentelemetry import trace
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

                from egglog import bindings

                provider = TracerProvider(resource=Resource.create({"service.name": "egglog"}))
                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
                trace.set_tracer_provider(provider)
                bindings.setup_tracing(exporter="console")
                """
            ).strip(),
            textwrap.dedent(extra_imports).strip(),
            textwrap.dedent(body).strip(),
            textwrap.dedent(
                """
                bindings.shutdown_tracing()
                provider.shutdown()
                """
            ).strip(),
        )
        if part
    )


HIGH_LEVEL_TRACE_SCRIPT = _console_tracing_script(
    """
    from egglog import EGraph, i64

    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("parent"):
        EGraph().extract(i64(0))
    """
)

LOW_LEVEL_TRACE_SCRIPT = _console_tracing_script(
    """
    tracer = trace.get_tracer("test")
    egraph = bindings.EGraph()

    def current_headers():
        carrier = {}
        propagate.inject(carrier)
        return carrier.get("traceparent"), carrier.get("tracestate")

    with tracer.start_as_current_span("run_program_parent"):
        traceparent, tracestate = current_headers()
        egraph.run_program(bindings.Push(1), traceparent=traceparent, tracestate=tracestate)

    lit = bindings.Lit(bindings.PanicSpan(), bindings.Int(0))
    with tracer.start_as_current_span("eval_expr_parent"):
        traceparent, tracestate = current_headers()
        sort, value = egraph.eval_expr(lit, traceparent=traceparent, tracestate=tracestate)

    with tracer.start_as_current_span("serialize_parent"):
        traceparent, tracestate = current_headers()
        egraph.serialize([], traceparent=traceparent, tracestate=tracestate)

    cost_model = bindings.CostModel(
        lambda head, head_cost, children_costs: head_cost + sum(children_costs),
        lambda func, args: 1,
        lambda sort, value, element_costs: 1,
        lambda sort, value: 1,
    )
    with tracer.start_as_current_span("extractor_new_parent"):
        traceparent, tracestate = current_headers()
        extractor = bindings.Extractor([sort], egraph, cost_model, traceparent=traceparent, tracestate=tracestate)

    with tracer.start_as_current_span("extract_best_parent"):
        traceparent, tracestate = current_headers()
        extractor.extract_best(egraph, bindings.TermDag(), value, sort, traceparent=traceparent, tracestate=tracestate)
    """,
    extra_imports="from opentelemetry import propagate",
)

SETUP_STATE_MACHINE_SCRIPT = textwrap.dedent(
    """
    from egglog import bindings

    bindings.setup_tracing(exporter="console")
    bindings.setup_tracing(exporter="console")

    try:
        bindings.setup_tracing(exporter="http", endpoint="http://127.0.0.1:4318/v1/traces")
    except RuntimeError:
        print("reconfigure_error")
    else:
        raise SystemExit("expected reconfigure error")

    egraph = bindings.EGraph()
    egraph.run_program(bindings.Push(1))
    bindings.shutdown_tracing()

    try:
        bindings.setup_tracing(exporter="console")
    except RuntimeError:
        print("shutdown_error")
    else:
        raise SystemExit("expected shutdown error")
    """
)

HTTP_SETUP_SCRIPT = textwrap.dedent(
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from egglog import bindings

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("content-length", "0"))
            if length:
                self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Connection", "close")
            self.end_headers()

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/traces"
        bindings.setup_tracing(exporter="http", endpoint=endpoint)
        bindings.shutdown_tracing()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    print("http_ok")
    """
)

HTTP_MISSING_ENDPOINT_SCRIPT = textwrap.dedent(
    """
    from egglog import bindings

    try:
        bindings.setup_tracing(exporter="http")
    except RuntimeError:
        print("missing_endpoint")
    else:
        raise SystemExit("expected missing endpoint error")
    """
)


class DummyConfig:
    def __init__(self, *, traces: str, endpoint: str | None = None) -> None:
        self._options = {
            "--otel-otlp-endpoint": endpoint,
            "--otel-traces": traces,
        }

    def getoption(self, name: str):
        return self._options[name]


def _run_script(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=True,
        text=True,
    )


def _parse_python_span(stdout: str, name: str) -> tuple[str, str, str | None]:
    match = re.search(
        rf'"name": "{re.escape(name)}".*?"trace_id": "0x([0-9a-f]+)".*?"span_id": "0x([0-9a-f]+)".*?"parent_id": (null|"0x([0-9a-f]+)")',
        stdout,
        re.DOTALL,
    )
    assert match is not None
    return match.group(1), match.group(2), match.group(4)


def _parse_rust_span(stdout: str, name: str) -> tuple[str, str | None]:
    match = re.search(
        rf"Name\s*: {re.escape(name)}\s+TraceId\s*: ([0-9a-f]+)\s+SpanId\s*: [0-9a-f]+\s+TraceFlags\s*: .*?\s+ParentSpanId: ([0-9a-f]+|None)",
        stdout,
        re.DOTALL,
    )
    assert match is not None
    parent_span_id = None if match.group(2) == "None" else match.group(2)
    return match.group(1), parent_span_id


def test_get_pytest_otel_config_defaults_to_off() -> None:
    config = get_pytest_otel_config(DummyConfig(traces="off"))
    assert config.traces == "off"
    assert config.endpoint is None


def test_get_pytest_otel_config_uses_default_jaeger_endpoint() -> None:
    config = get_pytest_otel_config(DummyConfig(traces="jaeger"))
    assert config.traces == "jaeger"
    assert config.endpoint == DEFAULT_OTLP_ENDPOINT


def test_get_pytest_otel_config_preserves_explicit_endpoint() -> None:
    config = get_pytest_otel_config(DummyConfig(traces="jaeger", endpoint="http://127.0.0.1:9999/v1/traces"))
    assert config.traces == "jaeger"
    assert config.endpoint == "http://127.0.0.1:9999/v1/traces"


def test_configure_pytest_otel_console_uses_bindings_setup() -> None:
    with (
        patch("egglog.bindings.setup_tracing") as setup_tracing,
        patch("opentelemetry.trace.set_tracer_provider"),
    ):
        provider = configure_pytest_otel(DummyConfig(traces="console"))

    assert provider is not None
    setup_tracing.assert_called_once_with(exporter="console")


def test_configure_pytest_otel_jaeger_uses_http_setup() -> None:
    with (
        patch("egglog.bindings.setup_tracing") as setup_tracing,
        patch("opentelemetry.trace.set_tracer_provider"),
    ):
        provider = configure_pytest_otel(DummyConfig(traces="jaeger"))

    assert provider is not None
    setup_tracing.assert_called_once_with(exporter="http", endpoint=DEFAULT_OTLP_ENDPOINT)


def test_bindings_setup_state_machine() -> None:
    result = _run_script(SETUP_STATE_MACHINE_SCRIPT)
    assert "bindings.run_program" in result.stdout
    assert "reconfigure_error" in result.stdout
    assert "shutdown_error" in result.stdout


def test_bindings_http_setup_requires_endpoint() -> None:
    result = _run_script(HTTP_MISSING_ENDPOINT_SCRIPT)
    assert "missing_endpoint" in result.stdout


def test_bindings_http_setup_succeeds_with_explicit_endpoint() -> None:
    result = _run_script(HTTP_SETUP_SCRIPT)
    assert "http_ok" in result.stdout


def test_console_export_smoke_uses_high_level_wrapper() -> None:
    result = _run_script(HIGH_LEVEL_TRACE_SCRIPT)
    assert '"name": "parent"' in result.stdout
    assert '"name": "extract"' in result.stdout
    assert "bindings.parse_and_run_program" in result.stdout

    parent_trace_id, parent_span_id, parent_parent_id = _parse_python_span(result.stdout, "parent")
    extract_trace_id, extract_span_id, extract_parent_id = _parse_python_span(result.stdout, "extract")
    rust_trace_id, rust_parent_id = _parse_rust_span(result.stdout, "bindings.parse_and_run_program")

    assert parent_parent_id is None
    assert extract_trace_id == parent_trace_id
    assert extract_parent_id == parent_span_id
    assert rust_trace_id == parent_trace_id
    assert rust_parent_id == extract_span_id


def test_low_level_explicit_context_propagates_to_rust_spans() -> None:
    result = _run_script(LOW_LEVEL_TRACE_SCRIPT)

    run_program_trace_id, run_program_span_id, _ = _parse_python_span(result.stdout, "run_program_parent")
    rust_run_trace_id, rust_run_parent_id = _parse_rust_span(result.stdout, "bindings.run_program")
    assert rust_run_trace_id == run_program_trace_id
    assert rust_run_parent_id == run_program_span_id

    eval_trace_id, eval_span_id, _ = _parse_python_span(result.stdout, "eval_expr_parent")
    rust_eval_trace_id, rust_eval_parent_id = _parse_rust_span(result.stdout, "bindings.eval_expr")
    assert rust_eval_trace_id == eval_trace_id
    assert rust_eval_parent_id == eval_span_id

    serialize_trace_id, serialize_span_id, _ = _parse_python_span(result.stdout, "serialize_parent")
    rust_serialize_trace_id, rust_serialize_parent_id = _parse_rust_span(result.stdout, "bindings.serialize")
    assert rust_serialize_trace_id == serialize_trace_id
    assert rust_serialize_parent_id == serialize_span_id

    extractor_new_trace_id, extractor_new_span_id, _ = _parse_python_span(result.stdout, "extractor_new_parent")
    rust_extractor_new_trace_id, rust_extractor_new_parent_id = _parse_rust_span(
        result.stdout, "bindings.extractor.new"
    )
    assert rust_extractor_new_trace_id == extractor_new_trace_id
    assert rust_extractor_new_parent_id == extractor_new_span_id

    extract_best_trace_id, extract_best_span_id, _ = _parse_python_span(result.stdout, "extract_best_parent")
    rust_extract_best_trace_id, rust_extract_best_parent_id = _parse_rust_span(
        result.stdout, "bindings.extractor.extract_best"
    )
    assert rust_extract_best_trace_id == extract_best_trace_id
    assert rust_extract_best_parent_id == extract_best_span_id
