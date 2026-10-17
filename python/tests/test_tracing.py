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

                provider = TracerProvider(resource=Resource.create({"service.name": "egglog"}))
                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
                trace.set_tracer_provider(provider)
                """
            ).strip(),
            textwrap.dedent(extra_imports).strip(),
            textwrap.dedent(body).strip(),
            textwrap.dedent(
                """
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


def test_configure_pytest_otel_console_uses_python_provider() -> None:
    with (
        patch("opentelemetry.trace.set_tracer_provider") as set_provider,
    ):
        provider = configure_pytest_otel(DummyConfig(traces="console"))

    assert provider is not None
    set_provider.assert_called_once_with(provider)


def test_configure_pytest_otel_jaeger_uses_python_provider() -> None:
    with (
        patch("opentelemetry.trace.set_tracer_provider") as set_provider,
    ):
        provider = configure_pytest_otel(DummyConfig(traces="jaeger"))

    assert provider is not None
    set_provider.assert_called_once_with(provider)


def test_console_export_smoke_uses_high_level_wrapper() -> None:
    result = _run_script(HIGH_LEVEL_TRACE_SCRIPT)
    assert '"name": "parent"' in result.stdout
    assert '"name": "extract"' in result.stdout

    parent_trace_id, parent_span_id, parent_parent_id = _parse_python_span(result.stdout, "parent")
    extract_trace_id, _, extract_parent_id = _parse_python_span(result.stdout, "extract")

    assert parent_parent_id is None
    assert extract_trace_id == parent_trace_id
    assert extract_parent_id == parent_span_id
