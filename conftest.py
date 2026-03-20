import os
import pathlib
import sys
from dataclasses import dataclass
from importlib import import_module
from typing import Any

ROOT_DIR = pathlib.Path(__file__).parent

# So that it finds the local typings
os.environ["MYPYPATH"] = str(ROOT_DIR / "python")


# Add mypy's test searcher so we can write test files to verify type checking
pytest_plugins = ["mypy.test.data"]
# Set this to the root directory so it finds the `test-data` directory
os.environ["MYPY_TEST_PREFIX"] = str(ROOT_DIR)

DEFAULT_OTLP_ENDPOINT = "http://127.0.0.1:4318/v1/traces"
SERVICE_NAME = "egglog"

sys.modules.setdefault("egglog_pytest_otel", sys.modules[__name__])


@dataclass(frozen=True)
class PytestOtelConfig:
    endpoint: str | None
    traces: str


def add_pytest_otel_options(parser: Any) -> None:
    group = parser.getgroup("egglog-otel")
    group.addoption(
        "--otel-traces",
        action="store",
        default="off",
        choices=("off", "console", "jaeger"),
        help="Export egglog traces during tests.",
    )
    group.addoption(
        "--otel-otlp-endpoint",
        action="store",
        default=None,
        help="OTLP/HTTP traces endpoint used when --otel-traces=jaeger.",
    )


def get_pytest_otel_config(config: Any) -> PytestOtelConfig:
    traces = config.getoption("--otel-traces")
    endpoint = config.getoption("--otel-otlp-endpoint")
    if traces == "jaeger" and not endpoint:
        endpoint = DEFAULT_OTLP_ENDPOINT
    return PytestOtelConfig(traces=traces, endpoint=endpoint)


def configure_pytest_otel(config: Any):
    otel_config = get_pytest_otel_config(config)
    if otel_config.traces == "off":
        return None

    trace = import_module("opentelemetry.trace")
    OTLPSpanExporter = import_module("opentelemetry.exporter.otlp.proto.http.trace_exporter").OTLPSpanExporter
    Resource = import_module("opentelemetry.sdk.resources").Resource
    TracerProvider = import_module("opentelemetry.sdk.trace").TracerProvider
    trace_export = import_module("opentelemetry.sdk.trace.export")
    BatchSpanProcessor = trace_export.BatchSpanProcessor
    ConsoleSpanExporter = trace_export.ConsoleSpanExporter
    SimpleSpanProcessor = trace_export.SimpleSpanProcessor

    bindings = import_module("egglog.bindings")

    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    if otel_config.traces == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    else:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otel_config.endpoint)))
    trace.set_tracer_provider(provider)
    if otel_config.traces == "console":
        bindings.setup_tracing(exporter="console")
    else:
        bindings.setup_tracing(exporter="http", endpoint=otel_config.endpoint)
    return provider


def pytest_addoption(parser):
    add_pytest_otel_options(parser)


def pytest_configure(config):
    provider = configure_pytest_otel(config)
    if provider is not None:
        config._egglog_otel_provider = provider


def pytest_unconfigure(config):
    provider = getattr(config, "_egglog_otel_provider", None)
    if provider is not None:
        import_module("egglog.bindings").shutdown_tracing()
        provider.shutdown()
