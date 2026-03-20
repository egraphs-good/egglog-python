# Tracing

`egglog` can emit OpenTelemetry spans from both the high-level Python wrapper and the Rust bindings.
The Python package stays library-style: it only depends on `opentelemetry-api`, and it starts emitting Python spans
once your application configures an OpenTelemetry tracer provider.

The Rust side uses the current Python trace context when one exists, so Rust spans can appear under the same parent
trace. To export Rust spans, call `egglog.bindings.setup_tracing(...)` before the traced Rust calls:

- `exporter="console"` writes Rust spans to stdout.
- `exporter="http"` sends Rust spans to an OTLP/HTTP endpoint.

For the contributor-oriented pytest workflow, see {doc}`../reference/contributing`.

## Trace A Host Application

This example configures Python tracing in an application that happens to call into `egglog`.
The Python spans come from the configured tracer provider, and the Rust spans join the same trace because the
bindings propagate the current `traceparent` and `tracestate` into the Rust tracing layer.

```python
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from egglog import EGraph, bindings, i64

provider = TracerProvider(resource=Resource.create({"service.name": "demo-app"}))
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
bindings.setup_tracing(exporter="console")

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("optimize"):
    EGraph().extract(i64(0))

bindings.shutdown_tracing()
provider.shutdown()
```

In that setup:

- Python spans use the module tracer names such as `egglog.egraph` and `egglog.egraph_state`.
- Python span names are the public method names such as `create`, `push`, `pop`, `register`, `run`, and `extract`, plus `run_schedule_to_egg` while schedules are lowered.
- Rust spans use names such as `bindings.run_program`, `bindings.serialize`, and `bindings.extractor.extract_best`.

If you call the low-level bindings directly, pass `traceparent=` and `tracestate=` yourself on the traced methods.
The high-level Python API does that automatically.

## Send Traces To Jaeger

The official Jaeger getting-started docs use this container:

```bash
docker run --rm \
  --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  -p 5778:5778 \
  -p 9411:9411 \
  cr.jaegertracing.io/jaegertracing/jaeger:2.16.0
```

Point both Python and Rust tracing at Jaeger over OTLP/HTTP:

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from egglog import bindings

provider = TracerProvider(resource=Resource.create({"service.name": "demo-app"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://127.0.0.1:4318/v1/traces")))
trace.set_tracer_provider(provider)
bindings.setup_tracing(exporter="http", endpoint="http://127.0.0.1:4318/v1/traces")
```

After that, open [http://localhost:16686](http://localhost:16686) and search for traces from `demo-app` and `egglog`.

## Local Test Runs

If you want the same tracing setup during `pytest`, use the built-in test flags documented in
{doc}`../reference/contributing`.

When using `--otel-traces=console` under `pytest`, pass `-s` so the console exporter output is shown as the test runs.
Console mode is best for short, targeted runs because it is intentionally verbose. For longer or hotter tests, prefer
OTLP/Jaeger tracing.
