use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use opentelemetry::{
    Context, ContextGuard,
    propagation::TextMapPropagator,
    trace::{TraceContextExt, Tracer as _, TracerProvider as _},
};
use opentelemetry_otlp::WithExportConfig as _;
use opentelemetry_sdk::{Resource, propagation::TraceContextPropagator, trace::SdkTracerProvider};
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::layer::SubscriberExt;

const SERVICE_NAME: &str = "egglog";
const TRACER_NAME: &str = "egglog.rust";

#[derive(Clone, Debug, PartialEq, Eq)]
enum TracingConfig {
    Console,
    Http { endpoint: String },
}

#[derive(Default)]
struct TracingBackend {
    config: Option<TracingConfig>,
    provider: Option<SdkTracerProvider>,
    shutdown: bool,
}

static TRACING_BACKEND: Mutex<TracingBackend> = Mutex::new(TracingBackend {
    config: None,
    provider: None,
    shutdown: false,
});

pub(crate) fn attach_parent_context(
    traceparent: Option<&str>,
    tracestate: Option<&str>,
) -> Option<ContextGuard> {
    extract_context_from_headers(traceparent, tracestate).map(|context| context.attach())
}

pub(crate) fn setup_tracing(exporter: &str, endpoint: Option<&str>) -> Result<(), String> {
    let config = parse_config(exporter, endpoint)?;
    {
        let backend = TRACING_BACKEND.lock().unwrap();
        if backend.shutdown {
            return Err("egglog tracing has already been shut down for this process".to_string());
        }
        if let Some(existing) = &backend.config {
            return if existing == &config {
                Ok(())
            } else {
                Err(format!(
                    "egglog tracing is already configured as {existing:?}; cannot reconfigure to {config:?}"
                ))
            };
        }
    }

    warmup_tracer_provider(&config)?;
    let provider = create_tracer_provider(&config)?;
    let otel_layer = OpenTelemetryLayer::new(provider.tracer(TRACER_NAME));
    let subscriber = tracing_subscriber::registry().with(otel_layer);
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|err| format!("could not install egglog tracing subscriber: {err}"))?;

    let mut backend = TRACING_BACKEND.lock().unwrap();
    if backend.shutdown {
        let _ = provider.shutdown();
        return Err("egglog tracing has already been shut down for this process".to_string());
    }
    match &backend.config {
        None => {
            backend.config = Some(config);
            backend.provider = Some(provider);
            Ok(())
        }
        Some(existing) if existing == &config => {
            let _ = provider.shutdown();
            Ok(())
        }
        Some(existing) => {
            let _ = provider.shutdown();
            Err(format!(
                "egglog tracing is already configured as {existing:?}; cannot reconfigure"
            ))
        }
    }
}

pub(crate) fn shutdown_tracing() -> Result<(), String> {
    let provider = {
        let mut backend = TRACING_BACKEND.lock().unwrap();
        if backend.shutdown {
            return Ok(());
        }
        backend.shutdown = true;
        backend.provider.take()
    };

    if let Some(provider) = provider {
        provider.shutdown().map_err(|err| err.to_string())?;
    }
    Ok(())
}

fn parse_config(exporter: &str, endpoint: Option<&str>) -> Result<TracingConfig, String> {
    match exporter {
        "console" => Ok(TracingConfig::Console),
        "http" => endpoint
            .map(|endpoint| TracingConfig::Http {
                endpoint: endpoint.to_string(),
            })
            .ok_or_else(|| "setup_tracing(exporter='http') requires an endpoint".to_string()),
        _ => Err(format!("unsupported tracing exporter {exporter:?}")),
    }
}

fn create_tracer_provider(config: &TracingConfig) -> Result<SdkTracerProvider, String> {
    let resource = Resource::builder().with_service_name(SERVICE_NAME).build();

    match config {
        TracingConfig::Console => Ok(SdkTracerProvider::builder()
            .with_resource(resource)
            .with_simple_exporter(opentelemetry_stdout::SpanExporter::default())
            .build()),
        TracingConfig::Http { endpoint } => {
            let exporter = build_http_exporter(endpoint)?;
            Ok(SdkTracerProvider::builder()
                .with_resource(resource)
                .with_batch_exporter(exporter)
                .build())
        }
    }
}

fn warmup_tracer_provider(config: &TracingConfig) -> Result<(), String> {
    let TracingConfig::Http { endpoint } = config else {
        return Ok(());
    };

    log::info!("warming up egglog rust tracing exporter at {endpoint}");
    let start = Instant::now();
    let resource = Resource::builder().with_service_name(SERVICE_NAME).build();
    let exporter = build_http_exporter(endpoint)?;
    let provider = SdkTracerProvider::builder()
        .with_resource(resource)
        .with_simple_exporter(exporter)
        .build();
    let tracer = provider.tracer(TRACER_NAME);
    tracer.in_span("bindings.setup_tracing", |_| {});
    provider.shutdown().map_err(|err| {
        log::warn!(
            "egglog rust tracing exporter warmup failed after {:?}: {err}",
            start.elapsed()
        );
        err.to_string()
    })?;
    log::info!(
        "warmed up egglog rust tracing exporter at {endpoint} in {:?}",
        start.elapsed()
    );
    Ok(())
}

fn build_http_exporter(endpoint: &str) -> Result<opentelemetry_otlp::SpanExporter, String> {
    opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_endpoint(endpoint)
        .build()
        .map_err(|err| err.to_string())
}

fn extract_context_from_headers(
    traceparent: Option<&str>,
    tracestate: Option<&str>,
) -> Option<Context> {
    let mut headers = HashMap::new();
    if let Some(traceparent) = traceparent {
        headers.insert("traceparent".to_string(), traceparent.to_string());
    }
    if let Some(tracestate) = tracestate {
        headers.insert("tracestate".to_string(), tracestate.to_string());
    }
    if headers.is_empty() {
        return None;
    }

    let propagator = TraceContextPropagator::new();
    let context = propagator.extract(&headers);
    if context.span().span_context().is_valid() {
        Some(context)
    } else {
        None
    }
}
