//! Private bytes-only access to the shared adapter. No Python syntax is lowered here.

use egglog_experimental::protobuf::{Engine, TransportError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass(name = "_ProtoEngine")]
#[derive(Default)]
pub struct ProtoEngine {
    engine: Engine,
}

impl ProtoEngine {
    fn request<'py>(
        &mut self,
        py: Python<'py>,
        request: &Bound<'py, PyBytes>,
        operation: fn(&mut Engine, &[u8]) -> Result<Vec<u8>, TransportError>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        // Own the input while the GIL is released; only encoded bytes cross
        // this boundary in either direction, including structured run errors.
        let bytes = request.as_bytes().to_vec();
        let response = py
            .detach(|| operation(&mut self.engine, &bytes))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(PyBytes::new(py, &response))
    }
}

#[pymethods]
impl ProtoEngine {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn create<'py>(
        &mut self,
        py: Python<'py>,
        request: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.request(py, request, Engine::create)
    }

    fn run<'py>(
        &mut self,
        py: Python<'py>,
        request: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.request(py, request, Engine::run)
    }

    fn clone_graph<'py>(
        &mut self,
        py: Python<'py>,
        request: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.request(py, request, Engine::clone_egraph)
    }

    fn destroy<'py>(
        &mut self,
        py: Python<'py>,
        request: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.request(py, request, Engine::destroy)
    }

    fn configure_resources<'py>(
        &mut self,
        py: Python<'py>,
        request: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.request(py, request, Engine::configure_resources)
    }
}
