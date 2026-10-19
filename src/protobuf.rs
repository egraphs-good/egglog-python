//! Encoded adapter access, with request-owned original Python callback errors.
//! No Python syntax is lowered here; ordinary results and errors remain bytes.

use crate::py_object_sort::{PyObjectErrorState, PyObjectExtension};
use egglog_experimental::builtin::PortableTypeInfo;
use egglog_experimental::proto as pb;
use egglog_experimental::protobuf::NativeExtension;
use egglog_experimental::protobuf::cost_model::{
    CostModelContract, CostModelRuntime, ModelSort, NativeCostModel, Objective,
};
use egglog_experimental::protobuf::source::{Renderer, Source, Submission};
use egglog_experimental::protobuf::{Engine, TransportError};
use prost::Message;
use pyo3::exceptions::{PyBaseException, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::cmp::Ordering;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;

/// Generation-time export from the actual registrar, not a parallel catalog.
#[pyfunction]
pub(crate) fn _export_pyobject_catalog(py: Python<'_>) -> PyResult<Bound<'_, PyBytes>> {
    let mut graph = egglog_experimental::new_experimental_egraph();
    crate::py_object_sort::PyObjectExtension::default()
        .initialize(&mut graph)
        .map_err(PyValueError::new_err)?;
    let catalog = graph
        .type_info()
        .portable_builtin_catalog()
        .map_err(PyValueError::new_err)?;
    eprintln!(
        "undescribed primitives: {:?}; families: {:?}; family primitives: {:?}; sorts: {:?}",
        catalog.undescribed_primitives,
        catalog.undescribed_families,
        catalog.undescribed_family_primitives,
        catalog.undescribed_sorts
    );
    Ok(PyBytes::new(py, &catalog.definitions.encode_to_vec()))
}

#[pyclass(name = "_ProtoEngine")]
pub struct ProtoEngine {
    engine: Engine,
    extension: Arc<PyObjectExtension>,
    py_error: PyObjectErrorState,
    recording: Option<Recording>,
}

enum Recording {
    Active {
        renderer: Box<Renderer>,
        staged: Option<(Box<Renderer>, String)>,
        source: String,
    },
    Invalid,
    Closed,
}

/// Own the actual sequential producer; only encoded Programs and scope counts escape.
#[pyclass(name = "_ProtoSource")]
struct ProtoSource {
    source: Source,
}

type EncodedSourceSubmission<'py> = (String, Option<Bound<'py, PyBytes>>, usize);

#[pymethods]
impl ProtoSource {
    /// Stop consuming on any production or execution failure, retaining its prefix.
    fn next_submission<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<EncodedSourceSubmission<'py>>> {
        self.source
            .next_submission()
            .map(|submission| {
                let submission =
                    submission.map_err(|error| PyValueError::new_err(error.to_string()))?;
                Ok(match submission {
                    Submission::Program(program) => (
                        "program".into(),
                        Some(PyBytes::new(py, &program.encode_to_vec())),
                        0,
                    ),
                    Submission::Push(count) => ("push".into(), None, count),
                    Submission::Pop { count, .. } => ("pop".into(), None, count),
                    Submission::PrintFunction(_) => {
                        return Err(PyValueError::new_err(
                            "source file/CSV presentation requires a host callback",
                        ));
                    }
                })
            })
            .transpose()
    }
}

impl Default for ProtoEngine {
    fn default() -> Self {
        let py_error = PyObjectErrorState::default();
        let extension = Arc::new(PyObjectExtension {
            py_error: py_error.clone(),
        });
        Self {
            engine: Engine::with_extension(extension.clone()),
            extension,
            py_error,
            recording: None,
        }
    }
}

impl ProtoEngine {
    fn stage_submission(&mut self, submission: Submission) -> PyResult<()> {
        match &mut self.recording {
            None => Ok(()),
            Some(Recording::Invalid) => Err(PyRuntimeError::new_err(
                "Recorded commands are disabled because partial effects cannot be replayed",
            )),
            Some(Recording::Closed) => Err(PyValueError::new_err(
                "The saved transcript has been closed",
            )),
            Some(Recording::Active {
                renderer, staged, ..
            }) => {
                if staged.is_some() {
                    return Err(PyRuntimeError::new_err(
                        "A source submission is already staged",
                    ));
                }
                let mut candidate = renderer.clone();
                let source = candidate
                    .render_submission(&submission)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?;
                *staged = Some((candidate, source));
                Ok(())
            }
        }
    }

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

    // Drain while this request still exclusively owns the engine. The returned
    // exception belongs to its response, not to a later mutable take_error slot.
    fn run_request<'py>(
        &mut self,
        py: Python<'py>,
        request: &Bound<'py, PyBytes>,
        operation: impl FnOnce(&mut Engine, &[u8]) -> Result<Vec<u8>, TransportError> + Send,
    ) -> PyResult<(Bound<'py, PyBytes>, Option<Py<PyBaseException>>)> {
        let bytes = request.as_bytes().to_vec();
        self.py_error.lock().unwrap().take();
        let result = catch_unwind(AssertUnwindSafe(|| {
            py.detach(|| operation(&mut self.engine, &bytes))
        }));
        let captured = self.py_error.lock().unwrap().take();
        let response = match result {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => {
                return Err(captured.unwrap_or_else(|| PyValueError::new_err(error.to_string())));
            }
            Err(error) => {
                if error.is::<PythonCostError>()
                    && let Some(captured) = captured
                {
                    return Err(captured);
                }
                resume_unwind(error);
            }
        };
        Ok((
            PyBytes::new(py, &response),
            captured.map(|error| error.into_value(py)),
        ))
    }
}

// Only this private payload is translated at the mutable request boundary.
// Native panics must never be disguised by an earlier Python error latch.
struct PythonCostError;

fn remember_cost_error(errors: &PyObjectErrorState, error: PyErr) -> egglog::Error {
    let mut first = errors.lock().unwrap();
    if first.is_none() {
        *first = Some(error);
    }
    egglog::Error::ExtractError("Python cost model failed".into())
}

fn host_cost_result<T>(errors: &PyObjectErrorState, result: PyResult<T>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            let _ = remember_cost_error(errors, error);
            resume_unwind(Box::new(PythonCostError))
        }
    }
}

struct PythonModelRegistration {
    sorts: Vec<pb::Sort>,
    annotation: u32,
    objective: u32,
    additive: bool,
    factory: Py<PyAny>,
    errors: PyObjectErrorState,
}

impl NativeCostModel for PythonModelRegistration {
    fn contract(&self) -> CostModelContract {
        CostModelContract {
            annotation: ModelSort {
                sorts: self.sorts.clone(),
                root: self.annotation,
            },
            objective: ModelSort {
                sorts: self.sorts.clone(),
                root: self.objective,
            },
            additive: self.additive,
        }
    }

    fn start(&self, observation: &pb::Program) -> Result<Arc<dyn CostModelRuntime>, egglog::Error> {
        let object = Python::attach(|py| {
            self.factory
                .call1(py, (PyBytes::new(py, &observation.encode_to_vec()),))
        })
        .map_err(|error| remember_cost_error(&self.errors, error))?;
        Ok(Arc::new(PythonModelRuntime {
            object,
            errors: self.errors.clone(),
        }))
    }
}

struct PythonModelRuntime {
    object: Py<PyAny>,
    errors: PyObjectErrorState,
}

impl CostModelRuntime for PythonModelRuntime {
    fn run(
        &self,
        operation: Box<dyn FnOnce() -> Result<(), egglog::Error> + '_>,
    ) -> Result<(), egglog::Error> {
        match catch_unwind(AssertUnwindSafe(operation)) {
            Ok(result) => result,
            Err(payload) if payload.is::<PythonCostError>() => Err(egglog::Error::ExtractError(
                "Python cost model failed".into(),
            )),
            Err(payload) => resume_unwind(payload),
        }
    }
    fn cost(&self, node: u32, children: &[Objective]) -> Objective {
        Python::attach(|py| {
            let children = children
                .iter()
                .map(|value| {
                    value
                        .downcast_ref::<Py<PyAny>>()
                        .expect("provider objective type")
                        .clone_ref(py)
                })
                .collect::<Vec<_>>();
            Arc::new(host_cost_result(
                &self.errors,
                self.object.call_method1(py, "cost", (node, children)),
            )) as Objective
        })
    }
    fn compare(&self, left: &Objective, right: &Objective) -> Ordering {
        Python::attach(|py| {
            host_cost_result(
                &self.errors,
                left.downcast_ref::<Py<PyAny>>()
                    .expect("provider objective type")
                    .bind(py)
                    .compare(
                        right
                            .downcast_ref::<Py<PyAny>>()
                            .expect("provider objective type")
                            .bind(py),
                    ),
            )
        })
    }
    fn combine(&self, left: &Objective, right: &Objective) -> Objective {
        Python::attach(|py| {
            let result = left
                .downcast_ref::<Py<PyAny>>()
                .expect("provider objective type")
                .bind(py)
                .add(
                    right
                        .downcast_ref::<Py<PyAny>>()
                        .expect("provider objective type")
                        .bind(py),
                );
            Arc::new(host_cost_result(&self.errors, result).unbind()) as Objective
        })
    }
    fn identity(&self) -> Objective {
        Python::attach(|py| {
            Arc::new(host_cost_result(
                &self.errors,
                self.object.getattr(py, "identity"),
            )) as Objective
        })
    }
    fn encode(&self, value: &Objective) -> Result<Option<pb::PrimitiveValue>, egglog::Error> {
        let payload = Python::attach(|py| {
            host_cost_result(
                &self.errors,
                crate::py_object_sort::dump(
                    value
                        .downcast_ref::<Py<PyAny>>()
                        .expect("provider objective type")
                        .bind(py)
                        .clone(),
                ),
            )
        });
        Ok(Some(pb::PrimitiveValue {
            value: Some(pb::primitive_value::Value::Custom(pb::CustomValue {
                payload: payload.0,
                args: vec![],
            })),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::py_object_sort::{PyObjectExtension, PyPickledValue, dump, load};
    use egglog_experimental::protobuf::NativeExtension;
    use egglog_experimental::{builtin::import_rooted_definition, proto as pb};
    use prost::Message;
    use pyo3::types::PyDict;
    use std::sync::Arc;

    #[test]
    fn recording_stages_commits_scopes_and_invalidates() {
        let mut engine = ProtoEngine::default();
        engine.enable_recording().unwrap();
        let program = Source::new(None, "(function recorded () i64 :merge new)")
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        engine
            .stage_submission(Submission::Program(program))
            .unwrap();
        assert_eq!(engine.recorded_source().unwrap(), "");
        engine.finish_recording(true);
        let declared = engine.recorded_source().unwrap();
        assert!(declared.contains("function"));
        engine.prepare_recording_scope(true).unwrap();
        assert_eq!(engine.recorded_source().unwrap(), declared);
        engine.finish_recording(true);
        engine.prepare_recording_scope(false).unwrap();
        engine.finish_recording(true);
        assert!(engine.recorded_source().unwrap().starts_with(&declared));
        assert!(
            engine
                .recorded_source()
                .unwrap()
                .contains("(push 1)\n(pop 1)")
        );
        let history = engine.recorded_source().unwrap();
        let unsupported = pb::Program {
            ir_version: 1,
            commands: vec![pb::Command {
                kind: Some(pb::command::Kind::Freeze(pb::Freeze::default())),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(
            engine
                .stage_submission(Submission::Program(unsupported))
                .is_err()
        );
        assert_eq!(engine.recorded_source().unwrap(), history);
        engine.finish_recording(false);
        assert!(matches!(&engine.recording, Some(Recording::Invalid)));
        engine.finish_recording(true);
        assert!(engine.recorded_source().is_err());
        engine.prepare_recording_scope(false).unwrap();
        engine.finish_recording(true);
        assert!(matches!(&engine.recording, Some(Recording::Invalid)));
        engine.finish_recording(false);
        assert!(matches!(&engine.recording, Some(Recording::Invalid)));
        assert!(engine.prepare_recording_scope(true).is_err());
        assert!(
            engine
                .stage_submission(Submission::Program(pb::Program::default()))
                .is_err()
        );
        engine.close_recording();
        engine.close_recording();
        assert!(matches!(&engine.recording, Some(Recording::Closed)));
        engine.prepare_recording_scope(false).unwrap();
        engine.finish_recording(true);
        engine.finish_recording(false);
        assert!(matches!(&engine.recording, Some(Recording::Closed)));
        assert!(engine.prepare_recording_scope(true).is_err());
        assert!(
            engine
                .stage_submission(Submission::Program(pb::Program::default()))
                .is_err()
        );
    }

    #[test]
    fn staged_panic_source_preserves_escaped_messages_without_execution() {
        for message in ["bad \"quote\"", "bad \\ slash"] {
            let mut engine = ProtoEngine::default();
            engine.enable_recording().unwrap();
            engine
                .stage_submission(Submission::Program(pb::Program {
                    ir_version: 1,
                    commands: vec![pb::Command {
                        kind: Some(pb::command::Kind::Action(pb::Action {
                            kind: Some(pb::action::Kind::Panic(message.into())),
                            ..Default::default()
                        })),
                        ..Default::default()
                    }],
                    ..Default::default()
                }))
                .unwrap();
            let Some(Recording::Active {
                staged: Some((_, text)),
                ..
            }) = &engine.recording
            else {
                panic!("source was not staged");
            };
            let parsed = Source::new(None, text).unwrap().next().unwrap().unwrap();
            let Some(pb::command::Kind::Action(pb::Action {
                kind: Some(pb::action::Kind::Panic(actual)),
                ..
            })) = &parsed.commands[0].kind
            else {
                panic!("source did not preserve the Panic action");
            };
            assert_eq!(actual, message);
            assert_eq!(engine.recorded_source().unwrap(), "");
        }
    }

    #[test]
    fn host_cost_boundary_preserves_unrelated_native_panics() {
        struct NativePanic;

        Python::initialize();
        let runtime = PythonModelRuntime {
            object: Python::attach(|py| py.None()),
            errors: PyObjectErrorState::default(),
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            runtime.run(Box::new(|| {
                let _ = remember_cost_error(
                    &runtime.errors,
                    PyValueError::new_err("earlier host error"),
                );
                resume_unwind(Box::new(NativePanic))
            }))
        }));
        assert!(result.unwrap_err().is::<NativePanic>());
        assert!(runtime.errors.lock().unwrap().is_some());
    }

    #[test]
    fn callback_error_handoff_preserves_response_and_identity() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::from_code(
                py,
                c"caught = None\ndef callback():\n    global caught\n    caught = ValueError('original callback error')\n    raise caught\n",
                c"pyerr_handoff.py",
                c"_egglog_pyerr_handoff",
            )
            .unwrap();
            PyModule::import(py, "sys")
                .unwrap()
                .getattr("modules")
                .unwrap()
                .set_item("_egglog_pyerr_handoff", &module)
                .unwrap();
            let callback = dump(module.getattr("callback").unwrap()).unwrap().0;
            let mut instance = ProtoEngine::default();
            let id = create(&mut instance.engine);
            let mut program = custom_program(callback);
            program.nodes.push(pb::Node {
                sort_id: 0,
                kind: Some(pb::node::Kind::Call(pb::Call {
                    func: "egglog.python.pyobject.call".into(),
                    args: vec![0],
                })),
                ..Default::default()
            });
            let Some(pb::command::Kind::Extract(extract)) = &mut program.commands[0].kind else {
                unreachable!()
            };
            extract.roots = vec![1];
            let request = pb::RunProgramRequest {
                egraph_id: id,
                program: Some(program),
                profile: true,
                ..Default::default()
            }
            .encode_to_vec();
            let instance = Py::new(py, instance).unwrap();
            // Dynamic invocation keeps the pre-fix witness compilable. Accept
            // its old bytes shape only to expose the lost actual callback error.
            let returned = instance
                .bind(py)
                .call_method1("run", (PyBytes::new(py, &request),))
                .unwrap();
            let (bytes, captured) = returned
                .extract::<(Vec<u8>, Option<Py<pyo3::exceptions::PyBaseException>>)>()
                .unwrap_or_else(|_| (returned.extract::<Vec<u8>>().unwrap(), None));
            let response = pb::RunProgramResponse::decode(bytes.as_slice()).unwrap();
            assert_eq!(response.encode_to_vec(), bytes);
            assert!(response.ruleset_publication.is_some());
            assert_eq!(
                response.error.unwrap().code,
                pb::ErrorCode::EvaluationFailed as i32
            );
            let original = module.getattr("caught").unwrap();
            assert!(original.is_instance_of::<PyValueError>());
            assert_eq!(
                original
                    .getattr("args")
                    .unwrap()
                    .extract::<(String,)>()
                    .unwrap()
                    .0,
                "original callback error"
            );
            let captured = captured.expect("actual callback error was lost after native execution");
            assert!(captured.bind(py).is(&original));
            let notes: Vec<String> = original.getattr("__notes__").unwrap().extract().unwrap();
            assert_eq!(notes.len(), 2);
            assert!(notes[0].starts_with("Calling function "));
            assert_eq!(notes[1], "While calling primitive 'py-call'");
            assert!(!original.getattr("__traceback__").unwrap().is_none());
        });
    }

    #[test]
    fn callback_error_request_drain_covers_worker_transport_and_unwind() {
        Python::initialize();
        Python::attach(|py| {
            let mut instance = ProtoEngine::default();
            let independent = ProtoEngine::default();
            assert!(!Arc::ptr_eq(&instance.py_error, &independent.py_error));
            for boundary in ["response", "transport", "unwind"] {
                *instance.py_error.lock().unwrap() = Some(PyValueError::new_err("stale"));
                let original = PyValueError::new_err(boundary);
                let value = original.value(py).clone().unbind();
                let errors = instance.py_error.clone();
                let result =
                    instance.run_request(py, &PyBytes::new(py, b"request"), move |engine, _| {
                        assert!(errors.lock().unwrap().is_none());
                        std::thread::spawn(move || {
                            *errors.lock().unwrap() = Some(original);
                        })
                        .join()
                        .unwrap();
                        match boundary {
                            "response" => Ok(pb::RunProgramResponse {
                                ruleset_publication: Some(pb::RulesetPublication::default()),
                                ..Default::default()
                            }
                            .encode_to_vec()),
                            "transport" => engine.run(&[0xff]),
                            _ => resume_unwind(Box::new(PythonCostError)),
                        }
                    });
                if boundary == "response" {
                    let (bytes, captured) = result.unwrap();
                    assert!(
                        pb::RunProgramResponse::decode(bytes.as_bytes())
                            .unwrap()
                            .ruleset_publication
                            .is_some()
                    );
                    assert!(captured.unwrap().bind(py).is(value.bind(py)));
                } else {
                    assert!(result.unwrap_err().value(py).is(value.bind(py)));
                }
                assert!(instance.py_error.lock().unwrap().is_none());
                assert!(independent.py_error.lock().unwrap().is_none());
                let (bytes, captured) = instance
                    .run_request(
                        py,
                        &PyBytes::new(py, b"next"),
                        |_, bytes| Ok(bytes.to_vec()),
                    )
                    .unwrap();
                assert_eq!(bytes.as_bytes(), b"next");
                assert!(captured.is_none());
            }
            let failure = instance
                .run_request(py, &PyBytes::new(py, &[0xff]), Engine::run)
                .unwrap_err();
            assert!(failure.is_instance_of::<PyValueError>(py));
            assert!(instance.py_error.lock().unwrap().is_none());
            struct NativePanic;
            for latched in [false, true] {
                let errors = instance.py_error.clone();
                let unwind = catch_unwind(AssertUnwindSafe(|| {
                    let _ = instance.run_request(py, &PyBytes::new(py, b"panic"), |_, _| {
                        if latched {
                            *errors.lock().unwrap() =
                                Some(PyValueError::new_err("prior host error"));
                        }
                        resume_unwind(Box::new(NativePanic))
                    });
                }));
                assert!(unwind.unwrap_err().is::<NativePanic>());
                assert!(instance.py_error.lock().unwrap().is_none());
            }
        });
    }

    #[test]
    fn provider_catalog_and_inert_string_bytes() {
        Python::initialize();
        let extension = Arc::new(PyObjectExtension::default());
        let mut graph = egglog_experimental::new_experimental_egraph();
        extension.initialize(&mut graph).unwrap();
        let catalog = graph.type_info().portable_builtin_catalog().unwrap();
        assert!(
            !catalog
                .undescribed_sorts
                .iter()
                .any(|name| name == "PyObject")
        );
        assert!(
            !catalog
                .undescribed_primitives
                .iter()
                .any(|name| name.starts_with("py-"))
        );
        let mut keys = catalog
            .definitions
            .declarations
            .iter()
            .filter_map(|d| match &d.kind {
                Some(pb::declaration::Kind::HostPrimitive(p))
                    if p.name.starts_with("egglog.python.pyobject.") =>
                {
                    Some(p.name.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        keys.sort();
        assert_eq!(
            keys,
            [
                "call",
                "call_extended",
                "dict",
                "dict_update",
                "eval",
                "exec",
                "from_int",
                "from_pickle",
                "from_string",
                "to_bool",
                "to_string"
            ]
            .map(|name| format!("egglog.python.pyobject.{name}"))
        );
        let family = catalog
            .definitions
            .declarations
            .iter()
            .find_map(|declaration| match &declaration.kind {
                Some(pb::declaration::Kind::HostSortFamily(family))
                    if family.name == "PyObject" =>
                {
                    Some(family)
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(family.arity, 0);
        assert_eq!(
            family
                .bindings
                .as_ref()
                .unwrap()
                .python
                .as_ref()
                .unwrap()
                .path,
            ["egglog", "builtins", "PyObject"]
        );
        assert!(family.bindings.as_ref().unwrap().rust.is_none());
        for (key, alias, inputs, tail, output) in [
            (
                "from_pickle",
                "py-object",
                vec!["String"],
                vec![],
                "PyObject",
            ),
            (
                "eval",
                "py-eval",
                vec!["String", "PyObject", "PyObject"],
                vec![],
                "PyObject",
            ),
            (
                "exec",
                "py-exec",
                vec!["String", "PyObject", "PyObject"],
                vec![],
                "PyObject",
            ),
            ("call", "py-call", vec![], vec!["PyObject"], "PyObject"),
            ("dict", "py-dict", vec![], vec!["PyObject"], "PyObject"),
            (
                "dict_update",
                "py-dict-update",
                vec![],
                vec!["PyObject"],
                "PyObject",
            ),
            (
                "call_extended",
                "py-call-extended",
                vec!["PyObject", "PyObject", "PyObject"],
                vec![],
                "PyObject",
            ),
            (
                "to_string",
                "py-to-string",
                vec!["PyObject"],
                vec![],
                "String",
            ),
            ("to_bool", "py-to-bool", vec!["PyObject"], vec![], "bool"),
            (
                "from_string",
                "py-from-string",
                vec!["String"],
                vec![],
                "PyObject",
            ),
            ("from_int", "py-from-int", vec!["i64"], vec![], "PyObject"),
        ] {
            let key = format!("egglog.python.pyobject.{key}");
            let declaration = catalog.definitions.declarations.iter().find(|d| matches!(&d.kind, Some(pb::declaration::Kind::HostPrimitive(p)) if p.name == key)).unwrap();
            let Some(pb::declaration::Kind::HostPrimitive(primitive)) = &declaration.kind else {
                unreachable!()
            };
            let Some(pb::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
                panic!("not a real closed signature")
            };
            assert!(signature.type_params.is_empty());
            let sort_name = |index: u32| {
                let Some(pb::sort::Kind::Family(family)) =
                    &catalog.definitions.sorts[index as usize].kind
                else {
                    panic!("not a host sort")
                };
                assert!(family.args.is_empty());
                family.name.as_str()
            };
            assert_eq!(
                signature
                    .inputs
                    .iter()
                    .map(|arg| sort_name(arg.sort))
                    .collect::<Vec<_>>(),
                inputs
            );
            assert_eq!(
                signature
                    .varargs
                    .iter()
                    .map(|arg| sort_name(arg.sort))
                    .collect::<Vec<_>>(),
                tail
            );
            assert_eq!(sort_name(signature.output.unwrap()), output);
            let bindings = declaration.bindings.as_ref().unwrap();
            assert_eq!(bindings.egglog.as_ref().unwrap().views[0].symbol, alias);
            if alias == "py-object" {
                assert!(bindings.python.is_none());
            } else {
                let views = &bindings
                    .python
                    .as_ref()
                    .expect("missing public PyObject view")
                    .views;
                assert_eq!(views.len(), 1);
                let view = &views[0];
                if matches!(alias, "py-eval" | "py-exec") {
                    assert_eq!(view.kind, pb::PythonCallKind::Function as i32);
                    assert_eq!(view.path, ["egglog", "builtins", &alias.replace('-', "_")]);
                    assert!(view.owner.is_none());
                    assert_eq!(view.receiver, None);
                    assert_eq!(view.params.len(), 3);
                    for (index, name) in ["code", "globals_", "locals_"].iter().enumerate() {
                        let parameter = &view.params[index];
                        assert_eq!(parameter.name, *name);
                        assert_eq!(parameter.core_input, Some(index as u32));
                        if index == 0 {
                            assert_eq!(parameter.default_expr, None);
                            continue;
                        }
                        let default = &catalog.definitions.nodes
                            [parameter.default_expr.expect("missing real dict default") as usize];
                        assert_eq!(sort_name(default.sort_id), "PyObject");
                        assert_eq!(
                            default.kind,
                            Some(pb::node::Kind::Call(pb::Call {
                                func: "egglog.python.pyobject.dict".into(),
                                args: vec![],
                            }))
                        );
                    }
                } else if matches!(alias, "py-call" | "py-dict-update") {
                    assert_eq!(view.kind, pb::PythonCallKind::Method as i32);
                    assert_eq!(view.receiver, Some(0));
                    assert_eq!(
                        view.owner.as_ref().unwrap().kind,
                        Some(pb::binding_owner::Kind::Sort(signature.varargs[0].sort))
                    );
                    assert_eq!(view.params.len(), 1);
                    assert_eq!(view.params[0].core_input, Some(1));
                    assert_eq!(view.params[0].default_expr, None);
                    assert_eq!(
                        view.path,
                        [if alias == "py-call" {
                            "__call__"
                        } else {
                            "dict_update"
                        }]
                    );
                }
            }
            // The alias and key refer to the same actual registration in every
            // native context. This resolves signatures; no callback is executed.
            let sorts = inputs
                .iter()
                .chain(&tail)
                .copied()
                .chain([output])
                .map(|name| graph.get_sort_by_name(name).unwrap().clone())
                .collect::<Vec<_>>();
            for context in [
                egglog_experimental::Context::Pure,
                egglog_experimental::Context::Read,
                egglog_experimental::Context::Write,
                egglog_experimental::Context::Full,
            ] {
                let mut resolve = |name: &str| {
                    egglog_experimental::ResolvedCall::from_resolution(
                        name,
                        &sorts,
                        graph.type_info(),
                        context,
                        &egglog::span!(),
                    )
                    .unwrap()
                };
                let (
                    egglog_experimental::ResolvedCall::Primitive(a),
                    egglog_experimental::ResolvedCall::Primitive(b),
                ) = (resolve(alias), resolve(&key))
                else {
                    panic!("not native primitives")
                };
                assert_eq!(a, b);
            }
        }
        let exported =
            Python::attach(|py| _export_pyobject_catalog(py).unwrap().as_bytes().to_vec());
        assert_eq!(exported, catalog.definitions.encode_to_vec());
        assert_eq!(
            pb::Program::decode(exported.as_slice())
                .unwrap()
                .encode_to_vec(),
            exported
        );
        assert_eq!(
            graph
                .type_info()
                .portable_builtin_catalog()
                .unwrap()
                .definitions,
            catalog.definitions
        );
        let payload = Python::attach(|py| dump("hello".into_pyobject(py).unwrap()).unwrap().0);
        let mut engine = Engine::with_extension(extension);
        let id = create(&mut engine);
        let catalog_result = run(&mut engine, id, catalog.definitions.clone());
        assert_eq!(catalog_result.error, None);
        let program = custom_program(payload.clone());
        let result = run(&mut engine, id, program);
        assert_eq!(result.error, None);
        assert_eq!(
            result.ruleset_publication.unwrap().names,
            Vec::<String>::new()
        );
        let Some(pb::command_output::Kind::Extraction(extract)) = &result.outputs[0].kind else {
            panic!("not an extract")
        };
        let output = extract.roots[0].variants[0].term;
        let node = &result.nodes[output as usize];
        assert!(
            matches!(&result.sorts[node.sort_id as usize].kind, Some(pb::sort::Kind::Family(f)) if f.name == "PyObject" && f.args.is_empty())
        );
        assert_eq!(
            node.kind,
            Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::Custom(pb::CustomValue {
                    payload: payload.clone(),
                    args: vec![]
                }))
            }))
        );
        Python::attach(|py| {
            assert_eq!(
                load(py, &PyPickledValue(payload))
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "hello"
            )
        });
    }

    #[test]
    fn pyobject_default_context_validation_and_installed_authority() {
        Python::initialize();
        let extension = Arc::new(PyObjectExtension::default());
        let mut graph = egglog_experimental::new_experimental_egraph();
        extension.initialize(&mut graph).unwrap();
        let catalog = graph
            .type_info()
            .portable_builtin_catalog()
            .unwrap()
            .definitions;
        let mut engine = Engine::with_extension(extension);
        for operation in ["eval", "exec"] {
            let key = format!("egglog.python.pyobject.{operation}");
            let helper = "egglog.python.pyobject.dict";
            // Export the actual callback-built context, not a second signature.
            let mut context = pb::Program::default();
            graph
                .type_info()
                .export_builtin_definition(&key, &mut context)
                .unwrap();
            let root = context
                .declarations
                .iter()
                .position(|d| {
                    matches!(&d.kind,
                Some(pb::declaration::Kind::HostPrimitive(p)) if p.name == key)
                })
                .unwrap();
            let default = context.declarations[root]
                .bindings
                .as_ref()
                .unwrap()
                .python
                .as_ref()
                .unwrap()
                .views[0]
                .params[1]
                .default_expr
                .unwrap() as usize;
            let Some(pb::declaration::Kind::HostPrimitive(primitive)) =
                &context.declarations[root].kind
            else {
                unreachable!()
            };
            let Some(pb::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
                unreachable!()
            };
            let string_sort = signature.inputs[0].sort;
            let mut imported = pb::Program::default();
            import_rooted_definition(&key, &context, &mut imported).unwrap();
            let before = imported.clone();
            import_rooted_definition(&key, &context, &mut imported).unwrap();
            assert_eq!(imported, before);

            // Standalone closure needs its helper; an installed P session
            // already owns that genuine declaration, so omission is allowed.
            let mut missing = context.clone();
            missing.declarations.retain(|d| {
                !matches!(&d.kind,
                Some(pb::declaration::Kind::HostPrimitive(p)) if p.name == helper)
            });
            let mut destination = pb::Program::default();
            assert_eq!(
                import_rooted_definition(&key, &missing, &mut destination).unwrap_err(),
                "unknown default callee"
            );
            assert_eq!(destination, pb::Program::default());
            let id = create(&mut engine);
            assert_eq!(run(&mut engine, id, missing).error, None);
            assert_eq!(run(&mut engine, id, context.clone()).error, None);

            for (case, standalone_error) in [
                ("forged_helper", None),
                ("forged_family", None),
                ("unknown_callee", Some("unknown default callee")),
                ("fixed_helper_arity", Some("default call arity mismatch")),
                ("default_sort", Some("default sort differs from parameter")),
                ("family_arity", Some("presentation family arity mismatch")),
            ] {
                let mut invalid = context.clone();
                match case {
                    "forged_helper" => {
                        let declaration = invalid
                            .declarations
                            .iter_mut()
                            .find(|d| {
                                matches!(&d.kind,
                            Some(pb::declaration::Kind::HostPrimitive(p)) if p.name == helper)
                            })
                            .unwrap();
                        declaration
                            .bindings
                            .as_mut()
                            .unwrap()
                            .python
                            .as_mut()
                            .unwrap()
                            .views[0]
                            .path = vec!["forged_dict".into()];
                    }
                    "forged_family" | "family_arity" => {
                        let family = invalid
                            .declarations
                            .iter_mut()
                            .find_map(|d| match &mut d.kind {
                                Some(pb::declaration::Kind::HostSortFamily(f))
                                    if f.name == "PyObject" =>
                                {
                                    Some(f)
                                }
                                _ => None,
                            })
                            .unwrap();
                        if case == "family_arity" {
                            family.arity = 1;
                        } else {
                            family
                                .bindings
                                .as_mut()
                                .unwrap()
                                .python
                                .as_mut()
                                .unwrap()
                                .path = vec!["forged".into(), "PyObject".into()];
                        }
                    }
                    "default_sort" => invalid.nodes[default].sort_id = string_sort,
                    _ => {
                        let callee = if case == "fixed_helper_arity" {
                            let callee = "egglog.python.pyobject.from_string";
                            graph
                                .type_info()
                                .export_builtin_definition(callee, &mut invalid)
                                .unwrap();
                            invalid.declarations.retain(|d| {
                                !matches!(&d.kind,
                                Some(pb::declaration::Kind::HostPrimitive(p)) if p.name == helper)
                            });
                            callee
                        } else {
                            "egglog.python.pyobject.missing_helper"
                        };
                        let Some(pb::node::Kind::Call(call)) = &mut invalid.nodes[default].kind
                        else {
                            unreachable!()
                        };
                        call.func = callee.into();
                    }
                }
                let mut destination = pb::Program::default();
                let result = import_rooted_definition(&key, &invalid, &mut destination);
                if let Some(expected) = standalone_error {
                    assert_eq!(result.unwrap_err(), expected, "{operation}/{case}");
                    assert_eq!(destination, pb::Program::default());
                } else {
                    // Rooted import assumes registered evidence; it does not
                    // confer native authority on self-consistent forged views.
                    result.unwrap();
                }
                let id = create(&mut engine);
                let response = run(&mut engine, id, invalid);
                assert_eq!(
                    response.error.unwrap().code,
                    pb::ErrorCode::InvalidProgram as i32,
                    "{operation}/{case}"
                );
                assert!(response.outputs.is_empty());
                assert!(response.ruleset_publication.unwrap().names.is_empty());
                assert_eq!(
                    run(&mut engine, id, context.clone()).error,
                    None,
                    "failed admission must not poison the genuine context: {operation}/{case}"
                );
            }

            // Dict has a homogeneous tail, not pair-width arity. A single
            // well-sorted argument is valid and the unchanged body ignores it.
            let mut odd = context.clone();
            let argument = odd.nodes.len() as u32;
            odd.nodes.push(odd.nodes[default].clone());
            let Some(pb::node::Kind::Call(call)) = &mut odd.nodes[default].kind else {
                unreachable!()
            };
            call.args.push(argument);
            import_rooted_definition(&key, &odd, &mut pb::Program::default()).unwrap();
            // Execute only the call, not a conflicting resupply of changed
            // eval/exec defaults. The installed catalog supplies its callee.
            odd.declarations.clear();
            odd.commands.push(pb::Command {
                kind: Some(pb::command::Kind::Extract(pb::Extract {
                    roots: vec![default as u32],
                    variants: 1,
                    extractor: pb::Extractor::Tree.into(),
                    ..Default::default()
                })),
                ..Default::default()
            });
            let response = run(&mut engine, id, odd);
            assert_eq!(response.error, None);
            let Some(pb::command_output::Kind::Extraction(extraction)) = &response.outputs[0].kind
            else {
                unreachable!()
            };
            let Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::Custom(value)),
            })) = &response.nodes[extraction.roots[0].variants[0].term as usize].kind
            else {
                panic!("expected an owned PyObject dict")
            };
            Python::attach(|py| {
                assert!(
                    load(py, &PyPickledValue(value.payload.clone()))
                        .unwrap()
                        .cast::<PyDict>()
                        .unwrap()
                        .is_empty()
                );
            });
        }
        assert_eq!(
            graph
                .type_info()
                .portable_builtin_catalog()
                .unwrap()
                .definitions,
            catalog
        );
    }

    #[test]
    fn custom_dict_bytes_survive_clone_fresh_handle_and_destroy() {
        Python::initialize();
        let payload = Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("answer", 42).unwrap();
            dict.set_item("text", "owned").unwrap();
            dump(dict).unwrap().0
        });
        let mut engine = Engine::with_extension(Arc::new(PyObjectExtension::default()));
        let id = create(&mut engine);
        let result = run(&mut engine, id, custom_program(payload.clone()));
        let replay = extracted_program(&result, &payload);
        let clone = pb::CloneEGraphResponse::decode(
            engine
                .clone_egraph(&pb::CloneEGraphRequest { egraph_id: id }.encode_to_vec())
                .unwrap()
                .as_slice(),
        )
        .unwrap()
        .egraph_id;
        let fresh = create(&mut engine);
        for handle in [id, clone, fresh] {
            let result = run(&mut engine, handle, replay.clone());
            extracted_program(&result, &payload);
            engine
                .destroy(&pb::DestroyEGraphRequest { egraph_id: handle }.encode_to_vec())
                .unwrap();
        }
        drop(engine);
        let Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
            value: Some(pb::primitive_value::Value::Custom(value)),
        })) = &replay.nodes[0].kind
        else {
            panic!("not Custom")
        };
        Python::attach(|py| {
            let loaded = load(py, &PyPickledValue(value.payload.clone())).unwrap();
            assert_eq!(
                loaded.get_item("answer").unwrap().extract::<i64>().unwrap(),
                42
            );
            assert_eq!(
                loaded
                    .get_item("text")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "owned"
            );
        });
    }

    #[test]
    fn custom_inspection_preparation_and_extraction_never_load_pickle() {
        Python::initialize();
        let payload = Python::attach(|py| {
            let module = PyModule::from_code(py, c"import builtins\nbuiltins._egglog_value_gate_loads = 0\ndef rebuild():\n    builtins._egglog_value_gate_loads += 1\n    return 'loaded explicitly'\nclass Probe:\n    def __reduce__(self):\n        return (rebuild, ())\n", c"value_gate.py", c"_egglog_value_gate").unwrap();
            dump(module.getattr("Probe").unwrap().call0().unwrap())
                .unwrap()
                .0
        });
        let value = PyPickledValue(payload.clone());
        assert!(format!("{value:?}").starts_with("PyPickledValue("));
        use std::hash::{Hash, Hasher};
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        value.hash(&mut hash);
        let _ = hash.finish();
        assert_eq!(value, value.clone());
        let extension = Arc::new(PyObjectExtension::default());
        let mut engine = Engine::with_extension(extension.clone());
        let id = create(&mut engine);
        let original = custom_program(payload.clone());
        let mut lowered = original.clone();
        let mut graph = egglog_experimental::new_experimental_egraph();
        extension.initialize(&mut graph).unwrap();
        let Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
            value: Some(pb::primitive_value::Value::Custom(custom)),
        })) = &original.nodes[0].kind
        else {
            unreachable!()
        };
        extension
            .lower_custom_value(
                &graph,
                graph.get_sort_by_name("PyObject").unwrap(),
                custom,
                None,
                &mut lowered,
            )
            .unwrap();
        assert_eq!(lowered.nodes[..original.nodes.len()], original.nodes);
        assert_eq!(lowered.sorts[..original.sorts.len()], original.sorts);
        assert_eq!(lowered.declarations, original.declarations);
        assert_eq!(lowered.nodes.len(), original.nodes.len() + 1);
        let mut recorder = ProtoEngine::default();
        recorder.enable_recording().unwrap();
        recorder
            .stage_submission(Submission::Program(original.clone()))
            .unwrap();
        assert_eq!(recorder.recorded_source().unwrap(), "");
        assert!(recorder.py_error.lock().unwrap().is_none());
        recorder.finish_recording(true);
        let source = recorder.recorded_source().unwrap();
        let phases = Source::with_extension(None, &source, recorder.extension.clone())
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(!phases.is_empty());
        let mut extraction_count = 0;
        for program in phases {
            let result = run(&mut engine, id, program);
            assert_eq!(result.error, None);
            assert!(result.ruleset_publication.is_some());
            if !result.outputs.is_empty() {
                assert_eq!(result.outputs.len(), 1);
                extracted_program(&result, &payload);
                extraction_count += 1;
            }
        }
        assert_eq!(extraction_count, 1);
        assert!(recorder.py_error.lock().unwrap().is_none());
        let mut unused = original.clone();
        unused.commands.clear();
        assert_eq!(run(&mut engine, id, unused).error, None);
        let mut invalid_later = original.clone();
        invalid_later.commands.push(pb::Command {
            kind: Some(pb::command::Kind::Extract(pb::Extract {
                roots: vec![99],
                variants: 1,
                extractor: pb::Extractor::Tree.into(),
                ..Default::default()
            })),
            ..Default::default()
        });
        let rejected = run(&mut engine, id, invalid_later);
        assert_eq!(
            rejected.error.unwrap().code,
            pb::ErrorCode::InvalidProgram as i32
        );
        assert!(rejected.outputs.is_empty());
        assert!(rejected.ruleset_publication.unwrap().names.is_empty());
        let result = run(&mut engine, id, original);
        let replay = extracted_program(&result, &payload);
        extracted_program(&run(&mut engine, id, replay), &payload);
        Python::attach(|py| {
            _export_pyobject_catalog(py).unwrap();
            let builtins = PyModule::import(py, "builtins").unwrap();
            assert_eq!(
                builtins
                    .getattr("_egglog_value_gate_loads")
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                0
            );
            assert_eq!(
                load(py, &value).unwrap().extract::<String>().unwrap(),
                "loaded explicitly"
            );
            assert_eq!(
                builtins
                    .getattr("_egglog_value_gate_loads")
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                1
            );
            builtins.delattr("_egglog_value_gate_loads").unwrap();
        });
        let bad_pickle = b"not a pickle".to_vec();
        assert!(format!("{:?}", PyPickledValue(bad_pickle.clone())).contains("PyPickledValue"));
        extracted_program(
            &run(&mut engine, id, custom_program(bad_pickle.clone())),
            &bad_pickle,
        );
        Python::attach(|py| assert!(load(py, &PyPickledValue(bad_pickle)).is_err()));
    }

    #[test]
    fn recording_provider_preparation_preserves_the_python_error_latch() {
        Python::initialize();
        let mut recorder = ProtoEngine::default();
        let original = PyValueError::new_err("retained provider error");
        let value = Python::attach(|py| original.value(py).clone().unbind());
        *recorder.py_error.lock().unwrap() = Some(original);
        recorder.enable_recording().unwrap();
        recorder
            .stage_submission(Submission::Program(custom_program(
                b"not a pickle".to_vec(),
            )))
            .unwrap();
        assert_eq!(recorder.recorded_source().unwrap(), "");
        recorder.finish_recording(true);
        Source::with_extension(
            None,
            &recorder.recorded_source().unwrap(),
            recorder.extension.clone(),
        )
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
        Python::attach(|py| {
            assert!(
                recorder
                    .py_error
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .value(py)
                    .is(value.bind(py))
            );
        });
    }

    #[test]
    fn custom_rejection_is_before_authored_effects() {
        use egglog_experimental::protobuf::source::Source;
        let phases = Source::new(
            None,
            "(function observed () i64 :merge new) (set (observed) 0) (check (= (observed) 0))",
        )
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
        for configured in [false, true] {
            let mut engine = if configured {
                Engine::with_extension(Arc::new(PyObjectExtension::default()))
            } else {
                Engine::default()
            };
            let id = create(&mut engine);
            for phase in &phases[..2] {
                assert_eq!(run(&mut engine, id, phase.clone()).error, None);
            }
            for invalid in 0..6 {
                let mut program = custom_program(b"inert invalid pickle".to_vec());
                program.sorts.push(pb::Sort {
                    kind: Some(pb::sort::Kind::Family(pb::HostSort {
                        name: "i64".into(),
                        args: vec![],
                    })),
                    ..Default::default()
                });
                program.nodes.push(pb::Node {
                    sort_id: 1,
                    kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                        value: Some(pb::primitive_value::Value::I64(1)),
                    })),
                    ..Default::default()
                });
                program.commands = vec![pb::Command {
                    kind: Some(pb::command::Kind::Action(pb::Action {
                        kind: Some(pb::action::Kind::Set(pb::Set {
                            target: Some(pb::Call {
                                func: "observed".into(),
                                args: vec![],
                            }),
                            value: Some(1),
                        })),
                        ..Default::default()
                    })),
                    ..Default::default()
                }];
                match invalid {
                    0 => {
                        let Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                            value: Some(pb::primitive_value::Value::Custom(value)),
                        })) = &mut program.nodes[0].kind
                        else {
                            unreachable!()
                        };
                        value.args.push(1);
                    }
                    1 => program.nodes[0].sort_id = 1,
                    2 => {
                        let Some(pb::sort::Kind::Family(family)) = &mut program.sorts[0].kind
                        else {
                            unreachable!()
                        };
                        family.args.push(1);
                    }
                    3 => program.commands.push(pb::Command {
                        kind: Some(pb::command::Kind::Extract(pb::Extract {
                            roots: vec![99],
                            variants: 1,
                            extractor: pb::Extractor::Tree.into(),
                            ..Default::default()
                        })),
                        ..Default::default()
                    }),
                    4 => {
                        program.sorts[0].kind = Some(pb::sort::Kind::Eq("PyObject".into()));
                        program.declarations.push(pb::Declaration {
                            kind: Some(pb::declaration::Kind::EqSort(pb::EqSort {
                                name: "PyObject".into(),
                                ..Default::default()
                            })),
                            ..Default::default()
                        });
                    }
                    _ => program.declarations.push(pb::Declaration {
                        kind: Some(pb::declaration::Kind::HostSortFamily(pb::HostSortFamily {
                            name: "PyObject".into(),
                            arity: 1,
                            bindings: None,
                        })),
                        ..Default::default()
                    }),
                }
                let response = run(&mut engine, id, program);
                assert_eq!(
                    response.error.unwrap().code,
                    pb::ErrorCode::InvalidProgram as i32,
                    "configured={configured}, invalid={invalid}"
                );
                assert!(response.outputs.is_empty());
                assert!(response.ruleset_publication.unwrap().names.is_empty());
                assert_eq!(
                    run(&mut engine, id, phases[2].clone()).error,
                    None,
                    "no preceding Set may run"
                );
            }
        }
    }

    #[test]
    fn nested_vec_custom_bytes_roundtrip() {
        let payload = b"inert nested bytes".to_vec();
        let mut program = custom_program(payload.clone());
        program.sorts.push(pb::Sort {
            kind: Some(pb::sort::Kind::Family(pb::HostSort {
                name: "Vec".into(),
                args: vec![0],
            })),
            ..Default::default()
        });
        program.nodes.push(pb::Node {
            sort_id: 1,
            kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::Vec(pb::ValueList {
                    items: vec![0, 0],
                })),
            })),
            ..Default::default()
        });
        let Some(pb::command::Kind::Extract(extract)) = &mut program.commands[0].kind else {
            unreachable!()
        };
        extract.roots = vec![1];
        let mut engine = Engine::with_extension(Arc::new(PyObjectExtension::default()));
        let id = create(&mut engine);
        for _ in 0..2 {
            let result = run(&mut engine, id, program);
            assert_eq!(result.error, None);
            assert!(
                result
                    .ruleset_publication
                    .as_ref()
                    .unwrap()
                    .names
                    .is_empty()
            );
            let Some(pb::command_output::Kind::Extraction(extract)) = &result.outputs[0].kind
            else {
                unreachable!()
            };
            let root = extract.roots[0].variants[0].term;
            let Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::Vec(values)),
            })) = &result.nodes[root as usize].kind
            else {
                panic!("not a Vec")
            };
            assert_eq!(values.items.len(), 2);
            assert_eq!(values.items[0], values.items[1]);
            let Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::Custom(value)),
            })) = &result.nodes[values.items[0] as usize].kind
            else {
                panic!("not Custom")
            };
            assert_eq!(value.payload, payload);
            program = pb::Program {
                ir_version: 1,
                nodes: result.nodes.clone(),
                sorts: result.sorts.clone(),
                commands: vec![pb::Command {
                    kind: Some(pb::command::Kind::Extract(pb::Extract {
                        roots: vec![root],
                        variants: 1,
                        extractor: pb::Extractor::Tree.into(),
                        ..Default::default()
                    })),
                    ..Default::default()
                }],
                ..Default::default()
            };
        }
    }

    // Validates the whole observation boundary and returns an independently
    // owned, re-ingestible arena. This is not an out-of-band result channel.
    fn extracted_program(result: &pb::RunProgramResponse, payload: &[u8]) -> pb::Program {
        assert_eq!(result.error, None);
        assert!(
            result
                .ruleset_publication
                .as_ref()
                .unwrap()
                .names
                .is_empty()
        );
        let Some(pb::command_output::Kind::Extraction(extract)) = &result.outputs[0].kind else {
            panic!("not extraction")
        };
        let extracted = &extract.roots[0].variants[0];
        let node = &result.nodes[extracted.term as usize];
        assert!(
            matches!(&result.sorts[node.sort_id as usize].kind, Some(pb::sort::Kind::Family(family)) if family.name == "PyObject" && family.args.is_empty())
        );
        assert_eq!(
            node.kind,
            Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::Custom(pb::CustomValue {
                    payload: payload.to_vec(),
                    args: vec![]
                }))
            }))
        );
        assert!(
            matches!(&result.nodes[extracted.cost.unwrap() as usize].kind, Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue { value: Some(pb::primitive_value::Value::I64(cost)) })) if *cost > 0)
        );
        pb::Program {
            ir_version: 1,
            nodes: result.nodes.clone(),
            sorts: result.sorts.clone(),
            commands: vec![pb::Command {
                kind: Some(pb::command::Kind::Extract(pb::Extract {
                    roots: vec![extracted.term],
                    variants: 1,
                    extractor: pb::Extractor::Tree.into(),
                    ..Default::default()
                })),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn create(engine: &mut Engine) -> u64 {
        let request = pb::CreateEGraphRequest {
            sorts: vec![pb::Sort {
                kind: Some(pb::sort::Kind::Family(pb::HostSort {
                    name: "i64".into(),
                    args: vec![],
                })),
                ..Default::default()
            }],
            options: Some(pb::EGraphOptions {
                cost_sort: Some(0),
                ..Default::default()
            }),
            ..Default::default()
        };
        pb::CreateEGraphResponse::decode(
            engine.create(&request.encode_to_vec()).unwrap().as_slice(),
        )
        .unwrap()
        .egraph_id
    }

    fn run(engine: &mut Engine, id: u64, program: pb::Program) -> pb::RunProgramResponse {
        let request = pb::RunProgramRequest {
            egraph_id: id,
            program: Some(program),
            profile: true,
            ..Default::default()
        };
        pb::RunProgramResponse::decode(engine.run(&request.encode_to_vec()).unwrap().as_slice())
            .unwrap()
    }

    fn custom_program(payload: Vec<u8>) -> pb::Program {
        pb::Program {
            ir_version: 1,
            sorts: vec![pb::Sort {
                kind: Some(pb::sort::Kind::Family(pb::HostSort {
                    name: "PyObject".into(),
                    args: vec![],
                })),
                ..Default::default()
            }],
            nodes: vec![pb::Node {
                sort_id: 0,
                kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                    value: Some(pb::primitive_value::Value::Custom(pb::CustomValue {
                        payload,
                        args: vec![],
                    })),
                })),
                ..Default::default()
            }],
            commands: vec![pb::Command {
                kind: Some(pb::command::Kind::Extract(pb::Extract {
                    roots: vec![0],
                    variants: 1,
                    extractor: pb::Extractor::Tree.into(),
                    ..Default::default()
                })),
                ..Default::default()
            }],
            ..Default::default()
        }
    }
}

#[pymethods]
impl ProtoEngine {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Typechecking-only source production, never native AST or graph exposure.
    #[staticmethod]
    fn source(py: Python<'_>, text: &str) -> PyResult<Py<ProtoSource>> {
        let source = Source::with_extension(None, text, Arc::new(PyObjectExtension::default()))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Py::new(py, ProtoSource { source })
    }

    /// Produce ordinary row actions from a descriptor without using the live engine.
    fn produce_input<'py>(
        &self,
        py: Python<'py>,
        descriptor: &Bound<'py, PyBytes>,
        table: &str,
        path: &str,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let descriptor = pb::Program::decode(descriptor.as_bytes())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let program = egglog_experimental::protobuf::source::produce_input(
            &descriptor,
            table,
            path,
            None,
            Some(self.extension.clone()),
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(PyBytes::new(py, &program.encode_to_vec()))
    }

    fn enable_recording(&mut self) -> PyResult<()> {
        if self.recording.is_some() {
            return Err(PyValueError::new_err(
                "Source recording is already configured",
            ));
        }
        let renderer = Renderer::with_extension(self.extension.clone())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.recording = Some(Recording::Active {
            renderer: Box::new(renderer),
            staged: None,
            source: String::new(),
        });
        Ok(())
    }

    /// Prepare the actual submitted Program without executing any user command.
    fn prepare_recording(&mut self, program: &Bound<'_, PyBytes>) -> PyResult<()> {
        let program = pb::Program::decode(program.as_bytes())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.stage_submission(Submission::Program(program))
    }

    fn prepare_recording_scope(&mut self, push: bool) -> PyResult<()> {
        // A disabled optional transcript must not block returning from a scope
        // that was pushed while recording was active.
        if !push
            && matches!(
                &self.recording,
                Some(Recording::Invalid | Recording::Closed)
            )
        {
            return Ok(());
        }
        self.stage_submission(if push {
            Submission::Push(1)
        } else {
            Submission::Pop {
                count: 1,
                span: egglog::ast::Span::Panic,
            }
        })
    }

    /// Commit only after Python has validated the actual transport acknowledgement.
    fn finish_recording(&mut self, success: bool) {
        if let Some(Recording::Active {
            renderer,
            staged,
            source,
        }) = &mut self.recording
        {
            if success {
                if let Some((candidate, text)) = staged.take() {
                    *renderer = candidate;
                    source.push_str(&text);
                }
            } else {
                self.recording = Some(Recording::Invalid);
            }
        }
    }

    fn recorded_source(&self) -> PyResult<String> {
        match &self.recording {
            None => Err(PyValueError::new_err(
                "Create the EGraph with save_egglog_string=True",
            )),
            Some(Recording::Closed) => Err(PyValueError::new_err(
                "The saved transcript has been closed",
            )),
            Some(Recording::Invalid) => Err(PyRuntimeError::new_err(
                "The saved transcript cannot be replayed after a failed engine operation",
            )),
            Some(Recording::Active { source, .. }) => Ok(source.clone()),
        }
    }

    fn close_recording(&mut self) {
        if self.recording.is_some() {
            self.recording = Some(Recording::Closed);
        }
    }

    /// Private native-provider registration; execution still submits Extract
    /// with the returned name through run(bytes). The explicit codec is the
    /// existing childless PyObject codec; C remains independently declared.
    fn register_cost_model(
        &mut self,
        handle: u64,
        sorts: &Bound<'_, PyBytes>,
        annotation: u32,
        objective: u32,
        additive: bool,
        factory: Py<PyAny>,
    ) -> PyResult<String> {
        let program = pb::Program::decode(sorts.as_bytes())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if program.ir_version != 1
            || !program.nodes.is_empty()
            || !program.declarations.is_empty()
            || !program.rulesets.is_empty()
            || !program.rules.is_empty()
            || !program.commands.is_empty()
            || !program.files.is_empty()
        {
            return Err(PyValueError::new_err(
                "model registration requires a sort-only Program",
            ));
        }
        let mut canonical = Vec::new();
        egglog_experimental::builtin::import_sort(&program.sorts, annotation, &mut canonical)
            .map_err(PyValueError::new_err)?;
        let result =
            egglog_experimental::builtin::import_sort(&program.sorts, objective, &mut canonical)
                .map_err(PyValueError::new_err)?;
        if !matches!(&canonical[result as usize].kind,
            Some(pb::sort::Kind::Family(family)) if family.name == "PyObject" && family.args.is_empty())
        {
            return Err(PyValueError::new_err(
                "Python model objective requires the explicit PyObject leaf codec",
            ));
        }
        if canonical
            .iter()
            .any(|sort| matches!(sort.kind, Some(pb::sort::Kind::Var(_))))
        {
            return Err(PyValueError::new_err("cost model sorts must be closed"));
        }
        self.engine
            .register_cost_model(
                handle,
                Arc::new(PythonModelRegistration {
                    sorts: program.sorts,
                    annotation,
                    objective,
                    additive,
                    factory,
                    errors: self.py_error.clone(),
                }),
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn release_cost_model(&mut self, handle: u64, name: &str) -> PyResult<()> {
        self.engine
            .release_cost_model(handle, name)
            .map_err(|error| PyValueError::new_err(error.to_string()))
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
    ) -> PyResult<(Bound<'py, PyBytes>, Option<Py<PyBaseException>>)> {
        self.run_request(py, request, Engine::run)
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

    fn return_to<'py>(
        &mut self,
        py: Python<'py>,
        request: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.request(py, request, Engine::return_to)
    }
}
