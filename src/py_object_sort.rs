use base64::{Engine as _, engine::general_purpose::STANDARD};
use core::fmt;
use egglog::{
    ArcSort, BaseValue, TermDag, TermId, Value, add_primitive,
    ast::Literal,
    prelude::{BaseSort, EGraph},
    sort::{BaseValues, S},
};
use egglog_experimental::{
    add_primitive as add_portable_primitive, builtin::PortableEGraph, proto as pb,
};
use std::{
    any::TypeId,
    env::temp_dir,
    ffi::{CStr, CString},
    fmt::Debug,
    fs::File,
    io::Write,
    sync::{Arc, Mutex},
};
use uuid::Uuid;

use crate::error::PrimitiveFailure;

use pyo3::{
    PyAny, PyErr, PyResult, PyTypeCheck, Python,
    prelude::*,
    types::{PyCode, PyCodeMethods as _, PyDict, PyTuple},
};

/// A pickled Python object.
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct PyPickledValue(pub Vec<u8>);

impl Debug for PyPickledValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("PyPickledValue").field(&self.0).finish()
    }
}

impl BaseValue for PyPickledValue {}

pub fn dump<X: PyTypeCheck>(obj: Bound<'_, X>) -> PyResult<PyPickledValue> {
    let cloudpickle = PyModule::import(obj.py(), "cloudpickle")?;

    let bytes: Vec<u8> = cloudpickle.getattr("dumps")?.call1((obj,))?.extract()?;
    Ok(PyPickledValue(bytes))
}

pub fn load<'py>(py: Python<'py>, pickled: &PyPickledValue) -> PyResult<Bound<'py, PyAny>> {
    let cloudpickle = PyModule::import(py, "cloudpickle")?;
    cloudpickle.getattr("loads")?.call1((&pickled.0,))
}

pub(crate) type PyObjectErrorState = Arc<Mutex<Option<PyErr>>>;

pub struct PyObjectSort {
    pub(crate) py_error: PyObjectErrorState,
}

impl Debug for PyObjectSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PyObjectSort").finish_non_exhaustive()
    }
}

impl BaseSort for PyObjectSort {
    type Base = PyPickledValue;

    fn name(&self) -> &str {
        "PyObject"
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        eg.type_info()
            .register_builtin_family(egglog::builtin::FamilyDescription {
                name: "PyObject".into(),
                arity: 0,
            })
            .expect("invalid PyObject native family");
        // (py-object <bytes>)
        add_primitive!(
            eg,
            "py-object"[id = "egglog.python.pyobject.from_pickle"] = |ident: S| -> PyPickledValue {
                PyPickledValue(STANDARD.decode(ident.as_bytes()).unwrap())
            }
        );
        // Register the real default-expression helper before its consumers.
        // (py-dict [<key-object> <value-object>]*)
        add_portable_primitive!(eg, "py-dict" [id = "egglog.python.pyobject.dict"] = {self.py_error.clone(): PyObjectErrorState} [xs: PyPickledValue] -?> PyPickledValue {
            attach(&self.ctx, "py-dict", |py| {
                let dict = PyDict::new(py);
                for i in xs.map(|x| load(py, &x)).collect::<PyResult<Vec<_>>>()?.chunks_exact(2) {
                    dict.set_item(i[0].clone(), i[1].clone())?;
                }
                dump(dict.into())
            })
        });
        // Supports calling (py-eval <str-obj> <globals-obj> <locals-obj>)
        add_portable_primitive!(eg, "py-eval" [id = "egglog.python.pyobject.eval"] = {self.py_error.clone(): PyObjectErrorState} |code: S, globals: PyPickledValue, locals: PyPickledValue| -?> PyPickledValue {
            attach(&self.ctx, "py-eval", |py| {
                dump(py.eval(
                    CString::new(code.to_string()).unwrap().as_c_str(),
                    Some(load(py, &globals)?.cast::<PyDict>()?),
                    Some(load(py, &locals)?.cast::<PyDict>()?),
                )?)
            })
        });
        // Copies the locals, execs the Python string, then returns the copied version of the locals with any updates
        // (py-exec <str-obj> <globals-obj> <locals-obj>)
        add_portable_primitive!(
            eg,
            "py-exec" [id = "egglog.python.pyobject.exec"] =
                {self.py_error.clone(): PyObjectErrorState}
                |code: S, globals: PyPickledValue, locals: PyPickledValue| -?> PyPickledValue {
                    attach(&self.ctx, "py-exec", |py| {
                        let locals = load(py, &locals)?;
                        // Copy code into temporary file
                        // Keep it around so that if errors occur we can debug them after the program exits
                        let mut path = temp_dir();
                        let file_name = format!("egglog-{}.py", Uuid::new_v4());
                        path.push(file_name);
                        let mut file = File::create(path.clone()).unwrap();
                        file.write_all(code.as_bytes()).unwrap();
                        let path = path.to_str().unwrap();
                        run_path(
                            py,
                            CString::new(code.into_inner()).unwrap().as_c_str(),
                            Some(load(py, &globals)?.cast::<PyDict>()?),
                            Some(locals.cast::<PyDict>()?),
                            CString::new(path).unwrap().as_c_str(),
                        )?;
                        dump(locals)
                    })
                }
        );
        // (py-call <fn-obj> [<arg-object>]*)
        add_portable_primitive!(eg, "py-call" [id = "egglog.python.pyobject.call"] = {self.py_error.clone(): PyObjectErrorState} [xs: PyPickledValue] -?> PyPickledValue {
            attach(&self.ctx, "py-call", |py| {
                let xs = xs.map(|x| load(py, &x)).collect::<PyResult<Vec<_>>>().map_err(|e| {if !e.is_instance_of::<PrimitiveFailure>(py) {let _ = e.add_note(py, "Loading arguments");} e})?;
                let fn_obj = &xs[0];
                let args = PyTuple::new(py, xs[1..].to_vec()).map_err(|e| {if !e.is_instance_of::<PrimitiveFailure>(py) {let _ = e.add_note(py, "Creating tuple");} e})?;
                dump(fn_obj.call1(args).map_err(|e| {if !e.is_instance_of::<PrimitiveFailure>(py) {let _ = e.add_note(py, format!("Calling function {}", fn_obj));} e})?)
            })
        });

        // (py-call-extended <fn-obj> <args-obj> <kwargs-obj>)
        add_portable_primitive!(eg, "py-call-extended" [id = "egglog.python.pyobject.call_extended"] = {self.py_error.clone(): PyObjectErrorState} |fn_: PyPickledValue, args: PyPickledValue, kwargs: PyPickledValue| -?> PyPickledValue {
            attach(&self.ctx, "py-call-extended", |py| {
                let fn_ = load(py, &fn_)?;
                let args = load(py, &args)?;
                let kwargs = load(py, &kwargs)?;
                dump(fn_.call(args.cast::<PyTuple>().map_err(|e| {e})?, Some(kwargs.cast::<PyDict>()?))?)
            })
        });

        // Supports calling (py-dict-update <dict-obj> [<key-object> <value-obj>]*)
        add_portable_primitive!(eg, "py-dict-update" [id = "egglog.python.pyobject.dict_update"] = {self.py_error.clone(): PyObjectErrorState} [xs: PyPickledValue] -?> PyPickledValue {{
            attach(&self.ctx, "py-dict-update", |py| {
                let xs = xs.map(|x| load(py, &x)).collect::<PyResult<Vec<_>>>()?;
                // Copy the dict so we can mutate it and return it
                let dict = xs[0].cast::<PyDict>()?;
                // Update the dict with the key-value pairs
                for i in xs[1..].chunks_exact(2) {
                    dict.set_item(i[0].clone(), i[1].clone())?;
                }
                dump(dict.clone())
            })
        }});
        // (py-to-string <obj>)
        add_portable_primitive!(
            eg,
            "py-to-string" [id = "egglog.python.pyobject.to_string"] = {self.py_error.clone(): PyObjectErrorState} |x: PyPickledValue| -?> S {
                {
                    let s: String = attach(&self.ctx, "py-to-string", move |py| load(py, &x)?.extract())?;
                    Some(s.into())
                }
            }
        );
        // (py-to-bool <obj>)
        add_portable_primitive!(
            eg,
            "py-to-bool" [id = "egglog.python.pyobject.to_bool"] = {self.py_error.clone(): PyObjectErrorState} |x: PyPickledValue| -?> bool {
                {
                    attach(&self.ctx, "py-to-bool", move |py| load(py, &x)?.extract())
                }
            }
        );
        // (py-from-string <str>)
        add_portable_primitive!(
            eg,
            "py-from-string" [id = "egglog.python.pyobject.from_string"] = {self.py_error.clone(): PyObjectErrorState} |x: S| -?> PyPickledValue {
                attach(&self.ctx, "py-from-string", |py| {
                    dump(x.to_string().into_pyobject(py)?)
                })
            }
        );
        // (py-from-int <int>)
        add_portable_primitive!(
            eg,
            "py-from-int" [id = "egglog.python.pyobject.from_int"] = {self.py_error.clone(): PyObjectErrorState} |x: i64| -?> PyPickledValue {
                attach(&self.ctx, "py-from-int", |py| {
                    dump(x.into_pyobject(py)?)
                })
            }
        );
    }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> TermId {
        let ident = base_values.unwrap::<PyPickledValue>(value);
        let arg = termdag.lit(Literal::String(STANDARD.encode(&ident.0).into()));
        termdag.app("py-object".into(), vec![arg])
    }
}

/// Explicit native Python capability. Each owning byte engine supplies its own
/// first-error latch; native handles and clones in that engine share it.
#[derive(Default)]
pub(crate) struct PyObjectExtension {
    pub(crate) py_error: PyObjectErrorState,
}

impl egglog_experimental::protobuf::NativeExtension for PyObjectExtension {
    fn initialize(&self, graph: &mut EGraph) -> Result<(), String> {
        egglog::prelude::add_base_sort(
            graph,
            PyObjectSort {
                py_error: self.py_error.clone(),
            },
            egglog::span!(),
        )
        .map_err(|error| error.to_string())
    }

    fn lower_custom_value(
        &self,
        graph: &EGraph,
        native_sort: &ArcSort,
        value: &pb::CustomValue,
        destination: &mut pb::Program,
    ) -> Result<pb::Call, String> {
        if native_sort.name() != "PyObject"
            || native_sort.is_eq_sort()
            || native_sort.is_container_sort()
            || native_sort.value_type() != Some(TypeId::of::<PyPickledValue>())
        {
            return Err("Custom value requires the actual native PyObject sort".into());
        }
        if !value.args.is_empty() {
            return Err("PyObject custom value children are not supported".into());
        }
        let sort_id = graph.export_sort(
            graph
                .get_sort_by_name("String")
                .ok_or("native String sort unavailable")?,
            &mut destination.sorts,
        )?;
        let index =
            u32::try_from(destination.nodes.len()).map_err(|_| "too many custom helper nodes")?;
        destination.nodes.push(pb::Node {
            sort_id,
            kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::String(
                    STANDARD.encode(&value.payload),
                )),
            })),
        });
        Ok(pb::Call {
            func: "egglog.python.pyobject.from_pickle".into(),
            args: vec![index],
        })
    }

    fn encode_observed_custom_value(
        &self,
        graph: &EGraph,
        sort: &ArcSort,
        value: Value,
    ) -> Result<Option<pb::CustomValue>, String> {
        if sort.value_type() != Some(TypeId::of::<PyPickledValue>()) {
            return Ok(None);
        }
        if !graph
            .get_sort_by_name("PyObject")
            .is_some_and(|actual| Arc::ptr_eq(actual, sort))
        {
            return Err("PyObject native sort provenance mismatch".into());
        }
        Ok(Some(pb::CustomValue {
            payload: graph.value_to_base::<PyPickledValue>(value).0.clone(),
            args: vec![],
        }))
    }
}

/// Attaches to the Python interpreter and runs the given closure.
///
/// Expected failures return None; other errors save the first one for the binding thread.
fn attach<F, R>(py_error: &PyObjectErrorState, name: &str, f: F) -> Option<R>
where
    F: for<'py> FnOnce(Python<'py>) -> PyResult<R>,
{
    Python::attach(|py| {
        if py_error.lock().unwrap().is_some() {
            return None;
        };
        match f(py) {
            Ok(val) => Some(val),
            Err(err) if err.is_instance_of::<PrimitiveFailure>(py) => None,
            Err(err) => {
                // User exception methods/state must not replace the first error.
                let _ = err.add_note(py, format!("While calling primitive '{}'", name));
                let mut first_error = py_error.lock().unwrap();
                if first_error.is_none() {
                    *first_error = Some(err);
                }
                None
            }
        }
    })
}
/// Runs the code in the given context with a certain path.
/// Copied from `run`, but allows specifying the path.
/// https://github.com/PyO3/pyo3/blob/55d379cff8e4157024ffe22215715bd04a5fb1a1/src/marker.rs#L667-L682
fn run_path<'py>(
    py: Python<'py>,
    code: &CStr,
    globals: Option<&Bound<'py, PyDict>>,
    locals: Option<&Bound<'py, PyDict>>,
    path: &CStr,
) -> PyResult<()> {
    let code = PyCode::compile(py, code, path, pyo3::types::PyCodeInput::File)?;
    code.run(globals, locals).map(|obj| {
        debug_assert!(obj.is_none());
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::PrimitiveFailure;
    use pyo3::exceptions::PyValueError;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn expected_failure_does_not_latch_or_suppress_later_callbacks() {
        Python::initialize();
        let state = PyObjectErrorState::default();
        let result: Option<()> = attach(&state, "py-call", |_py| {
            Err(PrimitiveFailure::new_err("expected miss"))
        });
        assert!(result.is_none());
        assert!(
            state.lock().unwrap().is_none(),
            "expected failure was latched"
        );
        assert_eq!(attach(&state, "py-call", |_py| Ok(7)), Some(7));

        let unexpected = PyValueError::new_err("unexpected error after expected miss");
        let original = Python::attach(|py| unexpected.value(py).clone().unbind());
        let result: Option<()> = attach(&state, "py-call", |_py| Err(unexpected));
        assert!(result.is_none());
        let result: Option<()> = attach(&state, "py-call", |_py| {
            panic!("an expected failure must not run after a latched unexpected error")
        });
        assert!(result.is_none());
        let captured = state.lock().unwrap().take().unwrap();
        Python::attach(|py| assert!(captured.value(py).is(original.bind(py))));
    }

    #[test]
    fn custom_codec_reads_actual_native_payload_and_checks_provenance() {
        use egglog_experimental::protobuf::NativeExtension;
        let extension = PyObjectExtension::default();
        let mut graph = egglog_experimental::new_experimental_egraph();
        extension.initialize(&mut graph).unwrap();
        let sort = graph.get_sort_by_name("PyObject").unwrap();
        let native = graph.base_to_value(PyPickledValue(b"inert\0payload".to_vec()));
        let value = extension
            .encode_observed_custom_value(&graph, sort, native)
            .unwrap()
            .unwrap();
        assert_eq!(value.payload, b"inert\0payload");
        assert!(value.args.is_empty());
        let string = graph.get_sort_by_name("String").unwrap();
        assert!(
            extension
                .encode_observed_custom_value(
                    &graph,
                    string,
                    graph.base_to_value(S::new("different type".into()))
                )
                .unwrap()
                .is_none()
        );
        let original = pb::Program::default();
        let mut destination = original.clone();
        assert!(
            extension
                .lower_custom_value(&graph, string, &value, &mut destination)
                .is_err()
        );
        assert_eq!(destination, original);
        let child_bearing = pb::CustomValue {
            args: vec![0],
            ..value.clone()
        };
        assert!(
            extension
                .lower_custom_value(&graph, sort, &child_bearing, &mut destination)
                .is_err()
        );
        assert_eq!(destination, original);
        let mut other = egglog_experimental::new_experimental_egraph();
        extension.initialize(&mut other).unwrap();
        assert!(
            extension
                .encode_observed_custom_value(&other, sort, native)
                .is_err()
        );
    }

    #[test]
    fn shares_only_the_first_primitive_error_across_threads() {
        Python::initialize();
        let error = PyObjectErrorState::default();
        let worker_error = error.clone();
        let called_after_error = Arc::new(AtomicBool::new(false));
        let worker_called_after_error = called_after_error.clone();

        std::thread::spawn(move || {
            let result: Option<()> = attach(&worker_error, "py-call", |_py| {
                Err(PyValueError::new_err("worker boom"))
            });
            assert!(result.is_none());

            let result = attach(&worker_error, "py-call", |_py| {
                worker_called_after_error.store(true, Ordering::Relaxed);
                Ok(())
            });
            assert!(result.is_none());
            Python::attach(|py| assert!(!PyErr::occurred(py)));
        })
        .join()
        .unwrap();

        assert!(!called_after_error.load(Ordering::Relaxed));
        let captured = error.lock().unwrap().take().unwrap();
        Python::attach(|py| {
            assert!(captured.is_instance_of::<PyValueError>(py));
            assert_eq!(captured.value(py).to_string(), "worker boom");
            let notes: Vec<String> = captured
                .value(py)
                .getattr("__notes__")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(notes, ["While calling primitive 'py-call'"]);
        });
    }
}
