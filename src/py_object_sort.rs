use base64::{Engine as _, engine::general_purpose::STANDARD};
use core::fmt;
/// This module defines a sort for Python objects in the Egglog framework.
///
/// Python objects are stored in a vec and referenced by their index.
///
/// We also keep two lookups, matching hashes and IDs to their indexes, so that when we add a new Python object, if
/// it's hashable, we can look it up by its hash, and if it's unhashable, we can look it up by its ID.
///
///
use egglog::{
    BaseValue, TermDag, TermId, Value, add_primitive,
    ast::Literal,
    prelude::{BaseSort, EGraph},
    sort::{BaseValues, S},
};
use std::{
    env::temp_dir,
    ffi::{CStr, CString},
    fmt::Debug,
    fs::File,
    io::Write,
    sync::{Arc, Mutex},
};
use uuid::Uuid;

use pyo3::{
    PyAny, PyErr, PyResult, Python,
    prelude::*,
    types::{PyCode, PyCodeMethods as _, PyDict, PyTuple},
};

/// A pickled Python object.
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct PyPickledValue(pub Vec<u8>);

impl Debug for PyPickledValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Python::attach(|py| load(py, self).unwrap().fmt(f))
    }
}

impl BaseValue for PyPickledValue {}

pub fn dump<X>(obj: Bound<X>) -> PyResult<PyPickledValue> {
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
        // (py-object <bytes>)
        add_primitive!(
            eg,
            "py-object" = |ident: S| -> PyPickledValue {
                PyPickledValue(STANDARD.decode(ident.as_bytes()).unwrap())
            }
        );
        // Supports calling (py-eval <str-obj> <globals-obj> <locals-obj>)
        add_primitive!(eg, "py-eval" = {self.py_error.clone(): PyObjectErrorState} |code: S, globals: PyPickledValue, locals: PyPickledValue| -?> PyPickledValue {
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
        add_primitive!(
            eg,
            "py-exec" =
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
        add_primitive!(eg, "py-call" = {self.py_error.clone(): PyObjectErrorState} [xs: PyPickledValue] -?> PyPickledValue {
            attach(&self.ctx, "py-call", |py| {
                let xs = xs.map(|x| load(py, &x)).collect::<PyResult<Vec<_>>>().map_err(|e| {e.add_note(py, "Loading arguments").unwrap(); e})?;
                let fn_obj = &xs[0];
                let args = PyTuple::new(py, xs[1..].to_vec()).map_err(|e| {e.add_note(py, "Creating tuple").unwrap(); e})?;
                dump(fn_obj.call1(args).map_err(|e| {e.add_note(py, format!("Calling function {}", fn_obj)).unwrap(); e})?)
            })
        });

        // (py-call-extended <fn-obj> <args-obj> <kwargs-obj>)
        add_primitive!(eg, "py-call-extended" = {self.py_error.clone(): PyObjectErrorState} |fn_: PyPickledValue, args: PyPickledValue, kwargs: PyPickledValue| -?> PyPickledValue {
            attach(&self.ctx, "py-call-extended", |py| {
                let fn_ = load(py, &fn_)?;
                let args = load(py, &args)?;
                let kwargs = load(py, &kwargs)?;
                dump(fn_.call(args.cast::<PyTuple>().map_err(|e| {e})?, Some(kwargs.cast::<PyDict>()?))?)
            })
        });

        // (py-dict [<key-object> <value-object>]*)
        add_primitive!(eg, "py-dict" = {self.py_error.clone(): PyObjectErrorState} [xs: PyPickledValue] -?> PyPickledValue {
            attach(&self.ctx, "py-dict", |py| {
                let dict = PyDict::new(py);
                for i in xs.map(|x| load(py, &x)).collect::<PyResult<Vec<_>>>()?.chunks_exact(2) {
                    dict.set_item(i[0].clone(), i[1].clone())?;
                }
                dump(dict.into())
            })
        });
        // Supports calling (py-dict-update <dict-obj> [<key-object> <value-obj>]*)
        add_primitive!(eg, "py-dict-update" = {self.py_error.clone(): PyObjectErrorState} [xs: PyPickledValue] -?> PyPickledValue {{
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
        add_primitive!(
            eg,
            "py-to-string" = {self.py_error.clone(): PyObjectErrorState} |x: PyPickledValue| -?> S {
                {
                    let s: String = attach(&self.ctx, "py-to-string", move |py| load(py, &x)?.extract())?;
                    Some(s.into())
                }
            }
        );
        // (py-to-bool <obj>)
        add_primitive!(
            eg,
            "py-to-bool" = {self.py_error.clone(): PyObjectErrorState} |x: PyPickledValue| -?> bool {
                {
                    attach(&self.ctx, "py-to-bool", move |py| load(py, &x)?.extract())
                }
            }
        );
        // (py-from-string <str>)
        add_primitive!(
            eg,
            "py-from-string" = {self.py_error.clone(): PyObjectErrorState} |x: S| -?> PyPickledValue {
                attach(&self.ctx, "py-from-string", |py| {
                    dump(x.to_string().into_pyobject(py)?)
                })
            }
        );
        // (py-from-int <int>)
        add_primitive!(
            eg,
            "py-from-int" = {self.py_error.clone(): PyObjectErrorState} |x: i64| -?> PyPickledValue {
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

/// Attaches to the Python interpreter and runs the given closure.
///
/// Also handles errors, by saving the first one for the binding thread and returning None.
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
            Err(err) => {
                err.add_note(py, format!("While calling primitive '{}'", name))
                    .unwrap();
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
    use pyo3::exceptions::PyValueError;
    use std::sync::atomic::{AtomicBool, Ordering};

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
