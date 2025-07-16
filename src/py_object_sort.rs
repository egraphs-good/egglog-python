/// This module defines a sort for Python objects in the Egglog framework.
///
/// Python objects are stored in a vec and referenced by their index.
///
/// We also keep two lookups, matching hashes and IDs to their indexes, so that when we add a new Python object, if
/// it's hashable, we can look it up by its hash, and if it's unhashable, we can look it up by its ID.
///
///
use egglog::{
    Term, TermDag, Value, add_primitive,
    ast::{Expr, Literal},
    call, lit,
    prelude::{BaseSort, EGraph},
    sort::{BaseValues, S},
};
use std::{
    collections::HashMap,
    env::temp_dir,
    ffi::CString,
    fmt::Debug,
    fs::File,
    io::Write,
    sync::{Arc, Mutex},
};
use uuid::Uuid;

use pyo3::{
    AsPyPointer, PyAny, PyErr, PyObject, PyResult, PyTraverseError, PyVisit, Python, ffi, intern,
    prelude::*, types::PyDict,
};

type PyObjectIdent = usize;

#[derive(Clone)]
#[pyclass]
pub struct PyObjectSort {
    /// All the Python objects stored in this sort.
    /// All objects with the same ID are only stored once.
    /// Also, for hashable objects, equal objects are only stored once.
    objects: Arc<Mutex<Vec<Py<PyAny>>>>,
    /// Maps from IDs to their index
    id_to_index: Arc<Mutex<HashMap<usize, PyObjectIdent>>>,
    /// Maps from all hashable objects to their index
    hashable_to_index: Arc<Py<PyDict>>,
}

impl Debug for PyObjectSort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let hashable_to_index =
            Python::with_gil(|py| self.hashable_to_index.bind(py).str().map_or_else(|_| "<error>".to_string(), |s| s.to_string()));
        f.debug_struct("PyObjectSort")
            .field("objects", &self.objects)
            .field("id_to_index", &self.id_to_index)
            .field("hashable_to_index", &hashable_to_index)
            .finish()
    }
}

impl PyObjectSort {
    /// Stores a Python object and return the index.
    ///
    /// If it has already been stored, it returns the existing index.
    /// If it is hashable, it will be stored in the `hashable_to_index` dictionary.
    pub fn store<'py>(&self, py: Python<'py>, obj: PyObject) -> PyResult<PyObjectIdent> {
        // 1. Check if the object is already stored
        let id = obj.as_ptr() as usize;
        let mut id_to_index = self.id_to_index.lock().unwrap();
        if let Some(index) = id_to_index.get(&id) {
            return Ok(*index);
        }
        // 2. If not, try looking the object up by its hash
        let hashable_to_index = self.hashable_to_index.bind(py);
        let mut objects = self.objects.lock().unwrap();
        match hashable_to_index.get_item(obj.clone_ref(py)) {
            // An error means it's not hashable, so we store it by ID
            Err(_) => {
                let index = objects.len();
                objects.push(obj);
                id_to_index.insert(id, index);
                Ok(index)
            }
            // OK means it's hashable
            Ok(result) => {
                match result {
                    // If the object is already stored, return its index
                    Some(index) => index.extract(),
                    // If not, store it in the hashable_to_index dictionary
                    None => {
                        let index = objects.len();
                        objects.push(obj.clone_ref(py));
                        id_to_index.insert(id, index);
                        hashable_to_index.set_item(obj, index)?;
                        Ok(index)
                    }
                }
            }
        }
    }

    /// Loads a Python object based on it's index
    pub fn load<'py>(&self, py: Python<'py>, ident: PyObjectIdent) -> Bound<'py, PyAny> {
        self.objects.lock().unwrap()[ident].bind(py).clone()
    }
}

#[pymethods]
impl PyObjectSort {
    #[new]
    fn new<'py>(py: Python<'py>) -> Self {
        Self {
            objects: Arc::new(Mutex::new(Vec::new())),
            id_to_index: Arc::new(Mutex::new(HashMap::new())),
            hashable_to_index: Arc::new(PyDict::new(py).into()),
        }
    }

    /// Store a Python object and return an Expr that points to it.
    #[pyo3(name="store", signature = (obj, /))]
    fn store_py<'py>(
        &mut self,
        py: Python<'py>,
        obj: PyObject,
    ) -> PyResult<crate::conversions::Expr> {
        let ident = self.store(py, obj)?;
        let arg = lit!(Literal::Int(ident as i64));
        Ok(call!("py-object", vec![arg]).into())
    }

    /// Retrieve the Python object from an expression
    #[pyo3(name="load", signature = (expr, /))]
    fn load_py<'py>(
        &self,
        py: Python<'py>,
        expr: crate::conversions::Expr,
    ) -> PyResult<Bound<'py, PyAny>> {
        let expr: Expr = expr.into();
        let err = Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected a py-object call with a single integer literal argument",
        ));
        let Expr::Call(_, name, args) = &expr else {
            return err;
        };
        if name != "py-object" {
            return err;
        }
        let [Expr::Lit(_, Literal::Int(i))] = &args[..] else {
            return err;
        };

        Ok(self.load(py, *i as usize))
    }

    // Integrate with Python garbage collector
    // https://pyo3.rs/main/class/protocols#garbage-collector-integration

    fn __traverse__<'py>(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(self.hashable_to_index.as_ref())?;
        self.objects
            .lock()
            .unwrap()
            .iter()
            .try_for_each(|obj| visit.call(obj))
    }

    fn __clear__<'py>(&mut self, py: Python<'py>) {
        self.hashable_to_index.bind(py).clear();
        self.objects.lock().unwrap().clear();
        self.id_to_index.lock().unwrap().clear();
    }
}

impl BaseSort for PyObjectSort {
    type Base = PyObjectIdent;

    fn name(&self) -> &str {
        "PyObject"
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        // (py-object <i64>)
        add_primitive!(
            eg,
            "py-object" = |ident: i64| -> PyObjectIdent { ident as usize }
        );
        // Supports calling (py-eval <str-obj> <globals-obj> <locals-obj>)
        add_primitive!(
            eg,
            "py-eval" = {self.clone(): PyObjectSort}
                |code: S, globals: PyObjectIdent, locals: PyObjectIdent| -> PyObjectIdent {
                    {
                        Python::with_gil(|py| {
                            let globals = self.ctx.load(py, globals);
                            let locals = self.ctx.load(py, locals);
                            let res = py
                                .eval(
                                    CString::new(code.to_string()).unwrap().as_c_str(),
                                    Some(globals.downcast::<PyDict>().unwrap()),
                                    Some(locals.downcast::<PyDict>().unwrap()),
                                )
                                .unwrap();
                            self.ctx.store(py, res.unbind()).unwrap()
                        })
                    }
                }
        );
        // Copies the locals, execs the Python string, then returns the copied version of the locals with any updates
        // (py-exec <str-obj> <globals-obj> <locals-obj>)
        add_primitive!(
            eg,
            "py-exec" = {self.clone(): PyObjectSort}
                |code: S, globals: PyObjectIdent, locals: PyObjectIdent| -> PyObjectIdent {
                    Python::with_gil(|py| {
                        let globals = self.ctx.load(py, globals);
                        let locals = self.ctx.load(py, locals);

                        // Copy the locals so we can mutate them and return them
                        let locals = locals.downcast::<PyDict>().unwrap().copy().unwrap();
                        // Copy code into temporary file
                        // Keep it around so that if errors occur we can debug them after the program exits
                        let mut path = temp_dir();
                        let file_name = format!("egglog-{}.py", Uuid::new_v4());
                        path.push(file_name);
                        let mut file = File::create(path.clone()).unwrap();
                        file.write_all(code.as_bytes()).unwrap();
                        let path = path.to_str().unwrap();
                        run_code_path(
                            py,
                            &code,
                            Some(globals.downcast::<PyDict>().unwrap()),
                            Some(&locals),
                            path,
                        )
                        .unwrap();
                        self.ctx.store(py, locals.unbind().into()).unwrap()
                    })
                }
        );

        // (py-dict [<key-object> <value-object>]*)
        add_primitive!(eg, "py-dict" = {self.clone(): PyObjectSort} [xs: PyObjectIdent] -> PyObjectIdent {
            Python::with_gil(|py| {
                let dict = PyDict::new(py);
                for i in xs.map(|x| self.ctx.load(py, x)).collect::<Vec<_>>().chunks_exact(2) {
                    dict.set_item(i[0].clone(), i[1].clone()).unwrap();
                }
                self.ctx.store(py, dict.unbind().into()).unwrap()
            })
        });
        // Supports calling (py-dict-update <dict-obj> [<key-object> <value-obj>]*)
        add_primitive!(eg, "py-dict-update" = {self.clone(): PyObjectSort} [xs: PyObjectIdent] -> PyObjectIdent {{
            Python::with_gil(|py| {
                let xs = xs.map(|x| self.ctx.load(py, x)).collect::<Vec<_>>();
                // Copy the dict so we can mutate it and return it
                let dict = xs[0].downcast::<PyDict>().unwrap().copy().unwrap();
                // Update the dict with the key-value pairs
                for i in xs[1..].chunks_exact(2) {
                    dict.set_item(i[0].clone(), i[1].clone()).unwrap();
                }
                self.ctx.store(py, dict.unbind().into()).unwrap()
            })
        }});
        // (py-to-string <obj>)
        add_primitive!(
            eg,
            "py-to-string" = {self.clone(): PyObjectSort} |x: PyObjectIdent| -> S {
                {
                    let s: String = Python::with_gil(move |py| self.ctx.load(py, x).extract().unwrap());
                    s.into()
                }
            }
        );
        // (py-to-bool <obj>)
        add_primitive!(
            eg,
            "py-to-bool" = {self.clone(): PyObjectSort} |x: PyObjectIdent| -> bool {
                {
                    Python::with_gil(move |py| self.ctx.load(py, x).extract().unwrap())
                }
            }
        );
        // (py-from-string <str>)
        add_primitive!(
            eg,
            "py-from-string" = {self.clone(): PyObjectSort} |x: S| -> PyObjectIdent {
                Python::with_gil(|py| {
                    let obj = x.to_string().into_pyobject(py).unwrap();
                    self.ctx.store(py, obj.unbind().into()).unwrap()
                })
            }
        );
        // (py-from-int <int>)
        add_primitive!(
            eg,
            "py-from-int" = {self.clone(): PyObjectSort} |x: i64| -> PyObjectIdent {
                Python::with_gil(|py| {
                    let obj = x.into_pyobject(py).unwrap();
                    self.ctx.store(py, obj.unbind().into()).unwrap()
                })
            }
        );
    }

    fn reconstruct_termdag(
        &self,
        base_values: &BaseValues,
        value: Value,
        termdag: &mut TermDag,
    ) -> Term {
        let ident = base_values.unwrap::<PyObjectIdent>(value);
        let arg = termdag.lit(Literal::Int(ident as i64));
        termdag.app("py-object".into(), vec![arg])
    }
}

/// Runs the code in the given context with a certain path.
/// Copied from `run_code`, but allows specifying the path.
/// https://github.com/PyO3/pyo3/blob/5d2f5b5702319150d41258de77f589119134ee74/src/marker.rs#L678
fn run_code_path<'py>(
    py: Python<'py>,
    code: &str,
    globals: Option<&Bound<'py, PyDict>>,
    locals: Option<&Bound<'py, PyDict>>,
    path: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let code = CString::new(code)?;
    let path = CString::new(path)?;
    unsafe {
        let mptr = ffi::PyImport_AddModule("__main__\0".as_ptr() as *const _);
        if mptr.is_null() {
            return Err(PyErr::fetch(py));
        }

        let globals = globals
            .map(AsPyPointer::as_ptr)
            .unwrap_or_else(|| ffi::PyModule_GetDict(mptr));
        let locals = locals.map(AsPyPointer::as_ptr).unwrap_or(globals);

        // If `globals` don't provide `__builtins__`, most of the code will fail if Python
        // version is <3.10. That's probably not what user intended, so insert `__builtins__`
        // for them.
        //
        // See also:
        // - https://github.com/python/cpython/pull/24564 (the same fix in CPython 3.10)
        // - https://github.com/PyO3/pyo3/issues/3370
        let builtins_s = intern!(py, "__builtins__").as_ptr();
        let has_builtins = ffi::PyDict_Contains(globals, builtins_s);
        if has_builtins == -1 {
            return Err(PyErr::fetch(py));
        }
        if has_builtins == 0 {
            // Inherit current builtins.
            let builtins = ffi::PyEval_GetBuiltins();

            // `PyDict_SetItem` doesn't take ownership of `builtins`, but `PyEval_GetBuiltins`
            // seems to return a borrowed reference, so no leak here.
            if ffi::PyDict_SetItem(globals, builtins_s, builtins) == -1 {
                return Err(PyErr::fetch(py));
            }
        }

        let code_obj = ffi::Py_CompileString(code.as_ptr(), path.as_ptr() as _, ffi::Py_file_input);
        if code_obj.is_null() {
            return Err(PyErr::fetch(py));
        }
        let res_ptr = ffi::PyEval_EvalCode(code_obj, globals, locals);
        ffi::Py_DECREF(code_obj);

        Bound::from_owned_ptr_or_err(py, res_ptr).map(|instance| instance.downcast_into_unchecked())
    }
}
