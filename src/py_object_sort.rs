use crate::error::EggResult;
use egglog::sort::IntoSort;
use egglog::{
    add_primitive,
    ast::{Expr, Literal, Span, Symbol},
    call,
    extract::{Cost, Extractor},
    lit,
    sort::{ColumnTy, FromSort, Sort},
    util::IndexMap,
    EGraph, Term, TermDag, Value,
};
use pyo3::{
    ffi, intern, prelude::*, types::PyDict, AsPyPointer, PyAny, PyErr, PyObject, PyResult,
    PyTraverseError, PyVisit, Python,
};

use std::{
    any::Any,
    env::temp_dir,
    ffi::CString,
    fs::File,
    io::Write,
    sync::{Arc, Mutex},
};
use uuid::Uuid;

const NAME: &str = "PyObject";

// TODO: Use a vec of objects with the index as the identifier instead of the hash or ID
// Then store two mappings: One from object ID to index and one from hash to list of indices
// When retrieving, try checking ID. If missing, try hash if hashable.
fn value(i: usize) -> Value {
    Value {
        #[cfg(debug_assertions)]
        tag: NAME.into(),
        bits: i as u64,
    }
}
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum PyObjectIdent {
    // Unhashable objects use the object ID as the key
    Unhashable(usize),
    // Hashable objects use the hash of the type as well as the hash of the object as the key
    // (need type hash as well, b/c different objects can have the same hash https://docs.python.org/3/library/functions.html#hash)
    Hashable(isize, isize),
}

impl PyObjectIdent {
    pub fn from_pyobject(obj: &PyObject) -> Self {
        Python::with_gil(|py| {
            let o = obj.bind(py);
            match o.hash() {
                Ok(hash) => PyObjectIdent::Hashable(o.get_type().hash().unwrap(), hash),
                Err(_) => PyObjectIdent::Unhashable(obj.as_ptr() as usize),
            }
        })
    }

    pub fn from_expr(expr: &Expr) -> Self {
        match expr {
            Expr::Call(_, head, args) => match head.as_str() {
                "py-object" => match args.as_slice() {
                    [Expr::Lit(_, Literal::Int(type_hash)), Expr::Lit(_, Literal::Int(hash))] => {
                        PyObjectIdent::Hashable(*type_hash as isize, *hash as isize)
                    }
                    [Expr::Lit(_, Literal::Int(id))] => PyObjectIdent::Unhashable(*id as usize),
                    _ => panic!("Unexpected children when loading PyObjectIdent"),
                },
                _ => panic!("Unexpected head when loading PyObjectIdent"),
            },
            _ => panic!("Unexpected expr when loading PyObjectIdent"),
        }
    }
    pub fn to_expr(self) -> Expr {
        let children = match self {
            PyObjectIdent::Unhashable(id) => {
                vec![lit!(Literal::Int(id as i64))]
            }
            PyObjectIdent::Hashable(type_hash, hash) => {
                vec![
                    lit!(Literal::Int(type_hash as i64)),
                    lit!(Literal::Int(hash as i64)),
                ]
            }
        };
        call!("py-object", children)
    }
}

#[pyclass]
#[derive(Debug)]
pub struct PyObjectSort(
    // Use an index map so that we can point to a value with an index we can store in the value
    Mutex<IndexMap<PyObjectIdent, PyObject>>,
);

impl PyObjectSort {
    fn new() -> Self {
        Self(Mutex::new(IndexMap::default()))
    }
    /// Store a Python object with its hash and return its index in the registry.
    pub fn insert_full(&self, key: PyObjectIdent, value: PyObject) -> usize {
        self.0.lock().unwrap().insert_full(key, value).0
    }

    /// Retrieves the Python object at the given index.
    pub fn get_index(&self, py: Python<'_>, index: usize) -> PyObject {
        self.0
            .lock()
            .unwrap()
            .get_index(index)
            .unwrap()
            .1
            .clone_ref(py)
    }

    /// Retrieves the index of the given key.
    pub fn get_index_of(&self, key: &PyObjectIdent) -> usize {
        self.0.lock().unwrap().get_index_of(key).unwrap()
    }

    pub fn load_ident(&self, value: &Value) -> PyObjectIdent {
        let objects = self.0.lock().unwrap();
        let i = value.bits as usize;
        let (ident, _) = objects.get_index(i).unwrap();
        ident.clone()
    }

    pub fn store(&self, obj: PyObject) -> Value {
        // Try hashing the object, if it fails, then it's unhashable, and store with ID
        let ident = PyObjectIdent::from_pyobject(&obj);
        let i = self.insert_full(ident, obj);
        value(i)
    }

    pub fn load(&self, py: Python<'_>, value: Value) -> PyObject {
        let i = value.bits as usize;
        self.get_index(py, i)
    }
}

// Clone + Hash + Eq + Any + Debug + Send + Sync

/// Implement wrapper struct so can implement foreign sort on it
#[derive(IntoPyObject, IntoPyObjectRef, Debug)]
pub struct MyPyObject(pub PyObject);

impl Clone for MyPyObject {
    fn clone(&self) -> Self {
        Python::with_gil(|py| MyPyObject(self.0.clone_ref(py)))
    }
}

impl PartialEq for MyPyObject {
    fn eq(&self, other: &Self) -> bool {
        self.0.is(&other.0)
    }
}

impl Eq for MyPyObject {}

impl std::hash::Hash for MyPyObject {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash the pointer to the object
        let ptr = self.0.as_ptr();
        state.write_usize(ptr as usize);
    }
}

impl IntoSort for MyPyObject {
    type Sort = PyObjectSort;
    fn store(self, sort: &Self::Sort) -> Value {
        sort.store(self.0)
    }
}

impl FromSort for MyPyObject {
    type Sort = PyObjectSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let obj = Python::with_gil(|py| sort.load(py, *value));
        MyPyObject(obj)
    }
}

#[pyclass(name = "PyObjectSort")]
#[derive(Debug, Clone)]
pub struct ArcPyObjectSort(
    // Use an index map so that we can point to a value with an index we can store in the value
    pub Arc<PyObjectSort>,
);

#[pymethods]
impl ArcPyObjectSort {
    #[new]
    fn new() -> Self {
        Self(Arc::new(PyObjectSort::new()))
    }

    /// Store a Python object and return an Expr that points to it.
    #[pyo3(name="store", signature = (obj, /))]
    fn store_py(&mut self, obj: PyObject) -> EggResult<crate::conversions::Expr> {
        let ident = PyObjectIdent::from_pyobject(&obj);
        self.0.insert_full(ident, obj);
        Ok(ident.to_expr().into())
    }

    // Retrieve the Python object from an expression
    #[pyo3(name="load", signature = (expr, /))]
    fn load_py(&self, expr: crate::conversions::Expr) -> PyObject {
        let expr: Expr = expr.into();
        let ident = PyObjectIdent::from_expr(&expr);
        let index = self.0.get_index_of(&ident);
        Python::with_gil(|py| self.0.get_index(py, index))
    }

    // Integrate with Python garbage collector
    // https://pyo3.rs/main/class/protocols#garbage-collector-integration

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.0
             .0
            .lock()
            .unwrap()
            .values()
            .try_for_each(|obj| visit.call(obj))
    }

    fn __clear__(&mut self) {
        self.0 .0.lock().unwrap().clear();
    }
}

impl Sort for PyObjectSort {
    fn name(&self) -> Symbol {
        NAME.into()
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn column_ty(&self, backend: &egglog_bridge::EGraph) -> ColumnTy {
        ColumnTy::Primitive(backend.primitives().get_ty::<PyObjectIdent>())
    }

    fn register_type(&self, backend: &mut egglog_bridge::EGraph) {
        backend.primitives_mut().register_type::<PyObjectIdent>();
    }

    fn register_primitives(self: Arc<Self>, eg: &mut EGraph) {
        // (py-object <i64> | <i64> <i64>)
        add_primitive!(eg, "py-object" = [xs: i64] -?> MyPyObject {{
            let ident = match xs.collect::<Vec<_>>().as_slice() {
                [id] => Some(PyObjectIdent::Unhashable(*id as usize)),
                [type_hash, hash] => Some(PyObjectIdent::Hashable(
                    *type_hash as isize,
                    *hash as isize,
                )),
                _ => None,
            };
            ident.map(|ident| {
                let index = self.__y.get_index_of(&ident);
                Python::with_gil(|py| {
                    MyPyObject(self.__y.get_index(py, index))
                })
            })
        }});
        // Supports calling (py-eval <str-obj> <globals-obj> <locals-obj>)
        add_primitive!(
            eg,
            "py-eval" = |code: Symbol, globals: MyPyObject, locals: MyPyObject| -> MyPyObject {
                {
                    let res_obj: PyObject = Python::with_gil(|py| {
                        let globals = globals.0.downcast_bound::<PyDict>(py).unwrap();
                        let locals = locals.0.downcast_bound::<PyDict>(py).unwrap();
                        py.eval(
                            CString::new(code.to_string()).unwrap().as_c_str(),
                            Some(globals),
                            Some(locals),
                        )
                        .unwrap()
                        .into()
                    });
                    MyPyObject(res_obj)
                }
            }
        );
        // Copies the locals, execs the Python string, then returns the copied version of the locals with any updates
        // (py-exec <str-obj> <globals-obj> <locals-obj>)
        add_primitive!(
            eg,
            "py-exec" = |code: Symbol, globals: MyPyObject, locals: MyPyObject| -> MyPyObject {
                {
                    let code: &str = code.into();
                    let res_obj: PyObject = Python::with_gil(|py| {
                        let globals = globals.0.downcast_bound::<PyDict>(py).unwrap();
                        // Copy the locals so we can mutate them and return them
                        let locals = locals
                            .0
                            .downcast_bound::<PyDict>(py)
                            .unwrap()
                            .copy()
                            .unwrap();
                        // Copy code into temporary file
                        // Keep it around so that if errors occur we can debug them after the program exits
                        let mut path = temp_dir();
                        let file_name = format!("egglog-{}.py", Uuid::new_v4());
                        path.push(file_name);
                        let mut file = File::create(path.clone()).unwrap();
                        file.write_all(code.as_bytes()).unwrap();
                        let path = path.to_str().unwrap();
                        run_code_path(py, code, Some(globals), Some(&locals), path).unwrap();
                        locals.into()
                    });
                    MyPyObject(res_obj)
                }
            }
        );

        // (py-dict [<key-object> <value-object>]*)
        add_primitive!(eg, "py-dict" = [xs: MyPyObject] -> MyPyObject {{
            let dict: PyObject = Python::with_gil(|py| {
                let dict = PyDict::new(py);
                // Update the dict with the key-value pairs
                for i in xs.collect::<Vec<_>>().chunks_exact(2) {
                    dict.set_item(&i[0].0, &i[1].0).unwrap();
                }
                dict.into()
            });
            MyPyObject(dict)
        }});
        // Supports calling (py-dict-update <dict-obj> [<key-object> <value-obj>]*)
        add_primitive!(eg, "py-dict-update" = [xs: MyPyObject] -> MyPyObject {{
            let dict: PyObject = Python::with_gil(|py| {
                let xs = xs.collect::<Vec<_>>();
                // Copy the dict so we can mutate it and return it
                let dict = xs[0].0.downcast_bound::<PyDict>(py).unwrap().copy().unwrap();
                // Update the dict with the key-value pairs
                for i in xs[1..].chunks_exact(2) {
                    dict.set_item(&i[0].0, &i[1].0).unwrap();
                }
                dict.into()
            });
            MyPyObject(dict)
        }});
        // (py-to-string <obj>)
        add_primitive!(
            eg,
            "py-to-string" = |x: MyPyObject| -> Symbol {
                {
                    let obj: String = Python::with_gil(|py| x.0.extract(py).unwrap());
                    let symbol: Symbol = obj.into();
                    symbol
                }
            }
        );
        // (py-to-bool <obj>)
        add_primitive!(
            eg,
            "py-to-bool" = |x: MyPyObject| -> bool {
                {
                    let obj: bool = Python::with_gil(|py| x.0.extract(py).unwrap());
                    obj
                }
            }
        );
        // (py-from-string <str>)
        add_primitive!(
            eg,
            "py-from-string" = |x: Symbol| -> MyPyObject {
                {
                    let obj: PyObject = Python::with_gil(|py| {
                        x.to_string()
                            .into_pyobject(py)
                            .unwrap()
                            .as_any()
                            .clone()
                            .unbind()
                    });
                    MyPyObject(obj)
                }
            }
        );
        // (py-from-int <int>)
        add_primitive!(
            eg,
            "py-from-int" = |x: i64| -> MyPyObject {
                {
                    let obj: PyObject = Python::with_gil(|py| {
                        x.into_pyobject(py).unwrap().as_any().clone().unbind()
                    });
                    MyPyObject(obj)
                }
            }
        );
    }
    fn extract_term(
        &self,
        _egraph: &EGraph,
        value: Value,
        _extractor: &Extractor,
        termdag: &mut TermDag,
    ) -> Option<(Cost, Term)> {
        #[cfg(debug_assertions)]
        assert!(value.tag == self.name());
        let children = match self.load_ident(&value) {
            PyObjectIdent::Unhashable(id) => {
                vec![termdag.lit(Literal::Int(id as i64))]
            }
            PyObjectIdent::Hashable(type_hash, hash) => {
                vec![
                    termdag.lit(Literal::Int(type_hash as i64)),
                    termdag.lit(Literal::Int(hash as i64)),
                ]
            }
        };
        Some((1, termdag.app("py-object".into(), children)))
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
