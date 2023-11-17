use crate::error::EggResult;
use egglog::{
    ast::{Expr, Literal, Symbol},
    sort::{BoolSort, FromSort, I64Sort, IntoSort as _, Sort, StringSort},
    util::IndexMap,
    ArcSort, EGraph, PrimitiveLike, TypeInfo, Value,
};
use pyo3::{
    ffi, intern, prelude::*, types::PyDict, AsPyPointer, IntoPy, PyAny, PyErr, PyObject, PyResult,
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

fn value(i: usize) -> Value {
    Value {
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
            let o = obj.as_ref(py);
            match o.hash() {
                Ok(hash) => PyObjectIdent::Hashable(o.get_type().hash().unwrap(), hash),
                Err(_) => PyObjectIdent::Unhashable(obj.as_ptr() as usize),
            }
        })
    }
    pub fn to_expr(self) -> Expr {
        let children = match self {
            PyObjectIdent::Unhashable(id) => {
                vec![Expr::Lit(Literal::Int(id as i64))]
            }
            PyObjectIdent::Hashable(type_hash, hash) => {
                vec![
                    Expr::Lit(Literal::Int(type_hash as i64)),
                    Expr::Lit(Literal::Int(hash as i64)),
                ]
            }
        };
        Expr::call("py-object", children)
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

    // /// Retrives the Python object at the given index.
    // pub fn get_index(&self, index: usize) -> PyObject {
    //     self.0.lock().unwrap().get_index(index).unwrap().1.clone()
    // }

    /// Retrieves the index of the given key.
    pub fn get_index_of(&self, key: &PyObjectIdent) -> usize {
        self.0.lock().unwrap().get_index_of(key).unwrap()
    }

    pub fn load_ident(&self, value: &Value) -> PyObjectIdent {
        self.get_index(value).0
    }

    fn get_index(&self, value: &Value) -> (PyObjectIdent, Py<PyAny>) {
        let objects = self.0.lock().unwrap();
        let i = value.bits as usize;
        let (ident, obj) = objects.get_index(i).unwrap();
        (*ident, obj.clone())
    }

    pub fn store(&self, obj: PyObject) -> Value {
        // Try hashing the object, if it fails, then it's unhashable, and store with ID
        let ident = PyObjectIdent::from_pyobject(&obj);
        let i = self.insert_full(ident, obj);
        value(i)
    }

    pub fn load(&self, value: Value) -> PyObject {
        let (_, obj) = self.get_index(&value);
        obj
    }
}

/// Implement wrapper struct so can implement foreign sort on it
pub struct MyPyObject(pub PyObject);

impl FromSort for MyPyObject {
    type Sort = PyObjectSort;
    fn load(sort: &Self::Sort, value: &Value) -> Self {
        let (_, obj) = sort.get_index(value);
        MyPyObject(obj)
    }
}

impl IntoPy<PyObject> for MyPyObject {
    fn into_py(self, _py: Python<'_>) -> PyObject {
        self.0
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

    #[rustfmt::skip]
    fn register_primitives(self: Arc<Self>, typeinfo: &mut TypeInfo) {
        typeinfo.add_primitive(Ctor {
            name: "py-object".into(),
            py_object: self.clone(),
            i64:typeinfo.get_sort(),
        });
        typeinfo.add_primitive(Eval {
            name: "py-eval".into(),
            py_object: self.clone(),
            string: typeinfo.get_sort(),
        });
        typeinfo.add_primitive(Exec {
            name: "py-exec".into(),
            py_object: self.clone(),
            string: typeinfo.get_sort(),
        });
        typeinfo.add_primitive(Dict {
            name: "py-dict".into(),
            py_object: self.clone(),
        });
        typeinfo.add_primitive(DictUpdate {
            name: "py-dict-update".into(),
            py_object: self.clone(),
        });
        typeinfo.add_primitive(ToString {
            name: "py-to-string".into(),
            py_object: self.clone(),
            string: typeinfo.get_sort(),
        });
        typeinfo.add_primitive(ToBool {
            name: "py-to-bool".into(),
            py_object: self.clone(),
            bool_: typeinfo.get_sort(),
        });
        typeinfo.add_primitive(FromString {
            name: "py-from-string".into(),
            py_object: self.clone(),
            string: typeinfo.get_sort(),
        });
        typeinfo.add_primitive(FromInt {
            name: "py-from-int".into(),
            py_object: self,
            int: typeinfo.get_sort(),
        });
    }
    fn make_expr(&self, _egraph: &EGraph, value: Value) -> (usize, Expr) {
        assert!(value.tag == self.name());
        let children = match self.load_ident(&value) {
            PyObjectIdent::Unhashable(id) => {
                vec![Expr::Lit(Literal::Int(id as i64))]
            }
            PyObjectIdent::Hashable(type_hash, hash) => {
                vec![
                    Expr::Lit(Literal::Int(type_hash as i64)),
                    Expr::Lit(Literal::Int(hash as i64)),
                ]
            }
        };
        (1, Expr::call("py-object", children))
    }
}

struct Ctor {
    name: Symbol,
    py_object: Arc<PyObjectSort>,
    i64: Arc<I64Sort>,
}

impl PrimitiveLike for Ctor {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [id] if id.name() == self.i64.name() => Some(self.py_object.clone()),
            [type_hash, hash]
                if type_hash.name() == self.i64.name() && hash.name() == self.i64.name() =>
            {
                Some(self.py_object.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let ident = match values {
            [id] => PyObjectIdent::Unhashable(i64::load(self.i64.as_ref(), id) as usize),
            [type_hash, hash] => PyObjectIdent::Hashable(
                i64::load(self.i64.as_ref(), type_hash) as isize,
                i64::load(self.i64.as_ref(), hash) as isize,
            ),
            _ => unreachable!(),
        };
        Some(value(self.py_object.get_index_of(&ident)))
    }
}

/// Supports calling (py-eval <str-obj> <globals-obj> <locals-obj>)
struct Eval {
    name: Symbol,
    py_object: Arc<PyObjectSort>,
    string: Arc<StringSort>,
}

impl PrimitiveLike for Eval {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [str, locals, globals]
                if str.name() == self.string.name()
                    && locals.name() == self.py_object.name()
                    && globals.name() == self.py_object.name() =>
            {
                Some(self.py_object.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let code: Symbol = Symbol::load(self.string.as_ref(), &values[0]);
        let res_obj: PyObject = Python::with_gil(|py| {
            let globals = self.py_object.load(values[1]);
            let globals = globals.downcast::<PyDict>(py).unwrap();
            let locals = self.py_object.load(values[2]);
            let locals = locals.downcast::<PyDict>(py).unwrap();
            py.eval(code.into(), Some(globals), Some(locals))
                .unwrap()
                .into()
        });
        Some(self.py_object.store(res_obj))
    }
}

/// Copies the locals, execs the Python string, then returns the copied version of the locals with any updates
/// (py-exec <str-obj> <globals-obj> <locals-obj>)
struct Exec {
    name: Symbol,
    py_object: Arc<PyObjectSort>,
    string: Arc<StringSort>,
}

impl PrimitiveLike for Exec {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [str, locals, globals]
                if str.name() == self.string.name()
                    && locals.name() == self.py_object.name()
                    && globals.name() == self.py_object.name() =>
            {
                Some(self.py_object.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let code: Symbol = Symbol::load(self.string.as_ref(), &values[0]);
        let code: &str = code.into();
        let locals: PyObject = Python::with_gil(|py| {
            let globals = self.py_object.load(values[1]);
            let globals = globals.downcast::<PyDict>(py).unwrap();
            let locals = self.py_object.load(values[2]);
            // Copy the locals so we can mutate them and return them
            let locals = locals.downcast::<PyDict>(py).unwrap().copy().unwrap();
            // Copy code into temporary file
            // Keep it around so that if errors occur we can debug them after the program exits
            let mut path = temp_dir();
            let file_name = format!("egglog-{}.py", Uuid::new_v4());
            path.push(file_name);
            let mut file = File::create(path.clone()).unwrap();
            file.write_all(code.as_bytes()).unwrap();
            let path = path.to_str().unwrap();
            run_code_path(py, code, Some(globals), Some(locals), path).unwrap();
            locals.into()
        });
        Some(self.py_object.store(locals))
    }
}

/// (py-dict [<key-object> <value-object>]*)
struct Dict {
    name: Symbol,
    py_object: Arc<PyObjectSort>,
}

impl PrimitiveLike for Dict {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        // Should have an even number of args
        if types.len() % 2 != 0 {
            return None;
        }
        for tp in types.iter() {
            // All tps should be object
            if tp.name() != self.py_object.name() {
                return None;
            }
        }
        Some(self.py_object.clone())
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let dict: PyObject = Python::with_gil(|py| {
            let dict = PyDict::new(py);
            // Update the dict with the key-value pairs
            for i in values.chunks_exact(2) {
                let key = self.py_object.load(i[0]);
                let value = self.py_object.load(i[1]);
                dict.set_item(key, value).unwrap();
            }
            dict.into()
        });
        Some(self.py_object.store(dict))
    }
}

/// Supports calling (py-dict-update <dict-obj> [<key-object> <value-obj>]*)
struct DictUpdate {
    name: Symbol,
    py_object: Arc<PyObjectSort>,
}

impl PrimitiveLike for DictUpdate {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        // Should have an odd number of args, with all the pairs plus the first arg
        if types.len() % 2 == 0 {
            return None;
        }
        for (i, tp) in types.iter().enumerate() {
            // First tp should be dict
            if i == 0 {
                if tp.name() != self.py_object.name() {
                    return None;
                }
            }
            // All other tps should be object
            else if tp.name() != self.py_object.name() {
                return None;
            }
        }
        Some(self.py_object.clone())
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let dict: PyObject = Python::with_gil(|py| {
            let dict = self.py_object.load(values[0]);
            // Copy the dict so we can mutate it and return it
            let dict = dict.downcast::<PyDict>(py).unwrap().copy().unwrap();
            // Update the dict with the key-value pairs
            for i in values[1..].chunks_exact(2) {
                let key = self.py_object.load(i[0]);
                let value = self.py_object.load(i[1]);
                dict.set_item(key, value).unwrap();
            }
            dict.into()
        });
        Some(self.py_object.store(dict))
    }
}

/// (py-to-string <obj>)
struct ToString {
    name: Symbol,
    py_object: Arc<PyObjectSort>,
    string: Arc<StringSort>,
}

impl PrimitiveLike for ToString {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [obj] if obj.name() == self.py_object.name() => Some(self.string.clone()),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let obj: String =
            Python::with_gil(|py| self.py_object.load(values[0]).extract(py).unwrap());
        let symbol: Symbol = obj.into();
        symbol.store(self.string.as_ref())
    }
}

/// (py-to-bool <obj>)
struct ToBool {
    name: Symbol,
    py_object: Arc<PyObjectSort>,
    bool_: Arc<BoolSort>,
}

impl PrimitiveLike for ToBool {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [obj] if obj.name() == self.py_object.name() => Some(self.bool_.clone()),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let obj: bool = Python::with_gil(|py| self.py_object.load(values[0]).extract(py).unwrap());
        obj.store(self.bool_.as_ref())
    }
}

/// (py-from-string <str>)
struct FromString {
    name: Symbol,
    py_object: Arc<PyObjectSort>,
    string: Arc<StringSort>,
}

impl PrimitiveLike for FromString {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [str] if str.name() == self.string.name() => Some(self.py_object.clone()),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let str = Symbol::load(self.string.as_ref(), &values[0]).to_string();
        let obj: PyObject = Python::with_gil(|py| str.into_py(py));
        Some(self.py_object.store(obj))
    }
}

// (py-from-int <int>)
struct FromInt {
    name: Symbol,
    py_object: Arc<PyObjectSort>,
    int: Arc<I64Sort>,
}

impl PrimitiveLike for FromInt {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [int] if int.name() == self.int.name() => Some(self.py_object.clone()),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value], _egraph: &EGraph) -> Option<Value> {
        let int = i64::load(self.int.as_ref(), &values[0]);
        let obj: PyObject = Python::with_gil(|py| int.into_py(py));
        Some(self.py_object.store(obj))
    }
}

/// Runs the code in the given context with a certain path.
/// Copied from `run_code`, but allows specifying the path.
fn run_code_path<'py>(
    py: Python<'py>,
    code: &str,
    globals: Option<&PyDict>,
    locals: Option<&PyDict>,
    path: &str,
) -> PyResult<&'py PyAny> {
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

        py.from_owned_ptr_or_err(res_ptr)
    }
}
