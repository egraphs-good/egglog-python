use std::{
    any::Any,
    sync::{Arc, Mutex},
};

use egglog::sort::FromSort;
use egglog::{
    ast::{Expr, Literal, Symbol},
    sort::{I64Sort, Sort},
    util::IndexMap,
    ArcSort, EGraph, PrimitiveLike, TypeInfo, Value,
};
use pyo3::{AsPyPointer, PyObject, Python};

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum PyObjectIdent {
    // Unhashable objects use the object ID as the key
    Unhashable(usize),
    // Hashable objects use the hash of the type as well as the hash of the object as the key
    // (need type hash as well, b/c different objects can have the same hash https://docs.python.org/3/library/functions.html#hash)
    Hashable(isize, isize),
}

#[derive(Debug)]
pub struct PyObjectSort {
    name: Symbol,
    // Use an index map so that we can point to a value with an index we can store in the value
    pub objects: Mutex<IndexMap<PyObjectIdent, PyObject>>,
}

impl PyObjectSort {
    pub fn new(name: Symbol) -> Self {
        Self {
            name,
            objects: Default::default(),
        }
    }

    pub fn load(&self, value: &Value) -> (PyObjectIdent, PyObject) {
        let objects = self.objects.lock().unwrap();
        let i = value.bits as usize;
        let (ident, obj) = objects.get_index(i).unwrap();
        (*ident, obj.clone())
    }
    pub fn store(&self, obj: PyObject) -> Value {
        // Try hashing the object, if it fails, then it's unhashable, and store with ID
        let ident = Python::with_gil(|py| {
            let o = obj.as_ref(py);
            match o.hash() {
                Ok(hash) => PyObjectIdent::Hashable(o.get_type().hash().unwrap(), hash),
                Err(_) => PyObjectIdent::Unhashable(obj.as_ptr() as usize),
            }
        });
        let mut objects = self.objects.lock().unwrap();
        let (i, _) = objects.insert_full(ident, obj.clone());
        Value {
            tag: self.name,
            bits: i as u64,
        }
    }
    fn get_value(&self, key: &PyObjectIdent) -> Value {
        let objects = self.objects.lock().unwrap();
        let i = objects.get_index_of(key).unwrap();
        Value {
            tag: self.name,
            bits: i as u64,
        }
    }
}

impl Sort for PyObjectSort {
    fn name(&self) -> Symbol {
        self.name
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    #[rustfmt::skip]
    fn register_primitives(self: Arc<Self>, typeinfo: &mut TypeInfo) {
        typeinfo.add_primitive(CtorID {
            name: "py-object-id".into(),
            py_object: self.clone(),
            i64:typeinfo.get_sort(),
        });
        typeinfo.add_primitive(CtorHash {
            name: "py-object-hash".into(),
            py_object: self,
            i64:typeinfo.get_sort(),
        });
    }
    fn make_expr(&self, _egraph: &EGraph, value: Value) -> Expr {
        assert!(value.tag == self.name());
        let (ident, _) = self.load(&value);
        match ident {
            PyObjectIdent::Unhashable(id) => {
                Expr::call("py-object-id", vec![Expr::Lit(Literal::Int(id as i64))])
            }
            PyObjectIdent::Hashable(type_hash, hash) => Expr::call(
                "py-object-hash",
                vec![
                    Expr::Lit(Literal::Int(type_hash as i64)),
                    Expr::Lit(Literal::Int(hash as i64)),
                ],
            ),
        }
    }
}

struct CtorID {
    name: Symbol,
    py_object: Arc<PyObjectSort>,
    i64: Arc<I64Sort>,
}

impl PrimitiveLike for CtorID {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [id] if id.name() == self.i64.name() => Some(self.py_object.clone()),
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let i = i64::load(self.i64.as_ref(), &values[0]);
        self.py_object
            .get_value(&PyObjectIdent::Unhashable(i as usize))
            .into()
    }
}

struct CtorHash {
    name: Symbol,
    py_object: Arc<PyObjectSort>,
    i64: Arc<I64Sort>,
}

impl PrimitiveLike for CtorHash {
    fn name(&self) -> Symbol {
        self.name
    }

    fn accept(&self, types: &[ArcSort]) -> Option<ArcSort> {
        match types {
            [type_hash, hash]
                if type_hash.name() == self.i64.name() && hash.name() == self.i64.name() =>
            {
                Some(self.py_object.clone())
            }
            _ => None,
        }
    }

    fn apply(&self, values: &[Value]) -> Option<Value> {
        let type_hash = i64::load(self.i64.as_ref(), &values[0]);
        let hash = i64::load(self.i64.as_ref(), &values[1]);
        self.py_object
            .get_value(&PyObjectIdent::Hashable(type_hash as isize, hash as isize))
            .into()
    }
}
