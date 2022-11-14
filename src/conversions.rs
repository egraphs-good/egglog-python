use std::time::Duration;

// Create wrappers around input types so that convert from pyobjects to them
// and then from them to the egg_smol types
//
// Converts from Python classes we define in pure python so we can use dataclasses
// to represent the input types
// TODO: Copy strings of these from egg-smol... Maybe actually wrap those isntead.
use pyo3::prelude::*;

// Execute the block and wrap the error in a type error
fn wrap_error<T>(tp: &str, obj: &'_ PyAny, block: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    block().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Error converting {} to {}: {}",
            obj, tp, e
        ))
    })
}

// Take the repr of a Python object
fn repr(py: Python, obj: PyObject) -> PyResult<String> {
    obj.call_method(py, "__repr__", (), None)?.extract(py)
}

// Create a dataclass-like repr, of the name of the class of the object
// called with the repr of the fields
fn data_repr(py: Python, obj: PyObject, field_names: Vec<&str>) -> PyResult<String> {
    let class_name: String = obj
        .getattr(py, "__class__")?
        .getattr(py, "__name__")?
        .extract(py)?;
    let field_strings: PyResult<Vec<String>> = field_names
        .iter()
        .map(|name| obj.getattr(py, *name).and_then(|x| repr(py, x)))
        .collect();
    Ok(format!("{}({})", class_name, field_strings?.join(", ")))
}

#[pyclass]
#[derive(Clone)]
pub struct Variant(pub egg_smol::ast::Variant);

#[pymethods]
impl Variant {
    #[new]
    fn new(name: String, types: Vec<String>, cost: Option<usize>) -> Self {
        Self(egg_smol::ast::Variant {
            name: name.into(),
            types: types.into_iter().map(|x| x.into()).collect(),
            cost,
        })
    }
    #[getter]
    fn name(&self) -> &str {
        self.0.name.into()
    }
    #[getter]
    fn types(&self) -> Vec<String> {
        self.0.types.iter().map(|x| x.to_string()).collect()
    }
    #[getter]
    fn cost(&self) -> Option<usize> {
        self.0.cost
    }

    fn __repr__(slf: PyRef<'_, Self>, py: Python) -> PyResult<String> {
        data_repr(py, slf.into_py(py), vec!["name", "types", "cost"])
    }

    fn __str__(&self) -> String {
        format!("{:#?}", self.0)
    }
}

// Wrapped version of FunctionDecl
pub struct WrappedFunctionDecl(egg_smol::ast::FunctionDecl);
impl FromPyObject<'_> for WrappedFunctionDecl {
    fn extract(obj: &'_ PyAny) -> PyResult<Self> {
        wrap_error("FunctionDecl", obj, || {
            Ok(WrappedFunctionDecl(egg_smol::ast::FunctionDecl {
                name: obj.getattr("name")?.extract::<String>()?.into(),
                schema: obj.getattr("schema")?.extract::<WrappedSchema>()?.into(),
                default: obj
                    .getattr("default")?
                    .extract::<Option<WrappedExpr>>()?
                    .map(|x| x.into()),
                merge: obj
                    .getattr("merge")?
                    .extract::<Option<WrappedExpr>>()?
                    .map(|x| x.into()),
                cost: obj.getattr("cost")?.extract()?,
            }))
        })
    }
}

impl From<WrappedFunctionDecl> for egg_smol::ast::FunctionDecl {
    fn from(other: WrappedFunctionDecl) -> Self {
        other.0
    }
}

// Wrapped version of Schema
pub struct WrappedSchema(egg_smol::ast::Schema);

impl FromPyObject<'_> for WrappedSchema {
    fn extract(obj: &'_ PyAny) -> PyResult<Self> {
        wrap_error("Schema", obj, || {
            Ok(WrappedSchema(egg_smol::ast::Schema {
                input: obj
                    .getattr("input")?
                    .extract::<Vec<String>>()?
                    .into_iter()
                    .map(|x| x.into())
                    .collect(),
                output: obj.getattr("output")?.extract::<String>()?.into(),
            }))
        })
    }
}

impl From<WrappedSchema> for egg_smol::ast::Schema {
    fn from(other: WrappedSchema) -> Self {
        other.0
    }
}

// Wrapped version of Expr
pub struct WrappedExpr(egg_smol::ast::Expr);

impl FromPyObject<'_> for WrappedExpr {
    fn extract(obj: &'_ PyAny) -> PyResult<Self> {
        wrap_error("Expr", obj, || 
            // Try extracting into each type of expression, and return the first one that works
            extract_expr_lit(obj)
                .or_else(|_| extract_expr_call(obj))
                .or_else(|_| extract_expr_var(obj))
                .map(WrappedExpr))
    }
}

fn extract_expr_lit(obj: &PyAny) -> PyResult<egg_smol::ast::Expr> {
    Ok(egg_smol::ast::Expr::Lit(
        obj.getattr("value")?.extract::<WrappedLiteral>()?.into(),
    ))
}

fn extract_expr_var(obj: &PyAny) -> PyResult<egg_smol::ast::Expr> {
    Ok(egg_smol::ast::Expr::Var(
        obj.getattr("name")?.extract::<String>()?.into(),
    ))
}

fn extract_expr_call(obj: &PyAny) -> PyResult<egg_smol::ast::Expr> {
    Ok(egg_smol::ast::Expr::Call(
        obj.getattr("name")?.extract::<String>()?.into(),
        obj.getattr("args")?
            .extract::<Vec<WrappedExpr>>()?
            .into_iter()
            .map(|x| x.into())
            .collect(),
    ))
}

impl From<WrappedExpr> for egg_smol::ast::Expr {
    fn from(other: WrappedExpr) -> Self {
        other.0
    }
}

impl From<egg_smol::ast::Expr> for WrappedExpr {
    fn from(other: egg_smol::ast::Expr) -> Self {
        WrappedExpr(other)
    }
}

// Wrapped version of Literal
pub struct WrappedLiteral(egg_smol::ast::Literal);

impl FromPyObject<'_> for WrappedLiteral {
    fn extract(obj: &'_ PyAny) -> PyResult<Self> {
        wrap_error("Literal", obj, || {
            extract_literal_int(obj)
                .or_else(|_| extract_literal_string(obj))
                .or_else(|_| extract_literal_unit(obj))
                .map(WrappedLiteral)
        })
    }
}

fn extract_literal_int(obj: &PyAny) -> PyResult<egg_smol::ast::Literal> {
    Ok(egg_smol::ast::Literal::Int(
        obj.getattr("value")?.extract()?,
    ))
}

fn extract_literal_string(obj: &PyAny) -> PyResult<egg_smol::ast::Literal> {
    Ok(egg_smol::ast::Literal::String(
        obj.getattr("value")?.extract::<String>()?.into(),
    ))
}
fn extract_literal_unit(obj: &PyAny) -> PyResult<egg_smol::ast::Literal> {
    if obj.is_none() {
        Ok(egg_smol::ast::Literal::Unit)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err("Expected None"))
    }
}

impl From<WrappedLiteral> for egg_smol::ast::Literal {
    fn from(other: WrappedLiteral) -> Self {
        other.0
    }
}

// Wrapped version of Rewrite
pub struct WrappedRewrite(egg_smol::ast::Rewrite);

impl FromPyObject<'_> for WrappedRewrite {
    fn extract(obj: &'_ PyAny) -> PyResult<Self> {
        wrap_error("Rewrite", obj, || {
            Ok(WrappedRewrite(egg_smol::ast::Rewrite {
                lhs: obj.getattr("lhs")?.extract::<WrappedExpr>()?.into(),
                rhs: obj.getattr("rhs")?.extract::<WrappedExpr>()?.into(),
                conditions: obj
                    .getattr("conditions")?
                    .extract::<Vec<WrappedFact>>()?
                    .into_iter()
                    .map(|x| x.into())
                    .collect(),
            }))
        })
    }
}

impl From<WrappedRewrite> for egg_smol::ast::Rewrite {
    fn from(other: WrappedRewrite) -> Self {
        other.0
    }
}

// Wrapped version of Fact
pub struct WrappedFact(egg_smol::ast::Fact);

impl FromPyObject<'_> for WrappedFact {
    fn extract(obj: &'_ PyAny) -> PyResult<Self> {
        wrap_error("Fact", obj, || {
            extract_fact_eq(obj)
                .or_else(|_| extract_fact_fact(obj))
                .map(WrappedFact)
        })
    }
}

fn extract_fact_eq(obj: &PyAny) -> PyResult<egg_smol::ast::Fact> {
    Ok(egg_smol::ast::Fact::Eq(
        obj.getattr("exprs")?
            .extract::<Vec<WrappedExpr>>()?
            .into_iter()
            .map(|x| x.into())
            .collect(),
    ))
}

fn extract_fact_fact(obj: &PyAny) -> PyResult<egg_smol::ast::Fact> {
    Ok(egg_smol::ast::Fact::Fact(
        obj.getattr("expr")?.extract::<WrappedExpr>()?.into(),
    ))
}

impl From<WrappedFact> for egg_smol::ast::Fact {
    fn from(other: WrappedFact) -> Self {
        other.0
    }
}

// Wrapped version of Duration
// Converts from a rust duration to a python timedelta
pub struct WrappedDuration(Duration);

impl From<Duration> for WrappedDuration {
    fn from(other: Duration) -> Self {
        WrappedDuration(other)
    }
}

impl IntoPy<PyObject> for WrappedDuration {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d = self.0;
        pyo3::types::PyDelta::new(
            py,
            0,
            0,
            d.as_millis()
                .try_into()
                .expect("Failed to convert miliseconds to int32 when converting duration"),
            true,
        )
        .expect("Failed to contruct timedelta")
        .into()
    }
}
