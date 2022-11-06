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

// Wrapped version of Variant
pub struct WrappedVariant(egg_smol::ast::Variant);

impl FromPyObject<'_> for WrappedVariant {
    fn extract(obj: &'_ PyAny) -> PyResult<Self> {
        wrap_error("Variant", obj, || {
            Ok(WrappedVariant(egg_smol::ast::Variant {
                name: obj.getattr("name")?.extract::<String>()?.into(),
                cost: obj.getattr("cost")?.extract()?,
                types: obj
                    .getattr("types")?
                    .extract::<Vec<String>>()?
                    .into_iter()
                    .map(|x| x.into())
                    .collect(),
            }))
        })
    }
}

impl From<WrappedVariant> for egg_smol::ast::Variant {
    fn from(other: WrappedVariant) -> Self {
        other.0
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
