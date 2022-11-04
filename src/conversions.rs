// Create wrappers around input types so that convert from pyobjects to them
// and then from them to the egg_smol types
//
// Converts from Python classes we define in pure python so we can use dataclasses
// to represent the input types
use pyo3::prelude::*;

pub struct WrappedVariant(egg_smol::ast::Variant);

impl FromPyObject<'_> for WrappedVariant {
    fn extract(obj: &'_ PyAny) -> PyResult<Self> {
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
    }
}

impl From<WrappedVariant> for egg_smol::ast::Variant {
    fn from(other: WrappedVariant) -> Self {
        other.0
    }
}
