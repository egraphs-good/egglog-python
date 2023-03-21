use pyo3::prelude::*;

use crate::conversions::*;
use crate::error::*;

#[pyfunction]
pub fn parse(input: &str) -> EggResult<Vec<Command>> {
    let parser = egg_smol::ast::parse::ProgramParser::new();
    let res = parser.parse(input)?;
    log::info!("Parsed {:?}", res);
    return Ok(res.into_iter().map(|x| x.into()).collect());
}
