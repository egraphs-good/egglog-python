// Wrapper around EGraph type

use crate::conversions::*;
use crate::error::EggResult;
use log::info;
use pyo3::prelude::*;

/// EGraph()
/// --
///
/// Create an empty EGraph.
#[pyclass(unsendable)]
pub struct EGraph {
    egraph: egg_smol::EGraph,
}

#[pymethods]
impl EGraph {
    #[new]
    fn new() -> Self {
        Self {
            egraph: egg_smol::EGraph::default(),
        }
    }

    /// Push a level onto the EGraph's stack.
    #[pyo3(text_signature = "($self)")]
    fn push(&mut self) {
        info!("Pushing egraph");
        self.egraph.push();
    }

    /// pop a level off the EGraph's stack.
    #[pyo3(text_signature = "($self)")]
    fn pop(&mut self) -> EggResult<()> {
        info!("Popping egraph");
        self.egraph.pop()?;
        Ok({})
    }

    /// Return a string representation of a function, up to n size
    #[pyo3(text_signature = "($self, name, n)")]
    fn print_function(&mut self, name: &str, n: usize) -> EggResult<String> {
        info!("Printing function {} up to size {}", name, n);
        Ok(self.egraph.print_function(name.into(), n)?)
    }

    /// Return a string representation of a function's size
    #[pyo3(text_signature = "($self, name)")]
    fn print_size(&self, name: &str) -> EggResult<String> {
        info!("Printing size of {}", name);
        Ok(self.egraph.print_size(name.into())?)
    }

    /// Clear all the nodes
    #[pyo3(text_signature = "($self)")]
    fn clear(&mut self) {
        info!("Clearing egraph");
        self.egraph.clear();
    }

    /// Clear all the rules
    #[pyo3(text_signature = "($self)")]
    fn clear_rules(&mut self) {
        info!("Clearing rules");
        self.egraph.clear_rules();
    }

    /// Extract the best expression of a given value. Will also return
    /// variants number of additional options.
    #[pyo3(signature = (expr, variants=0), text_signature = "($self, expr, variants=0)")]
    fn extract_expr(&mut self, expr: Expr, variants: usize) -> EggResult<(usize, Expr, Vec<Expr>)> {
        let expr_ast: egg_smol::ast::Expr = expr.into();
        info!("Extracting {:?}", expr_ast);
        let (cost, expr, exprs) = self.egraph.extract_expr(expr_ast, variants)?;
        Ok((
            cost,
            expr.into(),
            exprs.into_iter().map(|x| x.into()).collect(),
        ))
    }

    /// Check that a fact is true in the egraph.
    #[pyo3(text_signature = "($self, fact)")]
    fn check_fact(&mut self, fact: Fact_) -> EggResult<()> {
        let fact: egg_smol::ast::Fact = fact.into();
        info!("Checking {:?}", fact);
        self.egraph.check_fact(&fact.into())?;
        Ok({})
    }

    /// Run the rules on the egraph until it reaches a fixpoint, specifying the max number of iterations.
    /// Returns a tuple of the total time spen searching, applying, and rebuilding.
    #[pyo3(text_signature = "($self, limit)")]
    fn run_rules(
        &mut self,
        limit: usize,
    ) -> EggResult<(WrappedDuration, WrappedDuration, WrappedDuration)> {
        info!("Running rules with limit {}", limit);
        let [search, apply, rebuild] = self.egraph.run_rules(limit);
        Ok((search.into(), apply.into(), rebuild.into()))
    }

    /// Define a rewrite rule, returning the name of the rule
    #[pyo3(text_signature = "($self, rewrite)")]
    // Can be replaced with add_rule
    fn add_rewrite(&mut self, rewrite: Rewrite) -> EggResult<String> {
        let ast_rewrite: egg_smol::ast::Rewrite = rewrite.into();
        info!("Adding {:?}", ast_rewrite);
        let res = self.egraph.add_rewrite(ast_rewrite)?;
        Ok(res.to_string())
    }

    /// Run a number of actions on the egraph.
    #[pyo3(signature=(*actions), text_signature = "($self, *actions)")]
    fn eval_actions(&mut self, actions: Vec<Action>) -> EggResult<()> {
        let converted: Vec<egg_smol::ast::Action> = actions.into_iter().map(|x| x.into()).collect();
        info!("Evaling {:?}", converted);
        self.egraph.eval_actions(&converted)?;
        Ok({})
    }

    /// Define a rule, returning the name of it.
    #[pyo3(text_signature = "($self, rule)")]
    fn add_rule(&mut self, rule: Rule) -> EggResult<String> {
        let rule_ast: egg_smol::ast::Rule = rule.into();
        info!("Adding {:?}", rule_ast);
        let res = self.egraph.add_rule(rule_ast)?;
        Ok(res.to_string())
    }

    /// Define a new named value.
    #[pyo3(
        text_signature = "($self, name, expr, cost=None)",
        signature = (name, expr, cost=None)
    )]
    fn define(&mut self, name: String, expr: Expr, cost: Option<usize>) -> EggResult<()> {
        let expr_ast: egg_smol::ast::Expr = expr.into();
        info!("Defining {} as {:?}", name, expr_ast);
        self.egraph.define(name.into(), expr_ast, cost)?;
        Ok(())
    }

    /// Declare a new function definition.
    #[pyo3(text_signature = "($self, decl)")]
    fn declare_function(&mut self, decl: FunctionDecl) -> EggResult<()> {
        let decl: egg_smol::ast::FunctionDecl = decl.into();
        info!("Declaring {:?}", decl);
        self.egraph.declare_function(&decl)?;
        Ok(())
    }

    /// Declare a new sort with the given name.
    #[pyo3(
        text_signature = "($self, name, presort_and_args=None)",
        signature = (name, presort_and_args=None),
    )]
    fn declare_sort(
        &mut self,
        name: &str,
        presort_and_args: Option<(String, Vec<Expr>)>,
    ) -> EggResult<()> {
        info!("Declaring sort {}", name);
        // TODO: It would be cleaner to do this with a map function and call declare_sort
        // only once.
        // I had to move the function call inside the match, so that the lifetime
        // of the args slice lasts for as long as it is called in the declare_sort
        match presort_and_args {
            Some((presort, args)) => {
                let args_converted: Vec<egg_smol::ast::Expr> =
                    args.into_iter().map(|x| x.into()).collect();
                self.egraph
                    .declare_sort(name, Some((presort.into(), &args_converted[..])))?
            }
            None => self.egraph.declare_sort(name, None)?,
        };
        Ok({})
    }

    /// Declare a new datatype constructor.
    /// Can be replaced with declare_function
    #[pyo3(text_signature = "($self, variant, sort)")]
    fn declare_constructor(&mut self, variant: Variant, sort: &str) -> EggResult<()> {
        let variant_ast: egg_smol::ast::Variant = variant.into();
        info!("Declaring constructor {} {:?}", sort, variant_ast);
        self.egraph.declare_constructor(variant_ast, sort)?;
        Ok({})
    }

    /// Parse the input string as a program and run it on the EGraph.
    /// Returns a list of strings representing the output.
    /// An EggSmolError is raised if there is problem parsing or executing.
    #[pyo3(text_signature = "($self, input)")]
    fn parse_and_run_program(&mut self, input: &str) -> EggResult<Vec<String>> {
        info!("Running {}", input);
        let res = self.egraph.parse_and_run_program(input)?;
        Ok(res)
    }
}
