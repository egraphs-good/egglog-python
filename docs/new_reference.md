# Reference Documentation

```{module} egglog

```

This is a definitive reference of `egglog` module and the concepts in it.

## Terms


Ruleset
: A colleciton of rules

Rule
: Updates an EGraph by matching on a number of facts and then running a number of actions. Any variables in the facts can be used in the actions.

Fact
: A query on an EGraph, either by an expression or an equivalence between multiple expressions. 


Action
: A change to an EGraph, either unioning multiple expressing, setting the value of a function call, deleting an expression, or panicking.
Union
: Merges two equivalence classes of two expressions.
Set
: Similar to union, except can be used on primitive expressions, whereas union can only be used on user defined expressions.
Delete
: Remove a function call from an EGraph.



Schedule
: A composition of some rulesets, either composing them sequentially, running them repeatedly, running them till saturation, or running until some facts are met

EGraph ([](egglog.EGraph))
: An equivalence relation over a set of expressions.
: A collection of expressions where each expression is part of a distinct equivalence class.
: Can run actions, check facts, run schedules, or extract minimal cost expressions.


Expression ([](egglog.Expr))
: Either a function called with some number of argument expressions or a literal integer, float, or string, with a particular type.

Function ([](egglog.function))
: Defined by a unique name and a typing relation that will specify the return type based on the types of the argument expressions.
: These can either be builtin functions, which are implemented in Rust, or user defined function which have types for each argument and the return type.
: Relations ([](egglog.relation)), constants ([](egglog.constant)), methods ([](egglog.method)), classmethods, and class variables are all syntactic sugar for defining functions.

Type (called a "sort" in the rust bindings)
: A uniquely named entity, that is used to constraint the composition of functions.
: Can be either a primitive type if it is defined in Rust or a user defined type.



## Classes 

```{class} EGraph

```

https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc
https://www.sphinx-doc.org/en/master/usage/domains/python.html#signatures
https://myst-parser.readthedocs.io/en/latest/syntax/roles-and-directives.html

```{class} Expr

Subclass `Expr` to create a new type. Only direct subclasses are supported, subclasses of subclasses are not for now.


```

```{decorator} function

```

```{decorator} method

Any method can be decorated with this to customize it's behavior. This is only supported in classes which subclass [](egglog.Expr).

```

```{function} relation

Creates a function whose return type is [](egglog.Unit) and whose default value is `[](egglog.Unit)`.

```

```{function} constant

A "constant" is implemented as the instantiation of a value that takes no args.
This creates a function with `name` and return type `tp` and returns a value of it being called.

```

```{class} Unit

Primitive type with only one possible value.
```
