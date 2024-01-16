# Internals

## EGraph

## Functions

All functions are late bound, so that the type annotations can refer to types created after their definition. This is to support classes with methods that refer to each other's types.

## Expressions

All expressions have an `__egg_expr__` which is their contents as a an egg expression.

They also have an `__egg_context__` which contains the context needed to reproduce them in egg.


## Context

An egg context contains a number of fields neccesary to recompute the function in egg.


1. 

## Rulesets

Rulesets also contain an `__egg_context__`.



How to implement in iterative changes with all working code?

