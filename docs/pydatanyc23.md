# PyData NYC 2023 Talk Proposal

## Title

Egglog: Bringing e-graphs to Python

## Abstract
_Brief Summary – This informs attendees what the talk is about. Discloses the topic, domain and overall purpose. This is at most a few lines long, and will be printed in the conference programme._

E-graphs are a data structure that comes out of the automated theorem-proving community to store the equivalent terms in a language. The `egglog` library provides Pythonic high level bindings for the Rust library of the same name, supporting Python users to create e-graphs by writing replacement rules, and then extracting expressions from them. This talk will give an overview of the data structure itself, through visual examples, and also use cases in the PyData ecosystem, for optimizing expressions across different domains and libraries.

## Description

_Description – This is a self-contained statement that summarises the aspects of the talk. It should be structured and present the objective of the talk, its outline, central thesis and key takeaways. After reading the description, the audience should have an idea of the overall presentation and know what to expect. The description should also make clear what background knowledge is expected from the attendees. Both this and the summary will be included in the talk details online._

The goal of this talk is to introduce the e-graph data structure in Python, starting with some simpler interactive examples, and then moving on to larger realistic use cases. To guide the talk, we will focus on the example of building an Array API compatible library that can be used in scikit-learn. We will show how we can use this library to generate a semantic representation of the LDA method in sklearn and use that to generate an optimized version of the API using Numba.

The primary audience of this talk is data science library authors. By the end of it, they should understand how egglog can help them with delayed optimization and the possibility of interoperability between libraries.
