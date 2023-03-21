# Explanation

```{toctree}
explanation/compared_to_rust
explanation/optional_values
explanation/define_and_define
```

## Status of this project

This package is in development and is not ready for production use. `egg-smol` itself
is also subject to changes and is less stable than [`egg`](https://github.com/egraphs-good/egg).
If you are looking for a more stable e-graphs library in Python, you can try [`snake-egg`](https://github.com/egraphs-good/snake-egg), which wraps `egg`.

`egg-smol` is a rewrite of the egg library to use [relational e-matching](https://arxiv.org/abs/2108.02290):

> We have determined the relational e-matching is somewhat of a poor match for egg as it's currently structured for a couple reasons. (1) Most workloads that egg users encounter focus on small, simple patterns, so relational e-matching is no better and sometimes slower because you have to build the database "from scratch" every iteration. (2) egg gives the user a little too much flexibility in manipulating e-nodes (in particular, you the e-node list in each e-class is exposed); this makes it hard to incrementally maintain the datastructures you'd need for relational e-matching. egg-smol (name temporary) is a clean slate approach to this and some other shortcomings of egg. No promises on its stabilization or merging into egg yet.
>
> - [Max Willsey](https://egraphs.zulipchat.com/#narrow/stream/328972-general/topic/roadmap/near/297442354)

## How documentation is organized

We use the [Diátaxis framework](https://diataxis.fr/) to organize our documentation. This helps with figuring out where different content should live and how it should be organized.
