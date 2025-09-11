# mypy: disable-error-code="empty-body"

"""
Join Tree (custom costs)
========================

Example of using custom cost functions for jointree.

From https://egraphs.zulipchat.com/#narrow/stream/328972-general/topic/How.20can.20I.20find.20the.20tree.20associated.20with.20an.20extraction.3F
"""

from __future__ import annotations

from egglog import *


class JoinTree(Expr):
    def __init__(self, name: StringLike) -> None: ...

    def join(self, other: JoinTree) -> JoinTree: ...

    @method(merge=lambda old, new: old.min(new))  # type:ignore[prop-decorator]
    @property
    def size(self) -> i64: ...


ra = JoinTree("a")
rb = JoinTree("b")
rc = JoinTree("c")
rd = JoinTree("d")
re = JoinTree("e")
rf = JoinTree("f")

query = ra.join(rb).join(rc).join(rd).join(re).join(rf)

egraph = EGraph()
egraph.register(
    set_(ra.size).to(50),
    set_(rb.size).to(200),
    set_(rc.size).to(10),
    set_(rd.size).to(123),
    set_(re.size).to(10000),
    set_(rf.size).to(1),
)


@egraph.register
def _rules(s: String, a: JoinTree, b: JoinTree, c: JoinTree, asize: i64, bsize: i64):
    # cost of relation is its size minus 1, since the string arg will have a cost of 1 as well
    yield rule(JoinTree(s).size == asize).then(set_cost(JoinTree(s), asize - 1))
    # cost/size of join is product of sizes
    yield rule(a.join(b), a.size == asize, b.size == bsize).then(
        set_(a.join(b).size).to(asize * bsize), set_cost(a.join(b), asize * bsize)
    )
    # associativity
    yield rewrite(a.join(b)).to(b.join(a))
    # commutativity
    yield rewrite(a.join(b).join(c)).to(a.join(b.join(c)))


egraph.register(query)
egraph.run(1000)
print(egraph.extract(query))
print(egraph.extract(query.size))
