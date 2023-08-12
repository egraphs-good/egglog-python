"""
Goals for stringifying:

* Re-used sub expressions are extracted (given some threshold)
* Variables and copies are created for mutations iff they are needed (only add copy if existing var which is re-used)
* Works for python native objects surrounding expressions which are created after down conversions (tuples, slices)
* Only down convert when its safe (i.e. not the self object, but for args is ok)
* Works for stringigfying rules as well with mutations
"""

"""
OR: do the stringifying with numba, which is what I want for real.

to_string....

Try this as simplest, with no optimizations. Use hash of string for each expr.

x[y] = z

Context: Map<hash("setitem", x, y, z), (z,>



Context:
`op`_`hash(x, y z)` = clone(x)
`op`_`hash(x, y z)`[y] = z

-> `op`_`hash(x, y z)`

OR do we use python object and make it non functional?


"""
