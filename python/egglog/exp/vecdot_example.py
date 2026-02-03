from egglog.exp.array_api import *

# v = NDArray.matrix([[0.0, 5.0, 0.0], [0.0, 0.0, 10.0], [0.0, 6.0, 8.0]])
# n = NDArray.vector([0.0, 0.6, 0.8])
# smaller example
v = NDArray.matrix([[1, 2], [3, 4]])
n = NDArray.vector([3, 4])
res = vecdot(v, n)
# This fails with EggSmolError: Panic: Illegal merge attempted for function egglog_exp_array_api_Int_to_i64
# assert str(res.eval_numpy("float64")) == "array([ 3.,  8., 10.])"

# Trying to debug by inlining the code for eval


egraph = EGraph()
egraph.register(res.shape)
egraph.run(array_api_schedule)
assert eq(egraph.extract(res.shape)).to(TupleInt(Vec(Int(2))))
idxed = res.index((0,))
egraph.register(res.index((0,)))

# This is what fails
# egraph.run(array_api_schedule)
# print(egraph.extract(res.index((0,))))
# Trying to debug by running step by step

i = 0
# egraph.saturate(array_api_combined_ruleset + run(), visualize=True, n_inline_leaves=2, split_primitive_outputs=True)
while (report := egraph.run(array_api_combined_ruleset + run())).updated:
    print(f"Step {i}:")
    rules_applied = [(k, v) for k, v in report.num_matches_per_rule.items() if v > 0]
    for rule, count in rules_applied:
        print(f"  Applied rule {rule} {count} time(s)")
    # print out all e-classes
    egraph.debug_print()
    # If we want to look at which rules were applied:
    # matches = [k for k, v in report.num_matches_per_rule.items() if v > 0]
    # print(f"Step {i}: applied rules: {matches}")

    # If we want to see the current extraction:
    # print(egraph.extract(res.index((0,))))

    i += 1
