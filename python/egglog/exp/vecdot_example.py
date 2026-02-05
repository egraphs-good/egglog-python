from egglog.exp.array_api import *

v = NDArray([[1, 2], [3, 4]])
n = NDArray([3, 4])
res = vecdot(v, n)
egraph = EGraph()
egraph.register(res.to_recursive_value())
egraph.run(array_api_schedule)

new_res = egraph.extract(res.to_recursive_value())


egraph.debug_print()
print(new_res)

# new_egraph = EGraph()
# new_egraph.register(new_res)
# new_egraph.run(array_api_schedule)
# print(new_egraph.extract(new_res))


# smaller example


# print(res.eval_numpy("int64"))
# This fails with EggSmolError: Panic: Illegal merge attempted for function egglog_exp_array_api_Int_to_i64
# assert str(res.eval_numpy("float64")) == "array([ 3.,  8., 10.])"

# Trying to debug by inlining the code for eval


# egraph = EGraph()
# # egraph.set_report_level(StageInfo())
# egraph.register(res.index((0,)))
# # Trying to debug by running step by step

# i = 0
# prev_int01 = egraph.check_bool(eq(Int(0)).to(Int(1)))
# prev_tf = egraph.check_bool(eq(TRUE).to(FALSE))
# # egraph.saturate(array_api_combined_ruleset + run(), visualize=True, n_inline_leaves=2, split_primitive_outputs=True)
# while True:
#     try:
#         report = egraph.run(array_api_combined_ruleset + run())
#     except EggSmolError as e:
#         print(f"Step {i}: EggSmolError: {e}")
#         print("Int(0) == Int(1)?", egraph.check_bool(eq(Int(0)).to(Int(1))))
#         print("TRUE == FALSE?", egraph.check_bool(eq(TRUE).to(FALSE)))
#         egraph.debug_print()
#         raise
#     if not report.updated:
#         break
#     print(f"Step {i}:")
#     rules_applied = [(k, v) for k, v in report.num_matches_per_rule.items() if v > 0]
#     for rule, count in rules_applied:
#         print(f"  Applied rule {rule} {count} time(s)")
#     int01 = egraph.check_bool(eq(Int(0)).to(Int(1)))
#     tf = egraph.check_bool(eq(TRUE).to(FALSE))
#     if int01 != prev_int01 or tf != prev_tf:
#         print("=== New equality detected ===")
#         print("Int(0) == Int(1)?", int01)
#         print("TRUE == FALSE?", tf)
#         egraph.debug_print()
#         prev_int01 = int01
#         prev_tf = tf
#     i += 1
