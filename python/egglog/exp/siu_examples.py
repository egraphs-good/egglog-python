from __future__ import annotations


# https://github.com/sklam/pyasir/blob/c363ff4f8f91177700ad4108dd5042b9b97d8289/pyasir/tests/test_fib.py

# In progress - should be able to re-create this
# @df.func
# def fib_ir(n: pyasir.Int64) -> pyasir.Int64:
#     @df.switch(n <= 1)
#     def swt(n):
#         @df.case(1)
#         def case0(n):
#             return 1

#         @df.case(0)
#         def case1(n):
#             return fib_ir(n - 1) + fib_ir(n - 2)

#         yield case0
#         yield case1

#     r = swt(n)
#     return r


# With something like this:
# @egglog.function
# def fib(n: Int) -> Int:
#     return (n <= Int(1)).if_int(
#         Int(1),
#         fib(n - Int(1)) + fib(n - Int(2)),
#     )
