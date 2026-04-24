from egglog import *
from egglog.exp.param_eq.domain import *

# 10 + x + 5
# before constant folding
polynomial({{Num(10.0): 1}: 1.0, {Num.var("x"): 1}: 1.0, {Num(5.0): 1}: 1.0})
# after constant folding
polynomial({{}: 15.0, {Num.var("x"): 1}: 1.0})
# /var/folders/x4/rfvvhxcd53lft2b9844dw1lr0000gn/T/tmphnq__lg_.egg

# 2 * x * 3
# before constant folding
polynomial({{Num(2.0): 1, Num.var("x"): 1, Num(3.0): 1}: 1.0})
# after constant folding
polynomial({{Num.var("x"): 1}: 6.0})
# /var/folders/x4/rfvvhxcd53lft2b9844dw1lr0000gn/T/tmpzr3wtl47.egg

# (-0.7330341374049288 * x1 * (1.1635766746115828 * x0 * (x0 - 1.096491354684671 * x1 + 0.09649135468467125 * exp(x1) + 0.065716650770683) - 3.3628776435387486 * x0 - x1 + 0.5423590312635699) - 0.02765235981387666 * x1 + 0.02765235981387666 * exp(x0 ** 2.0) + 0.02765235981387666 * exp(x1) + 0.09299150260917513) / (-1.0 * x1 + exp(x0 ** 2.0) + exp(x1) + 3.3628776435387486)
# before constant folding
polynomial({
    {
        polynomial({
            {
                Num.var("x1"): 1,
                Num(-0.7330341374049288): 1,
                polynomial({
                    {Num.var("x1"): 1}: -1.0,
                    {
                        Num.var("x0"): 1,
                        Num(1.1635766746115828): 1,
                        polynomial({
                            {Num.var("x0"): 1}: 1.0,
                            {Num.var("x1"): 1, Num(1.096491354684671): 1}: -1.0,
                            {exp(Num.var("x1")): 1, Num(0.09649135468467125): 1}: 1.0,
                            {Num(0.065716650770683): 1}: 1.0,
                        }): 1,
                    }: 1.0,
                    {Num.var("x0"): 1, Num(3.3628776435387486): 1}: -1.0,
                    {Num(0.5423590312635699): 1}: 1.0,
                }): 1,
            }: 1.0,
            {Num.var("x1"): 1, Num(0.02765235981387666): 1}: -1.0,
            {Num(0.02765235981387666): 1, exp(polynomial({{Num.var("x0"): 2}: 1.0})): 1}: 1.0,
            {exp(Num.var("x1")): 1, Num(0.02765235981387666): 1}: 1.0,
            {Num(0.09299150260917513): 1}: 1.0,
        }): 1,
        polynomial({
            {exp(Num.var("x1")): 1}: 1.0,
            {Num(3.3628776435387486): 1}: 1.0,
            {exp(polynomial({{Num.var("x0"): 2}: 1.0})): 1}: 1.0,
            {Num.var("x1"): 1, Num(-1.0): 1}: 1.0,
        }): -1,
    }: 1.0
})
# after constant folding
polynomial({
    {
        polynomial({
            {}: 0.09299150260917513,
            {Num.var("x1"): 1}: -0.02765235981387666,
            {exp(Num.var("x1")): 1}: 0.02765235981387666,
            {exp(polynomial({{Num.var("x0"): 2}: 1.0})): 1}: 0.02765235981387666,
            {
                Num.var("x1"): 1,
                polynomial({
                    {}: 0.5423590312635699,
                    {Num.var("x1"): 1}: -1.0,
                    {Num.var("x0"): 1}: -3.3628776435387486,
                    {
                        Num.var("x0"): 1,
                        polynomial({
                            {}: 0.065716650770683,
                            {Num.var("x1"): 1}: -1.096491354684671,
                            {exp(Num.var("x1")): 1}: 0.09649135468467125,
                            {Num.var("x0"): 1}: 1.0,
                        }): 1,
                    }: 1.1635766746115828,
                }): 1,
            }: -0.7330341374049288,
        }): 1,
        polynomial({
            {}: 3.3628776435387486,
            {Num.var("x1"): 1}: -1.0,
            {exp(Num.var("x1")): 1}: 1.0,
            {exp(polynomial({{Num.var("x0"): 2}: 1.0})): 1}: 1.0,
        }): -1,
    }: 1.0
})
# /var/folders/x4/rfvvhxcd53lft2b9844dw1lr0000gn/T/tmpcgy5hqze.egg

# -0.8529414239783971 * (x1 * (x0 * (-2.824404573652556 + (x0 + x1 / -0.9119998946891651 + exp(x1) / 10.363622764629516)) + x1 / -1.1635766746115828 + 0.4661137019136418)) / (exp(x1) - x1 + exp(x0 ** 2.0) + 3.3628776435387486) + 0.02765235981387666
# before constant folding
polynomial({
    {
        Num.var("x1"): 1,
        Num(-0.8529414239783971): 1,
        polynomial({
            {
                Num.var("x0"): 1,
                polynomial({
                    {Num.var("x0"): 1}: 1.0,
                    {Num(-2.824404573652556): 1}: 1.0,
                    {Num.var("x1"): 1, Num(-0.9119998946891651): -1}: 1.0,
                    {exp(Num.var("x1")): 1, Num(10.363622764629516): -1}: 1.0,
                }): 1,
            }: 1.0,
            {Num.var("x1"): 1, Num(-1.1635766746115828): -1}: 1.0,
            {Num(0.4661137019136418): 1}: 1.0,
        }): 1,
        polynomial({
            {Num.var("x1"): 1}: -1.0,
            {exp(Num.var("x1")): 1}: 1.0,
            {exp(polynomial({{Num.var("x0"): 2}: 1.0})): 1}: 1.0,
            {Num(3.3628776435387486): 1}: 1.0,
        }): -1,
    }: 1.0,
    {Num(0.02765235981387666): 1}: 1.0,
})
# after constant folding
polynomial({
    {}: 0.02765235981387666,
    {
        Num.var("x1"): 1,
        polynomial({
            {}: 0.4661137019136418,
            {Num.var("x1"): 1}: -0.8594190841216486,
            {
                Num.var("x0"): 1,
                polynomial({
                    {}: -2.824404573652556,
                    {Num.var("x1"): 1}: -1.096491354684671,
                    {Num.var("x0"): 1}: 1.0,
                    {exp(Num.var("x1")): 1}: 0.09649135468467127,
                }): 1,
            }: 1.0,
        }): 1,
        polynomial({
            {}: 3.3628776435387486,
            {Num.var("x1"): 1}: -1.0,
            {exp(Num.var("x1")): 1}: 1.0,
            {exp(polynomial({{Num.var("x0"): 2}: 1.0})): 1}: 1.0,
        }): -1,
    }: -0.8529414239783971,
})
# /var/folders/x4/rfvvhxcd53lft2b9844dw1lr0000gn/T/tmparp2az5t.egg
