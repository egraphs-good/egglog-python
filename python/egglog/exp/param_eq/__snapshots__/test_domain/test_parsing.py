from egglog import *
from egglog.exp.param_eq.domain import *

# x * y + z
# n_params=0
Num.var("x") * Num.var("y") + Num.var("z")
# n_params=0
polynomial({{Num.var("x"): 1, Num.var("y"): 1}: 1.0, {Num.var("z"): 1}: 1.0})

# x * (y + z)
# n_params=0
Num.var("x") * (Num.var("y") + Num.var("z"))
# n_params=0
polynomial({{Num.var("x"): 1, polynomial({{Num.var("y"): 1}: 1.0, {Num.var("z"): 1}: 1.0}): 1}: 1.0})

# x * y * z
# n_params=0
Num.var("x") * Num.var("y") * Num.var("z")
# n_params=0
polynomial({{Num.var("x"): 1, Num.var("y"): 1, Num.var("z"): 1}: 1.0})

# (x ** 2) / y
# n_params=0
Num.var("x") ** Num(2.0) / Num.var("y")
# n_params=0
polynomial({{Num.var("x"): 2, Num.var("y"): -1}: 1.0})

# x + 1.0 * y
# n_params=0
Num.var("x") + Num(1.0) * Num.var("y")
# n_params=0
polynomial({{Num.var("x"): 1}: 1.0, {Num(1.0): 1, Num.var("y"): 1}: 1.0})

# x + -1.0 * y
# n_params=0
Num.var("x") + Num(-1.0) * Num.var("y")
# n_params=0
polynomial({{Num.var("x"): 1}: 1.0, {Num(-1.0): 1, Num.var("y"): 1}: 1.0})

# x - y
# n_params=0
Num.var("x") - Num.var("y")
# n_params=0
polynomial({{Num.var("x"): 1}: 1.0, {Num.var("y"): 1}: -1.0})

# -x
# n_params=0
Num(-1.0) * Num.var("x")
# n_params=0
polynomial({{Num(-1.0): 1, Num.var("x"): 1}: 1.0})

# 1 / x
# n_params=0
Num(1.0) / Num.var("x")
# n_params=0
polynomial({{Num.var("x"): -1}: 1.0})

# 2 / x
# n_params=0
Num(2.0) / Num.var("x")
# n_params=0
polynomial({{Num(2.0): 1, Num.var("x"): -1}: 1.0})

# x / 2
# n_params=0
Num.var("x") / Num(2.0)
# n_params=0
polynomial({{Num.var("x"): 1, Num(2.0): -1}: 1.0})

# x / 1
# n_params=0
Num.var("x") / Num(1.0)
# n_params=0
polynomial({{Num.var("x"): 1, Num(1.0): -1}: 1.0})

# x ** 1/2
# n_params=0
Num.var("x") ** Num(1.0) / Num(2.0)
# n_params=0
polynomial({{Num.var("x"): 1, Num(2.0): -1}: 1.0})

# 0.2306440753250631 + (0.03139967317000205)*(x1) + (-1.1634241022901022 + (exp(exp(-1 + x0)))**(-1))*(1.2522488356336676 - (exp((x0)*((exp(exp(-1 + x0)))**(-1)))))
# n_params=4
(
    Num(0.2306440753250631)
    + Num(0.03139967317000205) * Num.var("x1")
    + (Num(-1.1634241022901022) + exp(exp(Num(-1.0) + Num.var("x0"))) ** Num(-1.0))
    * (Num(1.2522488356336676) - exp(Num.var("x0") * exp(exp(Num(-1.0) + Num.var("x0"))) ** Num(-1.0)))
)
# n_params=4
_expr_1 = exp(exp(polynomial({{Num.var("x0"): 1}: 1.0, {Num(-1.0): 1}: 1.0})))
polynomial(
    {
        {Num(0.2306440753250631): 1}: 1.0,
        {Num(0.03139967317000205): 1, Num.var("x1"): 1}: 1.0,
        {
            polynomial({{_expr_1: -1}: 1.0, {Num(-1.1634241022901022): 1}: 1.0}): 1,
            polynomial({{Num(1.2522488356336676): 1}: 1.0, {exp(polynomial({{Num.var("x0"): 1, _expr_1: -1}: 1.0})): 1}: -1.0}): 1,
        }: 1.0,
    }
)

# (x0**2 - (0.04106910574307527*x0 + 0.043582355979073722*x1*(x1 - 4.735723943783631) + 0.01496006509706177 + I*pi)*exp(x0**2) + 0.00822065460724008)*exp(-x0**2)
# n_params=5
(
    Num.var("x0") ** Num(2.0)
    - (
        Num(0.04106910574307527) * Num.var("x0")
        + Num(0.04358235597907372) * Num.var("x1") * (Num.var("x1") - Num(4.735723943783631))
        + Num(0.01496006509706177)
        + Num.var("I") * Num.var("pi")
    )
    * exp(Num.var("x0") ** Num(2.0))
    + Num(0.00822065460724008)
) * exp(Num(-1.0) * Num.var("x0") ** Num(2.0))
# n_params=5
polynomial(
    {
        {
            polynomial(
                {
                    {Num.var("x0"): 2}: 1.0,
                    {
                        polynomial(
                            {
                                {Num.var("x0"): 1, Num(0.04106910574307527): 1}: 1.0,
                                {Num.var("x1"): 1, Num(0.04358235597907372): 1, polynomial({{Num.var("x1"): 1}: 1.0, {Num(4.735723943783631): 1}: -1.0}): 1}: 1.0,
                                {Num(0.01496006509706177): 1}: 1.0,
                                {Num.var("I"): 1, Num.var("pi"): 1}: 1.0,
                            }
                        ): 1,
                        exp(polynomial({{Num.var("x0"): 2}: 1.0})): 1,
                    }: -1.0,
                    {Num(0.00822065460724008): 1}: 1.0,
                }
            ): 1,
            exp(polynomial({{Num.var("x0"): 2, Num(-1.0): 1}: 1.0})): 1,
        }: 1.0
    }
)

# exp((plog((((plog((x1+0.385))*((-0.328/(-0.612))^3))/(-0.379))-plog((((-0.479/(-0.246/(-0.358)))-((-0.289/(-0.327))^3))-(x1/(-0.045-(((-0.464/x0)/x0)^2)))))))*((-0.293/(-0.389))^3)))
# n_params=13
exp(
    log(
        abs(
            log(abs(Num.var("x1") + Num(0.385))) * (Num(-0.328) / Num(-0.612)) ** Num(3.0) / Num(-0.379)
            - log(
                abs(
                    Num(-0.479) / (Num(-0.246) / Num(-0.358))
                    - (Num(-0.289) / Num(-0.327)) ** Num(3.0)
                    - Num.var("x1") / (Num(-0.045) - (Num(-0.464) / Num.var("x0") / Num.var("x0")) ** Num(2.0))
                )
            )
        )
    )
    * (Num(-0.293) / Num(-0.389)) ** Num(3.0)
)
# n_params=13
exp(
    polynomial(
        {
            {
                log(
                    abs(
                        polynomial(
                            {
                                {
                                    log(abs(polynomial({{Num.var("x1"): 1}: 1.0, {Num(0.385): 1}: 1.0}))): 1,
                                    polynomial({{Num(-0.328): 1, Num(-0.612): -1}: 1.0}): 3,
                                    Num(-0.379): -1,
                                }: 1.0,
                                {
                                    log(
                                        abs(
                                            polynomial(
                                                {
                                                    {Num(-0.479): 1, Num(-0.246): -1, Num(-0.358): 1}: 1.0,
                                                    {polynomial({{Num(-0.289): 1, Num(-0.327): -1}: 1.0}): 3}: -1.0,
                                                    {
                                                        Num.var("x1"): 1,
                                                        polynomial({{Num(-0.045): 1}: 1.0, {polynomial({{Num.var("x0"): -2, Num(-0.464): 1}: 1.0}): 2}: -1.0}): -1,
                                                    }: -1.0,
                                                }
                                            )
                                        )
                                    ): 1
                                }: -1.0,
                            }
                        )
                    )
                ): 1,
                polynomial({{Num(-0.293): 1, Num(-0.389): -1}: 1.0}): 3,
            }: 1.0
        }
    )
)

# (-0.7330341374049288 * x1 * (1.1635766746115828 * x0 * (x0 - 1.096491354684671 * x1 + 0.09649135468467125 * exp(x1) + 0.065716650770683) - 3.3628776435387486 * x0 - x1 + 0.5423590312635699) - 0.02765235981387666 * x1 + 0.02765235981387666 * exp(x0 ** 2.0) + 0.02765235981387666 * exp(x1) + 0.09299150260917513) / (-1.0 * x1 + exp(x0 ** 2.0) + exp(x1) + 3.3628776435387486)
# n_params=12
(
    Num(-0.7330341374049288)
    * Num.var("x1")
    * (
        Num(1.1635766746115828) * Num.var("x0") * (Num.var("x0") - Num(1.096491354684671) * Num.var("x1") + Num(0.09649135468467125) * exp(Num.var("x1")) + Num(0.065716650770683))
        - Num(3.3628776435387486) * Num.var("x0")
        - Num.var("x1")
        + Num(0.5423590312635699)
    )
    - Num(0.02765235981387666) * Num.var("x1")
    + Num(0.02765235981387666) * exp(Num.var("x0") ** Num(2.0))
    + Num(0.02765235981387666) * exp(Num.var("x1"))
    + Num(0.09299150260917513)
) / (Num(-1.0) * Num.var("x1") + exp(Num.var("x0") ** Num(2.0)) + exp(Num.var("x1")) + Num(3.3628776435387486))
# n_params=12
polynomial(
    {
        {
            polynomial(
                {
                    {
                        Num.var("x1"): 1,
                        Num(-0.7330341374049288): 1,
                        polynomial(
                            {
                                {Num.var("x1"): 1}: -1.0,
                                {
                                    Num.var("x0"): 1,
                                    Num(1.1635766746115828): 1,
                                    polynomial(
                                        {
                                            {Num.var("x0"): 1}: 1.0,
                                            {Num.var("x1"): 1, Num(1.096491354684671): 1}: -1.0,
                                            {exp(Num.var("x1")): 1, Num(0.09649135468467125): 1}: 1.0,
                                            {Num(0.065716650770683): 1}: 1.0,
                                        }
                                    ): 1,
                                }: 1.0,
                                {Num.var("x0"): 1, Num(3.3628776435387486): 1}: -1.0,
                                {Num(0.5423590312635699): 1}: 1.0,
                            }
                        ): 1,
                    }: 1.0,
                    {Num.var("x1"): 1, Num(0.02765235981387666): 1}: -1.0,
                    {Num(0.02765235981387666): 1, exp(polynomial({{Num.var("x0"): 2}: 1.0})): 1}: 1.0,
                    {exp(Num.var("x1")): 1, Num(0.02765235981387666): 1}: 1.0,
                    {Num(0.09299150260917513): 1}: 1.0,
                }
            ): 1,
            polynomial(
                {{exp(Num.var("x1")): 1}: 1.0, {Num(3.3628776435387486): 1}: 1.0, {exp(polynomial({{Num.var("x0"): 2}: 1.0})): 1}: 1.0, {Num.var("x1"): 1, Num(-1.0): 1}: 1.0}
            ): -1,
        }: 1.0
    }
)

# -0.8529414239783971 * (x1 * (x0 * (-2.824404573652556 + (x0 + x1 / -0.9119998946891651 + exp(x1) / 10.363622764629516)) + x1 / -1.1635766746115828 + 0.4661137019136418)) / (exp(x1) - x1 + exp(x0 ** 2.0) + 3.3628776435387486) + 0.02765235981387666
# n_params=8
Num(-0.8529414239783971) * (
    Num.var("x1")
    * (
        Num.var("x0") * (Num(-2.824404573652556) + (Num.var("x0") + Num.var("x1") / Num(-0.9119998946891651) + exp(Num.var("x1")) / Num(10.363622764629516)))
        + Num.var("x1") / Num(-1.1635766746115828)
        + Num(0.4661137019136418)
    )
) / (exp(Num.var("x1")) - Num.var("x1") + exp(Num.var("x0") ** Num(2.0)) + Num(3.3628776435387486)) + Num(0.02765235981387666)
# n_params=8
polynomial(
    {
        {
            Num.var("x1"): 1,
            Num(-0.8529414239783971): 1,
            polynomial(
                {
                    {
                        Num.var("x0"): 1,
                        polynomial(
                            {
                                {Num.var("x0"): 1}: 1.0,
                                {Num(-2.824404573652556): 1}: 1.0,
                                {Num.var("x1"): 1, Num(-0.9119998946891651): -1}: 1.0,
                                {exp(Num.var("x1")): 1, Num(10.363622764629516): -1}: 1.0,
                            }
                        ): 1,
                    }: 1.0,
                    {Num.var("x1"): 1, Num(-1.1635766746115828): -1}: 1.0,
                    {Num(0.4661137019136418): 1}: 1.0,
                }
            ): 1,
            polynomial({{Num.var("x1"): 1}: -1.0, {exp(Num.var("x1")): 1}: 1.0, {exp(polynomial({{Num.var("x0"): 2}: 1.0})): 1}: 1.0, {Num(3.3628776435387486): 1}: 1.0}): -1,
        }: 1.0,
        {Num(0.02765235981387666): 1}: 1.0,
    }
)
