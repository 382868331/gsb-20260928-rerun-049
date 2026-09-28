"""内置小例子：覆盖最优、空可行域、无界三种结论。"""

from __future__ import annotations

from fractions import Fraction

from .model import Constraint, ConstraintSense, ILP, ObjectiveSense, Var


def knapsack_like() -> ILP:
    # max 5x + 4y；x + y <= 5, 2x + y <= 8, 0<=x,y<=5 整数。
    # 穷举最优：x=3,y=2，值 23。
    return ILP.build(
        "knapsack",
        [Var("x", Fraction(0), Fraction(5)), Var("y", Fraction(0), Fraction(5))],
        [
            Constraint({"x": 1, "y": 1}, ConstraintSense.LE, Fraction(5), "c1"),
            Constraint({"x": 2, "y": 1}, ConstraintSense.LE, Fraction(8), "c2"),
        ],
        {"x": 5, "y": 4},
        ObjectiveSense.MAX,
    )


def minimize_cost() -> ILP:
    # min 3x + 7y；x + y >= 2, 0<=x,y<=4 整数。穷举最优 x=2,y=0 值 6。
    return ILP.build(
        "min_cost",
        [Var("x", 0, 4), Var("y", 0, 4)],
        [Constraint({"x": 1, "y": 1}, ConstraintSense.GE, Fraction(2), "demand")],
        {"x": 3, "y": 7},
        ObjectiveSense.MIN,
    )


def infeasible_bounds() -> ILP:
    # x + y >= 3 与 x <= 1, y <= 1 矛盾。
    return ILP.build(
        "infeasible",
        [Var("x", 0, 1), Var("y", 0, 1)],
        [Constraint({"x": 1, "y": 1}, ConstraintSense.GE, Fraction(3), "impossible")],
        {"x": 1, "y": 1},
        ObjectiveSense.MAX,
    )


def unbounded_integer() -> ILP:
    # max x + y；2x - y <= 0, x,y >= 0 且 y 无界。
    # (0,0) 可行；射线 (1,2) 可行（2*1-2=0）且目标每步 +3，故无界。
    return ILP.build(
        "unbounded",
        [Var("x", Fraction(0), None), Var("y", Fraction(0), None)],
        [Constraint({"x": 2, "y": -1}, ConstraintSense.LE, Fraction(0), "cone")],
        {"x": 1, "y": 1},
        ObjectiveSense.MAX,
    )


EXAMPLES = {
    "knapsack": "两变量背包式小问题，穷举最优 23",
    "min_cost": "最小化问题，穷举最优 6",
    "infeasible": "上下界互相矛盾，空可行域",
    "unbounded": "整数锥上目标无界",
}


def load_example(name: str) -> ILP:
    return {
        "knapsack": knapsack_like,
        "min_cost": minimize_cost,
        "infeasible": infeasible_bounds,
        "unbounded": unbounded_integer,
    }[name]()
