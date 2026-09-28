"""小例子穷举核对：分支定界结果必须与逐点枚举完全一致。"""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest

from optplatform.branch_bound import BranchBoundSolver, SolveStatus
from optplatform.examples import knapsack_like, minimize_cost
from optplatform.model import (
    Constraint,
    ConstraintSense,
    ILP,
    ObjectiveSense,
    Var,
)


def brute_force(problem: ILP):
    """枚举有限整数盒，返回 (status, incumbent, value)。"""
    names = problem.var_names
    ranges = []
    for v in problem.variables:
        lo = int(v.lb)
        hi = int(v.ub)
        assert v.lb.denominator == 1 and v.ub.denominator == 1
        ranges.append(range(lo, hi + 1))

    def feasible(point):
        for c in problem.constraints:
            lhs = sum(a * point[v] for v, a in c.terms.items())
            if c.sense is ConstraintSense.LE and not lhs <= c.rhs:
                return False
            if c.sense is ConstraintSense.GE and not lhs >= c.rhs:
                return False
            if c.sense is ConstraintSense.EQ and not lhs == c.rhs:
                return False
        return True

    maximizing = problem.objective.sense is ObjectiveSense.MAX
    best = None
    best_point = None
    for values in product(*ranges):
        point = dict(zip(names, values))
        if not feasible(point):
            continue
        val = problem.objective.value(point)
        if best is None or (val > best if maximizing else val < best):
            best, best_point = val, point
    if best is None:
        return SolveStatus.INFEASIBLE, None, None
    return SolveStatus.OPTIMAL, best_point, best


def solve(problem):
    return BranchBoundSolver().search(problem)[0]


@pytest.mark.parametrize("factory", [knapsack_like, minimize_cost])
def test_builtin_examples_match_bruteforce(factory):
    problem = factory()
    expected = brute_force(problem)
    result = solve(problem)
    assert result.status is expected[0]
    assert result.objective == expected[2]
    if expected[1] is not None:
        assert problem.objective.value(result.incumbent) == expected[2]


def test_knapsack_optimal_value():
    result = solve(knapsack_like())
    assert result.status is SolveStatus.OPTIMAL
    assert result.objective == Fraction(23)
    assert result.is_proven


def test_random_small_boxes_match_bruteforce():
    """随机生成若干 2~3 变量有界小问题，逐点枚举核对。"""
    import random

    rng = random.Random(20260929)
    for trial in range(40):
        nv = rng.choice((2, 3))
        names = [f"x{i}" for i in range(nv)]
        variables = [
            Var(n, Fraction(rng.randint(0, 1)), Fraction(rng.randint(2, 4)))
            for n in names
        ]
        constraints = []
        for k in range(rng.randint(1, 3)):
            terms = {n: rng.randint(-2, 3) for n in rng.sample(names, rng.choice((1, 2, nv)))}
            terms = {k: Fraction(v) for k, v in terms.items() if v != 0} or {names[0]: Fraction(1)}
            constraints.append(
                Constraint(terms, ConstraintSense.LE, Fraction(rng.randint(0, 5)), f"c{k}")
            )
        sense = rng.choice(list(ObjectiveSense))
        problem = ILP.build(
            f"rand{trial}",
            variables,
            constraints,
            {n: rng.randint(-3, 4) for n in names},
            sense,
        )
        expected = brute_force(problem)
        result = solve(problem)
        assert result.status is expected[0], trial
        assert result.objective == expected[2], trial
        if expected[1] is not None:
            assert problem.objective.value(result.incumbent) == expected[2]
