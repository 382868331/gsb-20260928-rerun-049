"""LP 松弛、相等约束、取消文件与序列化往返测试。"""

from __future__ import annotations

from fractions import Fraction

from optplatform.checkpoint import (
    CheckpointStore,
    problem_from_dict,
    problem_to_dict,
    state_from_dict,
    state_to_dict,
)
from optplatform.lp import LPStatus, solve_relaxation
from optplatform.examples import knapsack_like, unbounded_integer
from optplatform.model import (
    Constraint,
    ConstraintSense,
    ILP,
    ObjectiveSense,
    Var,
)
from optplatform.runner import run
from optplatform.branch_bound import SolveStatus


def test_relaxation_equality_and_shift():
    problem = ILP.build(
        "eq",
        [Var("x", Fraction(-2), Fraction(5)), Var("y", Fraction(1), Fraction(4))],
        [Constraint({"x": 1, "y": 1}, ConstraintSense.EQ, Fraction(3))],
        {"x": 1, "y": 2},
        ObjectiveSense.MAX,
    )
    lp = solve_relaxation(problem)
    assert lp.status is LPStatus.OPTIMAL
    assert lp.point["x"] + lp.point["y"] == 3
    assert lp.objective == Fraction(7)  # y=4, x=-1 时 -1+8=7


def test_relaxation_unbounded_ray():
    lp = solve_relaxation(unbounded_integer())
    assert lp.status is LPStatus.UNBOUNDED
    assert 2 * lp.ray["x"] - lp.ray["y"] <= 0


def test_problem_roundtrip_preserves_infinite_bound():
    problem = unbounded_integer()
    restored = problem_from_dict(problem_to_dict(problem))
    assert restored.var("y").ub is None
    assert restored == problem


def test_state_roundtrip_keeps_frontier_bounds():
    problem = knapsack_like()
    from optplatform.branch_bound import BranchBoundSolver, Budget, Node, ConstraintSense as CS

    _, s = BranchBoundSolver().search(problem, budget=Budget(max_nodes=1))
    restored = state_from_dict(state_to_dict(problem, s))
    assert restored.next_node_id == s.next_node_id
    assert len(restored.frontier) == len(s.frontier)
    for a, b in zip(restored.frontier, s.frontier):
        assert a.bound == b.bound
        assert a.unbounded == b.unbounded
        assert a.extra == b.extra


def test_cancel_file_stops_and_writes_search(tmp_path):
    problem = knapsack_like()
    cancel = tmp_path / "STOP"
    cancel.touch()
    result, state, _ = run(problem, tmp_path / "run", cancel_file=cancel)
    assert result.status is SolveStatus.CANCELLED
    assert (tmp_path / "run" / "search.json").exists()
    assert not (tmp_path / "run" / "result.json").exists()


def test_phase1_negative_rhs_rows():
    # 下界平移后出现负 RHS 行，需要人工变量第一阶段。
    problem = ILP.build(
        "neg",
        [Var("x", Fraction(3), Fraction(6))],
        [Constraint({"x": 1}, ConstraintSense.GE, Fraction(4))],
        {"x": -1},
        ObjectiveSense.MAX,
    )
    lp = solve_relaxation(problem)
    assert lp.status is LPStatus.OPTIMAL
    assert lp.objective == Fraction(-4)
