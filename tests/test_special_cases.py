"""空可行域、无界、取消/预算、检查点恢复的行为测试。"""

from __future__ import annotations
from fractions import Fraction

from optplatform.branch_bound import (
    BranchBoundSolver,
    Budget,
    FirstFractional,
    SearchState,
    SolveStatus,
)
from optplatform.checkpoint import CheckpointStore
from optplatform.examples import infeasible_bounds, unbounded_integer
from optplatform.model import (
    Constraint,
    ConstraintSense,
    ILP,
    ObjectiveSense,
    Var,
)
from optplatform.proof import CertificateType
from optplatform.runner import run


def test_empty_feasible_set_is_proven():
    result = BranchBoundSolver().search(infeasible_bounds())[0]
    assert result.status is SolveStatus.INFEASIBLE
    assert result.certificate is CertificateType.EMPTY_FEASIBLE_SET
    assert result.incumbent is None


def test_unbounded_has_integer_point_and_ray():
    problem = unbounded_integer()
    result = BranchBoundSolver().search(problem)[0]
    assert result.status is SolveStatus.UNBOUNDED
    assert result.certificate is CertificateType.UNBOUNDED_RAY
    point = {k: Fraction(v) for k, v in result.evidence["feasible_point"].items()}
    ray = {k: Fraction(v) for k, v in result.evidence["integer_ray"].items()}
    # 射线方向必须是整数、非零，且保持 cone 约束、改进目标。
    assert all(v.denominator == 1 for v in ray.values())
    assert 2 * ray["x"] - ray["y"] <= 0
    for t in range(0, 4):
        moved = {k: point[k] + t * ray[k] for k in point}
        assert 2 * moved["x"] - moved["y"] <= 0
        assert all(v >= 0 for v in moved.values())
    assert result.evidence["objective_gain_per_step"] == "3"


def _large_branching_problem():
    # 需要多次分支才能求解的有界问题。
    return ILP.build(
        "branchy",
        [Var("x", 0, 20), Var("y", 0, 20)],
        [
            Constraint({"x": 1, "y": 1}, ConstraintSense.LE, Fraction(13)),
            Constraint({"x": 2, "y": 7}, ConstraintSense.LE, Fraction(42)),
        ],
        {"x": 5, "y": 8},
        ObjectiveSense.MAX,
    )


def test_budget_reports_best_known_not_optimal():
    problem = _large_branching_problem()
    result, state = BranchBoundSolver(FirstFractional()).search(
        problem, budget=Budget(max_nodes=1)
    )
    assert result.status is SolveStatus.BUDGET
    assert result.certificate is CertificateType.BEST_KNOWN
    assert not result.is_proven
    assert len(state.frontier) >= 1  # 仍有未探索分支


def test_cancel_reports_best_known_not_optimal():
    problem = _large_branching_problem()
    result, _ = BranchBoundSolver().search(
        problem, budget=Budget(cancel_event=lambda: True)
    )
    assert result.status is SolveStatus.CANCELLED
    assert result.certificate is CertificateType.BEST_KNOWN


def test_resume_keeps_pruned_branches_and_finishes(tmp_path):
    problem = _large_branching_problem()
    store = CheckpointStore(tmp_path)
    store.save_problem(problem)

    result1, state1, resumed1 = run(problem, tmp_path, max_nodes=1)
    assert resumed1 is False
    assert result1.status is SolveStatus.BUDGET
    opened_first = state1.stats.nodes_opened
    assert opened_first == 1

    # 恢复后继续：不重新计数已打开节点，最终得到证明。
    result2, state2, resumed2 = run(problem, tmp_path)
    assert resumed2 is True
    assert result2.status is SolveStatus.OPTIMAL
    assert result2.certificate is CertificateType.OPTIMALITY
    # 恢复后新打开节点数是增量，已探索节点不会被重复打开。
    assert state2.stats.nodes_opened > opened_first

    # result.json 只在完成后存在；三类文件分开。
    assert (tmp_path / "problem.json").exists()
    assert (tmp_path / "search.json").exists()
    assert (tmp_path / "result.json").exists()
    loaded = store.load_result()
    assert loaded.objective == result2.objective


def test_budget_then_full_search_matches_direct_search(tmp_path):
    problem = _large_branching_problem()
    direct = BranchBoundSolver().search(problem)[0]

    run(problem, tmp_path / "a", max_nodes=1)
    result, _, _ = run(problem, tmp_path / "a")
    assert result.objective == direct.objective
    assert result.status is direct.status


def test_checkpoint_rejects_different_problem(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save_problem(unbounded_integer())
    state = SearchState.initial()
    store.save_state(unbounded_integer(), state)
    with __import__("pytest").raises(ValueError):
        store.load_state(infeasible_bounds())


def test_strategy_change_preserves_problem_meaning(tmp_path):
    problem = _large_branching_problem()
    r1 = BranchBoundSolver(FirstFractional()).search(problem)[0]
    r2 = BranchBoundSolver().search(problem)[0]
    assert r1.objective == r2.objective
    assert r1.status is r2.status


def test_finite_bounds_cannot_be_unbounded():
    problem = ILP.build(
        "bounded_objective",
        [Var("x", 0, 100)],
        [],
        {"x": 1},
        ObjectiveSense.MAX,
    )
    result = BranchBoundSolver().search(problem)[0]
    assert result.status is SolveStatus.OPTIMAL
    assert result.objective == Fraction(100)
