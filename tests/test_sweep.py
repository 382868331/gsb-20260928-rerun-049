import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from optplatform import (
    Constraint,
    ParameterAxis,
    ParameterSweep,
    Problem,
    ResultStatus,
    SearchBudget,
    SearchStore,
    Variable,
    problem_at,
)


def objective_sweep_problem() -> Problem:
    return Problem(
        "sway",
        [Variable("x", 0, None)],
        {"x": 0},
        [],
    )


def rhs_sweep_problem() -> Problem:
    return Problem(
        "capacity",
        [Variable("x", 0, 5)],
        {"x": 1},
        [Constraint("cap", {"x": 1}, 0)],
    )


class ProblemAtTests(unittest.TestCase):
    def test_objective_axis_replaces_coefficient(self) -> None:
        problem = problem_at(rhs_sweep_problem(), ParameterAxis(ParameterAxis.OBJECTIVE, "x"), 3)
        self.assertEqual(problem.objective, {"x": Fraction(3)})
        self.assertEqual(problem.constraints[0].rhs, Fraction(0))

    def test_rhs_axis_replaces_bound(self) -> None:
        problem = problem_at(rhs_sweep_problem(), ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap"), 4)
        self.assertEqual(problem.constraints[0].rhs, Fraction(4))
        self.assertEqual(problem.objective, {"x": Fraction(1)})

    def test_unknown_axis_target_rejected(self) -> None:
        with self.assertRaises(ValueError):
            problem_at(rhs_sweep_problem(), ParameterAxis(ParameterAxis.OBJECTIVE, "z"), 1)
        with self.assertRaises(ValueError):
            problem_at(rhs_sweep_problem(), ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "nope"), 1)


class ObjectiveSweepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_unbounded_point_verified_and_uncertifiable_points_kept_apart(self) -> None:
        sweep = ParameterSweep(
            self.directory / "obj",
            objective_sweep_problem(),
            ParameterAxis(ParameterAxis.OBJECTIVE, "x"),
            [-1, 0, 1],
        )
        sweep.run()
        self.assertEqual(sweep.pending_values(), [])
        verified = {point.value: point.status for point in sweep.verified_points()}
        self.assertEqual(verified, {1: ResultStatus.UNBOUNDED.value})
        for value in (-1, 0):
            point = sweep.point(value)
            self.assertFalse(point.verified)
            self.assertEqual(point.status, ResultStatus.UNSOLVED.value)
            self.assertEqual(point.stop_reason, "completed")
        self.assertEqual(sweep.summary()["unresolved"], [-1, 0])
        store = SearchStore(self.directory / "obj" / "points" / "1")
        self.assertTrue(store.review_saved().valid)
        unbounded = store.load_result()
        self.assertEqual(unbounded.direction, {"x": 1})

    def test_summary_does_not_claim_intervals(self) -> None:
        sweep = ParameterSweep(
            self.directory / "obj",
            objective_sweep_problem(),
            ParameterAxis(ParameterAxis.OBJECTIVE, "x"),
            [-1, 1],
        )
        sweep.run()
        summary = sweep.summary()
        self.assertEqual(summary["verified"], {"unbounded": [1]})
        self.assertEqual(summary["unresolved"], [-1])
        self.assertIn("no interval", summary["note"])

    def test_add_checkpoints_near_transition(self) -> None:
        sweep = ParameterSweep(
            self.directory / "obj",
            objective_sweep_problem(),
            ParameterAxis(ParameterAxis.OBJECTIVE, "x"),
            [-2, 2],
        )
        sweep.run()
        sweep.add_checkpoints([1, 3])
        self.assertEqual(sweep.pending_values(), [1, 3])
        sweep.run()
        self.assertEqual(sweep.pending_values(), [])
        self.assertEqual(len(sweep.verified_points()), 3)
        sweep.add_checkpoints([1])
        self.assertEqual(sweep.pending_values(), [])

    def test_derive_from_verified_point_keeps_original(self) -> None:
        base = Problem(
            "two-axis",
            [Variable("x", 0, 5)],
            {"x": 1},
            [Constraint("cap", {"x": 1}, 2)],
        )
        sweep = ParameterSweep(
            self.directory / "first",
            base,
            ParameterAxis(ParameterAxis.OBJECTIVE, "x"),
            [1],
        )
        sweep.run()
        derived = sweep.derive(
            1,
            ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap"),
            [-1, 0, 3],
            self.directory / "derived",
        )
        derived.run()
        statuses = {point.value: point.status for point in derived.verified_points()}
        self.assertEqual(statuses[-1], ResultStatus.INFEASIBLE.value)
        self.assertEqual(statuses[0], ResultStatus.OPTIMAL.value)
        self.assertEqual(statuses[3], ResultStatus.OPTIMAL.value)
        self.assertEqual(sweep.pending_values(), [])
        self.assertEqual(len(sweep.verified_points()), 1)
        with self.assertRaises(ValueError):
            sweep.derive(7, ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap"), [0], self.directory / "bad")


class RhsSweepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_infeasible_to_finite_transition(self) -> None:
        sweep = ParameterSweep(
            self.directory / "rhs",
            rhs_sweep_problem(),
            ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap"),
            [-1, 0, 3],
        )
        sweep.run()
        statuses = {point.value: point.status for point in sweep.verified_points()}
        self.assertEqual(statuses[-1], ResultStatus.INFEASIBLE.value)
        self.assertEqual(statuses[0], ResultStatus.OPTIMAL.value)
        self.assertEqual(statuses[3], ResultStatus.OPTIMAL.value)
        result = SearchStore(self.directory / "rhs" / "points" / "3").load_result()
        self.assertEqual(result.best_assignment, {"x": 3})
        self.assertEqual(result.best_objective, Fraction(3))

    def test_warm_start_candidate_is_rechecked_per_point(self) -> None:
        sweep = ParameterSweep(
            self.directory / "rhs",
            rhs_sweep_problem(),
            ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap"),
            [3],
        )
        sweep.run()
        self.assertEqual(sweep._warm_candidate(0), {"x": 3})
        sweep.add_checkpoints([0])
        sweep.run()
        point = sweep.point(0)
        self.assertTrue(point.verified)
        result = SearchStore(self.directory / "rhs" / "points" / "0").load_result()
        self.assertEqual(result.best_objective, Fraction(0))
        store = SearchStore(self.directory / "rhs" / "points" / "0")
        self.assertTrue(store.review_saved().valid)


class PauseResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_max_points_pauses_and_resume_loads_manifest(self) -> None:
        sweep = ParameterSweep(
            self.directory / "rhs",
            rhs_sweep_problem(),
            ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap"),
            [0, 1, 2],
        )
        sweep.run(max_points=1)
        self.assertEqual(sweep.pending_values(), [1, 2])
        resumed = ParameterSweep(
            self.directory / "rhs",
            rhs_sweep_problem(),
            ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap"),
            [0, 1, 2],
        )
        self.assertEqual(resumed.pending_values(), [1, 2])
        resumed.run()
        self.assertEqual(resumed.pending_values(), [])
        self.assertEqual(len(resumed.verified_points()), 3)

    def test_budget_exhaustion_keeps_point_pending_until_resolved(self) -> None:
        problem = ProblemProblem = Problem(
            "wide",
            [Variable("x", 0, 1000)],
            {"x": 1},
        )
        sweep = ParameterSweep(
            self.directory / "wide",
            problem,
            ParameterAxis(ParameterAxis.OBJECTIVE, "x"),
            [1],
            SearchBudget(max_nodes=1),
        )
        sweep.run()
        self.assertEqual(sweep.pending_values(), [1])
        point = sweep.point(1)
        self.assertIsNotNone(point)
        self.assertFalse(point.verified)
        self.assertEqual(point.stop_reason, "node_budget")
        self.assertEqual(point.status, ResultStatus.UNSOLVED.value)
        sweep.run(budget=SearchBudget())
        self.assertEqual(sweep.pending_values(), [])
        point = sweep.point(1)
        self.assertTrue(point.verified)
        self.assertEqual(point.status, ResultStatus.OPTIMAL.value)

    def test_manifest_rejects_mismatched_problem(self) -> None:
        sweep = ParameterSweep(
            self.directory / "rhs",
            rhs_sweep_problem(),
            ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap"),
            [0],
        )
        sweep.run()
        other = Problem("other", [Variable("x", 0, 5)], {"x": 1})
        with self.assertRaises(ValueError):
            ParameterSweep(
                self.directory / "rhs",
                other,
                ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap"),
                [0],
            )


if __name__ == "__main__":
    unittest.main()
