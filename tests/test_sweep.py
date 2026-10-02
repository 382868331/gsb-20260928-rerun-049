import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from optplatform import (
    AxisKind,
    Constraint,
    ParameterAxis,
    ParameterSweep,
    Problem,
    ResultStatus,
    SearchBudget,
    Variable,
    vary_problem,
)


def coefficient_problem() -> Problem:
    # maximize c*x with x in [0, +inf):
    # c > 0 -> unbounded (certified ray); c < 0 -> the polyhedron stays unbounded
    # in an objective-neutral direction, so the point remains unsolved/pending.
    return Problem("sweep-coef", [Variable("x", 0, None)], {"x": 1}, [])


def infeasible_unbounded_problem() -> Problem:
    # maximize x subject to y <= 0 and -y <= b with x in [0, +inf), y in [0, 1]:
    # b < 0 -> infeasible, b >= 0 -> unbounded (x grows freely)
    return Problem(
        "sweep-infeasible-unbounded",
        [Variable("x", 0, None), Variable("y", 0, 1)],
        {"x": 1},
        [Constraint("cap-y", {"y": 1}, 0), Constraint("floor-y", {"y": -1}, 0)],
    )


def rhs_problem() -> Problem:
    # maximize x subject to x <= b with x in [0, 2]: b < 0 -> infeasible, else optimal b
    return Problem(
        "sweep-rhs",
        [Variable("x", 0, 2)],
        {"x": 1},
        [Constraint("cap", {"x": 1}, 1)],
    )


class VaryProblemTests(unittest.TestCase):
    def test_objective_coefficient_variation(self) -> None:
        problem = vary_problem(
            coefficient_problem(), AxisKind.OBJECTIVE_COEFFICIENT, "x", Fraction(-3)
        )
        self.assertEqual(problem.objective, {"x": Fraction(-3)})
        self.assertNotEqual(problem.canonical_hash(), coefficient_problem().canonical_hash())

    def test_constraint_rhs_variation(self) -> None:
        problem = vary_problem(rhs_problem(), AxisKind.CONSTRAINT_RHS, "cap", Fraction(-1))
        self.assertEqual(problem.constraints[0].rhs, Fraction(-1))

    def test_unknown_target_rejected(self) -> None:
        with self.assertRaises(ValueError):
            vary_problem(rhs_problem(), AxisKind.CONSTRAINT_RHS, "missing", Fraction(0))
        with self.assertRaises(ValueError):
            vary_problem(rhs_problem(), AxisKind.OBJECTIVE_COEFFICIENT, "missing", Fraction(0))


class SweepTests(unittest.TestCase):
    def test_unbounded_boundary_keeps_uncertified_points_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sweep = ParameterSweep.create(
                coefficient_problem(),
                ParameterAxis(AxisKind.OBJECTIVE_COEFFICIENT, "x", [-2, -1, 1, 2]),
                Path(tmp) / "coef",
            )
            summary = sweep.run()
            unbounded = {p.value for p in summary.by_status(ResultStatus.UNBOUNDED)}
            self.assertEqual(unbounded, {Fraction(1), Fraction(2)})
            pending = {p.value for p in summary.pending}
            self.assertEqual(pending, {Fraction(-2), Fraction(-1)})
            for point in summary.pending:
                self.assertEqual(point.status, ResultStatus.UNSOLVED.value)
            for point in summary.verified:
                report = sweep.review_point(point.index)
                self.assertTrue(report.valid, report.errors)

    def test_infeasible_to_unbounded_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sweep = ParameterSweep.create(
                infeasible_unbounded_problem(),
                ParameterAxis(AxisKind.CONSTRAINT_RHS, "floor-y", [-1, 0, 1]),
                Path(tmp) / "floor",
            )
            summary = sweep.run()
            self.assertEqual(summary.pending, ())
            infeasible = {p.value for p in summary.by_status(ResultStatus.INFEASIBLE)}
            unbounded = {p.value for p in summary.by_status(ResultStatus.UNBOUNDED)}
            self.assertEqual(infeasible, {Fraction(-1)})
            self.assertEqual(unbounded, {Fraction(0), Fraction(1)})
            for point in summary.verified:
                self.assertTrue(sweep.review_point(point.index).valid, point.value)

    def test_infeasible_to_finite_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sweep = ParameterSweep.create(
                rhs_problem(),
                ParameterAxis(AxisKind.CONSTRAINT_RHS, "cap", [-1, 0, 1, 2]),
                Path(tmp) / "rhs",
            )
            summary = sweep.run()
            self.assertEqual(summary.pending, ())
            infeasible = {p.value for p in summary.by_status(ResultStatus.INFEASIBLE)}
            optimal = {p.value for p in summary.by_status(ResultStatus.OPTIMAL)}
            self.assertEqual(infeasible, {Fraction(-1)})
            self.assertEqual(optimal, {Fraction(0), Fraction(1), Fraction(2)})

    def test_budget_exhaustion_leaves_point_pending_then_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "rhs"
            sweep = ParameterSweep.create(
                rhs_problem(),
                ParameterAxis(AxisKind.CONSTRAINT_RHS, "cap", [0, 1, 2]),
                directory,
            )
            summary = sweep.run(SearchBudget(max_nodes=0))
            self.assertEqual(summary.verified, ())
            self.assertEqual(len(summary.pending), 3)
            for point in summary.pending:
                self.assertEqual(point.status, ResultStatus.UNSOLVED.value)
                self.assertEqual(point.budget["max_nodes"], 0)

            reloaded = ParameterSweep.load(directory)
            summary = reloaded.run()
            self.assertEqual(summary.pending, ())
            self.assertEqual(len(summary.verified), 3)

    def test_budget_exhaustion_before_lp_analysis_still_saves_proof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "coef"
            sweep = ParameterSweep.create(
                coefficient_problem(),
                ParameterAxis(AxisKind.OBJECTIVE_COEFFICIENT, "x", [1]),
                directory,
            )
            summary = sweep.run(SearchBudget(max_nodes=0))
            self.assertEqual(len(summary.pending), 1)
            self.assertEqual(summary.pending[0].status, ResultStatus.UNSOLVED.value)
            summary = ParameterSweep.load(directory).run()
            self.assertEqual(summary.pending, ())
            self.assertEqual(summary.verified[0].status, ResultStatus.UNBOUNDED.value)

    def test_pause_and_continue_keeps_finished_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "coef"
            sweep = ParameterSweep.create(
                rhs_problem(),
                ParameterAxis(AxisKind.CONSTRAINT_RHS, "cap", [0, 1, 2, 3]),
                directory,
            )
            calls = {"count": 0}

            def stop_after_two() -> bool:
                calls["count"] += 1
                return calls["count"] > 2

            first = sweep.run(should_stop=stop_after_two)
            self.assertEqual(len(first.verified), 2)
            self.assertEqual(len(first.pending), 2)

            resumed = ParameterSweep.load(directory).run()
            self.assertEqual(len(resumed.verified), 4)
            self.assertEqual(resumed.pending, ())

    def test_refine_adds_checkpoints_near_status_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "coef"
            sweep = ParameterSweep.create(
                infeasible_unbounded_problem(),
                ParameterAxis(AxisKind.CONSTRAINT_RHS, "floor-y", [-1, 1]),
                directory,
            )
            sweep.run()
            added = sweep.refine([Fraction(-1, 2), Fraction(1, 2)])
            self.assertEqual(len(added), 2)
            summary = ParameterSweep.load(directory).run()
            self.assertEqual(summary.pending, ())
            statuses = {p.value: p.status for p in summary.verified}
            self.assertEqual(statuses[Fraction(-1, 2)], ResultStatus.INFEASIBLE.value)
            self.assertEqual(statuses[Fraction(1, 2)], ResultStatus.UNBOUNDED.value)

    def test_derive_from_verified_point_keeps_original_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sweep = ParameterSweep.create(
                rhs_problem(),
                ParameterAxis(AxisKind.CONSTRAINT_RHS, "cap", [0, 1]),
                root / "rhs",
            )
            sweep.run()
            meta_before = (root / "rhs" / "sweep.json").read_text(encoding="utf-8")

            verified_index = next(p.index for p in sweep.points if p.value == 1)
            derived = sweep.derive(
                verified_index,
                ParameterAxis(AxisKind.OBJECTIVE_COEFFICIENT, "x", [1, 2]),
                root / "derived",
            )
            self.assertEqual(derived.base_problem.constraints[0].rhs, Fraction(1))
            self.assertEqual(derived.parent["point_index"], verified_index)
            derived.run()

            self.assertEqual((root / "rhs" / "sweep.json").read_text(encoding="utf-8"), meta_before)
            for point in sweep.points:
                self.assertTrue(sweep.review_point(point.index).valid)

    def test_derive_from_unverified_point_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sweep = ParameterSweep.create(
                rhs_problem(),
                ParameterAxis(AxisKind.CONSTRAINT_RHS, "cap", [0, 1]),
                Path(tmp) / "rhs",
            )
            with self.assertRaises(ValueError):
                sweep.derive(0, ParameterAxis(AxisKind.OBJECTIVE_COEFFICIENT, "x", [1]), Path(tmp) / "d")

    def test_warm_start_borrows_nearest_feasible_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "rhs"
            sweep = ParameterSweep.create(
                rhs_problem(),
                ParameterAxis(AxisKind.CONSTRAINT_RHS, "cap", [0, 1, 2]),
                directory,
            )
            sweep.run()
            loaded = ParameterSweep.load(directory)
            by_value = {p.value: p for p in loaded.points}
            self.assertIsNone(by_value[Fraction(0)].warm_start_from)
            self.assertEqual(
                by_value[Fraction(1)].warm_start_from,
                by_value[Fraction(0)].index,
            )
            self.assertEqual(
                by_value[Fraction(2)].warm_start_from,
                by_value[Fraction(1)].index,
            )
            for point in loaded.points:
                self.assertTrue(loaded.review_point(point.index).valid)

    def test_each_point_stores_problem_budget_and_proof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "rhs"
            sweep = ParameterSweep.create(
                rhs_problem(),
                ParameterAxis(AxisKind.CONSTRAINT_RHS, "cap", [0, 1]),
                directory,
            )
            sweep.run(SearchBudget(max_nodes=100))
            for point in sweep.points:
                store_dir = directory / point.store_dir
                for name in ("problem.json", "search.json", "result.json", "proof.json"):
                    self.assertTrue((store_dir / name).exists(), name)
                saved = json.loads((store_dir / "problem.json").read_text(encoding="utf-8"))
                self.assertEqual(Fraction(saved["constraints"][0]["rhs"]), point.value)
                self.assertEqual(point.budget["max_nodes"], 100)
                self.assertIsNotNone(point.problem_hash)


if __name__ == "__main__":
    unittest.main()
