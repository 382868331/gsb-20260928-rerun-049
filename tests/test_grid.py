
import json
import tempfile
import unittest
from pathlib import Path

from optplatform import (
    Constraint,
    ParameterAxis,
    ParameterGrid,
    Problem,
    RegionClaim,
    ResultStatus,
    SearchBudget,
    Variable,
    find_conflicts,
)


def band_problem() -> Problem:
    # feasible iff u + v >= 0; on the boundary u + v == 0 the feasible set is a
    # single integer point, and the feasible band narrows to zero width.
    return Problem(
        "band",
        [Variable("x", 0, 10)],
        {"x": 1},
        [
            Constraint("hi", {"x": 1}, 0),
            Constraint("lo", {"x": -1}, 0),
        ],
    )


def band_axes():
    return (
        ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "hi"),
        ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "lo"),
    )


def ray_problem() -> Problem:
    # maximize u * y with y unbounded above: unbounded for u > 0 and degenerate
    # (objective-neutral or worsening, never certified) exactly on u <= 0.
    return Problem(
        "ray",
        [Variable("x", 0, 5), Variable("y", 0, None)],
        {"y": 0},
        [Constraint("cap", {"x": 1}, 0)],
    )


def ray_axes():
    return (
        ParameterAxis(ParameterAxis.OBJECTIVE, "y"),
        ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap"),
    )


class GridTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()


class NarrowBandTests(GridTestCase):
    def make_grid(self, values_u, values_v) -> ParameterGrid:
        axis_u, axis_v = band_axes()
        return ParameterGrid(
            self.directory / "grid", band_problem(), axis_u, axis_v, values_u, values_v
        )

    def test_narrow_feasible_band_row(self) -> None:
        grid = self.make_grid([0, 1, 2, 3], [-2])
        grid.run()
        self.assertEqual(grid.point(0, -2).status, ResultStatus.INFEASIBLE.value)
        self.assertEqual(grid.point(1, -2).status, ResultStatus.INFEASIBLE.value)
        self.assertEqual(grid.point(2, -2).status, ResultStatus.OPTIMAL.value)
        self.assertEqual(grid.point(3, -2).status, ResultStatus.OPTIMAL.value)
        for cell in [(0, -2), (1, -2), (2, -2), (3, -2)]:
            self.assertTrue(grid.point(*cell).verified)

    def test_degenerate_boundary_region_claim(self) -> None:
        grid = self.make_grid([0, 1, 2, 3], [-2])
        grid.run()
        infeasible = grid.request_region(0, 1, -2, -2, ResultStatus.INFEASIBLE.value)
        self.assertTrue(infeasible.verified)
        self.assertTrue(infeasible.degenerate)
        optimal = grid.request_region(2, 3, -2, -2, ResultStatus.OPTIMAL.value)
        self.assertTrue(optimal.verified)
        # the whole row mixes statuses, so a single conclusion cannot cover it
        whole = grid.request_region(0, 3, -2, -2, ResultStatus.OPTIMAL.value)
        self.assertFalse(whole.verified)
        self.assertIn((0, -2), whole.missing)
        self.assertIn((1, -2), whole.missing)
        self.assertEqual(len(grid.claims()), 2)

    def test_region_claim_requires_full_coverage(self) -> None:
        grid = self.make_grid([2, 3], [-2])
        grid.run(max_points=1)
        claim = grid.request_region(2, 3, -2, -2, ResultStatus.OPTIMAL.value)
        self.assertFalse(claim.verified)
        self.assertEqual(claim.missing, ((3, -2),))
        self.assertEqual(grid.claims(), [])
        grid.run()
        claim = grid.request_region(2, 3, -2, -2, ResultStatus.OPTIMAL.value)
        self.assertTrue(claim.verified)
        self.assertEqual(set(claim.coverage), {(2, -2), (3, -2)})

    def test_each_point_keeps_problem_and_proof(self) -> None:
        grid = self.make_grid([2], [-2])
        grid.run()
        point_dir = self.directory / "grid" / "points" / "2_-2"
        self.assertTrue((point_dir / "problem.json").exists())
        self.assertTrue((point_dir / "proof.json").exists())
        self.assertTrue((point_dir / "result.json").exists())
        saved = json.loads((point_dir / "problem.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["name"], "band")


class CornerNotEnoughTests(GridTestCase):
    def test_verified_corners_do_not_cover_interior(self) -> None:
        axis_u, axis_v = ray_axes()
        grid = ParameterGrid(
            self.directory / "corners",
            ray_problem(),
            axis_u,
            axis_v,
            [1, 3],
            [1, 2],
        )
        grid.run()
        for cell in [(1, 1), (1, 2), (3, 1), (3, 2)]:
            self.assertEqual(grid.point(*cell).status, ResultStatus.UNBOUNDED.value)
            self.assertTrue(grid.point(*cell).verified)
        # four identical corner conclusions still say nothing about the interior
        claim = grid.request_region(1, 3, 1, 2, ResultStatus.UNBOUNDED.value)
        self.assertFalse(claim.verified)
        self.assertEqual(set(claim.missing), {(2, 1), (2, 2)})
        self.assertEqual(grid.claims(), [])
        grid.add_checkpoints([(2, 1), (2, 2)])
        grid.run()
        claim = grid.request_region(1, 3, 1, 2, ResultStatus.UNBOUNDED.value)
        self.assertTrue(claim.verified)
        self.assertEqual(len(claim.coverage), 6)

    def test_degenerate_line_stays_unresolved(self) -> None:
        axis_u, axis_v = ray_axes()
        grid = ParameterGrid(
            self.directory / "line", ray_problem(), axis_u, axis_v, [0, 1], [1, 2]
        )
        grid.run()
        # the u == 0 boundary is degenerate: never certified, never absorbed
        self.assertFalse(grid.point(0, 1).verified)
        self.assertFalse(grid.point(0, 2).verified)
        self.assertTrue(grid.point(1, 1).verified)
        claim = grid.request_region(0, 1, 1, 2, ResultStatus.UNBOUNDED.value)
        self.assertFalse(claim.verified)
        self.assertIn((0, 1), claim.missing)
        self.assertIn((0, 2), claim.missing)
        line = grid.request_region(1, 1, 1, 2, ResultStatus.UNBOUNDED.value)
        self.assertTrue(line.verified)
        self.assertTrue(line.degenerate)
        summary = grid.summary()
        self.assertIn([0, 1], summary["unresolved"])
        self.assertIn([0, 2], summary["unresolved"])
        self.assertEqual(len(summary["regions"]), 1)


class ConflictTests(GridTestCase):
    def test_find_conflicts_reports_overlap_and_evidence(self) -> None:
        first = RegionClaim(0, 3, 0, 3, "optimal", True, ((1, 1),))
        second = RegionClaim(2, 5, 1, 4, "infeasible", True, ((2, 1),))
        same = RegionClaim(0, 1, 0, 3, "optimal", True, ((0, 0),))
        conflicts = find_conflicts([first, second, same])
        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertEqual((conflict.u_min, conflict.u_max), (2, 3))
        self.assertEqual((conflict.v_min, conflict.v_max), (1, 3))
        self.assertEqual({claim.status for claim in conflict.claims}, {"optimal", "infeasible"})
        for claim in conflict.claims:
            self.assertTrue(claim.coverage)

    def test_unverified_or_disjoint_claims_do_not_conflict(self) -> None:
        first = RegionClaim(0, 1, 0, 1, "optimal", True, ((0, 0),))
        unverified = RegionClaim(0, 1, 0, 1, "infeasible", False)
        disjoint = RegionClaim(2, 3, 2, 3, "infeasible", True, ((2, 2),))
        self.assertEqual(find_conflicts([first, unverified, disjoint]), [])

    def test_grid_reports_contradictory_saved_claims(self) -> None:
        axis_u, axis_v = band_axes()
        directory = self.directory / "grid"
        grid = ParameterGrid(directory, band_problem(), axis_u, axis_v, [2, 3], [-2])
        grid.run()
        grid.request_region(2, 3, -2, -2, ResultStatus.OPTIMAL.value)
        # simulate a merged or hand-edited record with a contradictory claim
        data = json.loads((directory / "grid.json").read_text(encoding="utf-8"))
        data["claims"].append(
            RegionClaim(3, 4, -2, -2, "infeasible", True, ((3, -2),)).to_dict()
        )
        (directory / "grid.json").write_text(json.dumps(data), encoding="utf-8")
        reloaded = ParameterGrid(directory, band_problem(), axis_u, axis_v, [], [])
        conflicts = reloaded.conflicts()
        self.assertEqual(len(conflicts), 1)
        self.assertEqual((conflicts[0].u_min, conflicts[0].u_max), (3, 3))
        self.assertEqual(len(reloaded.summary()["conflicts"]), 1)


class PersistenceTests(GridTestCase):
    def test_reload_and_review(self) -> None:
        axis_u, axis_v = band_axes()
        directory = self.directory / "grid"
        grid = ParameterGrid(directory, band_problem(), axis_u, axis_v, [0, 1, 2], [-2])
        grid.run()
        grid.request_region(0, 1, -2, -2, ResultStatus.INFEASIBLE.value)
        reloaded = ParameterGrid(directory, band_problem(), axis_u, axis_v, [], [])
        report = reloaded.review_saved()
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(len(reloaded.claims()), 1)

    def test_review_flags_tampered_point_and_dependent_region(self) -> None:
        axis_u, axis_v = band_axes()
        directory = self.directory / "grid"
        grid = ParameterGrid(directory, band_problem(), axis_u, axis_v, [0, 1], [-2])
        grid.run()
        grid.request_region(0, 1, -2, -2, ResultStatus.INFEASIBLE.value)
        result_path = directory / "points" / "0_-2" / "result.json"
        data = json.loads(result_path.read_text(encoding="utf-8"))
        data["status"] = "optimal"
        result_path.write_text(json.dumps(data), encoding="utf-8")
        reloaded = ParameterGrid(directory, band_problem(), axis_u, axis_v, [], [])
        report = reloaded.review_saved()
        self.assertFalse(report.valid)
        self.assertTrue(any("(0, -2)" in error for error in report.errors))

    def test_review_flags_region_when_point_record_changes(self) -> None:
        axis_u, axis_v = band_axes()
        directory = self.directory / "grid"
        grid = ParameterGrid(directory, band_problem(), axis_u, axis_v, [0, 1], [-2])
        grid.run()
        grid.request_region(0, 1, -2, -2, ResultStatus.INFEASIBLE.value)
        data = json.loads((directory / "grid.json").read_text(encoding="utf-8"))
        for item in data["points"]:
            if item["u"] == 1:
                item["status"] = "optimal"
        (directory / "grid.json").write_text(json.dumps(data), encoding="utf-8")
        reloaded = ParameterGrid(directory, band_problem(), axis_u, axis_v, [], [])
        report = reloaded.review_saved()
        self.assertFalse(report.valid)
        self.assertTrue(any("depends on cell (1, -2)" in error for error in report.errors))

    def test_manifest_mismatch_rejected(self) -> None:
        axis_u, axis_v = band_axes()
        directory = self.directory / "grid"
        ParameterGrid(directory, band_problem(), axis_u, axis_v, [0], [-2]).run()
        other = Problem("band", [Variable("x", 0, 9)], {"x": 1}, list(band_problem().constraints))
        with self.assertRaises(ValueError):
            ParameterGrid(directory, other, axis_u, axis_v, [], [])


class BudgetAndDeriveTests(GridTestCase):
    def test_add_budget_resumes_unfinished_points(self) -> None:
        axis_u, axis_v = band_axes()
        directory = self.directory / "grid"
        grid = ParameterGrid(
            directory,
            band_problem(),
            axis_u,
            axis_v,
            [0, 1, 2, 3],
            [-2],
            budget=SearchBudget(max_nodes=0),
        )
        grid.run()
        self.assertTrue(grid.pending_cells())
        self.assertEqual(grid.verified_points(), [])
        grid.add_budget(SearchBudget())
        self.assertEqual(grid.pending_cells(), [])
        self.assertEqual(len(grid.verified_points()), 4)

    def test_derive_from_verified_point(self) -> None:
        axis_u, axis_v = band_axes()
        grid = ParameterGrid(self.directory / "grid", band_problem(), axis_u, axis_v, [2], [-2])
        grid.run()
        derived = grid.derive(
            2,
            -2,
            ParameterAxis(ParameterAxis.OBJECTIVE, "x"),
            axis_v,
            [1, 2],
            [-2, -1],
            self.directory / "derived",
        )
        derived.run()
        self.assertEqual(len(derived.verified_points()), 4)
        self.assertEqual(derived.base_problem.constraints[0].rhs, 2)

    def test_derive_requires_verified_source(self) -> None:
        axis_u, axis_v = band_axes()
        grid = ParameterGrid(self.directory / "grid", band_problem(), axis_u, axis_v, [2], [-2])
        with self.assertRaises(ValueError):
            grid.derive(2, -2, axis_u, axis_v, [1], [-2], self.directory / "derived")


if __name__ == "__main__":
    unittest.main()

