import json
import tempfile
import unittest
from pathlib import Path

from optplatform import (
    Constraint,
    CoverageConclusion,
    ParameterAxis,
    ParameterStudy2D,
    Problem,
    RegionConclusion,
    ResultStatus,
    SearchBudget,
    Variable,
    region_conflicts,
)

FLOOR = ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "floor")
CAP = ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap")


def band_problem() -> Problem:
    # "floor": -x <= rhs  => x >= -rhs ; "cap": x <= rhs
    # an integer x exists iff -floor <= cap, and on the line -floor == cap
    # the feasible set collapses to the single point x == cap
    return Problem(
        "band",
        [Variable("x", 0, 6)],
        {"x": 1},
        [
            Constraint("floor", {"x": -1}, 0),
            Constraint("cap", {"x": 1}, 6),
        ],
    )


def wide_problem() -> Problem:
    return Problem(
        "wide",
        [Variable("x", 0, 1000)],
        {"x": 1},
        [Constraint("cap", {"x": 1}, 1000)],
    )


def free_problem() -> Problem:
    return Problem(
        "free",
        [Variable("x", 0, None), Variable("y", 0, None)],
        {"x": 0, "y": 0},
    )


def triangle_problem() -> Problem:
    # "floor": -(x+y) <= rhs => x+y >= -rhs ; "sum": x+y <= rhs
    return Problem(
        "triangle",
        [Variable("x", 0, 5), Variable("y", 0, 5)],
        {"x": 1, "y": 1},
        [
            Constraint("floor", {"x": -1, "y": -1}, 0),
            Constraint("sum", {"x": 1, "y": 1}, 5),
        ],
    )


class GridStudyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.directory = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def make_study(self, name="study", floors=(-3, -2, -1, 0), caps=(0, 1, 2, 3), **kwargs):
        return ParameterStudy2D(
            self.directory / name,
            band_problem(),
            FLOOR,
            list(floors),
            CAP,
            list(caps),
            **kwargs,
        )

    def test_narrow_feasible_band_is_mapped_point_by_point(self) -> None:
        study = self.make_study()
        study.run()
        summary = study.summary()
        optimal = set(map(tuple, summary["verified"]["optimal"]))
        infeasible = set(map(tuple, summary["verified"]["infeasible"]))
        expected_optimal = {
            (f, c) for f in (-3, -2, -1, 0) for c in (0, 1, 2, 3) if -f <= c
        }
        self.assertEqual(optimal, expected_optimal)
        self.assertEqual(len(optimal) + len(infeasible), 16)
        self.assertEqual(summary["regions"], [])
        self.assertIn("no area", summary["note"])

    def test_region_conclusion_rejected_until_every_point_verified(self) -> None:
        study = self.make_study(floors=(), caps=())
        corners = [(-1, 1), (-1, 3), (0, 1), (0, 3)]
        study.add_checkpoints(corners)
        study.run()
        report = study.request_region_conclusion((-1, 0), (1, 3), status="optimal")
        self.assertFalse(report.verified)
        self.assertEqual(set(report.missing), {(-1, 2), (0, 2)})
        self.assertEqual(study.regions(), [])

        study.add_checkpoints([(-1, 2), (0, 2)])
        study.run()
        report = study.request_region_conclusion((-1, 0), (1, 3), status="optimal")
        self.assertTrue(report.verified)
        self.assertEqual(report.status, "optimal")
        self.assertEqual(len(report.conclusion.depends_on), 6)
        self.assertEqual(len(study.regions()), 1)

    def test_region_conclusion_reports_mismatched_interior(self) -> None:
        study = self.make_study()
        study.run()
        report = study.request_region_conclusion((-3, 0), (0, 3), status="optimal")
        self.assertFalse(report.verified)
        self.assertEqual(
            {coordinate for coordinate, _ in report.mismatched},
            {(f, c) for f in (-3, -2, -1, 0) for c in (0, 1, 2, 3) if -f > c},
        )
        self.assertEqual(study.regions(), [])

        band = study.request_region_conclusion((-1, 0), (2, 3))
        self.assertTrue(band.verified)
        self.assertEqual(band.status, "optimal")
        empty = study.request_region_conclusion((-3, -2), (0, 1))
        self.assertTrue(empty.verified)
        self.assertEqual(empty.status, "infeasible")
        self.assertEqual(study.conflicts(), [])

    def test_degenerate_boundary_line_is_stated_separately(self) -> None:
        study = self.make_study()
        study.run()
        # rectangle straddling the diagonal: boundary points are optimal but
        # the off-diagonal point (-2, 1) is infeasible, so no block conclusion
        report = study.request_region_conclusion((-2, -1), (1, 2))
        self.assertFalse(report.verified)
        self.assertEqual(set(report.mismatched) if False else set(report.missing), set())
        self.assertEqual(report.statuses[(-2, 1)], "infeasible")
        self.assertEqual(report.statuses[(-2, 2)], "optimal")
        # the degenerate line itself can be concluded point by point
        for f, c in [(-3, 3), (-2, 2), (-1, 1)]:
            line = study.request_region_conclusion((f, f), (c, c))
            self.assertTrue(line.verified)
            self.assertEqual(line.status, "optimal")
            self.assertEqual(line.conclusion.depends_on, ((f, c),))

    def test_unbounded_region_conclusion_and_unresolved_points_kept_apart(self) -> None:
        study = ParameterStudy2D(
            self.directory / "free",
            free_problem(),
            ParameterAxis(ParameterAxis.OBJECTIVE, "x"),
            [-1, 0, 1],
            ParameterAxis(ParameterAxis.OBJECTIVE, "y"),
            [-1, 0, 1],
        )
        study.run()
        unbounded = study.request_region_conclusion((1, 1), (-1, 1))
        self.assertTrue(unbounded.verified)
        self.assertEqual(unbounded.status, "unbounded")
        self.assertEqual(
            set(unbounded.conclusion.depends_on), {(1, -1), (1, 0), (1, 1)}
        )
        # decreasing or neutral objectives over free variables stay unresolved
        self.assertEqual(study.point(-1, -1).status, ResultStatus.UNSOLVED.value)
        self.assertFalse(study.point(-1, -1).verified)
        whole = study.request_region_conclusion((-1, 1), (-1, 1))
        self.assertFalse(whole.verified)
        self.assertEqual(
            set(whole.missing), {(-1, -1), (-1, 0), (0, -1), (0, 0)}
        )
        summary = study.summary()
        self.assertEqual(
            set(map(tuple, summary["unresolved"])),
            {(-1, -1), (-1, 0), (0, -1), (0, 0)},
        )

    def test_budget_exhaustion_pauses_and_reload_resumes(self) -> None:
        study = ParameterStudy2D(
            self.directory / "wide",
            wide_problem(),
            ParameterAxis(ParameterAxis.OBJECTIVE, "x"),
            [1],
            CAP,
            [5],
            SearchBudget(max_nodes=1),
        )
        study.run()
        self.assertEqual(study.pending_coordinates(), [(1, 5)])
        point = study.point(1, 5)
        self.assertFalse(point.verified)
        self.assertEqual(point.stop_reason, "node_budget")

        resumed = ParameterStudy2D(
            self.directory / "wide",
            wide_problem(),
            ParameterAxis(ParameterAxis.OBJECTIVE, "x"),
            [1],
            CAP,
            [5],
        )
        resumed.run()
        self.assertEqual(resumed.pending_coordinates(), [])
        self.assertTrue(resumed.point(1, 5).verified)
        report = resumed.request_region_conclusion((1, 1), (5, 5))
        self.assertTrue(report.verified)

    def test_reload_reviews_points_and_region_dependencies(self) -> None:
        study = self.make_study()
        study.run()
        study.request_region_conclusion((-1, 0), (2, 3))

        reloaded = self.make_study()
        self.assertEqual(len(reloaded.regions()), 1)
        self.assertTrue(reloaded.review_saved()["valid"])

        region = reloaded.regions()[0]
        target = reloaded.point(*region.depends_on[0])
        result_path = Path(target.run_dir) / "result.json"
        data = json.loads(result_path.read_text(encoding="utf-8"))
        data["status"] = "infeasible"
        result_path.write_text(json.dumps(data), encoding="utf-8")
        report = reloaded.review_saved()
        self.assertFalse(report["valid"])
        self.assertTrue(report["errors"])

    def test_conflicting_regions_are_reported_with_evidence(self) -> None:
        first = RegionConclusion(-1, 0, 1, 3, "optimal", ((-1, 2), (0, 2)))
        second = RegionConclusion(0, 1, 2, 4, "infeasible", ((0, 2), (1, 2)))
        conflicts = region_conflicts([first, second])
        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertEqual(
            conflict["overlap"], {"x_min": 0, "x_max": 0, "y_min": 2, "y_max": 3}
        )
        self.assertEqual(conflict["first"]["status"], "optimal")
        self.assertEqual(conflict["second"]["status"], "infeasible")
        self.assertTrue(conflict["first"]["evidence"])
        self.assertEqual(region_conflicts([first]), [])

    def test_regions_are_kept_not_overwritten(self) -> None:
        study = self.make_study()
        study.run()
        study.request_region_conclusion((-1, 0), (2, 3))
        study.request_region_conclusion((-3, -2), (0, 1))
        study.request_region_conclusion((-1, 0), (2, 3))
        statuses = [region.status for region in study.regions()]
        self.assertEqual(statuses, ["optimal", "infeasible"])

    def test_derive_spawns_sweep_from_verified_point_only(self) -> None:
        study = self.make_study()
        study.run()
        with self.assertRaises(ValueError):
            study.derive(5, 5, ParameterAxis(ParameterAxis.OBJECTIVE, "x"), [1, 2], self.directory / "nope")
        derived = study.derive(
            0,
            3,
            ParameterAxis(ParameterAxis.OBJECTIVE, "x"),
            [1, 2],
            self.directory / "derived",
        )
        derived.run()
        self.assertEqual(derived.summary()["verified"], {"optimal": [1, 2]})
        self.assertTrue(study.point(0, 3).verified)

    def test_manifest_rejects_mismatched_problem(self) -> None:
        self.make_study().run()
        other = Problem("other", [Variable("x", 0, 6)], {"x": 1})
        with self.assertRaises(ValueError):
            ParameterStudy2D(self.directory / "study", other, FLOOR, [0], CAP, [0])

    def make_triangle_study(self, name="triangle", **kwargs):
        return ParameterStudy2D(
            self.directory / name,
            triangle_problem(),
            FLOOR,
            [-2, -1, 0],
            ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "sum"),
            [4, 5],
            **kwargs,
        )

    def test_coverage_reports_are_saved_and_reviewed(self) -> None:
        study = self.make_triangle_study()
        study.run()
        subset = [(-2, 4), (-2, 5), (-1, 4), (-1, 5), (0, 5)]
        line = [(-2, 4), (-2, 5)]
        single = [(0, 5)]

        for cells in (subset, line, single):
            report = study.request_coverage(cells, status="optimal")
            self.assertTrue(report["verified"])
            self.assertIsNotNone(report["conclusion"])

        coverages = study.coverages()
        self.assertEqual(len(coverages), 3)
        model_version = triangle_problem().canonical_hash()
        for coverage, cells in zip(coverages, (subset, line, single)):
            self.assertEqual(coverage.coordinates, tuple(sorted(map(tuple, cells))))
            self.assertEqual(coverage.depends_on, coverage.coordinates)
            self.assertEqual(coverage.status, "optimal")
            self.assertEqual(coverage.model_version, model_version)

        # the hole (0, 4) inside the bounding rectangle is not covered
        self.assertFalse(coverages[0].covers(0, 4))
        self.assertTrue(coverages[0].covers(0, 5))

        # repeated requests do not duplicate or rewrite saved reports
        study.request_coverage(subset, status="optimal")
        self.assertEqual(study.coverages(), coverages)

        reloaded = self.make_triangle_study()
        self.assertEqual(reloaded.coverages(), coverages)
        self.assertEqual(reloaded.pending_coordinates(), [])
        self.assertTrue(reloaded.review_saved()["valid"])

        # tampering with a depended-on point invalidates the saved coverage
        manifest_path = self.directory / "triangle" / "study.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in data["points"]:
            if (item["x"], item["y"]) == (0, 5):
                item["status"] = "infeasible"
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        tampered = self.make_triangle_study()
        report = tampered.review_saved()
        self.assertFalse(report["valid"])
        self.assertTrue(any("coverage" in error for error in report["errors"]))

    def test_coverage_saved_before_budget_pause_survives_reload(self) -> None:
        study = self.make_triangle_study()
        study.run(max_points=2)
        self.assertEqual(len(study.pending_coordinates()), 4)
        report = study.request_coverage([(-2, 4), (-2, 5)], status="optimal")
        self.assertTrue(report["verified"])

        reloaded = self.make_triangle_study()
        self.assertEqual(len(reloaded.coverages()), 1)
        self.assertEqual(len(reloaded.pending_coordinates()), 4)
        reloaded.run()
        self.assertEqual(reloaded.pending_coordinates(), [])
        self.assertEqual(reloaded.coverages(), study.coverages())
        self.assertTrue(reloaded.review_saved()["valid"])


if __name__ == "__main__":
    unittest.main()
