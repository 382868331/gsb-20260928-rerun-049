import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from optplatform import (
    Constraint,
    ModelChange,
    ParameterAxis,
    ParameterStudy2D,
    Problem,
    RegionConclusion,
    ResultStatus,
    SearchBudget,
    SearchStore,
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

    def test_non_rectangular_region_covers_only_selected_lattice_points(self) -> None:
        study = self.make_study(floors=(0, 1), caps=(0, 1))
        study.run()
        selected = [(0, 0), (0, 1), (1, 1)]
        report = study.request_region_conclusion(None, None, coordinates=selected)
        self.assertTrue(report.verified)
        self.assertEqual(set(report.conclusion.coordinates), set(selected))
        self.assertTrue(report.conclusion.contains(0, 0))
        self.assertFalse(report.conclusion.contains(1, 0))

        point = study.request_region_conclusion(None, None, coordinates=[(0, 0)])
        self.assertTrue(point.verified)
        self.assertEqual(point.conclusion.coordinates, ((0, 0),))
        line = study.request_region_conclusion(None, None, coordinates=[(0, 0), (0, 1)])
        self.assertTrue(line.verified)
        self.assertEqual(line.conclusion.coordinates, ((0, 0), (0, 1)))

        reloaded = self.make_study(floors=(0, 1), caps=(0, 1))
        self.assertEqual(set(reloaded.regions()[0].coordinates), set(selected))
        self.assertTrue(reloaded.review_saved()["valid"])

    def test_branch_reuses_only_logic_preserving_status_and_keeps_version(self) -> None:
        parent = self.make_study("parent")
        parent.run()
        infeasible_region = [(-3, 0), (-3, 1), (-2, 0), (-2, 1)]
        optimal_region = [(-1, 2), (-1, 3), (0, 2), (0, 3)]
        self.assertTrue(parent.request_region_conclusion(None, None, coordinates=infeasible_region).verified)
        self.assertTrue(parent.request_region_conclusion(None, None, coordinates=optimal_region).verified)

        tightened_problem = Problem(
            "band",
            [Variable("x", 0, 6)],
            {"x": 1},
            [
                Constraint("floor", {"x": -1}, 0),
                Constraint("cap", {"x": 1}, 6),
                Constraint("extra", {"x": 1}, 1),
            ],
        )
        branch = parent.derive_branch(
            tightened_problem,
            [-3, -2, -1, 0],
            [0, 1, 2, 3],
            self.directory / "tight",
            ModelChange.CONSTRAINT_TIGHTENED,
            "tight-v1",
        )
        reused = branch.request_reused_region_conclusion(infeasible_region)
        self.assertTrue(reused.verified)
        self.assertTrue(reused.reused)
        self.assertEqual(reused.conclusion.model_version, "tight-v1")
        self.assertEqual(reused.conclusion.reused_from["model_version"], "current")
        self.assertEqual(branch.pending_coordinates()[:4], [(-3, 0), (-3, 1), (-3, 2), (-3, 3)])
        with self.assertRaises(ValueError):
            branch.request_reused_region_conclusion(optimal_region)
        self.assertFalse(branch.request_region_conclusion(None, None, coordinates=optimal_region).verified)

        reloaded = ParameterStudy2D(
            self.directory / "tight",
            tightened_problem,
            FLOOR,
            [-3, -2, -1, 0],
            CAP,
            [0, 1, 2, 3],
        )
        self.assertTrue(reloaded.review_saved()["valid"])

    def test_run_region_does_not_fill_outside_holes(self) -> None:
        study = self.make_study(floors=(0, 1), caps=(0, 1))
        study.run_region([(0, 0)])
        self.assertTrue(study.point(0, 0).verified)
        self.assertEqual(
            set(study.pending_coordinates()),
            {(0, 1), (1, 0), (1, 1)},
        )
        reloaded = self.make_study(floors=(0, 1), caps=(0, 1))
        self.assertEqual(
            set(reloaded.pending_coordinates()),
            {(0, 1), (1, 0), (1, 1)},
        )

    def test_range_branch_reuses_old_region_without_filling_new_holes(self) -> None:
        parent = self.make_study("parent", floors=(-1, 0), caps=(2, 3))
        parent.run(max_points=3)
        parent.add_checkpoints([(-1, 2), (-1, 3), (0, 2), (0, 3)])
        parent.run()
        region = [(-1, 2), (-1, 3), (0, 2), (0, 3)]
        self.assertTrue(parent.request_region_conclusion(None, None, coordinates=region).verified)

        branch = parent.derive_branch(
            band_problem(),
            [-1, 0, 1],
            [2, 3, 4],
            self.directory / "range",
            ModelChange.PARAMETER_RANGE,
            "range-v2",
        )
        report = branch.request_reused_region_conclusion(region)
        self.assertTrue(report.verified)
        pending = set(branch.pending_coordinates())
        self.assertIn((1, 4), pending)
        self.assertIn((-1, 2), pending)
        self.assertEqual(branch.verified_points(), [])
        self.assertTrue(branch.review_saved()["valid"])

    def test_crossing_constraint_changes_interior_optimum_and_requires_resolve(self) -> None:
        base = Problem(
            "crossing",
            [Variable("x", 0, 10), Variable("y", 0, 10)],
            {"x": 1, "y": 1},
            [
                Constraint("cap_x", {"x": 1}, 2),
                Constraint("cap_y", {"y": 1}, 2),
            ],
        )
        axis_x = ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap_x")
        axis_y = ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap_y")
        parent = ParameterStudy2D(self.directory / "cross-parent", base, axis_x, [0, 1, 2, 3], axis_y, [0, 1, 2, 3])
        parent.run()
        region = [(x, y) for x in range(4) for y in range(4)]
        self.assertTrue(parent.request_region_conclusion(None, None, coordinates=region).verified)

        crossed = Problem(
            "crossing",
            [Variable("x", 0, 10), Variable("y", 0, 10)],
            {"x": 1, "y": 1},
            [
                Constraint("cap_x", {"x": 1}, 2),
                Constraint("cap_y", {"y": 1}, 2),
                Constraint("diagonal", {"x": 1, "y": 1}, 2),
            ],
        )
        branch = parent.derive_branch(
            crossed,
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            self.directory / "cross-branch",
            ModelChange.CONSTRAINT_TIGHTENED,
            "cross-v2",
        )
        self.assertFalse(branch.request_region_conclusion(None, None, coordinates=region).verified)
        with self.assertRaises(ValueError):
            branch.request_reused_region_conclusion(region)
        branch.run()
        self.assertTrue(branch.request_region_conclusion(None, None, coordinates=region).verified)
        old_interior = SearchStore(parent.point(2, 2).run_dir).load_result().best_objective
        new_interior = SearchStore(branch.point(2, 2).run_dir).load_result().best_objective
        self.assertEqual((old_interior, new_interior), (Fraction(4), Fraction(2)))

    def test_relaxation_reuses_unbounded_but_objective_change_reuses_only_infeasible(self) -> None:
        constrained = Problem(
            "half-free",
            [Variable("x", 0, 5), Variable("y", 0, None)],
            {"x": 1, "y": 1},
            [
                Constraint("cap_x", {"x": 1}, 1),
                Constraint("floor_y", {"y": -1}, 1),
                Constraint("redundant_y", {"y": -1}, 5),
            ],
        )
        axis_x = ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap_x")
        axis_y = ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "floor_y")
        parent = ParameterStudy2D(
            self.directory / "half-free",
            constrained,
            axis_x,
            [0, 1],
            axis_y,
            [0, 1],
        )
        parent.run()
        region = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.assertTrue(parent.request_region_conclusion(None, None, coordinates=region).verified)
        relaxed = Problem(
            "half-free",
            [Variable("x", 0, 5), Variable("y", 0, None)],
            {"x": 1, "y": 1},
            [Constraint("cap_x", {"x": 1}, 1)],
        )
        relaxed_branch = parent.derive_branch(
            relaxed,
            [0, 1],
            [0, 1],
            self.directory / "relaxed",
            ModelChange.CONSTRAINT_RELAXED,
            "relaxed-v1",
        )
        reused = relaxed_branch.request_reused_region_conclusion(region)
        self.assertTrue(reused.verified)
        self.assertEqual(reused.status, ResultStatus.UNBOUNDED.value)

        band = self.make_study("band-parent")
        band.run()
        infeasible_region = [(-3, 0), (-3, 1), (-2, 0), (-2, 1)]
        self.assertTrue(
            band.request_region_conclusion(None, None, coordinates=infeasible_region).verified
        )
        changed_objective = Problem(
            "band",
            [Variable("x", 0, 6)],
            {"x": -1},
            [
                Constraint("floor", {"x": -1}, 0),
                Constraint("cap", {"x": 1}, 6),
            ],
        )
        objective_branch = band.derive_branch(
            changed_objective,
            [-3, -2, -1, 0],
            [0, 1, 2, 3],
            self.directory / "objective",
            ModelChange.OBJECTIVE_CHANGED,
            "objective-v1",
        )
        report = objective_branch.request_reused_region_conclusion(infeasible_region)
        self.assertTrue(report.verified)
        self.assertEqual(report.status, ResultStatus.INFEASIBLE.value)


if __name__ == "__main__":
    unittest.main()
