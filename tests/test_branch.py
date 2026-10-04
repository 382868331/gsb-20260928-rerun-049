import json
import tempfile
import unittest
from pathlib import Path

from optplatform import (
    Constraint,
    ParameterAxis,
    ParameterStudy2D,
    Problem,
    Variable,
    classify_problem_change,
)


def make_base(cap_rhs, objective=None):
    return Problem(
        "crossing",
        [Variable("x", 0, 4), Variable("y", 0, 4)],
        objective or {"x": 2, "y": 1},
        [
            Constraint("floor", {"x": -1, "y": -1}, -1),
            Constraint("c1", {"x": 1, "y": 2}, 0),
            Constraint("c2", {"x": 2, "y": 1}, 0),
            Constraint("cap", {"x": 1, "y": 1}, cap_rhs),
        ],
    )


AXIS_X = ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "c1")
AXIS_Y = ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "c2")
VALUES_X = [0, 1, 2, 3]
VALUES_Y = [2, 3]


class BranchTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old_dir = self.root / "old"
        self.study = ParameterStudy2D(
            self.old_dir, make_base(6), AXIS_X, VALUES_X, AXIS_Y, VALUES_Y
        )
        self.study.run()
        summary = self.study.summary()
        self.assertEqual(sorted(summary["verified"]["infeasible"]), [(0, 2), (0, 3)])
        self.assertEqual(
            sorted(summary["verified"]["optimal"]),
            [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3)],
        )

    def derive(self, new_base, name="branch", **kwargs):
        return self.study.derive_branch(new_base, self.root / name, **kwargs)

    def test_tightened_constraint_reuses_only_infeasible(self):
        branch = self.derive(make_base(1))
        summary = branch.reuse_summary()
        self.assertEqual(summary["reused"]["infeasible"], [(0, 2), (0, 3)])
        self.assertEqual(
            summary["requires_resolve"]["optimal"],
            [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3)],
        )
        self.assertEqual(summary["new_coordinates"], [])
        self.assertEqual(
            summary["pending"], [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3)]
        )
        self.assertEqual(summary["model_version"], make_base(1).canonical_hash())
        self.assertEqual(summary["derived_from"], str(self.old_dir.resolve()))
        for coordinate in [(0, 2), (0, 3)]:
            point = branch.point(*coordinate)
            self.assertTrue(point.verified)
            self.assertEqual(point.status, "infeasible")
        self.assertTrue(branch.review_saved()["valid"])

    def test_tightened_branch_resolves_changed_interior_optimum(self):
        branch = self.derive(make_base(1))
        branch.run()
        self.assertEqual(branch.pending_coordinates(), [])
        old_result = self.study.point(3, 3)
        new_result = branch.point(3, 3)
        self.assertEqual(old_result.status, "optimal")
        self.assertEqual(new_result.status, "optimal")
        self.assertNotEqual(old_result.problem_hash, new_result.problem_hash)
        self.assertTrue(branch.review_saved()["valid"])

    def test_relaxed_constraint_reuses_nothing(self):
        branch = self.derive(make_base(8))
        summary = branch.reuse_summary()
        self.assertEqual(summary["reused"], {})
        self.assertEqual(
            summary["requires_resolve"]["infeasible"], [(0, 2), (0, 3)]
        )
        self.assertEqual(
            summary["requires_resolve"]["optimal"],
            [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3)],
        )
        self.assertEqual(len(summary["pending"]), 8)

    def test_objective_change_reuses_only_infeasible(self):
        branch = self.derive(make_base(6, objective={"x": 3, "y": 1}))
        summary = branch.reuse_summary()
        self.assertEqual(summary["reused"]["infeasible"], [(0, 2), (0, 3)])
        self.assertEqual(len(summary["requires_resolve"]["optimal"]), 6)
        decision = next(d for d in branch.reuse_decisions() if d.coordinate == (0, 2))
        self.assertEqual(decision.rule, "objective_changed")

    def test_identical_model_reuses_everything(self):
        branch = self.derive(make_base(6))
        summary = branch.reuse_summary()
        self.assertEqual(summary["pending"], [])
        self.assertEqual(len(summary["reused"]["infeasible"]), 2)
        self.assertEqual(len(summary["reused"]["optimal"]), 6)
        self.assertTrue(branch.review_saved()["valid"])

    def test_range_extension_marks_new_coordinates(self):
        branch = self.derive(make_base(2), values_x=[0, 1, 2, 3, 4])
        summary = branch.reuse_summary()
        self.assertEqual(summary["new_coordinates"], [(4, 2), (4, 3)])
        self.assertIn((4, 2), summary["pending"])
        self.assertIsNone(branch.point(4, 2))

    def test_region_not_verified_from_corners_alone(self):
        branch = self.derive(make_base(1))
        report = branch.request_region_conclusion((0, 0), (2, 3), status="infeasible")
        self.assertTrue(report.verified)
        report = branch.request_region_conclusion((0, 2), (2, 2))
        self.assertFalse(report.verified)
        self.assertEqual(list(report.missing), [(1, 2), (2, 2)])
        self.assertEqual(report.statuses[(0, 2)], "infeasible")

    def test_cross_version_statuses_not_conflated(self):
        branch = self.derive(make_base(1))
        report = branch.request_region_conclusion((0, 0), (2, 3), status="optimal")
        self.assertFalse(report.verified)
        self.assertEqual(len(report.mismatched), 2)
        self.assertEqual(
            sorted(status for _, status in report.mismatched), ["infeasible"] * 2
        )

    def test_non_rectangular_subregion(self):
        branch = self.derive(make_base(1), subregion=[(0, 2), (2, 3), (3, 3)])
        summary = branch.reuse_summary()
        self.assertEqual(summary["reused"]["infeasible"], [(0, 2)])
        self.assertEqual(summary["pending"], [(2, 3), (3, 3)])
        self.assertIsNone(branch.point(1, 2))
        self.assertIsNone(branch.point(3, 2))

    def test_degenerate_line_and_point_regions(self):
        line = self.study.request_region_conclusion((2, 2), (2, 3), status="optimal")
        self.assertTrue(line.verified)
        self.assertEqual(line.conclusion.x_min, line.conclusion.x_max)
        point = self.study.request_region_conclusion((3, 3), (3, 3), status="optimal")
        self.assertTrue(point.verified)
        self.assertEqual(point.conclusion.depends_on, ((3, 3),))
        coverage = self.study.request_coverage([(2, 2), (3, 3)], status="optimal")
        self.assertTrue(coverage["verified"])
        coverage = self.study.request_coverage([(2, 2), (0, 2)], status="optimal")
        self.assertFalse(coverage["verified"])
        self.assertEqual(coverage["mismatched"], [((0, 2), "infeasible")])
        coverage = self.study.request_coverage([(2, 2), (9, 9)])
        self.assertEqual(coverage["missing"], [(9, 9)])

    def test_reopen_preserves_reuse_regions_and_holes(self):
        branch = self.derive(make_base(1))
        report = branch.request_region_conclusion((0, 0), (2, 3), status="infeasible")
        self.assertTrue(report.verified)
        branch.run(max_points=1)
        self.assertEqual(
            branch.pending_coordinates(), [(1, 3), (2, 2), (2, 3), (3, 2), (3, 3)]
        )

        reopened = ParameterStudy2D(
            self.root / "branch", make_base(1), AXIS_X, VALUES_X, AXIS_Y, VALUES_Y
        )
        summary = reopened.reuse_summary()
        self.assertEqual(summary["reused"]["infeasible"], [(0, 2), (0, 3)])
        self.assertEqual(
            reopened.pending_coordinates(), [(1, 3), (2, 2), (2, 3), (3, 2), (3, 3)]
        )
        self.assertEqual(len(reopened.regions()), 1)
        self.assertTrue(reopened.review_saved()["valid"])
        reopened.run()
        self.assertEqual(reopened.pending_coordinates(), [])
        self.assertTrue(reopened.review_saved()["valid"])

    def test_tampered_reuse_record_is_rejected(self):
        branch = self.derive(make_base(1))
        self.assertTrue(branch.review_saved()["valid"])
        manifest = json.loads((self.root / "branch" / "study.json").read_text())
        for record in manifest["reuse"]:
            if record["applied"]:
                record["status"] = "optimal"
        (self.root / "branch" / "study.json").write_text(json.dumps(manifest))
        reopened = ParameterStudy2D(
            self.root / "branch", make_base(1), AXIS_X, VALUES_X, AXIS_Y, VALUES_Y
        )
        report = reopened.review_saved()
        self.assertFalse(report["valid"])
        self.assertTrue(any("reuse" in error for error in report["errors"]))

    def test_classify_problem_change_rules(self):
        base = make_base(6)
        self.assertEqual(classify_problem_change(base, make_base(6)).kind, "identical")
        tightened = classify_problem_change(base, make_base(1))
        self.assertEqual(tightened.kind, "constraint_tightened")
        self.assertEqual(tightened.reusable_statuses, frozenset({"infeasible"}))
        relaxed = classify_problem_change(base, make_base(8))
        self.assertEqual(relaxed.kind, "constraint_relaxed")
        self.assertEqual(relaxed.reusable_statuses, frozenset())
        objective = classify_problem_change(base, make_base(6, {"x": 3, "y": 1}))
        self.assertEqual(objective.kind, "objective_changed")
        self.assertEqual(objective.reusable_statuses, frozenset({"infeasible"}))


if __name__ == "__main__":
    unittest.main()
