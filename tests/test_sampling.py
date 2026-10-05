import tempfile
import unittest
from pathlib import Path

from optplatform import (
    Constraint,
    HierarchicalSampler2D,
    ParameterAxis,
    ParameterStudy2D,
    Problem,
    Variable,
)


FLOOR = ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "floor")
CAP = ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap")


def band_problem() -> Problem:
    return Problem(
        "band",
        [Variable("x", 0, 10)],
        {"x": 1},
        [
            Constraint("floor", {"x": -1}, 0),
            Constraint("cap", {"x": 1}, 10),
        ],
    )


def domain():
    first_band = [(x, y) for x in (-2, -1, 0, 1) for y in (0, 1)]
    second_band = [(x, y) for x in (4, 5) for y in (3, 4)]
    return first_band + second_band + [(8, 2)]


class HierarchicalSamplingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "study"

    def make_study(self):
        return ParameterStudy2D(
            self.path,
            band_problem(),
            FLOOR,
            [-2, -1, 0, 4, 5, 8],
            CAP,
            [0, 1, 2, 3, 4],
        )

    def test_non_rectangular_initial_plan_required_points_and_refinement(self):
        study = self.make_study()
        sampler = HierarchicalSampler2D(
            study,
            domain(),
            total_budget=10,
            minimum_spacing=(1, 1),
            required=[(-2, 1), (0, 1), (5, 4), (8, 2)],
        )

        initial = sampler.candidate_report()
        self.assertNotIn((2, 2), initial["domain"])
        self.assertNotIn((-1, 1), study.pending_coordinates())
        self.assertEqual(initial["pending"][:3], [(-2, 1), (0, 1), (5, 4)])
        by_coordinate = {tuple(item["coordinate"]): item for item in initial["candidates"]}
        self.assertEqual(by_coordinate[(0, 1)]["source"], "required")
        self.assertEqual(by_coordinate[(8, 2)]["source"], "required")

        sampler.run()
        report = sampler.candidate_report()
        self.assertIn((-1, 1), report["computed"])
        adaptive = next(
            item
            for item in report["candidates"]
            if tuple(item["coordinate"]) == (-1, 1)
        )
        self.assertEqual(adaptive["source"], "adaptive")
        self.assertEqual(
            tuple(map(tuple, adaptive["depends_on"])), ((-2, 1), (0, 1))
        )
        self.assertIn("infeasible -> optimal", adaptive["reason"])
        self.assertEqual(report["spent_budget"], 6)
        self.assertEqual(report["pending"], [])

        coverage = sampler.request_coverage()
        self.assertFalse(coverage["verified"])
        self.assertEqual(set(coverage["coordinates"]), set(domain()))
        self.assertNotIn((2, 2), coverage["coordinates"])
        self.assertEqual(study.point(-2, 1).status, "infeasible")
        self.assertEqual(study.point(-1, 1).status, "optimal")

    def test_stop_and_reopen_preserves_budget_and_order(self):
        required = [(0, 0), (5, 4), (8, 2)]
        study = self.make_study()
        sampler = HierarchicalSampler2D(
            study,
            domain(),
            total_budget=3,
            minimum_spacing=(2, 2),
            required=required,
        )
        sampler.run()
        stopped = sampler.candidate_report()
        self.assertEqual(stopped["spent_budget"], 3)
        self.assertEqual(stopped["computed"], required)
        self.assertTrue(stopped["pending"])

        reopened_study = ParameterStudy2D(
            self.path,
            band_problem(),
            FLOOR,
            [-2, -1, 0, 4, 5, 8],
            CAP,
            [0, 1, 2, 3, 4],
        )
        reopened = HierarchicalSampler2D(
            reopened_study,
            domain(),
            total_budget=3,
            minimum_spacing=(2, 2),
            required=required,
        )
        reopened.run()
        report = reopened.candidate_report()
        self.assertEqual(report["spent_budget"], 3)
        self.assertEqual(report["computed"], required)
        self.assertEqual(report["pending"][0], (-2, 0))

    def test_manifest_rejects_model_version_change(self):
        study = self.make_study()
        HierarchicalSampler2D(
            study,
            domain(),
            total_budget=4,
            minimum_spacing=(2, 2),
            required=[(0, 0), (5, 4), (8, 2)],
        )
        replacement = Problem(
            "band",
            [Variable("x", 0, 10)],
            {"x": 1},
            [
                Constraint("floor", {"x": -1}, 0),
                Constraint("cap", {"x": 1}, 9),
            ],
        )
        with self.assertRaises(ValueError):
            HierarchicalSampler2D(
                ParameterStudy2D(
                    self.path,
                    replacement,
                    FLOOR,
                    [-2, -1, 0, 4, 5, 8],
                    CAP,
                    [0, 1, 2, 3, 4],
                ),
                domain(),
                total_budget=4,
                minimum_spacing=(2, 2),
                required=[(0, 0), (5, 4), (8, 2)],
            )


if __name__ == "__main__":
    unittest.main()
