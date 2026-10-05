import tempfile
import unittest
from pathlib import Path

from optplatform import (
    HierarchicalSamplingPlan2D,
    ParameterAxis,
    ParameterStudy2D,
    Problem,
    Constraint,
    Variable,
)


FLOOR = ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "floor")
CAP = ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap")


def band_problem(rhs_cap=6):
    return Problem(
        "band",
        [Variable("x", 0, 6)],
        {"x": 1},
        [
            Constraint("floor", {"x": -1}, 0),
            Constraint("cap", {"x": 1}, rhs_cap),
        ],
    )


def disconnected_domain():
    horizontal = [(x, 1) for x in range(0, 5)]
    vertical = [(-4, y) for y in range(1, 6)]
    isolated = [(5, 6)]
    return horizontal + vertical + isolated


class HierarchicalSamplingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def make_study(self, name="study", problem=None):
        return ParameterStudy2D(
            self.root / name,
            problem or band_problem(),
            FLOOR,
            [],
            CAP,
            [],
        )

    def make_plan(self, study=None, budget=6, required=((5, 6),)):
        return HierarchicalSamplingPlan2D(
            study or self.make_study(),
            disconnected_domain(),
            budget,
            (2, 2),
            required,
        )

    def test_initial_sampling_respects_shape_and_required_points(self):
        plan = self.make_plan(budget=10)
        initial = {(c.x, c.y): c for c in plan.candidates()}

        self.assertEqual(initial[(5, 6)].reason, "required observation")
        self.assertNotIn((-1, 1), initial)
        self.assertNotIn((3, 1), initial)
        self.assertNotIn((-4, 3), initial)
        self.assertEqual(
            {(c.x, c.y) for c in plan.pending_candidates()},
            {(-4, 1), (-4, 5), (0, 1), (4, 1), (5, 6)},
        )

    def test_local_refinement_uses_adjacent_change_without_filling_gaps(self):
        plan = self.make_plan()
        plan.run(max_points=2)
        refined = [candidate for candidate in plan.candidates() if candidate.level == 1]
        self.assertEqual([candidate.coordinate for candidate in refined], [(-4, 3)])
        self.assertIn("adjacent results changed", refined[0].reason)
        self.assertIn("infeasible", refined[0].reason)
        self.assertIn("optimal", refined[0].reason)
        self.assertNotIn((-3, 3), [candidate.coordinate for candidate in plan.candidates()])

        plan.run()
        observed = set(plan.observed_coordinates())
        self.assertEqual(plan.spent, 6)
        self.assertEqual(plan.pending_candidates(), [])
        self.assertIn((-4, 3), observed)
        self.assertNotIn((1, 1), observed)
        self.assertNotIn((2, 1), observed)
        self.assertNotIn((3, 1), observed)
        self.assertNotIn((-4, 2), observed)
        self.assertNotIn((-4, 4), observed)

        coverage = plan.request_domain_coverage()
        self.assertFalse(coverage["verified"])
        self.assertEqual(
            set(coverage["missing"]),
            {(1, 1), (2, 1), (3, 1), (-4, 2), (-4, 4)},
        )
        self.assertIsNone(coverage["conclusion"])

    def test_stop_and_reopen_preserves_observations_candidates_and_charge(self):
        study = self.make_study("resume")
        plan = self.make_plan(study)
        plan.run(max_points=2)
        self.assertEqual(plan.spent, 2)
        self.assertEqual(len(plan.observed_coordinates()), 2)

        reopened_study = self.make_study("resume")
        reopened_plan = self.make_plan(reopened_study)
        self.assertEqual(reopened_plan.spent, 2)
        self.assertEqual(len(reopened_plan.observed_coordinates()), 2)
        self.assertEqual(reopened_plan.pending_candidates()[0].coordinate, (-4, 3))

        reopened_plan.run(max_points=1)
        self.assertEqual(reopened_plan.spent, 3)
        self.assertEqual(reopened_plan.observed_coordinates()[2], (-4, 3))
        self.assertTrue(reopened_study.point(-4, 1).verified)

    def test_existing_same_version_observations_are_reused_without_recharge(self):
        study = self.make_study("existing")
        study.add_checkpoints([(-4, 1), (5, 6)])
        study.run()
        plan = self.make_plan(study, budget=6)

        reused = [candidate for candidate in plan.candidates() if "reused" in candidate.reason]
        self.assertEqual({candidate.coordinate for candidate in reused}, {(-4, 1), (5, 6)})
        self.assertEqual(plan.spent, 2)
        plan.run(max_points=1)
        self.assertEqual(plan.spent, 3)
        self.assertEqual(plan.observed_coordinates()[2], (-4, 5))
        self.assertEqual(plan.pending_candidates()[0].coordinate, (-4, 3))

    def test_model_replacement_keeps_legacy_and_resampling_conclusions_separate(self):
        old_study = self.make_study("old")
        plan = self.make_plan(old_study)
        plan.run()
        new_study = self.make_study("new", band_problem(rhs_cap=1))
        comparison = plan.compare_model_replacement(new_study)

        self.assertFalse(comparison["same_version"])
        self.assertEqual(comparison["legacy_observations"], plan.observed_coordinates())
        self.assertEqual(comparison["current_observations"], [])
        self.assertEqual(set(comparison["resample_candidates"]), set(disconnected_domain()))
        self.assertEqual(set(comparison["not_spliceable"]), set(plan.observed_coordinates()))

    def test_rectangular_study_queue_does_not_fill_sampling_holes(self):
        study = ParameterStudy2D(
            self.root / "rectangular-queue",
            band_problem(),
            FLOOR,
            list(range(-4, 6)),
            CAP,
            list(range(1, 7)),
        )
        plan = HierarchicalSamplingPlan2D(
            study,
            disconnected_domain(),
            6,
            (2, 2),
            ((5, 6),),
        )
        plan.run(max_points=2)

        self.assertIsNone(study.point(-3, 1))
        self.assertIsNone(study.point(-4, 2))
        self.assertIsNone(study.point(0, 2))
        self.assertTrue(study.point(-4, 1).verified)
        self.assertIn((0, 1), study.summary()["pending"])


if __name__ == "__main__":
    unittest.main()
