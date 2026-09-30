import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from optplatform import (
    BranchAndBound,
    Constraint,
    ExactSolver,
    ObjectiveSense,
    Problem,
    Proof,
    ResultStatus,
    SearchBudget,
    SearchStore,
    StopReason,
    Variable,
    review_proof,
)
from optplatform.exact import lp_rows
from optplatform.lp import FarkasCertificate, check_farkas
from optplatform.proof import build_proof


def run(problem, budget=None):
    solver = ExactSolver(problem)
    state = solver.initial_state()
    result = solver.solve(budget, state)
    return result, state


class UnboundedVariableTests(unittest.TestCase):
    def test_variable_without_upper_bound_enters_public_entry(self) -> None:
        problem = Problem("free-x", [Variable("x", 0, None)], {"x": 1}, [])
        result, state = run(problem)
        self.assertIs(result.status, ResultStatus.UNBOUNDED)
        self.assertEqual(result.best_assignment, {"x": 1})
        self.assertTrue(review_proof(problem, state, build_proof(problem, state, result)).valid)

    def test_unbounded_evidence_is_independently_checkable(self) -> None:
        problem = Problem(
            "ray-walk",
            [Variable("x", 0, None), Variable("y", 0, None)],
            {"x": 1, "y": 1},
            [Constraint("slope", {"x": 1, "y": -1}, 0)],
        )
        result, state = run(problem)
        self.assertIs(result.status, ResultStatus.UNBOUNDED)
        report = review_proof(problem, state, build_proof(problem, state, result))
        self.assertTrue(report.valid, report.problems)

    def test_finite_optimum_with_free_side_is_optimal_not_unsolved(self) -> None:
        problem = Problem(
            "capped",
            [Variable("x", 0, None), Variable("y", 0, 2)],
            {"x": 5, "y": 4},
            [Constraint("capacity", {"x": 2, "y": 1}, 3)],
        )
        result, state = run(problem)
        self.assertIs(result.status, ResultStatus.OPTIMAL)
        self.assertEqual(result.best_assignment, {"x": 1, "y": 1})
        self.assertEqual(result.best_objective, Fraction(9))
        self.assertIn("interior_box_radius", result.evidence)
        report = review_proof(problem, state, build_proof(problem, state, result))
        self.assertTrue(report.valid, report.problems)

    def test_infeasible_free_problem_uses_farkas_certificate(self) -> None:
        problem = Problem(
            "empty-relaxation",
            [Variable("x", 0, None)],
            {"x": 1},
            [Constraint("forbid", {"x": 1}, -1)],
        )
        result, state = run(problem)
        self.assertIs(result.status, ResultStatus.INFEASIBLE)
        certificate = FarkasCertificate.from_dict(result.evidence["farkas"])
        self.assertTrue(check_farkas(certificate, lp_rows(problem)))
        report = review_proof(problem, state, build_proof(problem, state, result))
        self.assertTrue(report.valid, report.problems)

    def test_minimize_unbounded_direction(self) -> None:
        problem = Problem(
            "downhill",
            [Variable("x", None, 0)],
            {"x": 1},
            [],
            ObjectiveSense.MINIMIZE,
        )
        result, state = run(problem)
        self.assertIs(result.status, ResultStatus.UNBOUNDED)
        report = review_proof(problem, state, build_proof(problem, state, result))
        self.assertTrue(report.valid, report.problems)

    def test_budget_exhaustion_is_never_called_unbounded(self) -> None:
        problem = Problem("free-x", [Variable("x", 0, None)], {"x": 1}, [])
        solver = ExactSolver(problem)
        state = solver.initial_state()
        partial = solver.solve(SearchBudget(max_nodes=1), state)
        self.assertIn(partial.status, {ResultStatus.UNSOLVED, ResultStatus.BEST_KNOWN})
        self.assertIs(partial.stop_reason, StopReason.NODE_BUDGET)
        self.assertIsNot(partial.status, ResultStatus.UNBOUNDED)

        resumed = ExactSolver(problem).solve(state=state)
        self.assertIs(resumed.status, ResultStatus.UNBOUNDED)
        self.assertGreater(resumed.nodes_evaluated, partial.nodes_evaluated)
        self.assertTrue(review_proof(problem, state, build_proof(problem, state, resumed)).valid)

    def test_budget_then_resume_for_finite_free_problem(self) -> None:
        problem = Problem(
            "capped-resume",
            [Variable("x", 0, None)],
            {"x": 1},
            [Constraint("cap", {"x": 1}, 3)],
        )
        state = ExactSolver(problem).initial_state()
        partial = ExactSolver(problem).solve(SearchBudget(max_nodes=1), state)
        self.assertIs(partial.stop_reason, StopReason.NODE_BUDGET)
        final = ExactSolver(problem).solve(state=state)
        self.assertIs(final.status, ResultStatus.OPTIMAL)
        self.assertEqual(final.best_assignment, {"x": 3})


class StoredReviewTests(unittest.TestCase):
    def test_review_entry_reloads_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SearchStore(directory)
            problem = Problem("free-x", [Variable("x", 0, None)], {"x": 1}, [])
            solver = ExactSolver(problem, checkpoint_callback=store.save_state)
            store.save_problem(problem)
            result = solver.solve()
            state = store.load_state()
            store.save_result(result)
            store.save_proof(problem, state, result)
            report = store.review()
            self.assertTrue(report.valid, report.problems)

    def test_changing_a_constraint_invalidates_old_proof_but_keeps_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SearchStore(directory)
            problem = Problem(
                "cap",
                [Variable("x", 0, None)],
                {"x": 1},
                [Constraint("cap", {"x": 1}, 3)],
            )
            solver = ExactSolver(problem, checkpoint_callback=store.save_state)
            store.save_problem(problem)
            result = solver.solve()
            state = store.load_state()
            store.save_result(result)
            store.save_proof(problem, state, result)

            changed = Problem(
                "cap",
                [Variable("x", 0, None)],
                {"x": 1},
                [Constraint("cap", {"x": 1}, 30)],
            )
            store.save_problem(changed)

            history = list((Path(directory) / "history").iterdir())
            self.assertEqual(len(history), 1)
            archived = history[0]
            for name in ("problem.json", "search.json", "result.json", "proof.json"):
                self.assertTrue((archived / name).is_file())

            old_proof = Proof.from_dict(json.loads((archived / "proof.json").read_text()))
            report = review_proof(changed, None, old_proof)
            self.assertFalse(report.valid)
            self.assertTrue(any("problem_hash" in item for item in report.problems))

    def test_tampered_ray_is_flagged(self) -> None:
        problem = Problem("free-x", [Variable("x", 0, None)], {"x": 1}, [])
        result, state = run(problem)
        proof = build_proof(problem, state, result)
        unbounded = dict(proof.evidence["unbounded"])
        unbounded["ray"] = {"x": -1}
        evidence = dict(proof.evidence)
        evidence["unbounded"] = unbounded
        object.__setattr__(proof, "evidence", evidence)
        report = review_proof(problem, None, proof)
        self.assertFalse(report.valid)
        self.assertTrue(any("ray" in item for item in report.problems))


if __name__ == "__main__":
    unittest.main()
