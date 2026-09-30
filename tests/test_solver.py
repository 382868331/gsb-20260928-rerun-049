import tempfile
import unittest
from fractions import Fraction
from itertools import product
from pathlib import Path

from optplatform import (
    BranchAndBound,
    Constraint,
    ObjectiveSense,
    Problem,
    review_proof,
    ResultStatus,
    SearchBudget,
    SearchStore,
    StopReason,
    Variable,
)
from optplatform.proof import verify_proof
from optplatform.search import SearchState, WidestDomain


def make_problem() -> Problem:
    return Problem(
        "knapsack-like",
        [Variable("x", 0, 2), Variable("y", 0, 2)],
        {"x": 5, "y": 4},
        [Constraint("capacity", {"x": 2, "y": 1}, 3)],
    )


def unbounded_problem() -> Problem:
    return Problem("unbounded", [Variable("x", 0, None)], {"x": 1}, [])


def finite_unbounded_variable_problem() -> Problem:
    return Problem(
        "finite-unbounded-variable",
        [Variable("x", 0, None)],
        {"x": 1},
        [Constraint("upper", {"x": 1}, 2)],
    )


def infeasible_unbounded_variable_problem() -> Problem:
    return Problem(
        "infeasible-unbounded-variable",
        [Variable("x", 0, None)],
        {"x": 1},
        [Constraint("impossible", {"x": 1}, -1)],
    )


class SolverTests(unittest.TestCase):
    def test_optimal_solution(self) -> None:
        result = BranchAndBound(make_problem()).solve()
        self.assertIs(result.status, ResultStatus.OPTIMAL)
        self.assertEqual(result.best_assignment, {"x": 1, "y": 1})
        self.assertEqual(result.best_objective, Fraction(9))
        self.assertEqual(result.bound, Fraction(9))

    def test_empty_feasible_domain(self) -> None:
        problem = Problem(
            "empty",
            [Variable("x", 0, 2)],
            {"x": 1},
            [Constraint("impossible", {"x": 1}, -1)],
        )
        result = BranchAndBound(problem).solve()
        self.assertIs(result.status, ResultStatus.INFEASIBLE)
        self.assertIsNone(result.best_assignment)

    def test_unbounded_variable_is_accepted_and_certified(self) -> None:
        problem = unbounded_problem()
        result = BranchAndBound(problem).solve()
        self.assertIs(result.status, ResultStatus.UNBOUNDED)
        self.assertTrue(problem.is_feasible(result.best_assignment))
        self.assertEqual(result.direction, {"x": 1})
        self.assertEqual(problem.objective_value(result.direction), Fraction(1))

    def test_unbounded_variable_with_finite_optimum(self) -> None:
        result = BranchAndBound(finite_unbounded_variable_problem()).solve()
        self.assertIs(result.status, ResultStatus.OPTIMAL)
        self.assertEqual(result.best_assignment, {"x": 2})
        self.assertEqual(result.best_objective, Fraction(2))

    def test_unbounded_variable_with_infeasible_domain(self) -> None:
        result = BranchAndBound(infeasible_unbounded_variable_problem()).solve()
        self.assertIs(result.status, ResultStatus.INFEASIBLE)
        self.assertIsNone(result.best_assignment)

    def test_unbounded_cases_resume_after_exhausted_budget(self) -> None:
        cases = [
            (unbounded_problem(), ResultStatus.UNBOUNDED),
            (finite_unbounded_variable_problem(), ResultStatus.OPTIMAL),
            (infeasible_unbounded_variable_problem(), ResultStatus.INFEASIBLE),
        ]
        for problem, expected in cases:
            with self.subTest(problem=problem.name):
                state = BranchAndBound(problem).initial_state()
                stopped = BranchAndBound(problem).solve(SearchBudget(max_nodes=0), state)
                self.assertIs(stopped.status, ResultStatus.UNSOLVED)
                self.assertIs(stopped.stop_reason, StopReason.NODE_BUDGET)
                resumed = BranchAndBound(problem).solve(state=state)
                self.assertIs(resumed.status, expected)
                self.assertIs(resumed.stop_reason, StopReason.COMPLETED)

    def test_budget_returns_best_known_not_optimal(self) -> None:
        problem = make_problem()
        state = BranchAndBound(problem).initial_state()
        first = BranchAndBound(problem).solve(SearchBudget(max_nodes=1), state)
        self.assertIn(first.status, {ResultStatus.BEST_KNOWN, ResultStatus.UNSOLVED})
        self.assertIs(first.stop_reason, StopReason.NODE_BUDGET)
        self.assertIsNotNone(first.bound)

        second = BranchAndBound(problem).solve(state=state)
        self.assertIs(second.status, ResultStatus.OPTIMAL)
        self.assertEqual(second.best_objective, Fraction(9))
        self.assertGreaterEqual(second.nodes_evaluated, first.nodes_evaluated)

    def test_cancel_reports_best_known(self) -> None:
        calls = iter([False, True, True])
        result = BranchAndBound(make_problem()).solve(
            SearchBudget(is_cancelled=lambda: next(calls, True))
        )
        self.assertIs(result.stop_reason, StopReason.CANCELLED)
        self.assertIn(result.status, {ResultStatus.BEST_KNOWN, ResultStatus.UNSOLVED})
        self.assertFalse(result.proven_optimal)

    def test_checkpoint_keeps_closed_branches(self) -> None:
        problem = make_problem()
        store = SearchStore(tempfile.mkdtemp())
        solver = BranchAndBound(problem, checkpoint_callback=store.save_state)
        first = solver.solve(SearchBudget(max_nodes=2))
        restored = store.load_state()
        evaluated_before = restored.nodes_evaluated
        closed_before = len(restored.closed_nodes)
        pending_before = len(restored.pending)
        final = solver.solve(state=restored)
        self.assertEqual(evaluated_before, first.nodes_evaluated)
        self.assertEqual(closed_before, first.nodes_evaluated)
        self.assertGreaterEqual(pending_before, 1)
        self.assertIs(final.status, ResultStatus.OPTIMAL)
        self.assertGreaterEqual(final.nodes_evaluated - evaluated_before, pending_before)

    def test_rejects_checkpoint_from_changed_problem(self) -> None:
        state = BranchAndBound(make_problem()).initial_state()
        changed = Problem("knapsack-like", [Variable("x", 0, 3), Variable("y", 0, 2)], {"x": 5, "y": 4}, [Constraint("capacity", {"x": 2, "y": 1}, 3)])
        with self.assertRaises(ValueError):
            BranchAndBound(changed).solve(state=state)

    def test_store_separates_problem_search_result_and_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SearchStore(directory)
            problem = make_problem()
            solver = BranchAndBound(problem, checkpoint_callback=store.save_state)
            result = solver.solve()
            state = SearchState.from_dict(__import__("json").loads(Path(store.search_path).read_text()))
            store.save_problem(problem)
            store.save_result(result)
            proof = store.save_proof(problem, state, result)
            for filename in ["problem.json", "search.json", "result.json", "proof.json"]:
                self.assertTrue((Path(directory) / filename).is_file())
            self.assertTrue(verify_proof(store.load_problem(), store.load_state(), store.load_proof()))
            self.assertEqual(proof.status, ResultStatus.OPTIMAL)

    def test_saved_unbounded_proof_rechecks_and_rejects_changed_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SearchStore(directory)
            problem = unbounded_problem()
            solver = BranchAndBound(problem, checkpoint_callback=store.save_state)
            result = solver.solve()
            state = store.load_state()
            store.save_problem(problem)
            store.save_result(result)
            proof = store.save_proof(problem, state, result)
            self.assertTrue(store.review_saved().valid)

            changed = Problem(
                problem.name,
                list(problem.variables),
                dict(problem.objective),
                [Constraint("new-upper", {"x": 1}, 0)],
            )
            review = store.review_saved(changed)
            self.assertFalse(review.valid)
            self.assertTrue(any("changed constraint invalidates" in message for message in review.errors))
            self.assertEqual(store.load_proof().direction, proof.direction)
            self.assertEqual(store.load_problem().constraints, ())

    def test_proof_review_reports_invalid_unbounded_direction(self) -> None:
        from dataclasses import replace
        from optplatform.proof import build_proof

        problem = unbounded_problem()
        state = BranchAndBound(problem).initial_state()
        result = BranchAndBound(problem).solve(state=state)
        valid_proof = build_proof(problem, state, result)
        invalid_proof = replace(valid_proof, direction={"x": -1})
        review = review_proof(problem, invalid_proof, state, result)
        self.assertFalse(review.valid)
        self.assertTrue(any("decreases unbounded variable" in message for message in review.errors))

    def test_minimization_and_branching_strategy_do_not_change_meaning(self) -> None:
        problem = Problem(
            "min",
            [Variable("x", 0, 3), Variable("y", 0, 3)],
            {"x": 1, "y": 2},
            [Constraint("c", {"x": 1, "y": 1}, 2)],
            ObjectiveSense.MINIMIZE,
        )
        default = BranchAndBound(problem).solve()
        widest = BranchAndBound(problem, WidestDomain()).solve()
        self.assertEqual(default.best_objective, widest.best_objective)
        self.assertEqual(default.best_assignment, widest.best_assignment)
        self.assertEqual(default.best_objective, Fraction(0))

    def test_small_examples_match_enumeration(self) -> None:
        problem = make_problem()
        best = max(
            (
                problem.objective_value({"x": x, "y": y}),
                {"x": x, "y": y},
            )
            for x, y in product(range(3), range(3))
            if problem.is_feasible({"x": x, "y": y})
        )
        result = BranchAndBound(problem).solve()
        self.assertEqual((result.best_objective, result.best_assignment), best)


if __name__ == "__main__":
    unittest.main()
