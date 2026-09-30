from __future__ import annotations

import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

from . import lp
from .model import Constraint, ObjectiveSense, Problem, Variable
from .search import BranchAndBound, SearchBudget
from .types import ResultStatus, SolveResult, StopReason


INITIAL_RADIUS = 1


@dataclass(frozen=True)
class UnboundedEvidence:
    witness: dict[str, int]
    witness_objective: Fraction
    ray: dict[str, int]
    ray_improvement: Fraction

    def to_dict(self) -> dict:
        return {
            "witness": self.witness,
            "witness_objective": str(self.witness_objective),
            "ray": self.ray,
            "ray_improvement": str(self.ray_improvement),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UnboundedEvidence":
        return cls(
            dict(data["witness"]),
            Fraction(data["witness_objective"]),
            dict(data["ray"]),
            Fraction(data["ray_improvement"]),
        )


@dataclass(frozen=True)
class InfeasibleEvidence:
    certificate: lp.FarkasCertificate

    def to_dict(self) -> dict:
        return {"certificate": self.certificate.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "InfeasibleEvidence":
        return cls(lp.FarkasCertificate.from_dict(data["certificate"]))


class ExactSolver:
    """Unified exact entry point for bounded and unbounded integer variables.

    Bounded problems use the interval branch-and-bound directly.  Problems
    with a free side use Fourier-Motzkin pre-checks and expanding finite
    boxes.  ``unbounded`` is returned only together with an independently
    re-checkable feasible witness and improving recession ray; running out
    of budget always yields ``best_known``/``unsolved`` with open branches.
    """

    def __init__(
        self,
        problem: Problem,
        checkpoint_callback: Callable | None = None,
    ):
        self.problem = problem
        self.checkpoint_callback = checkpoint_callback

    def initial_state(self):
        if self.problem.all_bounded:
            return BranchAndBound(self.problem).initial_state()
        from .search import SearchState

        state = SearchState(self.problem.name, self.problem.canonical_hash(), [])
        return state

    def solve(self, budget: SearchBudget | None = None, state=None) -> SolveResult:
        budget = budget or SearchBudget()
        if self.problem.all_bounded:
            return BranchAndBound(self.problem, checkpoint_callback=self.checkpoint_callback).solve(budget, state)

        return _UnboundedRunner(self.problem, self.checkpoint_callback).run(budget, state)


def lp_rows(problem: Problem) -> list[tuple[str, dict, Fraction]]:
    rows: list[tuple[str, dict, Fraction]] = []
    for variable in problem.variables:
        if variable.lower is not None:
            rows.append((f"lower:{variable.name}", {variable.name: Fraction(-1)}, Fraction(-variable.lower)))
        if variable.upper is not None:
            rows.append((f"upper:{variable.name}", {variable.name: Fraction(1)}, Fraction(variable.upper)))
    for constraint in problem.constraints:
        rows.append((
            f"constraint:{constraint.name}",
            dict(constraint.coefficients),
            Fraction(constraint.rhs),
        ))
    return rows


def transformed_objective(problem: Problem) -> dict[str, Fraction]:
    sign = 1 if problem.sense == ObjectiveSense.MAXIMIZE else -1
    return {name: sign * Fraction(value) for name, value in problem.objective.items()}


class _UnboundedRunner:
    def __init__(self, problem: Problem, checkpoint_callback):
        self.problem = problem
        self.checkpoint_callback = checkpoint_callback
        self.rows = lp_rows(problem)
        self.objective = transformed_objective(problem)

    def run(self, budget, state) -> SolveResult:
        from .search import SearchState

        if state is None:
            state = SearchState(self.problem.name, self.problem.canonical_hash(), [])
        if state.problem_name != self.problem.name:
            raise ValueError("checkpoint belongs to a different problem")
        if state.problem_hash and state.problem_hash != self.problem.canonical_hash():
            raise ValueError("checkpoint does not match the immutable problem definition")

        decision = lp.certify_feasibility(self.problem.variable_names, self.rows)
        if isinstance(decision, lp.FarkasCertificate):
            self.infeasible_evidence = InfeasibleEvidence(decision)
            state.pending.clear()
            state.stop_reason = StopReason.COMPLETED
            self._checkpoint(state)
            return SolveResult(
                ResultStatus.INFEASIBLE,
                None,
                None,
                None,
                StopReason.COMPLETED,
                state.nodes_evaluated,
                "continuous relaxation is empty; Farkas certificate proves integer infeasibility",
                {"farkas": decision.to_dict()},
            )

        ray = lp.improving_ray(self.problem.variable_names, self.rows, self.objective)
        self.ray = ray

        if not state.pending:
            radius = state.box_radius
            if radius == 0:
                radius = INITIAL_RADIUS
                state.box_radius = radius
            boxed = self._boxed_problem(radius)
            inner = BranchAndBound(boxed)
            root = inner.initial_state().pending[0]
            state.pending.append(root)

        started = time.monotonic()
        result = None
        while True:
            if budget.max_nodes is not None and state.nodes_evaluated >= budget.max_nodes:
                state.stop_reason = StopReason.NODE_BUDGET
                break
            if budget.max_seconds is not None:
                if time.monotonic() - started >= budget.max_seconds:
                    state.stop_reason = StopReason.TIME_BUDGET
                    break
            if budget.is_cancelled is not None and budget.is_cancelled():
                state.stop_reason = StopReason.CANCELLED
                break

            radius = state.box_radius
            boxed = self._boxed_problem(radius)
            inner = BranchAndBound(boxed, checkpoint_callback=self.checkpoint_callback)
            result = inner.solve(budget, state, verify_problem_hash=False)

            if state.pending:
                return self._budget_result(state, result)

            if result.status is ResultStatus.INFEASIBLE:
                self._open_next_box(state, radius * 2)
                continue

            best_assignment = state.best_assignment
            best_objective = state.best_objective
            if best_assignment is None:
                continue

            if self.ray is not None:
                direction = self.ray
                improvement = sum(
                    self.objective.get(name, Fraction(0)) * direction.get(name, 0)
                    for name in self.problem.variable_names
                )
                witness = dict(best_assignment)
                original_objective = self._original(best_objective)
                self.unbounded_evidence = UnboundedEvidence(
                    witness=witness,
                    witness_objective=original_objective,
                    ray=direction,
                    ray_improvement=improvement if self.problem.sense == ObjectiveSense.MAXIMIZE else -improvement,
                )
                state.pending.clear()
                state.stop_reason = StopReason.COMPLETED
                self._checkpoint(state)
                return SolveResult(
                    ResultStatus.UNBOUNDED,
                    witness,
                    original_objective,
                    None,
                    StopReason.COMPLETED,
                    state.nodes_evaluated,
                    "a feasible integer point plus an improving recession ray proves unboundedness",
                    {"unbounded": self.unbounded_evidence.to_dict()},
                )

            if self._is_interior(best_assignment, radius):
                original_objective = self._original(best_objective)
                state.pending.clear()
                state.stop_reason = StopReason.COMPLETED
                self._checkpoint(state)
                return SolveResult(
                    ResultStatus.OPTIMAL,
                    dict(best_assignment),
                    original_objective,
                    original_objective,
                    StopReason.COMPLETED,
                    state.nodes_evaluated,
                    "box optimum is strictly interior on every free side, so larger boxes cannot improve it",
                    {"interior_box_radius": radius},
                )

            self._open_next_box(state, radius * 2)

        self._checkpoint(state)
        return self._budget_result(state, result)

    def _budget_result(self, state, result) -> SolveResult:
        if state.best_assignment is not None:
            objective = self._original(state.best_objective)
            return SolveResult(
                ResultStatus.BEST_KNOWN,
                dict(state.best_assignment),
                objective,
                None,
                state.stop_reason or StopReason.NODE_BUDGET,
                state.nodes_evaluated,
                "stopped with open expanding-box branches; no optimum or unboundedness is claimed",
            )
        return SolveResult(
            ResultStatus.UNSOLVED,
            None,
            None,
            None,
            state.stop_reason or StopReason.NODE_BUDGET,
            state.nodes_evaluated,
            "stopped with open expanding-box branches and no feasible point; nothing is claimed",
        )

    def _is_interior(self, assignment: dict[str, int], radius: int) -> bool:
        """True if the assignment does not touch any artificial box face.

        When no improving recession ray exists, a box optimum strictly inside
        every free-side face is also the integer optimum of the full domain:
        a better point beyond a face would yield a recession ray contradicting
        the pre-check.  Integer problems need at least one unit of slack.
        """

        for variable in self.problem.variables:
            value = assignment[variable.name]
            if variable.lower is None and value <= -radius + 1:
                return False
            if variable.upper is None and value >= radius - 1:
                return False
        return True

    def _boxed_problem(self, radius: int) -> Problem:
        variables = []
        for variable in self.problem.variables:
            lower = -radius if variable.lower is None else variable.lower
            upper = radius if variable.upper is None else variable.upper
            variables.append(Variable(variable.name, lower, upper))
        return Problem(
            self.problem.name,
            variables,
            dict(self.problem.objective),
            [
                Constraint(c.name, dict(c.coefficients), c.rhs)
                for c in self.problem.constraints
            ],
            self.problem.sense,
        )

    def _open_next_box(self, state, radius: int) -> None:
        state.box_radius = radius
        inner = BranchAndBound(self._boxed_problem(radius))
        state.pending.append(inner.initial_state().pending[0])

    def _checkpoint(self, state) -> None:
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(state)

    def _original(self, transformed: Fraction | None) -> Fraction | None:
        if transformed is None:
            return None
        return transformed if self.problem.sense == ObjectiveSense.MAXIMIZE else -transformed
