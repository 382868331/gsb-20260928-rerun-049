from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

from .model import Problem
from .model import ObjectiveSense
from .search import SearchState
from .types import ResultStatus, SolveResult


@dataclass(frozen=True)
class VerificationReport:
    valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class Proof:
    problem_hash: str
    search_digest: str
    status: ResultStatus
    claim: str
    assignment: dict[str, int] | None
    objective: Fraction | None
    open_branch_count: int
    closed_branch_count: int
    direction: dict[str, int] | None = None

    def to_dict(self) -> dict:
        return {
            "problem_hash": self.problem_hash,
            "search_digest": self.search_digest,
            "status": self.status.value,
            "claim": self.claim,
            "assignment": self.assignment,
            "objective": None if self.objective is None else str(self.objective),
            "open_branch_count": self.open_branch_count,
            "closed_branch_count": self.closed_branch_count,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "Proof":
        objective = data.get("objective")
        return cls(
            data["problem_hash"],
            data["search_digest"],
            ResultStatus(data["status"]),
            data["claim"],
            dict(data["assignment"]) if data.get("assignment") is not None else None,
            Fraction(objective) if objective is not None else None,
            int(data["open_branch_count"]),
            int(data["closed_branch_count"]),
            dict(data["direction"]) if data.get("direction") is not None else None,
        )


def build_proof(problem: Problem, search: SearchState, result: SolveResult) -> Proof:
    if result.status is ResultStatus.OPTIMAL:
        claim = "incumbent is optimal because no open branches remain"
    elif result.status is ResultStatus.INFEASIBLE:
        claim = "feasible region is empty because every branch was closed without a solution"
    elif result.status is ResultStatus.UNBOUNDED:
        claim = "an improving feasible direction establishes unboundedness"
    elif result.status is ResultStatus.BEST_KNOWN:
        claim = "only the incumbent is known; no optimum is claimed"
    else:
        claim = "no feasible solution was found before search stopped; no optimum is claimed"
    objective = result.best_objective
    if objective is None and result.best_assignment is not None:
        objective = problem.objective_value(result.best_assignment)
    return Proof(
        problem.canonical_hash(),
        search_digest(search),
        result.status,
        claim,
        dict(result.best_assignment) if result.best_assignment is not None else None,
        objective,
        len(search.pending),
        len(search.closed_nodes),
        dict(result.direction) if result.direction is not None else None,
    )


def verify_proof(problem: Problem, search: SearchState, proof: Proof) -> bool:
    return review_proof(problem, proof, search).valid


def review_proof(
    problem: Problem,
    proof: Proof,
    search: SearchState | None = None,
    result: SolveResult | None = None,
) -> VerificationReport:
    errors: list[str] = []

    if proof.problem_hash != problem.canonical_hash():
        errors.append("proof problem_hash does not match the saved problem; a changed constraint invalidates this proof")

    if search is not None:
        if proof.search_digest != search_digest(search):
            errors.append("proof search_digest does not match the saved search state")
        if proof.open_branch_count != len(search.pending):
            errors.append("proof open_branch_count does not match the saved search state")
        if proof.closed_branch_count != len(search.closed_nodes):
            errors.append("proof closed_branch_count does not match the saved search state")

    assignment_error = _check_assignment(problem, proof)
    if assignment_error:
        errors.append(assignment_error)
    direction_error = _check_direction(problem, proof)
    if direction_error:
        errors.append(direction_error)

    if search is not None:
        if proof.status is ResultStatus.OPTIMAL:
            if search.pending or search.best_assignment != proof.assignment:
                errors.append("optimal proof requires no open branches and a matching incumbent")
        elif proof.status is ResultStatus.INFEASIBLE:
            if search.pending or search.best_assignment is not None:
                errors.append("infeasibility proof requires no open branches and no incumbent")
        elif proof.status is ResultStatus.BEST_KNOWN:
            if not search.pending or proof.assignment is None:
                errors.append("best_known proof requires open branches and a feasible incumbent")
        elif proof.status is ResultStatus.UNSOLVED:
            if not search.pending or proof.assignment is not None:
                errors.append("unsolved proof requires open branches and must not present an incumbent")
        elif proof.status is ResultStatus.UNBOUNDED:
            if proof.assignment is None or proof.direction is None:
                errors.append("unbounded proof requires both a feasible integer point and an improving direction")
        else:
            errors.append("proof contains an unsupported result status")

    if result is not None:
        if result.status != proof.status:
            errors.append("saved result status does not match proof status")
        if result.direction != proof.direction:
            errors.append("saved result direction does not match proof direction")

    return VerificationReport(not errors, tuple(errors))


def _check_assignment(problem: Problem, proof: Proof) -> str | None:
    if proof.assignment is None:
        if proof.status in (ResultStatus.OPTIMAL, ResultStatus.BEST_KNOWN):
            return f"{proof.status.value} proof is missing the required feasible integer point"
        return None
    if set(proof.assignment) != set(problem.variable_names):
        return "proof assignment does not name exactly the problem variables"
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in proof.assignment.values()):
        return "proof assignment contains a non-integer value"
    if not problem.is_feasible(proof.assignment):
        return "proof assignment is not feasible for the saved problem"
    if proof.objective != problem.objective_value(proof.assignment):
        return "proof objective does not equal the independently recomputed assignment objective"
    return None


def _check_direction(problem: Problem, proof: Proof) -> str | None:
    if proof.status is not ResultStatus.UNBOUNDED:
        if proof.direction is not None:
            return "only an unbounded proof may carry an improving recession direction"
        return None
    if proof.direction is None:
        return "unbounded proof is missing an improving integer recession direction"
    if set(proof.direction) != set(problem.variable_names):
        return "proof direction does not name exactly the problem variables"
    if any(not isinstance(value, int) or isinstance(value, bool) for value in proof.direction.values()):
        return "proof direction contains a non-integer value"

    for variable in problem.variables:
        value = proof.direction[variable.name]
        if variable.upper is None and value < 0:
            return f"direction decreases unbounded variable {variable.name!r} and eventually leaves its domain"

    for constraint in problem.constraints:
        left = sum(
            Fraction(coefficient) * Fraction(proof.direction.get(name, 0))
            for name, coefficient in constraint.coefficients.items()
        )
        if left > 0:
            return f"direction violates recession condition for constraint {constraint.name!r}"

    sign = 1 if problem.sense == ObjectiveSense.MAXIMIZE else -1
    improvement = sign * problem.objective_value(proof.direction)
    if improvement <= 0:
        return "direction does not strictly improve the saved objective"
    return None


def search_digest(search: SearchState) -> str:
    encoded = json.dumps(search.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
