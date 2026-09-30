from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

from . import lp
from .exact import InfeasibleEvidence, UnboundedEvidence, lp_rows, transformed_objective
from .model import ObjectiveSense, Problem
from .search import SearchState
from .types import ResultStatus, SolveResult


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
    evidence: dict | None = None

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
            "evidence": self.evidence,
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
            data.get("evidence"),
        )


def build_proof(problem: Problem, search: SearchState, result: SolveResult) -> Proof:
    if result.status is ResultStatus.OPTIMAL:
        claim = "incumbent is optimal because no open branches remain"
    elif result.status is ResultStatus.INFEASIBLE:
        claim = "feasible region is empty; a Farkas certificate or the closed search tree proves it"
    elif result.status is ResultStatus.UNBOUNDED:
        claim = "a feasible integer witness plus an improving recession ray establishes unboundedness"
    elif result.status is ResultStatus.BEST_KNOWN:
        claim = "only the incumbent is known; no optimum is claimed"
    else:
        claim = "no feasible solution was found before search stopped; no optimum is claimed"
    return Proof(
        problem.canonical_hash(),
        search_digest(search),
        result.status,
        claim,
        dict(result.best_assignment) if result.best_assignment is not None else None,
        result.best_objective,
        len(search.pending),
        len(search.closed_nodes),
        result.evidence,
    )


@dataclass(frozen=True)
class ReviewReport:
    valid: bool
    problems: tuple[str, ...]

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise InvalidProofError("; ".join(self.problems))


class InvalidProofError(ValueError):
    pass


def review_proof(problem: Problem, search: SearchState | None, proof: Proof) -> ReviewReport:
    """Re-check a saved conclusion independently of how it was produced."""

    problems: list[str] = []
    if proof.problem_hash != problem.canonical_hash():
        problems.append(
            "proof problem_hash does not match the saved problem: the proof belongs to different assumptions"
        )

    if proof.assignment is not None and not problem.is_feasible(proof.assignment):
        problems.append("proof assignment is not feasible for the saved problem")
    if (
        proof.assignment is not None
        and proof.objective is not None
        and proof.objective != problem.objective_value(proof.assignment)
    ):
        problems.append("proof objective does not equal the objective of its assignment")

    if search is not None:
        if proof.search_digest != search_digest(search):
            problems.append("proof search_digest does not match the saved search state")
        if proof.open_branch_count != len(search.pending):
            problems.append("open branch count disagrees with the saved search state")
        if proof.closed_branch_count != len(search.closed_nodes):
            problems.append("closed branch count disagrees with the saved search state")

    status = proof.status
    if status is ResultStatus.OPTIMAL:
        problems.extend(_review_optimal(problem, search, proof))
    elif status is ResultStatus.INFEASIBLE:
        problems.extend(_review_infeasible(problem, search, proof))
    elif status is ResultStatus.UNBOUNDED:
        problems.extend(_review_unbounded(problem, proof))
    elif status in (ResultStatus.BEST_KNOWN, ResultStatus.UNSOLVED):
        problems.extend(_review_incomplete(search, proof))
    return ReviewReport(not problems, tuple(problems))


def verify_proof(problem: Problem, search: SearchState | None, proof: Proof) -> bool:
    return review_proof(problem, search, proof).valid


def _review_optimal(problem: Problem, search: SearchState | None, proof: Proof) -> list[str]:
    problems: list[str] = []
    if proof.assignment is None or proof.objective is None:
        problems.append("optimal proof is missing a feasible assignment and objective")
    if problem.all_bounded:
        if search is None:
            problems.append("optimal proof for a bounded problem requires the saved search state")
        elif search.pending:
            problems.append("optimal proof still has open branches, so optimality is not established")
    else:
        problems.extend(_review_interior_optimum(problem, proof))
    return problems


def _review_interior_optimum(problem: Problem, proof: Proof) -> list[str]:
    evidence = proof.evidence or {}
    radius = evidence.get("interior_box_radius")
    if radius is None:
        return ["optimal proof for unbounded variables is missing the interior-box evidence"]
    if not isinstance(radius, int) or radius <= 0:
        return ["interior_box_radius must be a positive integer"]
    transformed = transformed_objective(problem)
    ray = lp.improving_ray(problem.variable_names, lp_rows(problem), transformed)
    if ray is not None:
        return ["an improving recession ray exists, so a box optimum cannot be globally optimal"]
    if proof.assignment is None:
        return ["optimal proof is missing its assignment"]
    for variable in problem.variables:
        value = proof.assignment[variable.name]
        if variable.lower is None and value <= -radius + 1:
            return [f"assignment touches the artificial lower face of {variable.name!r}"]
        if variable.upper is None and value >= radius - 1:
            return [f"assignment touches the artificial upper face of {variable.name!r}"]
    return []


def _review_infeasible(problem: Problem, search: SearchState | None, proof: Proof) -> list[str]:
    evidence = proof.evidence or {}
    data = evidence.get("farkas")
    if data is not None:
        certificate = lp.FarkasCertificate.from_dict(data)
        if not lp.check_farkas(certificate, lp_rows(problem)):
            return ["infeasibility Farkas certificate is invalid for the saved constraints"]
        return []
    if search is None:
        return ["infeasible proof without a Farkas certificate requires the saved search state"]
    if search.pending:
        return ["infeasible proof still has open branches, so emptiness is not established"]
    if search.best_assignment is not None:
        return ["search recorded a feasible point but the proof claims infeasibility"]
    return []


def _review_unbounded(problem: Problem, proof: Proof) -> list[str]:
    evidence = proof.evidence or {}
    data = evidence.get("unbounded")
    if data is None:
        return ["unbounded proof is missing the witness and ray evidence"]
    unbounded = UnboundedEvidence.from_dict(data)
    problems: list[str] = []
    if not problem.is_feasible(unbounded.witness):
        problems.append("unbounded witness point is not feasible")
    if problem.objective_value(unbounded.witness) != unbounded.witness_objective:
        problems.append("unbounded witness objective value is inconsistent")
    ray = unbounded.ray
    sign = Fraction(1) if problem.sense == ObjectiveSense.MAXIMIZE else Fraction(-1)
    improvement = sum(
        Fraction(problem.objective.get(name, 0)) * ray.get(name, 0)
        for name in problem.variable_names
    )
    transformed_improvement = sign * improvement
    if transformed_improvement <= 0:
        problems.append("ray does not strictly improve the stated objective")
    if unbounded.ray_improvement != improvement:
        problems.append("recorded ray improvement does not match the saved ray")
    for constraint in problem.constraints:
        value = sum(
            Fraction(coefficient) * ray.get(name, 0)
            for name, coefficient in constraint.coefficients.items()
        )
        if value > 0:
            problems.append(f"ray leaves the feasible region of constraint {constraint.name!r}")
    for variable in problem.variables:
        step = ray.get(variable.name, 0)
        if variable.lower is not None and step < 0:
            problems.append(f"ray violates the lower bound of variable {variable.name!r}")
        if variable.upper is not None and step > 0:
            problems.append(f"ray violates the upper bound of variable {variable.name!r}")
    return problems


def _review_incomplete(search: SearchState | None, proof: Proof) -> list[str]:
    problems: list[str] = []
    if proof.status is ResultStatus.BEST_KNOWN and proof.assignment is None:
        problems.append("best_known proof must carry a feasible incumbent")
    if proof.status is ResultStatus.UNSOLVED and proof.assignment is not None:
        problems.append("unsolved proof must not claim an assignment")
    if search is not None and not search.pending:
        problems.append("budget conclusion has no open branches; the search actually finished")
    return problems


def search_digest(search: SearchState) -> str:
    encoded = json.dumps(search.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
