from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

from .model import Problem
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
    return Proof(
        problem.canonical_hash(),
        search_digest(search),
        result.status,
        claim,
        dict(result.best_assignment) if result.best_assignment is not None else None,
        result.best_objective,
        len(search.pending),
        len(search.closed_nodes),
    )


def verify_proof(problem: Problem, search: SearchState, proof: Proof) -> bool:
    if proof.problem_hash != problem.canonical_hash():
        return False
    if proof.search_digest != search_digest(search):
        return False
    if proof.open_branch_count != len(search.pending):
        return False
    if proof.closed_branch_count != len(search.closed_nodes):
        return False

    if proof.assignment is not None:
        if not problem.is_feasible(proof.assignment):
            return False
        if proof.objective != problem.objective_value(proof.assignment):
            return False

    if proof.status is ResultStatus.OPTIMAL:
        return search.pending == [] and search.best_assignment == proof.assignment
    if proof.status is ResultStatus.INFEASIBLE:
        return search.pending == [] and search.best_assignment is None
    if proof.status is ResultStatus.BEST_KNOWN:
        return bool(search.pending) and proof.assignment is not None
    if proof.status is ResultStatus.UNSOLVED:
        return bool(search.pending) and proof.assignment is None
    return proof.status is ResultStatus.UNBOUNDED


def search_digest(search: SearchState) -> str:
    encoded = json.dumps(search.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
