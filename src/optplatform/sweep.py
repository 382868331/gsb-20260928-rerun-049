from __future__ import annotations

import json
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

from .model import Constraint, InvalidProblemError, ObjectiveSense, Problem
from .search import BranchAndBound, SearchBudget
from .store import SearchStore
from .types import ResultStatus, SolveResult


TERMINAL_STATUSES = {ResultStatus.OPTIMAL, ResultStatus.INFEASIBLE, ResultStatus.UNBOUNDED}
BUDGET_STOP_REASONS = {"node_budget", "time_budget", "cancelled"}


@dataclass(frozen=True)
class ParameterAxis:
    OBJECTIVE = "objective"
    CONSTRAINT_RHS = "constraint_rhs"

    kind: str
    name: str

    def __post_init__(self) -> None:
        if self.kind not in (ParameterAxis.OBJECTIVE, ParameterAxis.CONSTRAINT_RHS):
            raise InvalidProblemError("axis kind must be 'objective' or 'constraint_rhs'")
        if not self.name:
            raise InvalidProblemError("axis name must be non-empty")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name}

    @classmethod
    def from_dict(cls, data: Mapping) -> "ParameterAxis":
        return cls(data["kind"], data["name"])


def problem_at(problem: Problem, axis: ParameterAxis, value: int) -> Problem:
    if not isinstance(value, int) or isinstance(value, bool):
        raise InvalidProblemError("parameter values must be integers")
    if axis.kind == ParameterAxis.OBJECTIVE:
        if axis.name not in problem.variable_names:
            raise InvalidProblemError(f"unknown objective variable {axis.name!r}")
        objective = dict(problem.objective)
        objective[axis.name] = Fraction(value)
        return Problem(
            problem.name,
            problem.variables,
            objective,
            problem.constraints,
            problem.sense,
        )
    constraints = []
    matched = False
    for constraint in problem.constraints:
        if constraint.name == axis.name:
            matched = True
            constraints.append(Constraint(constraint.name, constraint.coefficients, Fraction(value)))
        else:
            constraints.append(constraint)
    if not matched:
        raise InvalidProblemError(f"unknown constraint {axis.name!r}")
    return Problem(problem.name, problem.variables, problem.objective, constraints, problem.sense)


@dataclass(frozen=True)
class SweepPoint:
    value: int
    status: str | None
    verified: bool
    stop_reason: str | None
    nodes_evaluated: int
    problem_hash: str
    run_dir: str

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "status": self.status,
            "verified": self.verified,
            "stop_reason": self.stop_reason,
            "nodes_evaluated": self.nodes_evaluated,
            "problem_hash": self.problem_hash,
            "run_dir": self.run_dir,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "SweepPoint":
        return cls(
            int(data["value"]),
            data.get("status"),
            bool(data.get("verified", False)),
            data.get("stop_reason"),
            int(data.get("nodes_evaluated", 0)),
            data.get("problem_hash", ""),
            data.get("run_dir", ""),
        )


class ParameterSweep:
    def __init__(
        self,
        directory: str | Path,
        base_problem: Problem,
        axis: ParameterAxis,
        values: Sequence[int],
        budget: SearchBudget | None = None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "sweep.json"
        self.base_problem = base_problem
        self.axis = axis
        self.budget = budget or SearchBudget()
        self._points: dict[int, SweepPoint] = {}
        self._pending: list[int] = []
        if self.manifest_path.exists():
            self._load()
        else:
            self._pending = sorted({int(value) for value in values})
            self._save()

    def _load(self) -> None:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        saved_problem = Problem.from_dict(data["base_problem"])
        if saved_problem.canonical_hash() != self.base_problem.canonical_hash():
            raise ValueError("sweep manifest belongs to a different base problem")
        saved_axis = ParameterAxis.from_dict(data["axis"])
        if saved_axis != self.axis:
            raise ValueError("sweep manifest belongs to a different parameter axis")
        self._points = {
            int(item["value"]): SweepPoint.from_dict(item) for item in data.get("points", [])
        }
        self._pending = sorted(int(value) for value in data.get("pending", []))

    def _save(self) -> None:
        payload = {
            "base_problem": self.base_problem.to_dict(),
            "axis": self.axis.to_dict(),
            "points": [self._points[value].to_dict() for value in sorted(self._points)],
            "pending": sorted(self._pending),
        }
        temporary = self.manifest_path.with_name(f".{self.manifest_path.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(self.manifest_path)

    def pending_values(self) -> list[int]:
        return sorted(self._pending)

    def verified_points(self) -> list[SweepPoint]:
        return [point for _, point in sorted(self._points.items()) if point.verified]

    def point(self, value: int) -> SweepPoint | None:
        return self._points.get(int(value))

    def add_checkpoints(self, values: Sequence[int]) -> None:
        for value in values:
            value = int(value)
            existing = self._points.get(value)
            if existing is not None and existing.verified:
                continue
            if value not in self._pending:
                self._pending.append(value)
        self._pending.sort()
        self._save()

    def summary(self) -> dict:
        grouped: dict[str, list[int]] = {}
        for point in self.verified_points():
            grouped.setdefault(point.status or "unknown", []).append(point.value)
        return {
            "verified": grouped,
            "unresolved": [
                value for value, point in sorted(self._points.items()) if not point.verified
            ],
            "pending": self.pending_values(),
            "note": "only the listed parameter points are verified; no interval between points is claimed",
        }

    def run(self, max_points: int | None = None, budget: SearchBudget | None = None) -> None:
        budget = budget or self.budget
        completed = 0
        while self._pending:
            if max_points is not None and completed >= max_points:
                break
            value = self._pending.pop(0)
            point = self._solve_point(value, budget)
            self._points[value] = point
            if not point.verified:
                if point.stop_reason in BUDGET_STOP_REASONS:
                    self._pending.insert(0, value)
                    self._save()
                    return
            completed += 1
            self._save()

    def _solve_point(self, value: int, budget: SearchBudget) -> SweepPoint:
        problem = problem_at(self.base_problem, self.axis, value)
        run_dir = (self.directory / "points" / str(value)).resolve()
        store = SearchStore(run_dir)
        store.save_problem(problem)

        solver = BranchAndBound(problem, checkpoint_callback=store.save_state)
        state = solver.initial_state()
        warm = self._warm_candidate(value)
        if warm is not None and problem.is_feasible(warm):
            state.best_assignment = dict(warm)
            state.best_objective = _transformed(problem, problem.objective_value(warm))

        result = solver.solve(budget, state)
        store.save_state(state)
        store.save_result(result)
        try:
            store.save_proof(problem, state, result)
        except RuntimeError:
            pass

        verified = result.status in TERMINAL_STATUSES and store.review_saved().valid
        return SweepPoint(
            value,
            result.status.value,
            verified,
            result.stop_reason.value,
            result.nodes_evaluated,
            problem.canonical_hash(),
            str(run_dir),
        )

    def _warm_candidate(self, value: int) -> dict[str, int] | None:
        best_value = None
        best_assignment = None
        for point_value, point in self._points.items():
            if not point.verified or point.status != ResultStatus.OPTIMAL.value:
                continue
            if best_value is None or abs(point_value - value) < abs(best_value - value):
                result = _load_result(point.run_dir)
                if result is not None and result.best_assignment is not None:
                    best_value = point_value
                    best_assignment = result.best_assignment
        return best_assignment

    def derive(
        self,
        source_value: int,
        axis: ParameterAxis,
        values: Sequence[int],
        directory: str | Path,
        budget: SearchBudget | None = None,
    ) -> "ParameterSweep":
        point = self._points.get(int(source_value))
        if point is None or not point.verified:
            raise ValueError(f"parameter point {source_value} is not verified; cannot derive from it")
        base = problem_at(self.base_problem, self.axis, int(source_value))
        return ParameterSweep(directory, base, axis, values, budget or self.budget)


def _transformed(problem: Problem, value: Fraction) -> Fraction:
    return value if problem.sense == ObjectiveSense.MAXIMIZE else -value


def _load_result(run_dir: str) -> SolveResult | None:
    store = SearchStore(run_dir)
    if not store.result_path.exists():
        return None
    return store.load_result()
