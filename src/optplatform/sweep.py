from __future__ import annotations

import json
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .model import Constraint, Problem
from .proof import VerificationReport
from .search import BranchAndBound, SearchBudget
from .store import SearchStore
from .types import ResultStatus


class AxisKind:
    OBJECTIVE_COEFFICIENT = "objective_coefficient"
    CONSTRAINT_RHS = "constraint_rhs"


VERIFIED_STATUSES = frozenset(
    {ResultStatus.OPTIMAL, ResultStatus.INFEASIBLE, ResultStatus.UNBOUNDED}
)


@dataclass(frozen=True)
class ParameterAxis:
    kind: str
    target: str
    values: tuple[Fraction, ...]

    def __init__(self, kind: str, target: str, values: Sequence[int | Fraction]):
        if kind not in (AxisKind.OBJECTIVE_COEFFICIENT, AxisKind.CONSTRAINT_RHS):
            raise ValueError("axis kind must be objective_coefficient or constraint_rhs")
        if not target:
            raise ValueError("axis target must be non-empty")
        cleaned = tuple(Fraction(value) for value in values)
        if not cleaned:
            raise ValueError("axis must contain at least one parameter point")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("axis values must be distinct")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "values", tuple(sorted(cleaned)))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "target": self.target, "values": [str(v) for v in self.values]}

    @classmethod
    def from_dict(cls, data: Mapping) -> "ParameterAxis":
        return cls(data["kind"], data["target"], [Fraction(v) for v in data["values"]])


def vary_problem(problem: Problem, axis_kind: str, target: str, value: Fraction) -> Problem:
    if axis_kind == AxisKind.OBJECTIVE_COEFFICIENT:
        if target not in problem.variable_names:
            raise ValueError(f"unknown objective variable {target!r}")
        objective = dict(problem.objective)
        if value == 0:
            objective.pop(target, None)
        else:
            objective[target] = value
        constraints = list(problem.constraints)
    elif axis_kind == AxisKind.CONSTRAINT_RHS:
        names = {constraint.name for constraint in problem.constraints}
        if target not in names:
            raise ValueError(f"unknown constraint {target!r}")
        objective = dict(problem.objective)
        constraints = [
            Constraint(c.name, dict(c.coefficients), value if c.name == target else c.rhs)
            for c in problem.constraints
        ]
    else:
        raise ValueError("axis kind must be objective_coefficient or constraint_rhs")
    return Problem(
        f"{problem.name}[{target}={value}]",
        list(problem.variables),
        objective,
        constraints,
        problem.sense,
    )


@dataclass
class SweepPoint:
    index: int
    value: Fraction
    status: str = "pending"
    problem_hash: str | None = None
    store_dir: str | None = None
    budget: dict | None = None
    warm_start_from: int | None = None
    message: str = ""

    @property
    def verified(self) -> bool:
        return self.status in {s.value for s in VERIFIED_STATUSES}

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "value": str(self.value),
            "status": self.status,
            "problem_hash": self.problem_hash,
            "store_dir": self.store_dir,
            "budget": self.budget,
            "warm_start_from": self.warm_start_from,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "SweepPoint":
        return cls(
            int(data["index"]),
            Fraction(data["value"]),
            data.get("status", "pending"),
            data.get("problem_hash"),
            data.get("store_dir"),
            data.get("budget"),
            data.get("warm_start_from"),
            data.get("message", ""),
        )


@dataclass(frozen=True)
class SweepSummary:
    verified: tuple[SweepPoint, ...]
    pending: tuple[SweepPoint, ...]

    def by_status(self, status: ResultStatus) -> tuple[SweepPoint, ...]:
        return tuple(point for point in self.verified if point.status == status.value)


class ParameterSweep:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.directory / "sweep.json"
        self.base_problem: Problem | None = None
        self.axis: ParameterAxis | None = None
        self.parent: dict | None = None
        self.points: list[SweepPoint] = []

    @classmethod
    def create(
        cls,
        problem: Problem,
        axis: ParameterAxis,
        directory: str | Path,
        parent: dict | None = None,
    ) -> "ParameterSweep":
        sweep = cls(directory)
        if sweep.meta_path.exists():
            raise FileExistsError(f"sweep already exists in {sweep.directory}")
        sweep.base_problem = problem
        sweep.axis = axis
        sweep.parent = parent
        sweep.points = [SweepPoint(i, value) for i, value in enumerate(axis.values)]
        sweep._save_meta()
        return sweep

    @classmethod
    def load(cls, directory: str | Path) -> "ParameterSweep":
        sweep = cls(directory)
        data = _read_json(sweep.meta_path)
        sweep.base_problem = Problem.from_dict(data["base_problem"])
        sweep.axis = ParameterAxis.from_dict(data["axis"])
        sweep.parent = data.get("parent")
        sweep.points = [SweepPoint.from_dict(item) for item in data["points"]]
        return sweep

    def derive(
        self,
        point_index: int,
        axis: ParameterAxis,
        directory: str | Path,
    ) -> "ParameterSweep":
        point = self._require_point(point_index)
        if not point.verified:
            raise ValueError("only a verified point can seed a derived sweep")
        base = vary_problem(self.base_problem, self.axis.kind, self.axis.target, point.value)
        parent = {
            "sweep_dir": str(self.directory),
            "point_index": point.index,
            "point_value": str(point.value),
        }
        return ParameterSweep.create(base, axis, directory, parent=parent)

    def refine(self, extra_values: Sequence[int | Fraction]) -> tuple[int, ...]:
        existing = {point.value for point in self.points}
        added: list[int] = []
        for value in sorted({Fraction(v) for v in extra_values} - existing):
            index = len(self.points)
            self.points.append(SweepPoint(index, value))
            added.append(index)
        self.points.sort(key=lambda point: (point.value, point.index))
        if added:
            self._save_meta()
        return tuple(added)

    def run(
        self,
        budget: SearchBudget | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> SweepSummary:
        budget = budget or SearchBudget()
        for point in self.points:
            if point.verified:
                continue
            if should_stop is not None and should_stop():
                break
            self._solve_point(point, budget)
            self._save_meta()
        return self.summary()

    def summary(self) -> SweepSummary:
        verified = tuple(point for point in self.points if point.verified)
        pending = tuple(point for point in self.points if not point.verified)
        return SweepSummary(verified, pending)

    def review_point(self, point_index: int) -> VerificationReport:
        point = self._require_point(point_index)
        if point.store_dir is None:
            return VerificationReport(False, ("point has no saved run to review",))
        return SearchStore(self.directory / point.store_dir).review_saved()

    def point_problem(self, point_index: int) -> Problem:
        point = self._require_point(point_index)
        return vary_problem(self.base_problem, self.axis.kind, self.axis.target, point.value)

    def _solve_point(self, point: SweepPoint, budget: SearchBudget) -> None:
        problem = self.point_problem(point.index)
        store = SearchStore(self.directory / f"points/{point.index:04d}")
        store.save_problem(problem)

        solver = BranchAndBound(problem)
        state = solver.initial_state()
        seed_index, seed_assignment = self._warm_start_candidate(point, problem)
        if seed_assignment is not None:
            state.best_assignment = seed_assignment
            state.best_objective = solver._transformed_value(seed_assignment)
            point.warm_start_from = seed_index

        result = solver.solve(budget, state)
        store.save_state(state)
        store.save_result(result)
        store.save_proof(problem, state, result)

        point.status = result.status.value
        point.problem_hash = problem.canonical_hash()
        point.store_dir = f"points/{point.index:04d}"
        point.budget = {
            "max_nodes": budget.max_nodes,
            "max_seconds": budget.max_seconds,
        }
        point.message = result.message

    def _warm_start_candidate(
        self, point: SweepPoint, problem: Problem
    ) -> tuple[int | None, dict[str, int] | None]:
        if not problem.has_finite_bounds():
            return None, None
        best_index: int | None = None
        best_assignment: dict[str, int] | None = None
        best_distance: Fraction | None = None
        for other in self.points:
            if other.index == point.index or other.store_dir is None:
                continue
            try:
                result = SearchStore(self.directory / other.store_dir).load_result()
            except (OSError, ValueError, KeyError):
                continue
            if result.best_assignment is None:
                continue
            if not problem.is_feasible(result.best_assignment):
                continue
            distance = abs(other.value - point.value)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = other.index
                best_assignment = dict(result.best_assignment)
        return best_index, best_assignment

    def _require_point(self, point_index: int) -> SweepPoint:
        for point in self.points:
            if point.index == point_index:
                return point
        raise KeyError(f"no sweep point with index {point_index}")

    def _save_meta(self) -> None:
        data = {
            "base_problem": self.base_problem.to_dict(),
            "axis": self.axis.to_dict(),
            "parent": self.parent,
            "points": [point.to_dict() for point in self.points],
        }
        temporary = self.meta_path.with_name(f".{self.meta_path.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(self.meta_path)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
