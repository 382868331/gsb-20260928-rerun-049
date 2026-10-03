from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .model import Problem
from .search import BranchAndBound, SearchBudget
from .store import SearchStore
from .sweep import BUDGET_STOP_REASONS, TERMINAL_STATUSES, ParameterAxis, problem_at


@dataclass(frozen=True)
class GridPoint:
    u: int
    v: int
    status: str | None
    verified: bool
    stop_reason: str | None
    nodes_evaluated: int
    problem_hash: str
    run_dir: str

    def to_dict(self) -> dict:
        return {
            "u": self.u,
            "v": self.v,
            "status": self.status,
            "verified": self.verified,
            "stop_reason": self.stop_reason,
            "nodes_evaluated": self.nodes_evaluated,
            "problem_hash": self.problem_hash,
            "run_dir": self.run_dir,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "GridPoint":
        return cls(
            int(data["u"]),
            int(data["v"]),
            data.get("status"),
            bool(data.get("verified", False)),
            data.get("stop_reason"),
            int(data.get("nodes_evaluated", 0)),
            data.get("problem_hash", ""),
            data.get("run_dir", ""),
        )


@dataclass(frozen=True)
class RegionClaim:
    u_min: int
    u_max: int
    v_min: int
    v_max: int
    status: str
    verified: bool
    coverage: tuple = ()
    missing: tuple = ()

    @property
    def degenerate(self) -> bool:
        return self.u_min == self.u_max or self.v_min == self.v_max

    def cells(self) -> list:
        return [
            (u, v)
            for u in range(self.u_min, self.u_max + 1)
            for v in range(self.v_min, self.v_max + 1)
        ]

    def contains(self, u: int, v: int) -> bool:
        return self.u_min <= u <= self.u_max and self.v_min <= v <= self.v_max

    def to_dict(self) -> dict:
        return {
            "u_min": self.u_min,
            "u_max": self.u_max,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "status": self.status,
            "verified": self.verified,
            "coverage": [list(cell) for cell in self.coverage],
            "degenerate": self.degenerate,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "RegionClaim":
        return cls(
            int(data["u_min"]),
            int(data["u_max"]),
            int(data["v_min"]),
            int(data["v_max"]),
            data["status"],
            bool(data.get("verified", False)),
            tuple((int(cell[0]), int(cell[1])) for cell in data.get("coverage", [])),
        )


@dataclass(frozen=True)
class RegionConflict:
    u_min: int
    u_max: int
    v_min: int
    v_max: int
    claims: tuple

    def to_dict(self) -> dict:
        return {
            "overlap": {
                "u_min": self.u_min,
                "u_max": self.u_max,
                "v_min": self.v_min,
                "v_max": self.v_max,
            },
            "claims": [claim.to_dict() for claim in self.claims],
        }


def find_conflicts(claims: Sequence) -> list:
    conflicts = []
    verified = [claim for claim in claims if claim.verified]
    for index, first in enumerate(verified):
        for second in verified[index + 1:]:
            if first.status == second.status:
                continue
            u_min = max(first.u_min, second.u_min)
            u_max = min(first.u_max, second.u_max)
            v_min = max(first.v_min, second.v_min)
            v_max = min(first.v_max, second.v_max)
            if u_min <= u_max and v_min <= v_max:
                conflicts.append(RegionConflict(u_min, u_max, v_min, v_max, (first, second)))
    return conflicts


@dataclass(frozen=True)
class GridReviewReport:
    valid: bool
    errors: tuple


class ParameterGrid:
    def __init__(
        self,
        directory,
        base_problem: Problem,
        axis_u: ParameterAxis,
        axis_v: ParameterAxis,
        values_u: Sequence,
        values_v: Sequence,
        budget: SearchBudget | None = None,
    ):
        if axis_u == axis_v:
            raise ValueError("the two parameter axes must be distinct")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "grid.json"
        self.base_problem = base_problem
        self.axis_u = axis_u
        self.axis_v = axis_v
        self.budget = budget or SearchBudget()
        self._points = {}
        self._pending = []
        self._claims = []
        if self.manifest_path.exists():
            self._load()
        else:
            self._pending = sorted({(int(u), int(v)) for u in values_u for v in values_v})
            self._save()

    def problem_at_point(self, u: int, v: int) -> Problem:
        return problem_at(problem_at(self.base_problem, self.axis_u, u), self.axis_v, v)

    def _load(self) -> None:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        saved_problem = Problem.from_dict(data["base_problem"])
        if saved_problem.canonical_hash() != self.base_problem.canonical_hash():
            raise ValueError("grid manifest belongs to a different base problem")
        saved_u = ParameterAxis.from_dict(data["axis_u"])
        saved_v = ParameterAxis.from_dict(data["axis_v"])
        if saved_u != self.axis_u or saved_v != self.axis_v:
            raise ValueError("grid manifest belongs to different parameter axes")
        self._points = {
            (int(item["u"]), int(item["v"])): GridPoint.from_dict(item)
            for item in data.get("points", [])
        }
        self._pending = sorted(
            (int(cell[0]), int(cell[1])) for cell in data.get("pending", [])
        )
        self._claims = [RegionClaim.from_dict(item) for item in data.get("claims", [])]

    def _save(self) -> None:
        payload = {
            "base_problem": self.base_problem.to_dict(),
            "axis_u": self.axis_u.to_dict(),
            "axis_v": self.axis_v.to_dict(),
            "points": [self._points[key].to_dict() for key in sorted(self._points)],
            "pending": [list(cell) for cell in self._pending],
            "claims": [claim.to_dict() for claim in self._claims],
        }
        temporary = self.manifest_path.with_name(
            "." + self.manifest_path.name + ".tmp-" + str(os.getpid())
        )
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(self.manifest_path)

    def pending_cells(self) -> list:
        return list(self._pending)

    def verified_points(self) -> list:
        return [point for _, point in sorted(self._points.items()) if point.verified]

    def point(self, u: int, v: int):
        return self._points.get((int(u), int(v)))

    def claims(self) -> list:
        return list(self._claims)

    def add_checkpoints(self, cells: Sequence) -> None:
        for u, v in cells:
            cell = (int(u), int(v))
            existing = self._points.get(cell)
            if existing is not None and existing.verified:
                continue
            if cell not in self._pending:
                self._pending.append(cell)
        self._pending.sort()
        self._save()

    def run(self, max_points=None, budget: SearchBudget | None = None) -> None:
        budget = budget or self.budget
        completed = 0
        while self._pending:
            if max_points is not None and completed >= max_points:
                break
            cell = self._pending.pop(0)
            point = self._solve_point(cell, budget)
            self._points[cell] = point
            if not point.verified:
                if point.stop_reason in BUDGET_STOP_REASONS:
                    self._pending.insert(0, cell)
                    self._save()
                    return
            completed += 1
            self._save()

    def add_budget(self, budget: SearchBudget) -> None:
        for cell, point in sorted(self._points.items()):
            if point.verified or cell in self._pending:
                continue
            self._pending.append(cell)
        self._pending.sort()
        self._save()
        self.run(budget=budget)

    def _solve_point(self, cell, budget: SearchBudget) -> GridPoint:
        u, v = cell
        problem = self.problem_at_point(u, v)
        run_dir = (self.directory / "points" / (str(u) + "_" + str(v))).resolve()
        store = SearchStore(run_dir)
        store.save_problem(problem)

        solver = BranchAndBound(problem, checkpoint_callback=store.save_state)
        state = solver.initial_state()
        result = solver.solve(budget, state)
        store.save_state(state)
        store.save_result(result)
        try:
            store.save_proof(problem, state, result)
        except RuntimeError:
            pass

        verified = result.status in TERMINAL_STATUSES and store.review_saved().valid
        return GridPoint(
            u,
            v,
            result.status.value,
            verified,
            result.stop_reason.value,
            result.nodes_evaluated,
            problem.canonical_hash(),
            str(run_dir),
        )

    def request_region(self, u_min, u_max, v_min, v_max, status: str) -> RegionClaim:
        u_min, u_max, v_min, v_max = int(u_min), int(u_max), int(v_min), int(v_max)
        if u_min > u_max or v_min > v_max:
            raise ValueError("region bounds are inverted")
        claim = RegionClaim(u_min, u_max, v_min, v_max, status, False)
        missing = []
        for cell in claim.cells():
            point = self._points.get(cell)
            if point is None or not point.verified or point.status != status:
                missing.append(cell)
        if missing:
            return RegionClaim(
                u_min, u_max, v_min, v_max, status, False, (), tuple(missing)
            )
        verified_claim = RegionClaim(
            u_min, u_max, v_min, v_max, status, True, tuple(claim.cells())
        )
        self._claims.append(verified_claim)
        self._save()
        return verified_claim

    def conflicts(self) -> list:
        return find_conflicts(self._claims)

    def summary(self) -> dict:
        grouped = {}
        for point in self.verified_points():
            grouped.setdefault(point.status or "unknown", []).append([point.u, point.v])
        return {
            "verified": grouped,
            "unresolved": [
                [u, v]
                for (u, v), point in sorted(self._points.items())
                if not point.verified
            ],
            "pending": [list(cell) for cell in self._pending],
            "regions": [claim.to_dict() for claim in self._claims if claim.verified],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts()],
            "note": (
                "only the listed (u, v) positions are individually verified; "
                "verified regions cover exactly their listed cells; "
                "no conclusion is implied for any other position in the plane"
            ),
        }

    def derive(
        self,
        source_u: int,
        source_v: int,
        axis_u: ParameterAxis,
        axis_v: ParameterAxis,
        values_u: Sequence,
        values_v: Sequence,
        directory,
        budget: SearchBudget | None = None,
    ) -> "ParameterGrid":
        point = self._points.get((int(source_u), int(source_v)))
        if point is None or not point.verified:
            raise ValueError(
                "grid position (%s, %s) is not verified; cannot derive from it"
                % (source_u, source_v)
            )
        base = self.problem_at_point(int(source_u), int(source_v))
        return ParameterGrid(
            directory, base, axis_u, axis_v, values_u, values_v, budget or self.budget
        )

    def review_saved(self) -> GridReviewReport:
        errors = []
        terminal = {status.value for status in TERMINAL_STATUSES}
        for (u, v), point in sorted(self._points.items()):
            expected = self.problem_at_point(u, v).canonical_hash()
            if point.problem_hash != expected:
                errors.append(
                    "point (%d, %d): recorded problem hash does not match the grid axes"
                    % (u, v)
                )
                continue
            store = SearchStore(point.run_dir)
            try:
                report = store.review_saved()
            except (OSError, ValueError, KeyError) as exc:
                errors.append("point (%d, %d): saved run unreadable: %s" % (u, v, exc))
                continue
            if not report.valid:
                errors.extend(
                    "point (%d, %d): %s" % (u, v, error) for error in report.errors
                )
            if point.verified and point.status not in terminal:
                errors.append(
                    "point (%d, %d): verified flag on non-terminal status %r"
                    % (u, v, point.status)
                )
        for claim in self._claims:
            if not claim.verified:
                continue
            for cell in claim.cells():
                point = self._points.get(cell)
                if point is None or not point.verified or point.status != claim.status:
                    errors.append(
                        "region %r over u=[%d,%d] v=[%d,%d] depends on cell %s "
                        "which is no longer verified with that status"
                        % (
                            claim.status,
                            claim.u_min,
                            claim.u_max,
                            claim.v_min,
                            claim.v_max,
                            cell,
                        )
                    )
        return GridReviewReport(not errors, tuple(errors))
