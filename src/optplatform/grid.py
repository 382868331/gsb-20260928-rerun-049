from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .model import Problem
from .search import BranchAndBound, SearchBudget
from .store import SearchStore
from .branch import ReuseDecision, classify_problem_change, review_reuse_decisions
from .sweep import (
    BUDGET_STOP_REASONS,
    TERMINAL_STATUSES,
    ParameterAxis,
    ParameterSweep,
    _transformed,
    problem_at,
)
from .types import ResultStatus


Coordinate = tuple[int, int]


@dataclass(frozen=True)
class GridPoint:
    x: int
    y: int
    status: str | None
    verified: bool
    stop_reason: str | None
    nodes_evaluated: int
    problem_hash: str
    run_dir: str

    @property
    def coordinate(self) -> Coordinate:
        return (self.x, self.y)

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
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
            int(data["x"]),
            int(data["y"]),
            data.get("status"),
            bool(data.get("verified", False)),
            data.get("stop_reason"),
            int(data.get("nodes_evaluated", 0)),
            data.get("problem_hash", ""),
            data.get("run_dir", ""),
        )


@dataclass(frozen=True)
class RegionConclusion:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    status: str
    depends_on: tuple[Coordinate, ...]

    def contains(self, x: int, y: int) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def to_dict(self) -> dict:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "status": self.status,
            "depends_on": [list(coordinate) for coordinate in self.depends_on],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "RegionConclusion":
        return cls(
            int(data["x_min"]),
            int(data["x_max"]),
            int(data["y_min"]),
            int(data["y_max"]),
            data["status"],
            tuple((int(pair[0]), int(pair[1])) for pair in data.get("depends_on", [])),
        )


@dataclass(frozen=True)
class RegionReport:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    verified: bool
    status: str | None
    missing: tuple[Coordinate, ...]
    mismatched: tuple[tuple[Coordinate, str], ...]
    statuses: Mapping[Coordinate, str]
    conclusion: RegionConclusion | None


def region_overlap(first: RegionConclusion, second: RegionConclusion) -> tuple[int, int, int, int] | None:
    x_min = max(first.x_min, second.x_min)
    x_max = min(first.x_max, second.x_max)
    y_min = max(first.y_min, second.y_min)
    y_max = min(first.y_max, second.y_max)
    if x_min > x_max or y_min > y_max:
        return None
    return (x_min, x_max, y_min, y_max)


def region_conflicts(regions: Sequence[RegionConclusion]) -> list[dict]:
    conflicts = []
    for index, first in enumerate(regions):
        for second in regions[index + 1 :]:
            if first.status == second.status:
                continue
            overlap = region_overlap(first, second)
            if overlap is None:
                continue
            x_min, x_max, y_min, y_max = overlap
            conflicts.append(
                {
                    "overlap": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
                    "first": {"status": first.status, "evidence": sorted(first.depends_on)},
                    "second": {"status": second.status, "evidence": sorted(second.depends_on)},
                }
            )
    return conflicts


class ParameterStudy2D:
    def __init__(
        self,
        directory: str | Path,
        base_problem: Problem,
        axis_x: ParameterAxis,
        values_x: Sequence[int],
        axis_y: ParameterAxis,
        values_y: Sequence[int],
        budget: SearchBudget | None = None,
    ):
        if axis_x == axis_y:
            raise ValueError("the two parameter axes must be distinct")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "study.json"
        self.base_problem = base_problem
        self.axis_x = axis_x
        self.axis_y = axis_y
        self.budget = budget or SearchBudget()
        self._points: dict[Coordinate, GridPoint] = {}
        self._pending: list[Coordinate] = []
        self._regions: list[RegionConclusion] = []
        self._reuse: list[ReuseDecision] = []
        self._derived_from: str | None = None
        if self.manifest_path.exists():
            self._load()
        else:
            self._pending = sorted({(int(x), int(y)) for x in values_x for y in values_y})
            self._save()

    def _load(self) -> None:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        saved_problem = Problem.from_dict(data["base_problem"])
        if saved_problem.canonical_hash() != self.base_problem.canonical_hash():
            raise ValueError("study manifest belongs to a different base problem")
        if ParameterAxis.from_dict(data["axis_x"]) != self.axis_x:
            raise ValueError("study manifest belongs to a different x parameter axis")
        if ParameterAxis.from_dict(data["axis_y"]) != self.axis_y:
            raise ValueError("study manifest belongs to a different y parameter axis")
        self._points = {
            (int(item["x"]), int(item["y"])): GridPoint.from_dict(item)
            for item in data.get("points", [])
        }
        self._pending = sorted(
            (int(pair[0]), int(pair[1])) for pair in data.get("pending", [])
        )
        self._regions = [RegionConclusion.from_dict(item) for item in data.get("regions", [])]
        self._reuse = [ReuseDecision.from_dict(item) for item in data.get("reuse", [])]
        self._derived_from = data.get("derived_from")

    def _save(self) -> None:
        payload = {
            "base_problem": self.base_problem.to_dict(),
            "axis_x": self.axis_x.to_dict(),
            "axis_y": self.axis_y.to_dict(),
            "points": [self._points[key].to_dict() for key in sorted(self._points)],
            "pending": [list(pair) for pair in sorted(self._pending)],
            "regions": [region.to_dict() for region in self._regions],
            "reuse": [decision.to_dict() for decision in self._reuse],
            "derived_from": self._derived_from,
        }
        temporary = self.manifest_path.with_name(f".{self.manifest_path.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(self.manifest_path)

    def pending_coordinates(self) -> list[Coordinate]:
        return sorted(self._pending)

    def verified_points(self) -> list[GridPoint]:
        return [point for _, point in sorted(self._points.items()) if point.verified]

    def point(self, x: int, y: int) -> GridPoint | None:
        return self._points.get((int(x), int(y)))

    def regions(self) -> list[RegionConclusion]:
        return list(self._regions)

    def reuse_decisions(self) -> list[ReuseDecision]:
        return list(self._reuse)

    def add_checkpoints(self, coordinates: Sequence[Sequence[int]]) -> None:
        for pair in coordinates:
            coordinate = (int(pair[0]), int(pair[1]))
            existing = self._points.get(coordinate)
            if existing is not None and existing.verified:
                continue
            if coordinate not in self._pending:
                self._pending.append(coordinate)
        self._pending.sort()
        self._save()

    def summary(self) -> dict:
        grouped: dict[str, list[Coordinate]] = {}
        for point in self.verified_points():
            grouped.setdefault(point.status or "unknown", []).append(point.coordinate)
        return {
            "verified": grouped,
            "unresolved": [
                coordinate
                for coordinate, point in sorted(self._points.items())
                if not point.verified
            ],
            "pending": self.pending_coordinates(),
            "regions": [region.to_dict() for region in self._regions],
            "note": (
                "only the listed discrete points are verified; no area between or around "
                "points is claimed unless a region conclusion explicitly covers it"
            ),
        }

    def run(self, max_points: int | None = None, budget: SearchBudget | None = None) -> None:
        budget = budget or self.budget
        completed = 0
        while self._pending:
            if max_points is not None and completed >= max_points:
                break
            coordinate = self._pending.pop(0)
            point = self._solve_point(coordinate, budget)
            self._points[coordinate] = point
            if not point.verified and point.stop_reason in BUDGET_STOP_REASONS:
                self._pending.insert(0, coordinate)
                self._save()
                return
            completed += 1
            self._save()

    def request_region_conclusion(
        self,
        x_range: Sequence[int],
        y_range: Sequence[int],
        status: str | None = None,
    ) -> RegionReport:
        x_min, x_max = int(x_range[0]), int(x_range[1])
        y_min, y_max = int(y_range[0]), int(y_range[1])
        if x_min > x_max or y_min > y_max:
            raise ValueError("region ranges must be ordered (min, max)")
        if status is not None and status not in {item.value for item in ResultStatus}:
            raise ValueError(f"unknown result status {status!r}")

        missing: list[Coordinate] = []
        mismatched: list[tuple[Coordinate, str]] = []
        statuses: dict[Coordinate, str] = {}
        depends_on: list[Coordinate] = []
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                point = self._points.get((x, y))
                if point is None or not point.verified or point.status is None:
                    missing.append((x, y))
                    continue
                statuses[(x, y)] = point.status
                depends_on.append((x, y))
                if status is not None and point.status != status:
                    mismatched.append(((x, y), point.status))

        distinct = set(statuses.values())
        unanimous = len(distinct) == 1 and (status is None or distinct == {status})
        verified = not missing and not mismatched and unanimous
        conclusion = None
        if verified:
            conclusion = RegionConclusion(
                x_min, x_max, y_min, y_max, next(iter(distinct)), tuple(sorted(depends_on))
            )
            if conclusion not in self._regions:
                self._regions.append(conclusion)
                self._save()
        return RegionReport(
            x_min,
            x_max,
            y_min,
            y_max,
            verified,
            next(iter(distinct)) if verified else None,
            tuple(missing),
            tuple(mismatched),
            statuses,
            conclusion,
        )

    def conflicts(self) -> list[dict]:
        return region_conflicts(self._regions)

    def review_saved(self) -> dict:
        errors: list[str] = []
        for coordinate, point in sorted(self._points.items()):
            if not point.verified:
                continue
            report = SearchStore(point.run_dir).review_saved()
            if not report.valid:
                errors.extend(f"point {coordinate}: {error}" for error in report.errors)
        for region in self._regions:
            for coordinate in region.depends_on:
                point = self._points.get(coordinate)
                if point is None or not point.verified:
                    errors.append(
                        f"region {region.to_dict()} depends on unverified point {coordinate}"
                    )
                elif point.status != region.status:
                    errors.append(
                        f"region {region.to_dict()} depends on point {coordinate} "
                        f"whose status {point.status!r} no longer matches"
                    )
        errors.extend(review_reuse_decisions(self))
        return {"valid": not errors, "errors": errors}

    def derive_branch(
        self,
        new_base_problem: Problem,
        directory: str | Path,
        values_x: Sequence[int] | None = None,
        values_y: Sequence[int] | None = None,
        subregion: Sequence[Sequence[int]] | None = None,
        budget: SearchBudget | None = None,
    ) -> "ParameterStudy2D":
        if values_x is None:
            values_x = sorted({x for x, _ in list(self._points) + self._pending})
        if values_y is None:
            values_y = sorted({y for _, y in list(self._points) + self._pending})
        if not values_x or not values_y:
            raise ValueError("cannot derive a branch without any parameter values")
        if subregion is None:
            target = {(int(x), int(y)) for x in values_x for y in values_y}
        else:
            target = {(int(pair[0]), int(pair[1])) for pair in subregion}
            if not target:
                raise ValueError("subregion must contain at least one coordinate")

        branch = ParameterStudy2D(
            directory,
            new_base_problem,
            self.axis_x,
            values_x,
            self.axis_y,
            values_y,
            budget or self.budget,
        )
        branch._derived_from = str(self.directory.resolve())
        branch._pending = sorted(c for c in branch._pending if c in target)

        decisions = list(branch._reuse)
        decided = {(d.x, d.y) for d in decisions}
        for coordinate in sorted(target):
            if coordinate in decided or coordinate in branch._points:
                continue
            old_point = self._points.get(coordinate)
            if old_point is None or not old_point.verified or old_point.status is None:
                continue
            old_effective = problem_at(
                problem_at(self.base_problem, self.axis_x, coordinate[0]),
                self.axis_y,
                coordinate[1],
            )
            new_effective = problem_at(
                problem_at(new_base_problem, self.axis_x, coordinate[0]),
                self.axis_y,
                coordinate[1],
            )
            change = classify_problem_change(old_effective, new_effective)
            applied = old_point.status in change.reusable_statuses
            decisions.append(
                ReuseDecision(
                    coordinate[0],
                    coordinate[1],
                    old_point.status,
                    change.kind,
                    applied,
                    str(self.directory.resolve()),
                    self.base_problem.canonical_hash(),
                    new_base_problem.canonical_hash(),
                    old_point.run_dir,
                )
            )
            if applied:
                branch._points[coordinate] = GridPoint(
                    coordinate[0],
                    coordinate[1],
                    old_point.status,
                    True,
                    None,
                    0,
                    new_effective.canonical_hash(),
                    old_point.run_dir,
                )
                if coordinate in branch._pending:
                    branch._pending.remove(coordinate)
        branch._reuse = decisions
        branch._save()
        return branch

    def reuse_summary(self) -> dict:
        reused: dict[str, list[Coordinate]] = {}
        rejected: dict[str, list[Coordinate]] = {}
        decided = set()
        for decision in self._reuse:
            decided.add(decision.coordinate)
            bucket = reused if decision.applied else rejected
            bucket.setdefault(decision.status, []).append(decision.coordinate)
        new_coordinates = [
            coordinate
            for coordinate in self.pending_coordinates()
            if coordinate not in decided
        ]
        return {
            "model_version": self.base_problem.canonical_hash(),
            "derived_from": self._derived_from,
            "reused": {status: sorted(coords) for status, coords in sorted(reused.items())},
            "requires_resolve": {
                status: sorted(coords) for status, coords in sorted(rejected.items())
            },
            "new_coordinates": new_coordinates,
            "pending": self.pending_coordinates(),
            "note": (
                "reused conclusions apply only to the recorded model version and "
                "coordinates; points under requires_resolve and new_coordinates must "
                "be solved before any region conclusion covers them"
            ),
        }

    def request_coverage(
        self,
        coordinates: Sequence[Sequence[int]],
        status: str | None = None,
    ) -> dict:
        if status is not None and status not in {item.value for item in ResultStatus}:
            raise ValueError(f"unknown result status {status!r}")
        cells = sorted({(int(pair[0]), int(pair[1])) for pair in coordinates})
        if not cells:
            raise ValueError("coverage request must contain at least one coordinate")
        missing: list[Coordinate] = []
        mismatched: list[tuple[Coordinate, str]] = []
        statuses: dict[Coordinate, str] = {}
        for cell in cells:
            point = self._points.get(cell)
            if point is None or not point.verified or point.status is None:
                missing.append(cell)
                continue
            statuses[cell] = point.status
            if status is not None and point.status != status:
                mismatched.append((cell, point.status))
        distinct = set(statuses.values())
        verified = (
            not missing
            and not mismatched
            and len(distinct) == 1
            and (status is None or distinct == {status})
        )
        return {
            "coordinates": cells,
            "verified": verified,
            "status": next(iter(distinct)) if verified else None,
            "missing": missing,
            "mismatched": mismatched,
            "statuses": statuses,
        }

    def derive(
        self,
        x: int,
        y: int,
        axis: ParameterAxis,
        values: Sequence[int],
        directory: str | Path,
        budget: SearchBudget | None = None,
    ) -> ParameterSweep:
        point = self._points.get((int(x), int(y)))
        if point is None or not point.verified:
            raise ValueError(
                f"parameter point ({x}, {y}) is not verified; cannot derive from it"
            )
        base = problem_at(
            problem_at(self.base_problem, self.axis_x, int(x)), self.axis_y, int(y)
        )
        return ParameterSweep(directory, base, axis, values, budget or self.budget)

    def _solve_point(self, coordinate: Coordinate, budget: SearchBudget) -> GridPoint:
        x, y = coordinate
        problem = problem_at(problem_at(self.base_problem, self.axis_x, x), self.axis_y, y)
        run_dir = (self.directory / "points" / f"{x}__{y}").resolve()
        store = SearchStore(run_dir)
        store.save_problem(problem)

        solver = BranchAndBound(problem, checkpoint_callback=store.save_state)
        state = solver.initial_state()
        warm = self._warm_candidate(coordinate)
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
        return GridPoint(
            x,
            y,
            result.status.value,
            verified,
            result.stop_reason.value,
            result.nodes_evaluated,
            problem.canonical_hash(),
            str(run_dir),
        )

    def _warm_candidate(self, coordinate: Coordinate) -> dict[str, int] | None:
        best_distance = None
        best_assignment = None
        for point_coordinate, point in self._points.items():
            if not point.verified or point.status != ResultStatus.OPTIMAL.value:
                continue
            distance = abs(point_coordinate[0] - coordinate[0]) + abs(
                point_coordinate[1] - coordinate[1]
            )
            if best_distance is None or distance < best_distance:
                store = SearchStore(point.run_dir)
                if not store.result_path.exists():
                    continue
                result = store.load_result()
                if result.best_assignment is not None:
                    best_distance = distance
                    best_assignment = result.best_assignment
        return best_assignment
