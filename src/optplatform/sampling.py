from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .grid import Coordinate, ParameterStudy2D
from .sweep import problem_at
from .store import SearchStore


@dataclass(frozen=True)
class SamplingCandidate:
    x: int
    y: int
    reason: str
    level: int
    state: str = "pending"
    order: int | None = None

    @property
    def coordinate(self) -> Coordinate:
        return (self.x, self.y)

    def observed(self, order: int) -> "SamplingCandidate":
        return SamplingCandidate(self.x, self.y, self.reason, self.level, "observed", order)

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "reason": self.reason,
            "level": self.level,
            "state": self.state,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "SamplingCandidate":
        return cls(
            int(data["x"]),
            int(data["y"]),
            data["reason"],
            int(data.get("level", 0)),
            data.get("state", "pending"),
            None if data.get("order") is None else int(data["order"]),
        )


class HierarchicalSamplingPlan2D:
    def __init__(
        self,
        study: ParameterStudy2D,
        domain: Sequence[Sequence[int]],
        total_budget: int,
        min_spacing: Sequence[int],
        required: Sequence[Sequence[int]] = (),
    ):
        self.study = study
        self.path = study.directory / "sampling.json"
        cells = sorted({(int(pair[0]), int(pair[1])) for pair in domain})
        required_cells = sorted({(int(pair[0]), int(pair[1])) for pair in required})
        if not cells:
            raise ValueError("sampling domain must contain at least one coordinate")
        outside = [coordinate for coordinate in required_cells if coordinate not in set(cells)]
        if outside:
            raise ValueError(f"required coordinates are outside the domain: {outside}")
        spacing_x, spacing_y = int(min_spacing[0]), int(min_spacing[1])
        if spacing_x <= 0 or spacing_y <= 0:
            raise ValueError("minimum coordinate spacing must be positive")
        if int(total_budget) <= 0:
            raise ValueError("total sampling budget must be positive")

        self.domain = cells
        self.required = required_cells
        self.total_budget = int(total_budget)
        self.spacing = (spacing_x, spacing_y)
        self.model_version = study.base_problem.canonical_hash()
        self.spent = 0
        self.candidates_record: list[SamplingCandidate] = []

        if self.path.exists():
            self._load()
        else:
            self._adopt_existing_observations()
            self._initial_candidates()
            self.refine()
            self._save()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        actual = (
            [(int(x), int(y)) for x, y in data.get("domain", [])],
            [(int(x), int(y)) for x, y in data.get("required", [])],
            int(data.get("total_budget", 0)),
            tuple(int(value) for value in data.get("min_spacing", [])),
            data.get("model_version"),
        )
        expected = (
            self.domain,
            self.required,
            self.total_budget,
            self.spacing,
            self.model_version,
        )
        if actual != expected:
            raise ValueError("saved sampling plan configuration does not match the current study")
        self.spent = int(data.get("spent", 0))
        self.candidates_record = [
            SamplingCandidate.from_dict(item) for item in data.get("candidates", [])
        ]

    def _save(self) -> None:
        payload = {
            "domain": [list(coordinate) for coordinate in self.domain],
            "required": [list(coordinate) for coordinate in self.required],
            "total_budget": self.total_budget,
            "min_spacing": list(self.spacing),
            "model_version": self.model_version,
            "spent": self.spent,
            "candidates": [candidate.to_dict() for candidate in self.candidates_record],
        }
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(self.path)

    def _adopt_existing_observations(self) -> None:
        for coordinate in self.domain:
            if self._current_observation(coordinate) is None:
                continue
            order = len(self.candidates_record)
            self.candidates_record.append(
                SamplingCandidate(
                    coordinate[0],
                    coordinate[1],
                    "reused an existing observation computed under the same model version",
                    0,
                    "observed",
                    order,
                )
            )
            self.spent += 1

    def _initial_candidates(self) -> None:
        min_x = min(x for x, _ in self.domain)
        min_y = min(y for _, y in self.domain)
        spacing_x, spacing_y = self.spacing[0] * 2, self.spacing[1] * 2
        observed = {candidate.coordinate for candidate in self.candidates_record}
        for coordinate in self.domain:
            if coordinate in observed:
                continue
            x, y = coordinate
            if coordinate in set(self.required):
                reason = "required observation"
            elif (x - min_x) % spacing_x == 0 and (y - min_y) % spacing_y == 0:
                reason = (
                    f"initial lattice point: x offset {x - min_x} is a multiple of {spacing_x}, "
                    f"and y offset {y - min_y} is a multiple of {spacing_y}"
                )
            else:
                continue
            self.candidates_record.append(SamplingCandidate(x, y, reason, 0))

    def candidates(self) -> list[SamplingCandidate]:
        return list(self.candidates_record)

    def pending_candidates(self) -> list[SamplingCandidate]:
        return sorted(
            (candidate for candidate in self.candidates_record if candidate.state == "pending"),
            key=lambda candidate: (-candidate.level, candidate.coordinate),
        )

    def observed_coordinates(self) -> list[Coordinate]:
        return [
            candidate.coordinate
            for candidate in sorted(
                (candidate for candidate in self.candidates_record if candidate.state == "observed"),
                key=lambda candidate: candidate.order if candidate.order is not None else -1,
            )
        ]

    def run(self, max_points: int | None = None) -> None:
        completed = 0
        while self.spent < self.total_budget:
            if max_points is not None and completed >= max_points:
                return
            pending = self.pending_candidates()
            if not pending:
                self.refine()
                pending = self.pending_candidates()
                if not pending:
                    return
            candidate = pending[0]
            self.study.add_checkpoints([candidate.coordinate])
            point = self.study.run_point(candidate.x, candidate.y)
            observation = self._current_observation(candidate.coordinate)
            if observation is None:
                self._save()
                return
            index = self.candidates_record.index(candidate)
            self.candidates_record[index] = candidate.observed(self.spent)
            self.spent += 1
            completed += 1
            self.refine()
            self._save()

    def refine(self) -> list[SamplingCandidate]:
        observed = [
            candidate
            for candidate in self.candidates_record
            if candidate.state == "observed" and self._current_observation(candidate.coordinate) is not None
        ]
        occupied = {candidate.coordinate for candidate in self.candidates_record}
        domain = set(self.domain)
        additions: list[SamplingCandidate] = []
        by_coordinate = {candidate.coordinate: candidate for candidate in observed}
        for index, first in enumerate(observed):
            for second in observed[index + 1 :]:
                midpoint = self._refinement_between(first.coordinate, second.coordinate, domain, occupied)
                if midpoint is None:
                    continue
                first_point = by_coordinate[first.coordinate]
                second_point = by_coordinate[second.coordinate]
                first_signature = self._signature(first.coordinate)
                second_signature = self._signature(second.coordinate)
                if first_signature == second_signature:
                    continue
                refined = SamplingCandidate(
                    midpoint[0],
                    midpoint[1],
                    (
                        f"local refinement between {first.coordinate} ({first_signature}) "
                        f"and {second.coordinate} ({second_signature}); adjacent results changed"
                    ),
                    max(first_point.level, second_point.level) + 1,
                )
                additions.append(refined)
                occupied.add(refined.coordinate)
        additions.sort(key=lambda candidate: (candidate.level, candidate.coordinate))
        self.candidates_record.extend(additions)
        if additions:
            self._save()
        return additions

    def _refinement_between(
        self,
        first: Coordinate,
        second: Coordinate,
        domain: set[Coordinate],
        occupied: set[Coordinate],
    ) -> Coordinate | None:
        x1, y1 = first
        x2, y2 = second
        if x1 == x2:
            distance = abs(y2 - y1)
            spacing = self.spacing[1]
            midpoint = (x1, (y1 + y2) // 2)
            segment = {(x1, y) for y in range(min(y1, y2), max(y1, y2) + 1)}
        elif y1 == y2:
            distance = abs(x2 - x1)
            spacing = self.spacing[0]
            midpoint = ((x1 + x2) // 2, y1)
            segment = {(x, y1) for x in range(min(x1, x2), max(x1, x2) + 1)}
        else:
            return None
        if distance < 2 * spacing or distance % 2 == 1:
            return None
        if midpoint in occupied or midpoint not in domain or not segment <= domain:
            return None
        return midpoint

    def request_domain_coverage(self, status: str | None = None) -> dict:
        return self.study.request_coverage(self.domain, status)

    def compare_model_replacement(self, new_study: ParameterStudy2D) -> dict:
        if new_study.axis_x != self.study.axis_x or new_study.axis_y != self.study.axis_y:
            raise ValueError("model replacement comparison requires the same two parameter axes")
        old_version = self.model_version
        new_version = new_study.base_problem.canonical_hash()
        legacy = self.observed_coordinates()
        current: list[Coordinate] = []
        resampled: list[Coordinate] = []
        reused: list[Coordinate] = []
        pending: list[Coordinate] = []
        reused_runs = {
            decision.coordinate: decision.source_run_dir
            for decision in new_study.reuse_decisions()
            if decision.applied
        }
        for coordinate in self.domain:
            point = new_study.point(*coordinate)
            new_effective = problem_at(
                problem_at(new_study.base_problem, new_study.axis_x, coordinate[0]),
                new_study.axis_y,
                coordinate[1],
            )
            if point is None or not point.verified or point.problem_hash != new_effective.canonical_hash():
                pending.append(coordinate)
                continue
            current.append(coordinate)
            if point.run_dir == reused_runs.get(coordinate):
                reused.append(coordinate)
            elif new_version != old_version:
                resampled.append(coordinate)
        same_version = new_version == old_version
        not_spliceable = [] if same_version else sorted(set(legacy) - set(resampled))
        fresh_resample_candidates = [] if same_version else sorted(set(self.domain) - set(resampled))
        return {
            "old_model_version": old_version,
            "new_model_version": new_version,
            "same_version": same_version,
            "legacy_observations": legacy,
            "current_observations": current,
            "reused_observations": reused,
            "resampled_observations": resampled,
            "resample_candidates": fresh_resample_candidates,
            "pending": pending,
            "not_spliceable": not_spliceable,
            "note": (
                "legacy observations remain comparable as old-model evidence only; they cannot be "
                "joined with new-model coverage until the same coordinates are resampled or the "
                "model versions are identical"
            ),
        }

    def _current_observation(self, coordinate: Coordinate):
        point = self.study.point(*coordinate)
        if point is None or not self._is_completed(point):
            return None
        effective = problem_at(
            problem_at(self.study.base_problem, self.study.axis_x, coordinate[0]),
            self.study.axis_y,
            coordinate[1],
        )
        if point.problem_hash != effective.canonical_hash():
            return None
        return point

    def _is_completed(self, point) -> bool:
        return point.verified or point.stop_reason == "completed"

    def _signature(self, coordinate: Coordinate) -> str:
        point = self._current_observation(coordinate)
        if point is None:
            return "unobserved"
        objective = "none"
        result_path = Path(point.run_dir) / "result.json"
        if result_path.exists():
            result = SearchStore(point.run_dir).load_result()
            if result.best_objective is not None:
                objective = str(result.best_objective)
        return f"{point.status}:{objective}"
