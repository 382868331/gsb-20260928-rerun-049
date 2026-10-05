from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

from .grid import Coordinate, ParameterStudy2D


@dataclass(frozen=True)
class SamplingCandidate:
    coordinate: Coordinate
    reason: str
    source: str
    depends_on: tuple[Coordinate, ...] = ()

    def to_dict(self) -> dict:
        return {
            "coordinate": list(self.coordinate),
            "reason": self.reason,
            "source": self.source,
            "depends_on": [list(point) for point in self.depends_on],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "SamplingCandidate":
        return cls(
            (int(data["coordinate"][0]), int(data["coordinate"][1])),
            data["reason"],
            data.get("source", "unknown"),
            tuple((int(p[0]), int(p[1])) for p in data.get("depends_on", [])),
        )


class HierarchicalSampler2D:
    def __init__(
        self,
        study: ParameterStudy2D,
        domain: Sequence[Sequence[int]],
        *,
        total_budget: int,
        minimum_spacing: int | tuple[int, int] = 1,
        required: Sequence[Sequence[int]] = (),
    ):
        self.study = study
        self.domain = tuple(sorted({(int(p[0]), int(p[1])) for p in domain}))
        if not self.domain:
            raise ValueError("sampling domain must contain at least one coordinate")
        self.domain_set = set(self.domain)
        if isinstance(minimum_spacing, int):
            spacing = (int(minimum_spacing), int(minimum_spacing))
        else:
            spacing = (int(minimum_spacing[0]), int(minimum_spacing[1]))
        if spacing[0] < 1 or spacing[1] < 1:
            raise ValueError("minimum spacing must be positive")
        self.minimum_spacing = spacing
        self.total_budget = int(total_budget)
        if self.total_budget < 0:
            raise ValueError("total budget cannot be negative")
        self.required = tuple(sorted({(int(p[0]), int(p[1])) for p in required}))
        for coordinate in self.required:
            if coordinate not in self.domain_set:
                raise ValueError(f"required coordinate {coordinate} is outside the domain")
        self._validate_required_spacing()

        self.manifest_path = self.study.directory / "sampling.json"
        self.candidates: list[SamplingCandidate] = []
        self.queued: list[Coordinate] = []
        self.computed: list[Coordinate] = []
        self.charged: list[Coordinate] = []
        self.spent_budget = 0
        if self.manifest_path.exists():
            self._load()
        self.study._pending = [
            coordinate for coordinate in self.study._pending if coordinate in self.domain_set
        ]
        self.study._save()
        if not self.manifest_path.exists():
            self._build_initial_plan()
            self._adopt_existing_observations()
            self._sync_study_pending()
            self._save()
        else:
            self._sync_study_pending()

    def _sync_study_pending(self) -> None:
        self.study._pending = list(self.queued)
        self.study._save()

    def _validate_required_spacing(self) -> None:
        for index, first in enumerate(self.required):
            for second in self.required[index + 1 :]:
                if not self._respects_spacing(first, {second}):
                    raise ValueError("required coordinates violate the minimum spacing")

    def _build_initial_plan(self) -> None:
        occupied = set()
        for component in self._connected_components():
            required_in_component = sorted(
                coordinate for coordinate in self.required if coordinate in component
            )
            non_required = sorted(component - set(required_in_component))
            seed = required_in_component[0] if required_in_component else non_required[0]
            if seed not in occupied:
                self._add_candidate(
                    seed,
                    "first observation in a disconnected component",
                    "required" if seed in set(self.required) else "component_seed",
                )
                occupied.add(seed)
            for coordinate in required_in_component:
                if coordinate not in occupied:
                    self._add_candidate(
                        coordinate,
                        "required observation retained before initial sampling",
                        "required",
                    )
                    occupied.add(coordinate)

        x_min = min(x for x, _ in self.domain)
        y_min = min(y for _, y in self.domain)
        step_x = self._coarse_step(0)
        step_y = self._coarse_step(1)
        coarse = [
            coordinate
            for coordinate in self.domain
            if (coordinate[0] - x_min) % step_x == 0
            and (coordinate[1] - y_min) % step_y == 0
        ]
        for coordinate in coarse:
            if self._respects_spacing(coordinate, occupied):
                self._add_candidate(
                    coordinate,
                    "initial coarse-grid observation inside the exact non-rectangular domain",
                    "coarse",
                )
                occupied.add(coordinate)

    def _coarse_step(self, axis: int) -> int:
        minimum = self.minimum_spacing[axis]
        values = sorted({point[axis] for point in self.domain})
        span = values[-1] - values[0]
        step = minimum
        while step * 2 <= max(span, minimum * 2):
            step *= 2
        return step

    def _connected_components(self) -> list[set[Coordinate]]:
        remaining = set(self.domain_set)
        components: list[set[Coordinate]] = []
        while remaining:
            start = min(remaining)
            stack = [start]
            remaining.remove(start)
            component = {start}
            while stack:
                x, y = stack.pop()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        neighbor = (x + dx, y + dy)
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            component.add(neighbor)
                            stack.append(neighbor)
            components.append(component)
        return components

    def _respects_spacing(self, coordinate: Coordinate, occupied: set[Coordinate]) -> bool:
        for other in occupied:
            dx = abs(coordinate[0] - other[0])
            dy = abs(coordinate[1] - other[1])
            too_close_x = dx != 0 and dx < self.minimum_spacing[0]
            too_close_y = dy != 0 and dy < self.minimum_spacing[1]
            if too_close_x or too_close_y:
                return False
        return True

    def _add_candidate(
        self,
        coordinate: Coordinate,
        reason: str,
        source: str,
        depends_on: Sequence[Coordinate] = (),
    ) -> bool:
        if coordinate not in self.domain_set:
            return False
        if coordinate in self.computed or coordinate in self.queued:
            return False
        if source != "required" and not self._respects_spacing(
            coordinate, set(self.computed + self.queued)
        ):
            return False
        self.candidates.append(
            SamplingCandidate(coordinate, reason, source, tuple(depends_on))
        )
        self.queued.append(coordinate)
        return True

    def _adopt_existing_observations(self) -> None:
        for coordinate in list(self.queued):
            point = self.study.point(*coordinate)
            if point is None or not point.verified:
                continue
            effective = self._effective_hash(coordinate)
            if point.problem_hash != effective:
                continue
            self._mark_computed(coordinate, charge=False)

    def _effective_hash(self, coordinate: Coordinate) -> str:
        from .sweep import problem_at

        problem = problem_at(
            problem_at(self.study.base_problem, self.study.axis_x, coordinate[0]),
            self.study.axis_y,
            coordinate[1],
        )
        return problem.canonical_hash()

    def _verified_observations(self) -> dict[Coordinate, str]:
        observations = {}
        for coordinate in self.computed:
            point = self.study.point(*coordinate)
            if (
                point is not None
                and point.verified
                and point.status is not None
                and point.problem_hash == self._effective_hash(coordinate)
            ):
                observations[coordinate] = point.status
        return observations

    def _add_refinement_candidates(self, changed_from: Coordinate) -> None:
        observations = self._verified_observations()
        new_status = observations.get(changed_from)
        if new_status is None:
            return
        for coordinate, status in sorted(observations.items()):
            if coordinate == changed_from or status == new_status:
                continue
            dx = changed_from[0] - coordinate[0]
            dy = changed_from[1] - coordinate[1]
            candidates: list[Coordinate] = []
            if dy == 0 and dx % 2 == 0 and abs(dx) >= self.minimum_spacing[0] * 2:
                candidates.append((coordinate[0] + dx // 2, coordinate[1]))
            if dx == 0 and dy % 2 == 0 and abs(dy) >= self.minimum_spacing[1] * 2:
                candidates.append((coordinate[0], coordinate[1] + dy // 2))
            if dx % 2 == 0 and dy % 2 == 0:
                candidates.append((coordinate[0] + dx // 2, coordinate[1] + dy // 2))
            for midpoint in candidates:
                valid_horizontal = midpoint[1] == changed_from[1] and abs(dx) >= self.minimum_spacing[0] * 2
                valid_vertical = midpoint[0] == changed_from[0] and abs(dy) >= self.minimum_spacing[1] * 2
                valid_diagonal = (
                    midpoint[0] != changed_from[0]
                    and midpoint[1] != changed_from[1]
                    and abs(dx) >= self.minimum_spacing[0] * 2
                    and abs(dy) >= self.minimum_spacing[1] * 2
                )
                if not (valid_horizontal or valid_vertical or valid_diagonal):
                    continue
                self._add_candidate(
                    midpoint,
                    f"local refinement: adjacent results change {status} -> {new_status}",
                    "adaptive",
                    (coordinate, changed_from),
                )

    def run(self, max_points: int | None = None) -> None:
        points_done = 0
        while self.queued:
            if self.spent_budget >= self.total_budget:
                break
            if max_points is not None and points_done >= max_points:
                break
            coordinate = self.queued[0]
            point = self.study._solve_point(coordinate, self.study.budget)
            self.study._points[coordinate] = point
            self.study._save()
            budget_paused = point.stop_reason in {"node_budget", "time_budget", "cancelled"}
            if not point.verified:
                if budget_paused:
                    self._mark_computed(coordinate, charge=True)
                    self._sync_study_pending()
                    self._save()
                    return
                self.queued.pop(0)
                if coordinate not in self.computed:
                    self.computed.append(coordinate)
                    self.computed.sort()
                if coordinate not in self.charged and self.spent_budget < self.total_budget:
                    self.spent_budget += 1
                    self.charged.append(coordinate)
                    self.charged.sort()
                self._sync_study_pending()
                self._save()
                points_done += 1
                continue
            self._mark_computed(coordinate, charge=True)
            points_done += 1
            self._add_refinement_candidates(coordinate)
            self._sync_study_pending()
            self._save()

    def _mark_computed(self, coordinate: Coordinate, *, charge: bool) -> None:
        point = self.study.point(*coordinate)
        verified = point is not None and point.verified
        exhausted = (
            point is not None
            and point.stop_reason in {"node_budget", "time_budget", "cancelled"}
            and not verified
        )
        if coordinate in self.queued and (verified or exhausted):
            self.queued.remove(coordinate)
        if coordinate not in self.computed:
            self.computed.append(coordinate)
            self.computed.sort()
        if charge and coordinate not in self.charged and self.spent_budget < self.total_budget:
            self.spent_budget += 1
            self.charged.append(coordinate)
            self.charged.sort()

    def stop(self) -> None:
        self._save()

    def candidate_report(self) -> dict:
        return {
            "domain": list(self.domain),
            "minimum_spacing": list(self.minimum_spacing),
            "model_version": self.study.base_problem.canonical_hash(),
            "total_budget": self.total_budget,
            "spent_budget": self.spent_budget,
            "computed": list(self.computed),
            "pending": list(self.queued),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "reasons": {
                str(list(candidate.coordinate)): candidate.to_dict()
                for candidate in self.candidates
            },
            "note": "pending coordinates are candidates; holes and disconnected components are represented by the exact domain only",
        }

    def request_coverage(self, status: str | None = None) -> dict:
        return self.study.request_coverage(self.domain, status)

    def _save(self) -> None:
        payload = {
            "domain": [list(point) for point in self.domain],
            "minimum_spacing": list(self.minimum_spacing),
            "total_budget": self.total_budget,
            "required": [list(point) for point in self.required],
            "model_version": self.study.base_problem.canonical_hash(),
            "spent_budget": self.spent_budget,
            "computed": [list(point) for point in self.computed],
            "charged": [list(point) for point in self.charged],
            "queued": [list(point) for point in self.queued],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }
        temporary = self.manifest_path.with_name(f".{self.manifest_path.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(self.manifest_path)

    def _load(self) -> None:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        saved_domain = tuple((int(p[0]), int(p[1])) for p in data.get("domain", []))
        if saved_domain != self.domain:
            raise ValueError("sampling manifest belongs to a different domain")
        saved_spacing = tuple(int(value) for value in data.get("minimum_spacing", [1, 1]))
        if saved_spacing != self.minimum_spacing:
            raise ValueError("sampling manifest has different minimum spacing")
        if int(data.get("total_budget", self.total_budget)) != self.total_budget:
            raise ValueError("sampling manifest has a different total budget")
        saved_required = tuple((int(p[0]), int(p[1])) for p in data.get("required", []))
        if saved_required != self.required:
            raise ValueError("sampling manifest has different required observations")
        if data.get("model_version") != self.study.base_problem.canonical_hash():
            raise ValueError("sampling manifest belongs to a different model version")
        self.spent_budget = int(data.get("spent_budget", 0))
        self.computed = [(int(p[0]), int(p[1])) for p in data.get("computed", [])]
        self.charged = [(int(p[0]), int(p[1])) for p in data.get("charged", self.computed)]
        self.queued = [(int(p[0]), int(p[1])) for p in data.get("queued", [])]
        self.candidates = [SamplingCandidate.from_dict(item) for item in data.get("candidates", [])]
