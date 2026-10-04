from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .model import Problem


TERMINAL_STATUS_STRINGS = frozenset({"optimal", "infeasible", "unbounded"})


@dataclass(frozen=True)
class ModelChange:
    kind: str
    reusable_statuses: frozenset[str]
    details: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "reusable_statuses": sorted(self.reusable_statuses),
            "details": list(self.details),
        }


def classify_problem_change(old: Problem, new: Problem) -> ModelChange:
    if (
        old.variables != new.variables
        or old.sense != new.sense
        or {c.name for c in old.constraints} != {c.name for c in new.constraints}
    ):
        return ModelChange(
            "model_changed",
            frozenset(),
            ("variable domains, objective sense, or constraint set differ",),
        )

    kinds: list[str] = []
    details: list[str] = []
    reusable = set(TERMINAL_STATUS_STRINGS)

    if dict(old.objective) != dict(new.objective):
        kinds.append("objective_changed")
        reusable &= {"infeasible"}
        details.append("objective coefficients differ; feasibility is preserved but optimality is not")

    old_constraints = {c.name: c for c in old.constraints}
    new_constraints = {c.name: c for c in new.constraints}
    tightened = relaxed = reshaped = False
    for name in sorted(old_constraints):
        before = old_constraints[name]
        after = new_constraints[name]
        if dict(before.coefficients) == dict(after.coefficients) and before.rhs == after.rhs:
            continue
        if dict(before.coefficients) == dict(after.coefficients):
            if after.rhs < before.rhs:
                tightened = True
                details.append(f"constraint {name!r} tightened: rhs {before.rhs} -> {after.rhs}")
            else:
                relaxed = True
                details.append(f"constraint {name!r} relaxed: rhs {before.rhs} -> {after.rhs}")
        else:
            reshaped = True
            details.append(f"constraint {name!r} coefficients changed")

    if reshaped or (tightened and relaxed):
        kinds.append("constraint_changed")
        reusable = set()
    elif tightened:
        kinds.append("constraint_tightened")
        reusable &= {"infeasible"}
    elif relaxed:
        kinds.append("constraint_relaxed")
        reusable = set()

    if not kinds:
        return ModelChange("identical", frozenset(TERMINAL_STATUS_STRINGS), ())
    return ModelChange("+".join(kinds), frozenset(reusable), tuple(details))


@dataclass(frozen=True)
class ReuseDecision:
    x: int
    y: int
    status: str
    rule: str
    applied: bool
    source_study: str
    source_version: str
    model_version: str
    source_run_dir: str

    @property
    def coordinate(self) -> tuple[int, int]:
        return (self.x, self.y)

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "status": self.status,
            "rule": self.rule,
            "applied": self.applied,
            "source_study": self.source_study,
            "source_version": self.source_version,
            "model_version": self.model_version,
            "source_run_dir": self.source_run_dir,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "ReuseDecision":
        return cls(
            int(data["x"]),
            int(data["y"]),
            data["status"],
            data["rule"],
            bool(data.get("applied", False)),
            data.get("source_study", ""),
            data.get("source_version", ""),
            data.get("model_version", ""),
            data.get("source_run_dir", ""),
        )


def review_reuse_decisions(study) -> list[str]:
    errors: list[str] = []
    decisions = study.reuse_decisions()
    if not decisions:
        return errors
    model_version = study.base_problem.canonical_hash()
    source_cache: dict[str, dict] = {}
    for decision in decisions:
        label = f"reuse ({decision.x}, {decision.y})"
        if decision.model_version != model_version:
            errors.append(
                f"{label}: recorded model version {decision.model_version!r} does not match "
                f"the branch base problem {model_version!r}"
            )
            continue
        source_dir = Path(decision.source_study)
        if decision.source_study not in source_cache:
            manifest_path = source_dir / "study.json"
            if not manifest_path.exists():
                source_cache[decision.source_study] = {"error": f"missing manifest {manifest_path}"}
            else:
                with manifest_path.open("r", encoding="utf-8") as handle:
                    source_cache[decision.source_study] = json.load(handle)
        source = source_cache[decision.source_study]
        if "error" in source:
            errors.append(f"{label}: {source['error']}")
            continue
        source_problem = Problem.from_dict(source["base_problem"])
        if source_problem.canonical_hash() != decision.source_version:
            errors.append(f"{label}: source study base problem no longer matches the recorded version")
            continue
        source_point = None
        for item in source.get("points", []):
            if int(item["x"]) == decision.x and int(item["y"]) == decision.y:
                source_point = item
                break
        if (
            source_point is None
            or not source_point.get("verified", False)
            or source_point.get("status") != decision.status
        ):
            errors.append(
                f"{label}: source point ({decision.x}, {decision.y}) is not verified "
                f"with status {decision.status!r} in the source study"
            )
            continue
        from .sweep import ParameterAxis, problem_at

        axis_x = ParameterAxis.from_dict(source["axis_x"])
        axis_y = ParameterAxis.from_dict(source["axis_y"])
        old_eff = problem_at(
            problem_at(source_problem, axis_x, decision.x), axis_y, decision.y
        )
        new_eff = problem_at(
            problem_at(study.base_problem, axis_x, decision.x), axis_y, decision.y
        )
        change = classify_problem_change(old_eff, new_eff)
        if change.kind != decision.rule:
            errors.append(
                f"{label}: recorded rule {decision.rule!r} no longer matches recomputed "
                f"change {change.kind!r}"
            )
            continue
        point = study.point(decision.x, decision.y)
        if decision.applied:
            if decision.status not in change.reusable_statuses:
                errors.append(
                    f"{label}: status {decision.status!r} is not reusable under {change.kind!r}"
                )
            elif point is None or not point.verified or point.status != decision.status:
                errors.append(
                    f"{label}: branch point is not carrying the reused status {decision.status!r}"
                )
    return errors
