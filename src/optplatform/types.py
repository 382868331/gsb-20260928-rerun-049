from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Mapping


class ResultStatus(str, Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    BEST_KNOWN = "best_known"
    UNSOLVED = "unsolved"


class StopReason(str, Enum):
    COMPLETED = "completed"
    NODE_BUDGET = "node_budget"
    TIME_BUDGET = "time_budget"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SolveResult:
    status: ResultStatus
    best_assignment: dict[str, int] | None
    best_objective: Fraction | None
    bound: Fraction | None
    improving_direction: dict[str, int] | None
    stop_reason: StopReason
    nodes_evaluated: int
    message: str = ""

    @property
    def proven_optimal(self) -> bool:
        return self.status is ResultStatus.OPTIMAL

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "best_assignment": self.best_assignment,
            "best_objective": _fraction(self.best_objective),
            "bound": _fraction(self.bound),
            "improving_direction": self.improving_direction,
            "stop_reason": self.stop_reason.value,
            "nodes_evaluated": self.nodes_evaluated,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "SolveResult":
        best_objective = data.get("best_objective")
        bound = data.get("bound")
        return cls(
            ResultStatus(data["status"]),
            dict(data["best_assignment"]) if data.get("best_assignment") is not None else None,
            Fraction(best_objective) if best_objective is not None else None,
            Fraction(bound) if bound is not None else None,
            dict(data["improving_direction"]) if data.get("improving_direction") is not None else None,
            StopReason(data["stop_reason"]),
            int(data["nodes_evaluated"]),
            data.get("message", ""),
        )


def _fraction(value: Fraction | None) -> str | None:
    return None if value is None else str(value)
