from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Mapping

from .model import InvalidProblemError, ObjectiveSense, Problem
from .types import ResultStatus, SolveResult, StopReason


@dataclass(frozen=True)
class SearchBudget:
    max_nodes: int | None = None
    max_seconds: float | None = None
    is_cancelled: Callable[[], bool] | None = None


@dataclass(frozen=True)
class Node:
    node_id: int
    parent_id: int | None
    depth: int
    bounds: dict[str, tuple[int, int]]
    optimistic_objective: Fraction | None


@dataclass
class SearchState:
    problem_name: str
    problem_hash: str = ""
    pending: list[Node] = field(default_factory=list)
    next_node_id: int = 0
    nodes_evaluated: int = 0
    best_assignment: dict[str, int] | None = None
    best_objective: Fraction | None = None
    closed_nodes: list[dict] = field(default_factory=list)
    stop_reason: StopReason | None = None
    box_radius: int = 0

    def to_dict(self) -> dict:
        return {
            "problem_name": self.problem_name,
            "problem_hash": self.problem_hash,
            "next_node_id": self.next_node_id,
            "nodes_evaluated": self.nodes_evaluated,
            "best_assignment": self.best_assignment,
            "best_objective": _fraction(self.best_objective),
            "stop_reason": None if self.stop_reason is None else self.stop_reason.value,
            "box_radius": self.box_radius,
            "pending": [_node_to_dict(node) for node in self.pending],
            "closed_nodes": self.closed_nodes,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "SearchState":
        stop = data.get("stop_reason")
        return cls(
            data["problem_name"],
            data.get("problem_hash", ""),
            [_node_from_dict(item) for item in data.get("pending", [])],
            int(data["next_node_id"]),
            int(data["nodes_evaluated"]),
            dict(data["best_assignment"]) if data.get("best_assignment") is not None else None,
            Fraction(data["best_objective"]) if data.get("best_objective") is not None else None,
            list(data.get("closed_nodes", [])),
            None if stop is None else StopReason(stop),
            int(data.get("box_radius", 0)),
        )


class BranchingStrategy:
    def select_variable(self, problem: Problem, bounds: Mapping[str, tuple[int, int]]) -> str:
        raise NotImplementedError


class WidestDomain(BranchingStrategy):
    def select_variable(self, problem: Problem, bounds: Mapping[str, tuple[int, int]]) -> str:
        candidates = [
            (high - low, name) for name, (low, high) in bounds.items() if high > low
        ]
        if not candidates:
            raise InvalidProblemError("cannot branch on a singleton node")
        return max(candidates)[1]


class BranchAndBound:
    def __init__(
        self,
        problem: Problem,
        branching_strategy: BranchingStrategy | None = None,
        checkpoint_callback: Callable[[SearchState], None] | None = None,
    ):
        self.problem = problem
        self.branching_strategy = branching_strategy or WidestDomain()
        self.checkpoint_callback = checkpoint_callback

    def initial_state(self) -> SearchState:
        root_bounds = self.problem.initial_bounds()
        root = Node(0, None, 0, root_bounds, self._optimistic(root_bounds))
        return SearchState(self.problem.name, self.problem.canonical_hash(), [root], 1)

    def solve(
        self,
        budget: SearchBudget | None = None,
        state: SearchState | None = None,
        verify_problem_hash: bool = True,
    ) -> SolveResult:
        if not self.problem.all_bounded:
            from .exact import ExactSolver

            return ExactSolver(self.problem, self.checkpoint_callback).solve(budget, state)
        budget = budget or SearchBudget()
        search = state or self.initial_state()
        if search.problem_name != self.problem.name:
            raise ValueError("checkpoint belongs to a different problem")
        if verify_problem_hash and search.problem_hash and search.problem_hash != self.problem.canonical_hash():
            raise ValueError("checkpoint does not match the immutable problem definition")

        search.stop_reason = None
        started = time.monotonic()

        while search.pending:
            if budget.max_nodes is not None and search.nodes_evaluated >= budget.max_nodes:
                search.stop_reason = StopReason.NODE_BUDGET
                break
            if budget.max_seconds is not None and time.monotonic() - started >= budget.max_seconds:
                search.stop_reason = StopReason.TIME_BUDGET
                break
            if budget.is_cancelled is not None and budget.is_cancelled():
                search.stop_reason = StopReason.CANCELLED
                break

            node = search.pending.pop()
            search.nodes_evaluated += 1
            closed = self._evaluate(node, search)
            search.closed_nodes.append(closed)
            if self.checkpoint_callback is not None:
                self.checkpoint_callback(search)

        return self._result(search)

    def _evaluate(self, node: Node, search: SearchState) -> dict:
        if any(low > high for low, high in node.bounds.values()):
            return {"node_id": node.node_id, "reason": "empty_integer_domain"}

        for constraint in self.problem.constraints:
            if not self._constraint_possible(constraint.coefficients, Fraction(constraint.rhs), node.bounds):
                return {
                    "node_id": node.node_id,
                    "reason": "constraint_box_infeasible",
                    "constraint": constraint.name,
                }

        if search.best_objective is not None and node.optimistic_objective is not None:
            if node.optimistic_objective <= search.best_objective:
                return {
                    "node_id": node.node_id,
                    "reason": "incumbent_bound",
                    "optimistic_objective": str(node.optimistic_objective),
                    "best_objective": str(search.best_objective),
                }

        singleton = all(low == high for low, high in node.bounds.values())
        if singleton:
            assignment = {name: bounds[0] for name, bounds in node.bounds.items()}
            if not self.problem.is_feasible(assignment):
                return {"node_id": node.node_id, "reason": "singleton_infeasible"}
            value = self._transformed_value(assignment)
            if search.best_objective is None or value > search.best_objective:
                search.best_objective = value
                search.best_assignment = assignment
            return {
                "node_id": node.node_id,
                "reason": "integer_solution",
                "assignment": assignment,
                "objective": str(value),
            }

        variable = self.branching_strategy.select_variable(self.problem, node.bounds)
        low, high = node.bounds[variable]
        split = low + (high - low) // 2
        left_bounds = dict(node.bounds)
        right_bounds = dict(node.bounds)
        left_bounds[variable] = (low, split)
        right_bounds[variable] = (split + 1, high)
        children = [search.next_node_id, search.next_node_id + 1]
        search.next_node_id += 2
        left = Node(children[0], node.node_id, node.depth + 1, left_bounds, self._optimistic(left_bounds))
        right = Node(children[1], node.node_id, node.depth + 1, right_bounds, self._optimistic(right_bounds))
        search.pending.extend((left, right))
        return {
            "node_id": node.node_id,
            "reason": "branched",
            "variable": variable,
            "split": split,
            "children": children,
        }

    def _optimistic(self, bounds: Mapping[str, tuple[int, int]]) -> Fraction | None:
        value = Fraction(0)
        sign = Fraction(1) if self.problem.sense == ObjectiveSense.MAXIMIZE else Fraction(-1)
        for name, coefficient in self.problem.objective.items():
            coefficient = sign * Fraction(coefficient)
            low, high = bounds[name]
            endpoint = high if Fraction(coefficient) > 0 else low
            value += Fraction(coefficient) * endpoint
        return value

    def _constraint_possible(
        self,
        coefficients: Mapping[str, Fraction],
        rhs: Fraction,
        bounds: Mapping[str, tuple[int, int]],
    ) -> bool:
        minimum = Fraction(0)
        for name, coefficient in coefficients.items():
            low, high = bounds[name]
            endpoint = low if coefficient > 0 else high
            minimum += coefficient * endpoint
        return minimum <= rhs

    def _result(self, search: SearchState) -> SolveResult:
        best_objective = self._original(search.best_objective)
        bound = self.current_bound(search)

        if search.pending:
            status = ResultStatus.BEST_KNOWN if search.best_assignment is not None else ResultStatus.UNSOLVED
            return SolveResult(
                status,
                _copy(search.best_assignment),
                best_objective,
                bound,
                search.stop_reason or StopReason.NODE_BUDGET,
                search.nodes_evaluated,
                "stopped before every open branch was closed",
            )

        if search.best_assignment is None:
            return SolveResult(
                ResultStatus.INFEASIBLE,
                None,
                None,
                None,
                StopReason.COMPLETED,
                search.nodes_evaluated,
                "every branch is infeasible or empty",
            )

        return SolveResult(
            ResultStatus.OPTIMAL,
            _copy(search.best_assignment),
            best_objective,
            best_objective,
            StopReason.COMPLETED,
            search.nodes_evaluated,
            "every branch is closed and the incumbent bound is tight",
        )

    def current_bound(self, search: SearchState) -> Fraction | None:
        if not search.pending:
            return self._original(search.best_objective)
        values = [node.optimistic_objective for node in search.pending if node.optimistic_objective is not None]
        if not values:
            return None
        value = max(values)
        return value if self.problem.sense == ObjectiveSense.MAXIMIZE else -value

    def _original(self, transformed: Fraction | None) -> Fraction | None:
        if transformed is None:
            return None
        return transformed if self.problem.sense == ObjectiveSense.MAXIMIZE else -transformed

    def _transformed_value(self, assignment: Mapping[str, int]) -> Fraction:
        value = self.problem.objective_value(assignment)
        return value if self.problem.sense == ObjectiveSense.MAXIMIZE else -value


def _node_to_dict(node: Node) -> dict:
    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "depth": node.depth,
        "bounds": {name: list(bounds) for name, bounds in node.bounds.items()},
        "optimistic_objective": None if node.optimistic_objective is None else str(node.optimistic_objective),
    }


def _node_from_dict(data: Mapping) -> Node:
    optimistic = data.get("optimistic_objective")
    return Node(
        int(data["node_id"]),
        None if data.get("parent_id") is None else int(data["parent_id"]),
        int(data["depth"]),
        {name: (int(values[0]), int(values[1])) for name, values in data["bounds"].items()},
        None if optimistic is None else Fraction(optimistic),
    )


def _copy(value: Mapping[str, int] | None) -> dict[str, int] | None:
    return None if value is None else dict(value)


def _fraction(value: Fraction | None) -> str | None:
    return None if value is None else str(value)
