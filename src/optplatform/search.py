from __future__ import annotations

from math import gcd, lcm
import time
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Callable, Mapping

from .model import Constraint, InvalidProblemError, ObjectiveSense, Problem, Variable
from .types import ResultStatus, SolveResult, StopReason


Bound = int | None
Bounds = dict[str, tuple[Bound, Bound]]


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
    bounds: Bounds
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
    improving_point: dict[str, int] | None = None
    improving_direction: dict[str, int] | None = None
    stop_reason: StopReason | None = None

    def to_dict(self) -> dict:
        return {
            "problem_name": self.problem_name,
            "problem_hash": self.problem_hash,
            "next_node_id": self.next_node_id,
            "nodes_evaluated": self.nodes_evaluated,
            "best_assignment": self.best_assignment,
            "best_objective": _fraction(self.best_objective),
            "stop_reason": None if self.stop_reason is None else self.stop_reason.value,
            "pending": [_node_to_dict(node) for node in self.pending],
            "closed_nodes": self.closed_nodes,
            "improving_point": self.improving_point,
            "improving_direction": self.improving_direction,
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
            dict(data.get("improving_point")) if data.get("improving_point") is not None else None,
            dict(data.get("improving_direction")) if data.get("improving_direction") is not None else None,
            None if stop is None else StopReason(stop),
        )


class BranchingStrategy:
    def select_variable(self, problem: Problem, bounds: Mapping[str, tuple[Bound, Bound]]) -> str:
        raise NotImplementedError


class WidestDomain(BranchingStrategy):
    def select_variable(self, problem: Problem, bounds: Mapping[str, tuple[Bound, Bound]]) -> str:
        candidates = [
            (high - low, name)
            for name, (low, high) in bounds.items()
            if low is not None and high is not None and high > low
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

    def solve(self, budget: SearchBudget | None = None, state: SearchState | None = None) -> SolveResult:
        budget = budget or SearchBudget()
        search = state or self.initial_state()
        if search.problem_name != self.problem.name:
            raise ValueError("checkpoint belongs to a different problem")
        if search.problem_hash and search.problem_hash != self.problem.canonical_hash():
            raise ValueError("checkpoint does not match the immutable problem definition")

        search.stop_reason = None
        started = time.monotonic()

        if search.improving_direction is not None:
            return self._result(search)

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
        tightened_bounds = self._tighten_bounds(node.bounds)
        node = replace(
            node,
            bounds=tightened_bounds,
            optimistic_objective=self._optimistic(tightened_bounds),
        )
        if any(low is not None and high is not None and low > high for low, high in node.bounds.values()):
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

        singleton = all(low is not None and high is not None and low == high for low, high in node.bounds.values())
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

        finite_candidates = [
            name for name, (low, high) in node.bounds.items()
            if low is not None and high is not None and high > low
        ]
        if not finite_candidates:
            return self._find_improving_direction(node, search)

        variable = self.branching_strategy.select_variable(self.problem, node.bounds)
        low, high = node.bounds[variable]
        split = low + ((high - low + 1) // 2) - 1
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

    def _optimistic(self, bounds: Mapping[str, tuple[Bound, Bound]]) -> Fraction | None:
        value = Fraction(0)
        sign = Fraction(1) if self.problem.sense == ObjectiveSense.MAXIMIZE else Fraction(-1)
        for name, coefficient in self.problem.objective.items():
            coefficient = sign * Fraction(coefficient)
            low, high = bounds[name]
            if coefficient > 0 and high is None:
                return None
            if coefficient < 0 and low is None:
                return None
            endpoint = high if coefficient > 0 else low
            value += Fraction(coefficient) * endpoint
        return value

    def _constraint_possible(
        self,
        coefficients: Mapping[str, Fraction],
        rhs: Fraction,
        bounds: Mapping[str, tuple[Bound, Bound]],
    ) -> bool:
        minimum = Fraction(0)
        for name, coefficient in coefficients.items():
            low, high = bounds[name]
            if coefficient > 0 and low is None:
                return True
            if coefficient < 0 and high is None:
                return True
            endpoint = low if coefficient > 0 else high
            minimum += coefficient * endpoint
        return minimum <= rhs

    def _tighten_bounds(self, bounds: Bounds) -> Bounds:
        tightened = dict(bounds)
        changed = True
        while changed:
            changed = False
            for constraint in self.problem.constraints:
                for target, target_coefficient in constraint.coefficients.items():
                    lower, upper = tightened[target]
                    other_minimum = Fraction(0)
                    missing_minimum = False
                    for name, coefficient in constraint.coefficients.items():
                        if name == target:
                            continue
                        other_lower, other_upper = tightened[name]
                        if coefficient > 0:
                            if other_lower is None:
                                missing_minimum = True
                            else:
                                other_minimum += coefficient * other_lower
                        else:
                            if other_upper is None:
                                missing_minimum = True
                            else:
                                other_minimum += coefficient * other_upper

                    rhs = Fraction(constraint.rhs)
                    target_coefficient = Fraction(target_coefficient)
                    if target_coefficient > 0 and not missing_minimum:
                        derived_upper = (rhs - other_minimum) // target_coefficient
                        if upper is None or derived_upper < upper:
                            upper = int(derived_upper)
                            changed = True
                    if target_coefficient < 0 and not missing_minimum:
                        derived_lower = -((rhs - other_minimum) // -target_coefficient)
                        if lower is None or derived_lower > lower:
                            lower = int(derived_lower)
                            changed = True
                    tightened[target] = (lower, upper)
                    if lower is not None and upper is not None and lower > upper:
                        return tightened
        return tightened

    def _find_improving_direction(self, node: Node, search: SearchState) -> dict:
        infinite = {
            name: bounds for name, bounds in node.bounds.items()
            if bounds[0] is None or bounds[1] is None
        }
        endpoint = self._ray_base(node.bounds, {name: 0 for name in self.problem.variable_names})
        if endpoint is not None and self.problem.is_feasible(endpoint):
            value = self._transformed_value(endpoint)
            if search.best_objective is None or value > search.best_objective:
                search.best_objective = value
                search.best_assignment = endpoint
            if not infinite:
                return {
                    "node_id": node.node_id,
                    "reason": "integer_solution",
                    "assignment": endpoint,
                    "objective": str(value),
                }

        base = self._ray_base(node.bounds, {name: 0 for name in self.problem.variable_names})
        if base is not None:
            direction = self._find_integer_direction(node.bounds, base)
            if direction is not None:
                witness = self._feasible_ray_point(base, direction)
                if witness is not None:
                    search.improving_point = witness
                    search.improving_direction = direction
                    search.pending.clear()
                    return {
                        "node_id": node.node_id,
                        "reason": "improving_direction",
                        "assignment": witness,
                        "direction": search.improving_direction,
                    }

        return {"node_id": node.node_id, "reason": "unbounded_box_exhausted_without_direction"}

    def _find_integer_direction(
        self,
        bounds: Mapping[str, tuple[Bound, Bound]],
        base: Mapping[str, int],
    ) -> dict[str, int] | None:
        infinite_names = [
            name for name, (low, high) in bounds.items()
            if low is None or high is None
        ]
        if not infinite_names:
            return None

        signs = {
            name: Fraction(-1 if bounds[name][0] is None else 1)
            for name in infinite_names
        }
        coefficient_limit = 0
        for constraint in self.problem.constraints:
            row_scale = lcm(*(
                Fraction(coefficient).denominator
                for coefficient in constraint.coefficients.values()
            )) if constraint.coefficients else 1
            row_max = max(
                (
                    abs(Fraction(coefficient).numerator * row_scale // Fraction(coefficient).denominator)
                    for coefficient in constraint.coefficients.values()
                ),
                default=1,
            )
            coefficient_limit = max(
                coefficient_limit,
                row_max,
            )
        for coefficient in self.problem.objective.values():
            coefficient_limit = max(coefficient_limit, abs(Fraction(coefficient).numerator))
        direction_bound = max(len(infinite_names) * max(coefficient_limit, 1), 1) ** len(infinite_names)

        variables = [Variable(f"__direction_{name}", 0, direction_bound) for name in infinite_names]
        constraints = []
        for constraint in self.problem.constraints:
            base_left = sum(
                Fraction(coefficient) * Fraction(base[name])
                for name, coefficient in constraint.coefficients.items()
            )
            constraints.append(
                Constraint(
                    f"__direction_{constraint.name}",
                    {
                        f"__direction_{name}": Fraction(constraint.coefficients.get(name, 0)) * signs[name]
                        for name in infinite_names
                        if constraint.coefficients.get(name, 0) != 0
                    },
                    Fraction(constraint.rhs) - base_left,
                )
            )

        objective_sign = Fraction(1) if self.problem.sense == ObjectiveSense.MAXIMIZE else Fraction(-1)
        direction_problem = Problem(
            f"__direction_{self.problem.name}",
            variables,
            {
                f"__direction_{name}": objective_sign * Fraction(self.problem.objective.get(name, 0)) * signs[name]
                for name in infinite_names
                if self.problem.objective.get(name, 0) != 0
            },
            constraints,
        )
        direction_result = BranchAndBound(direction_problem).solve()
        if direction_result.best_objective is None or direction_result.best_objective <= 0:
            return None
        scale = 0
        for name in infinite_names:
            scale = gcd(scale, abs(direction_result.best_assignment[f"__direction_{name}"]))
        if scale == 0:
            return None
        direction = {name: 0 for name in self.problem.variable_names}
        for name in infinite_names:
            value = direction_result.best_assignment[f"__direction_{name}"] // scale
            direction[name] = int(signs[name] * value)
        return {name: value for name, value in direction.items() if value != 0}

    def _feasible_ray_point(self, base: Mapping[str, int], direction: Mapping[str, int]) -> dict[str, int] | None:
        steps = 0
        for constraint in self.problem.constraints:
            left = sum(
                coefficient * Fraction(base[name])
                for name, coefficient in constraint.coefficients.items()
            )
            slope = sum(
                coefficient * Fraction(direction.get(name, 0))
                for name, coefficient in constraint.coefficients.items()
            )
            violation = left - Fraction(constraint.rhs)
            if violation <= 0:
                continue
            if slope >= 0:
                return None
            needed = -(-violation // -slope)
            if needed > steps:
                steps = int(needed)
        return {name: value + direction.get(name, 0) * steps for name, value in base.items()}

    def _ray_base(self, bounds: Mapping[str, tuple[Bound, Bound]], direction: Mapping[str, int]) -> dict[str, int] | None:
        base = {}
        for name, (low, high) in bounds.items():
            if direction[name] > 0:
                if low is None:
                    return None
                base[name] = low
            elif direction[name] < 0:
                if high is None:
                    return None
                base[name] = high
            elif low is not None:
                base[name] = low
            elif high is not None:
                base[name] = high
            else:
                return None
        return base

    def _transformed_direction(self, direction: Mapping[str, int]) -> Fraction:
        sign = Fraction(1) if self.problem.sense == ObjectiveSense.MAXIMIZE else Fraction(-1)
        return sign * sum(
            Fraction(coefficient) * Fraction(direction.get(name, 0))
            for name, coefficient in self.problem.objective.items()
        )

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
                _copy(search.improving_direction),
                search.stop_reason or StopReason.NODE_BUDGET,
                search.nodes_evaluated,
                "stopped before every open branch was closed",
            )

        if search.improving_direction is not None:
            return SolveResult(
                ResultStatus.UNBOUNDED,
                _copy(search.improving_point),
                self.problem.objective_value(search.improving_point) if search.improving_point is not None else None,
                None,
                _copy(search.improving_direction),
                StopReason.COMPLETED,
                search.nodes_evaluated,
                "a feasible integer point and strictly improving feasible direction prove unboundedness",
            )

        if search.best_assignment is None:
            return SolveResult(
                ResultStatus.INFEASIBLE,
                None,
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
            None,
            StopReason.COMPLETED,
            search.nodes_evaluated,
            "every branch is closed and the incumbent bound is tight",
        )

    def current_bound(self, search: SearchState) -> Fraction | None:
        if not search.pending:
            return self._original(search.best_objective)
        values = [node.optimistic_objective for node in search.pending]
        if any(value is None for value in values):
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
        "bounds": {name: [bounds[0], bounds[1]] for name, bounds in node.bounds.items()},
        "optimistic_objective": None if node.optimistic_objective is None else str(node.optimistic_objective),
    }


def _node_from_dict(data: Mapping) -> Node:
    optimistic = data.get("optimistic_objective")
    return Node(
        int(data["node_id"]),
        None if data.get("parent_id") is None else int(data["parent_id"]),
        int(data["depth"]),
        {
            name: (
                None if values[0] is None else int(values[0]),
                None if values[1] is None else int(values[1]),
            )
            for name, values in data["bounds"].items()
        },
        None if optimistic is None else Fraction(optimistic),
    )


def _copy(value: Mapping[str, int] | None) -> dict[str, int] | None:
    return None if value is None else dict(value)


def _fraction(value: Fraction | None) -> str | None:
    return None if value is None else str(value)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
