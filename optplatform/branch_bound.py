"""可恢复的分支定界搜索。

搜索过程（待探索节点栈、已完成剪枝统计、incumbent 与全局界限）全部
封装在可序列化的 :class:`SearchState` 中。中断（预算耗尽/取消）后写入
检查点，重新加载即可继续；已剪枝的分支不会被重复探索。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from enum import Enum
from math import gcd
from typing import Callable, Protocol

from .lp import LPStatus, solve_relaxation
from .model import ConstraintSense, ILP, ObjectiveSense
from .proof import CertificateType, Result


class SolveStatus(str, Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    BUDGET = "budget_exhausted"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Node:
    """一个子问题：在原问题上追加的线性约束列表。

    每项为 ``(terms, sense, rhs)``；父节点的所有约束通过 ``extra``
    完整携带，因此节点可独立从检查点恢复。
    """

    node_id: int
    extra: tuple[tuple[frozenset[tuple[str, Fraction]], ConstraintSense, Fraction], ...]
    bound: Fraction | None = None  # 继承的松弛界限（max 上界/min 下界）
    unbounded: bool = False       # 该子问题松弛无界

    def runtime_extra(self):
        return [({k: v for k, v in terms}, sense, rhs) for terms, sense, rhs in self.extra]


@dataclass
class SearchStats:
    nodes_opened: int = 0
    nodes_infeasible: int = 0
    nodes_integer: int = 0
    nodes_pruned_by_bound: int = 0
    nodes_branched: int = 0


class BranchingStrategy(Protocol):
    """分支变量选择策略。更换策略只改变搜索次序，不改变问题含义。"""

    def choose(self, problem: ILP, point: dict[str, Fraction]) -> str: ...


class MostFractional:
    """选择离整数最远的分数变量（默认策略）。"""

    def choose(self, problem: ILP, point: dict[str, Fraction]) -> str:
        def frac(name: str) -> Fraction:
            x = point[name]
            return min(x - x.__floor__(), x.__ceil__() - x)

        candidates = [v.name for v in problem.variables if point[v.name].denominator != 1]
        return max(candidates, key=lambda nm: (frac(nm), nm))


class FirstFractional:
    """按变量声明顺序选第一个分数变量（确定性更强，便于核对）。"""

    def choose(self, problem: ILP, point: dict[str, Fraction]) -> str:
        return next(v.name for v in problem.variables if point[v.name].denominator != 1)


def _better(a: Fraction, b: Fraction | None, maximizing: bool) -> bool:
    return b is None or (a > b if maximizing else a < b)


def _integer_ray(ray: dict[str, Fraction]) -> dict[str, int] | None:
    """把有理衰退方向放大为原始整数方向（分母最小公倍数/内容约分）。

    若衰退锥中存在有理射线，则通分放大后即为整数方向；沿它走任意
    非负整数步都保持可行，故可用于无界性证明。
    """
    nonzero = {k: v for k, v in ray.items() if v != 0}
    if not nonzero:
        return None
    scale = 1
    for v in nonzero.values():
        scale = scale * v.denominator // gcd(scale, v.denominator)
    ints = {k: int(v * scale) for k, v in nonzero.items()}
    common = 0
    for v in ints.values():
        common = gcd(common, abs(v))
    if common > 1:
        ints = {k: v // common for k, v in ints.items()}
    return ints


def _is_integer_point(point: dict[str, Fraction]) -> bool:
    return all(v.denominator == 1 for v in point.values())


@dataclass
class SearchState:
    """搜索过程的完整状态，可序列化为 JSON 并原样恢复。"""

    frontier: list[Node] = field(default_factory=list)
    next_node_id: int = 1
    incumbent: dict[str, Fraction] | None = None
    incumbent_value: Fraction | None = None
    bound: Fraction | None = None  # 所有开放/未探索节点的最佳松弛界限
    finished: bool = False
    stats: SearchStats = field(default_factory=SearchStats)

    @classmethod
    def initial(cls, root_extra=()) -> "SearchState":
        node = Node(0, tuple(root_extra))
        return cls(frontier=[node])


@dataclass
class Budget:
    """节点预算；None 表示不限。取消通过 cancel_event 外部置位。"""

    max_nodes: int | None = None
    cancel_event: "Callable[[], bool] | None" = None


class BranchBoundSolver:
    def __init__(self, strategy: BranchingStrategy | None = None) -> None:
        self.strategy = strategy or MostFractional()

    def search(
        self,
        problem: ILP,
        state: SearchState | None = None,
        budget: Budget | None = None,
    ) -> tuple[Result, SearchState]:
        budget = budget or Budget()
        state = state or SearchState.initial()
        maximizing = problem.objective.sense is ObjectiveSense.MAX
        self._maximizing = maximizing
        opened_this_run = 0
        stop_reason: str | None = None

        while state.frontier:
            if budget.cancel_event is not None and budget.cancel_event():
                stop_reason = SolveStatus.CANCELLED
                break
            if budget.max_nodes is not None and opened_this_run >= budget.max_nodes:
                stop_reason = SolveStatus.BUDGET
                break

            node = state.frontier.pop()
            opened_this_run += 1
            state.stats.nodes_opened += 1
            # 用继承界限在重新求解前即可剪枝（检查点恢复后依然有效）。
            if (
                not node.unbounded
                and node.bound is not None
                and state.incumbent_value is not None
                and (
                    (maximizing and node.bound <= state.incumbent_value)
                    or (not maximizing and node.bound >= state.incumbent_value)
                )
            ):
                state.stats.nodes_pruned_by_bound += 1
                state.bound = self._global_bound(state)
                continue
            lp = solve_relaxation(problem, node.runtime_extra())

            if lp.status is LPStatus.INFEASIBLE:
                state.stats.nodes_infeasible += 1
            elif lp.status is LPStatus.UNBOUNDED:
                integer_ray = _integer_ray(lp.ray)
                feasible = state.incumbent
                if feasible is None and lp.point is not None and _is_integer_point(lp.point):
                    feasible = lp.point
                gain = sum(
                    problem.objective_coeff(k) * Fraction(d)
                    for k, d in (integer_ray or {}).items()
                )
                if problem.objective.sense is ObjectiveSense.MIN:
                    gain = -gain
                if integer_ray is not None and feasible is not None and gain > 0:
                    state.finished = True
                    return self._unbounded(problem, state, feasible, integer_ray, gain), state
                # 尚无整数可行点或射线不改进目标：沿分数顶点继续分支，
                # 找到整数可行点后再用同一射线完成无界证明。
                branched = self._branch(problem, state, node, lp.point, None, True)
                if branched is None:
                    # 顶点已整数却未完成证明（理论上不会发生）：保守中止。
                    state.frontier.append(node)
                    stop_reason = stop_reason or SolveStatus.CANCELLED
                    break
            else:
                bound_value = lp.objective if maximizing else -lp.objective
                # incumbent 剪枝。
                if state.incumbent_value is not None and (
                    (maximizing and bound_value <= state.incumbent_value)
                    or (not maximizing and bound_value >= state.incumbent_value)
                ):
                    state.stats.nodes_pruned_by_bound += 1
                elif _is_integer_point(lp.point):
                    state.stats.nodes_integer += 1
                    if _better(bound_value, state.incumbent_value, maximizing):
                        state.incumbent = dict(lp.point)
                        state.incumbent_value = bound_value
                        # incumbent 更新后，frontier 中可剪枝节点惰性处理。
                else:
                    branched = self._branch(
                        problem, state, node, lp.point, bound_value, False
                    )
                    if branched is not None:
                        state.bound = self._global_bound(state)
                        continue

            state.bound = self._global_bound(state)

        state.finished = not state.frontier and stop_reason is None
        return self._finalize(problem, state, stop_reason), state

    def _branch(
        self,
        problem,
        state: SearchState,
        node: Node,
        point,
        bound: Fraction | None,
        unbounded: bool,
    ) -> int | None:
        if point is None or all(
            point[v.name].denominator == 1 for v in problem.variables
        ):
            return None
        name = self.strategy.choose(problem, point)
        value = point[name]
        floor = value.__floor__()
        terms = frozenset({name: Fraction(1)}.items())
        child_extra_lo = node.extra + ((terms, ConstraintSense.LE, floor),)
        child_extra_hi = node.extra + ((terms, ConstraintSense.GE, floor + 1),)
        children = (
            Node(state.next_node_id, child_extra_lo, bound, unbounded),
            Node(state.next_node_id + 1, child_extra_hi, bound, unbounded),
        )
        state.next_node_id += 2
        state.frontier.extend(children)  # 先压上界侧，优先探索下界侧
        state.stats.nodes_branched += 1
        return 1

    def _global_bound(self, state: SearchState) -> Fraction | None:
        """所有开放子问题继承界限中的最佳者。

        任一开放子问题松弛无界时无法给出有限界限，返回 None；
        frontier 关闭后界限即为 incumbent（最优值）。
        """
        if not state.frontier:
            return state.incumbent_value
        if any(n.unbounded for n in state.frontier):
            return None
        finite = [n.bound for n in state.frontier if n.bound is not None]
        if not finite:
            return state.incumbent_value
        return max(finite) if self._maximizing else min(finite)

    def _finalize(
        self, problem: ILP, state: SearchState, stop_reason: str | None
    ) -> Result:
        stats = state.stats
        if state.finished and state.frontier == [] and state.incumbent is not None:
            return Result(
                SolveStatus.OPTIMAL,
                dict(state.incumbent),
                state.incumbent_value,
                state.incumbent_value,
                CertificateType.OPTIMALITY,
                {
                    "reason": "所有节点已探索或被剪枝，界限与可行解重合",
                    "stats": stats.__dict__,
                },
            )
        if state.finished and state.incumbent is None:
            return Result(
                SolveStatus.INFEASIBLE,
                None,
                None,
                None,
                CertificateType.EMPTY_FEASIBLE_SET,
                {"reason": "根松弛或所有分支均不可行", "stats": stats.__dict__},
            )
        # 被预算/取消打断：只报告已知最好结果。
        status = SolveStatus(stop_reason) if stop_reason else SolveStatus.CANCELLED
        return Result(
            status,
            dict(state.incumbent) if state.incumbent else None,
            state.incumbent_value,
            None,
            CertificateType.BEST_KNOWN,
            {
                "reason": "搜索未完成；未探索分支仍存在，不能声称最优",
                "open_nodes": len(state.frontier),
                "stats": stats.__dict__,
            },
        )

    def _unbounded(
        self,
        problem: ILP,
        state: SearchState,
        feasible: dict,
        ray: dict[str, int],
        gain: Fraction,
    ) -> Result:
        return Result(
            SolveStatus.UNBOUNDED,
            dict(feasible),
            None,
            None,
            CertificateType.UNBOUNDED_RAY,
            {
                "feasible_point": {k: str(v) for k, v in feasible.items()},
                "integer_ray": {k: str(v) for k, v in ray.items()},
                "objective_gain_per_step": str(gain),
                "stats": state.stats.__dict__,
            },
        )
