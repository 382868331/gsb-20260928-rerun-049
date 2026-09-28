"""结果与证明层：保存求解结论及其依据，与搜索过程记录分开。

``OPTIMAL`` 带最优性证明（根/开放节点界限与 incumbent 重合、剪枝统计）；
``INFEASIBLE`` 与 ``UNBOUNDED`` 带相应证据；``BUDGET`` / ``CANCELLED``
只报告已知最好结果，绝不断言最优。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from enum import Enum
from typing import Any


class CertificateType(str, Enum):
    OPTIMALITY = "optimality"
    EMPTY_FEASIBLE_SET = "empty_feasible_set"
    UNBOUNDED_RAY = "unbounded_ray"
    BEST_KNOWN = "best_known"  # 预算耗尽或取消：不是最优性证明


@dataclass(frozen=True)
class Result:
    status: "SolveStatusLike"
    incumbent: dict[str, Fraction] | None
    objective: Fraction | None
    bound: Fraction | None  # max 为上界、min 为下界；None 表示尚未建立
    certificate: CertificateType
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def is_proven(self) -> bool:
        return self.certificate in (
            CertificateType.OPTIMALITY,
            CertificateType.EMPTY_FEASIBLE_SET,
            CertificateType.UNBOUNDED_RAY,
        )

    def gap(self) -> Fraction | None:
        if self.objective is None or self.bound is None:
            return None
        return abs(self.bound - self.objective)

    def assert_claim(self) -> str:
        """给出人类可读的结论强度说明。"""
        if self.certificate is CertificateType.BEST_KNOWN:
            return (
                f"{self.status.value}: 仅报告已知最好可行解，目标值 {self.objective}；"
                "搜索未完成，不声称最优。"
            )
        if self.certificate is CertificateType.OPTIMALITY:
            return f"OPTIMAL: 最优值 {self.objective}，界限重合。"
        if self.certificate is CertificateType.EMPTY_FEASIBLE_SET:
            return "INFEASIBLE: 可行域为空（松弛无解）。"
        return "UNBOUNDED: 存在整数可行点与整数改进射线，目标无界。"


# 避免与 branch_bound 产生循环导入，仅做结构类型约束。
class SolveStatusLike(str, Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    BUDGET = "budget_exhausted"
    CANCELLED = "cancelled"
