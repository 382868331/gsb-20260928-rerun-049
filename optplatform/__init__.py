"""整数线性优化求解平台（第一版：带上下界的整数线性规划分支搜索）。

设计原则：问题描述 (:mod:`optplatform.model`)、搜索过程
(:mod:`optplatform.branch_bound`) 与结果证明 (:mod:`optplatform.proof`)
三者彼此独立；更换搜索策略不改变问题含义。
"""

from .model import Constraint, ILP, Objective, ObjectiveSense, Var, VarType
from .branch_bound import BranchBoundSolver, SearchStats, SolveStatus
from .proof import Result

__all__ = [
    "Constraint",
    "ILP",
    "Objective",
    "ObjectiveSense",
    "Var",
    "VarType",
    "BranchBoundSolver",
    "SearchStats",
    "SolveStatus",
    "Result",
]
