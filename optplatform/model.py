"""问题描述层：带上下界的整数线性规划模型。

本模块只描述问题，不包含任何搜索算法。任何搜索策略都只读取这里的
不可变数据结构，因此更换策略不会改变问题含义。

形式上，一个 ILP 为::

    optimize  sum(c[j] * x[j])
    s.t.      对每条约束 lhs <= rhs、lhs >= rhs 或 lhs == rhs
              lb[j] <= x[j] <= ub[j]   (ub 可为 None，表示 +inf)
              x[j] 为整数
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Iterable, Mapping, Sequence


def _num(value) -> Fraction:
    """把 int/str/Fraction 统一转成精确有理数。"""
    return value if isinstance(value, Fraction) else Fraction(value)


class ObjectiveSense(str, Enum):
    MAX = "max"
    MIN = "min"


class VarType(str, Enum):
    INTEGER = "integer"


class ConstraintSense(str, Enum):
    LE = "<="
    GE = ">="
    EQ = "=="


@dataclass(frozen=True)
class Var:
    """决策变量：名字、整数类型、有限或无限上下界。"""

    name: str
    lb: Fraction = Fraction(0)
    ub: Fraction | None = None
    vtype: VarType = VarType.INTEGER

    def __post_init__(self) -> None:
        if self.ub is not None and self.ub < self.lb:
            raise ValueError(f"变量 {self.name!r} 的上界 {self.ub} 小于下界 {self.lb}")

    @property
    def is_finite_bounded(self) -> bool:
        return self.ub is not None


@dataclass(frozen=True)
class Constraint:
    """线性约束 sum(terms[k] * x[name[k]]) sense rhs。"""

    terms: Mapping[str, Fraction]
    sense: ConstraintSense
    rhs: Fraction
    name: str = ""

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("约束必须包含至少一个变量")

    def normalized_terms(self) -> dict[str, Fraction]:
        """返回统一为 <= 形式的系数（GE 取负，EQ 保持）。"""
        if self.sense is ConstraintSense.GE:
            return {v: -a for v, a in self.terms.items()}
        return dict(self.terms)

    def normalized_rhs(self) -> Fraction:
        return -self.rhs if self.sense is ConstraintSense.GE else self.rhs


@dataclass(frozen=True)
class Objective:
    coefficients: Mapping[str, Fraction]
    sense: ObjectiveSense

    def value(self, point: Mapping[str, Fraction]) -> Fraction:
        return sum((self.coefficients.get(v, Fraction(0)) * val) for v, val in point.items())


@dataclass(frozen=True)
class ILP:
    """完整的整数线性规划问题描述（不可变）。"""

    name: str
    variables: tuple[Var, ...]
    constraints: tuple[Constraint, ...]
    objective: Objective

    def __post_init__(self) -> None:
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError("变量名必须唯一")
        valid = set(names)
        for c in self.constraints:
            unknown = set(c.terms) - valid
            if unknown:
                raise ValueError(f"约束引用了不存在的变量: {sorted(unknown)}")
        unknown = set(self.objective.coefficients) - valid
        if unknown:
            raise ValueError(f"目标引用了不存在的变量: {sorted(unknown)}")

    @classmethod
    def build(
        cls,
        name: str,
        variables: Iterable[Var],
        constraints: Iterable[Constraint],
        objective: Mapping[str, int | Fraction | str],
        sense: ObjectiveSense | str = ObjectiveSense.MAX,
    ) -> "ILP":
        sense = ObjectiveSense(sense)
        obj = Objective({v: _num(a) for v, a in objective.items()}, sense)
        return cls(
            name=name,
            variables=tuple(variables),
            constraints=tuple(constraints),
            objective=obj,
        )

    @property
    def var_names(self) -> list[str]:
        return [v.name for v in self.variables]

    def var(self, name: str) -> Var:
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(name)

    def objective_coeff(self, name: str) -> Fraction:
        return self.objective.coefficients.get(name, Fraction(0))
