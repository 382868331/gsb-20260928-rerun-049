"""精确有理两阶段单纯形法，求解整数问题的线性松弛。

所有数值使用 :class:`fractions.Fraction`，不存在浮点误差，因此界限、
剪枝与整数判定都是可证明的。统一把变量平移为非负变量，约束统一为
``<=`` 行；等号拆成一对不等式。b<0 的行用人工变量起步（第一阶段）。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from enum import Enum

from .model import ConstraintSense, ILP


class LPStatus(str, Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"


@dataclass(frozen=True)
class LPResult:
    status: LPStatus
    point: dict[str, Fraction] | None = None
    objective: Fraction | None = None
    """无界时给出沿目标改进方向的整数/有理射线方向（原变量空间）。"""
    ray: dict[str, Fraction] | None = None


ZERO = Fraction(0)


def solve_relaxation(problem: ILP, extra: list | None = None) -> LPResult:
    """求解线性松弛。

    extra 中每项为 ``(terms: dict[str, Fraction], sense, rhs)``，表示
    分支时追加到当前节点的约束（作用于原变量）。
    """
    names = problem.var_names
    n = len(names)

    # 平移 x = lb + y，y >= 0；ub 转成行约束 y <= ub - lb。
    shift: list[Fraction] = []
    rows: list[tuple[list[Fraction], Fraction]] = []
    for j, var in enumerate(problem.variables):
        lb = var.lb
        shift.append(lb)
        if var.ub is not None:
            coeff = [ZERO] * n
            coeff[j] = Fraction(1)
            rows.append((coeff, var.ub - lb))

    def add_row(terms, sense: ConstraintSense, rhs: Fraction) -> None:
        coeff = [ZERO] * n
        for vname, a in terms.items():
            coeff[names.index(vname)] += a
        if sense is ConstraintSense.GE:
            coeff = [-a for a in coeff]
            rhs = -rhs
        rhs = rhs - sum(coeff[j] * shift[j] for j in range(n))
        rows.append((coeff, rhs))
        if sense is ConstraintSense.EQ:
            rows.append(([-a for a in coeff], -rhs))

    for c in problem.constraints:
        add_row(dict(c.terms), c.sense, c.rhs)
    for terms, sense, rhs in extra or []:
        add_row(dict(terms), sense, rhs)

    # 目标统一为最大化；平移常数单独保留。
    sign = 1 if problem.objective.sense.value == "max" else -1
    cost = [sign * problem.objective_coeff(names[j]) for j in range(n)]
    const = sum(cost[j] * shift[j] for j in range(n))

    m = len(rows)
    if m == 0:
        # 无任何约束：任一非零正系数方向都使问题无界；否则常数最优。
        q = next((j for j in range(n) if cost[j] > 0), None)
        if q is None:
            point = {names[j]: shift[j] for j in range(n)}
            return LPResult(LPStatus.OPTIMAL, point, const)
        ray = {names[j]: ZERO for j in range(n)}
        ray[names[q]] = Fraction(1)
        return LPResult(LPStatus.UNBOUNDED, None, None, ray)

    # 结构：n 个 y 列、m 个松弛列、m 个人工列、1 个 RHS 列。
    slack0 = n
    art0 = n + m
    ncols = n + 2 * m
    tableau: list[list[Fraction]] = []
    for i, (coeff, rhs) in enumerate(rows):
        row = [ZERO] * ncols + [rhs]
        for j in range(n):
            row[j] = coeff[j]
        row[slack0 + i] = Fraction(1)
        row[art0 + i] = Fraction(1)
        if rhs < 0:
            row = [-a for a in row[:-1]] + [-rhs]
        tableau.append(row)

    # 初始基：rhs>=0 用松弛，rhs<0（取负后）用人工。
    basics: dict[int, int] = {}
    for i, (_, rhs) in enumerate(rows):
        basics[i] = art0 + i if rhs < 0 else slack0 + i

    art_cols = set(range(art0, art0 + m))

    def pivot(col: int) -> None:
        ratios = []
        for i, row in enumerate(tableau):
            if row[col] > 0:
                ratios.append((row[-1] / row[col], i))
        if not ratios:
            raise ZeroDivisionError
        _, r = min(ratios, key=lambda t: (t[0], basics[t[1]]))
        pivot_row = tableau[r]
        a = pivot_row[col]
        tableau[r] = [v / a for v in pivot_row]
        for i, row in enumerate(tableau):
            if i == r:
                continue
            f = row[col]
            if f != 0:
                tableau[i] = [v - f * tableau[r][k] for k, v in enumerate(row)]
        basics[r] = col

    def reduced(costs, allow_cols) -> int | None:
        for col in sorted(allow_cols):
            z = ZERO
            for i, row in enumerate(tableau):
                z += costs[basics[i]] * row[col]
            if costs[col] - z > 0:
                return col
        return None

    all_cols = set(range(ncols))

    # ---------- 第一阶段：最小化人工之和（最大化其相反数） ----------
    cost1 = [ZERO] * ncols
    for col in art_cols:
        cost1[col] = Fraction(-1)

    while True:
        col = reduced(cost1, all_cols)
        if col is None:
            break
        try:
            pivot(col)
        except ZeroDivisionError:
            break  # 第一阶段不会无界（人工有上界），防御性处理

    art_sum = ZERO
    for i, row in enumerate(tableau):
        bcol = basics[i]
        if bcol in art_cols:
            art_sum += row[-1]
    if art_sum > 0:
        return LPResult(LPStatus.INFEASIBLE)

    # 把值为 0 的人工基变量换出；无法换出说明该行冗余，删除。
    drop_rows: set[int] = set()
    for i, col in list(basics.items()):
        if col not in art_cols:
            continue
        row = tableau[i]
        candidate = next(
            (k for k in range(art0) if row[k] != 0 and k not in art_cols), None
        )
        if candidate is None:
            drop_rows.add(i)
        else:
            a = row[candidate]
            tableau[i] = [v / a for v in row]
            for k2, other in enumerate(tableau):
                if k2 != i:
                    f = other[candidate]
                    if f != 0:
                        tableau[k2] = [v - f * tableau[i][t] for t, v in enumerate(other)]
            basics[i] = candidate
    if drop_rows:
        keep = [i for i in range(m) if i not in drop_rows]
        tableau = [tableau[i] for i in keep]
        basics = {r: basics[i] for r, i in enumerate(keep)}

    # ---------- 第二阶段：真正的目标（最大化） ----------
    cost2 = list(cost) + [ZERO] * (ncols - n)
    allow = all_cols - art_cols
    unbounded_col: int | None = None
    while True:
        col = reduced(cost2, allow)
        if col is None:
            break
        try:
            pivot(col)
        except ZeroDivisionError:
            unbounded_col = col
            break

    def primal_vector() -> list[Fraction]:
        values = [ZERO] * ncols
        for i, row in enumerate(tableau):
            bc = basics[i]
            if bc < ncols:
                values[bc] = row[-1]
        return values

    if unbounded_col is not None:
        values = primal_vector()
        current = {names[j]: values[j] + shift[j] for j in range(n)}
        direction = [ZERO] * ncols
        direction[unbounded_col] = Fraction(1)
        for i, row in enumerate(tableau):
            direction[basics[i]] = row[-1] - row[unbounded_col]
        ray = {names[j]: direction[j] for j in range(n)}
        return LPResult(LPStatus.UNBOUNDED, current, None, ray)

    values = primal_vector()
    obj_val = sum(cost[j] * values[j] for j in range(n)) + const
    point = {names[j]: values[j] + shift[j] for j in range(n)}
    return LPResult(LPStatus.OPTIMAL, point, obj_val)
