"""约束交叉导致内部最优解变化：有效复用与必须重新求解的区别。

模型：maximize 2x + y，x,y in [0, 4]
    floor: x + y >= 1
    c1:    x + 2y <= b1   (x 轴参数)
    c2:    2x + y <= b2   (y 轴参数)
    cap:   x + y <= t     (新测量把 t 从 6 收紧到 1)

c1 与 c2 的交叉决定内部最优解；cap 收紧后交叉点被切掉，
可行域退化为 x + y = 1 的线段，内部最优解随之改变。
"""

import shutil
from pathlib import Path

from optplatform import (
    Constraint,
    ParameterAxis,
    ParameterStudy2D,
    Problem,
    SearchStore,
    Variable,
)


def make_base(cap_rhs):
    return Problem(
        "crossing",
        [Variable("x", 0, 4), Variable("y", 0, 4)],
        {"x": 2, "y": 1},
        [
            Constraint("floor", {"x": -1, "y": -1}, -1),
            Constraint("c1", {"x": 1, "y": 2}, 0),
            Constraint("c2", {"x": 2, "y": 1}, 0),
            Constraint("cap", {"x": 1, "y": 1}, cap_rhs),
        ],
    )


AXIS_X = ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "c1")
AXIS_Y = ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "c2")
VALUES_X = [0, 1, 2, 3]
VALUES_Y = [2, 3]


def best_of(run_dir):
    return SearchStore(run_dir).load_result().best_assignment


def main():
    root = Path("runs/constraint-crossing")
    shutil.rmtree(root, ignore_errors=True)

    old = ParameterStudy2D(root / "v1-cap6", make_base(6), AXIS_X, VALUES_X, AXIS_Y, VALUES_Y)
    old.run()
    print("旧模型 (cap: x+y<=6) 已核实:", old.summary()["verified"])
    print("  (3,3) 处旧最优解:", best_of(old.point(3, 3).run_dir))

    tightened = old.derive_branch(make_base(1), root / "v2-cap1")
    summary = tightened.reuse_summary()
    print("\n约束收紧 (cap: x+y<=1) 派生分支:")
    print("  直接复用 (不可行在收紧下仍然不可行):", summary["reused"])
    print("  必须重新求解 (旧最优结论不再有效):", summary["requires_resolve"])

    report = tightened.request_region_conclusion((0, 3), (2, 3))
    print("\n角点都有旧结果不足以验证整个区域:")
    print("  全区域申请 verified =", report.verified, " 缺失:", list(report.missing))
    line = tightened.request_region_conclusion((0, 0), (2, 3), status="infeasible")
    print("  退化成线的窄区域 (x=0) verified =", line.verified)

    tightened.run()
    print("\n重新求解后 (3,3) 处新最优解:", best_of(tightened.point(3, 3).run_dir))
    print("  可行域退化为 x+y=1 的线段，最优从 (1,1) 变为 (1,0)")
    print("  复核:", tightened.review_saved())

    relaxed = old.derive_branch(make_base(8), root / "v3-cap8")
    print("\n约束放宽 (cap: x+y<=8): 任何结论都不可复用:",
          relaxed.reuse_summary()["reused"])

    objective = old.derive_branch(
        make_base(6).__class__(
            "crossing",
            make_base(6).variables,
            {"x": 3, "y": 1},
            make_base(6).constraints,
        ),
        root / "v4-objective",
    )
    print("目标变化: 只复用不可行结论:", objective.reuse_summary()["reused"])


if __name__ == "__main__":
    main()
