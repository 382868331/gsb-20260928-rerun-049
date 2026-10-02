"""参数扫描示例：比较约束边界变化下的结论。"""

import tempfile
from pathlib import Path

from optplatform import (
    AxisKind,
    Constraint,
    ParameterAxis,
    ParameterSweep,
    Problem,
    SearchBudget,
    Variable,
)

problem = Problem(
    "capacity-study",
    [Variable("x", 0, 2)],
    {"x": 1},
    [Constraint("cap", {"x": 1}, 1)],
)

with tempfile.TemporaryDirectory() as tmp:
    sweep = ParameterSweep.create(
        problem,
        ParameterAxis(AxisKind.CONSTRAINT_RHS, "cap", [-1, 0, 1, 2]),
        Path(tmp) / "cap-sweep",
    )
    summary = sweep.run(SearchBudget(max_nodes=1000))
    for point in summary.verified:
        print(f"cap={point.value}: {point.status} (proof in {point.store_dir})")
    for point in summary.pending:
        print(f"cap={point.value}: pending ({point.status})")
