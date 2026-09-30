# Python 优化求解平台

当前版本支持至少一侧有限的整数域、线性目标和 `<=` 线性约束的整数线性问题。`lower` 或 `upper` 可写为 `None`（JSON 中为 `null`）表示该方向无限；不允许两侧同时无限。模型定义是不可变对象；搜索状态、结果和证明分别保存，不与问题描述混写。

## 核心结构

- `Problem` / `Variable` / `Constraint`：不可变问题定义，带规范化后的 SHA-256 指纹。
- `BranchAndBound`：精确整数区间分支、约束区间传播和整数射线证明，不使用浮点运算。
- `BranchingStrategy`：只决定分支变量；换策略不会改变问题含义或最优值。
- `SearchState`：待开节点、已关闭分支、当前最好整数解和节点计数，可序列化为检查点。
- `SolveResult`：明确区分 `optimal`、`infeasible`、`best_known`、`unsolved` 和 `unbounded`，并在 `improving_direction` 中保存改善方向。
- `SearchStore`：分别写入 `problem.json`、`search.json`、`result.json`、`proof.json`。

## 结果语义

预算耗尽或取消时，只要仍有待开分支，结果只能是 `best_known` 或 `unsolved`，不会标记为最优，也不会因为搜索尚未停止而标记无界。每个待开节点保留其当前乐观界，恢复后继续搜索，已关闭分支记录不会被重新探索。

空可行域在所有分支关闭后返回 `infeasible`。只有找到可独立核对的整数可行点 `x` 和非零整数方向 `d`，且对每个约束的方向斜率 `<= 0`、目标沿求解方向严格改善时，才返回 `unbounded`。这证明 `x + t*d`（`t` 为任意非负整数）始终可行并无限改善；有限截断问题的最优值不会被当作原问题最优值。

检查点会同时保存当前 incumbent、开放分支、已关闭节点、无界可行点和改善方向。恢复后已完成的最优、不可行或无界证明仍可独立验证。

## 快速开始

```python
from optplatform import BranchAndBound, Constraint, Problem, Variable

problem = Problem(
    "demo",
    [Variable("x", 0, 2), Variable("y", 0, 2)],
    {"x": 5, "y": 4},
    [Constraint("capacity", {"x": 2, "y": 1}, 3)],
)
result = BranchAndBound(problem).solve()
```

无界示例：

```python
unbounded = Problem(
    "unbounded-demo",
    [Variable("x", 0, None)],
    {"x": 1},
    [Constraint("start_after_3", {"x": -1}, -3)],
)
result = BranchAndBound(unbounded).solve()
assert result.status is ResultStatus.UNBOUNDED
assert result.best_assignment == {"x": 3}
assert result.improving_direction == {"x": 1}
```

检查点示例见 `examples/small_example.py`。

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

测试包含小规模穷举对照、有限截断与原无界问题对照、空可行域、预算/取消后的非最优/非无界声明、整数射线证书、检查点恢复、问题指纹校验和独立证明文件校验。
