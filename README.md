# Python 优化求解平台

当前版本支持整数变量、线性目标和 `<=` 线性约束。变量可以用 `Variable(name, lower, None)` 表示缺少有限上界；模型定义是不可变对象，搜索状态、结果和证明分别保存，不与问题描述混写。

## 核心结构

- `Problem` / `Variable` / `Constraint`：不可变问题定义，带规范化后的 SHA-256 指纹。
- `BranchAndBound`：精确整数区间分支搜索，不使用浮点运算。
- `BranchingStrategy`：只决定分支变量；换策略不会改变问题含义或最优值。
- `SearchState`：待开节点、已关闭分支、当前最好整数解和节点计数，可序列化为检查点。
- `SolveResult`：明确区分 `optimal`、`infeasible`、`best_known`、`unsolved` 和 `unbounded`。
- `SearchStore`：分别写入 `problem.json`、`search.json`、`result.json`、`proof.json`。
- `review_proof` / `SearchStore.review_saved`：重新读取并独立核对问题、搜索、结果和证据，返回具体不匹配原因。

## 结果语义

预算耗尽或取消时，只要仍有待开分支，结果只能是 `best_known` 或 `unsolved`，不会标记为最优。每个待开节点保留其当前乐观界，恢复后继续搜索，已关闭分支记录不会被重新探索。

空可行域在线性预分析或所有整数分支关闭后返回 `infeasible`。存在目标改进方向时，只有同时找到可行整数点和整数回收方向才返回 `unbounded`；证据中的方向必须满足每条约束 `A d <= 0`，并按目标方向严格改进。预算耗尽、取消，或尚未找到整数见证时都不会被当作无界证明。

若线性多面体在目标方向有界，求解器会用精确有理数计算无界变量的有限上界，再进入原有有限范围整数分支搜索。若仍存在目标中性但变量方向无界、且整数结论尚未完成证书化，结果保持 `unsolved`，不会误报有限最优或无界。

```python
unbounded = Problem("free-x", [Variable("x", 0, None)], {"x": 1})
result = BranchAndBound(unbounded).solve()
# result.status == ResultStatus.UNBOUNDED
# result.best_assignment 是可重新代入的整数点
# result.direction 是可逐条检查 A d <= 0 和 c d > 0 的整数方向
```

复核保存的运行：

```python
report = SearchStore("runs/example").review_saved()
if not report.valid:
    print("\n".join(report.errors))
```

把修改后的约束传给 `review_saved(changed_problem)` 只会复核，不会覆盖保存的原问题和旧证明；指纹不匹配会作为错误明确报告。

## 参数扫描

`ParameterSweep` 比较同一问题在单个参数（目标系数或约束边界）取不同整数值时的结论。每个参数点在 `points/<值>/` 下独立保存问题、搜索状态、结果和证明，结论必须对当前参数重新求解并复核通过才标记为 `verified`；相邻点的最优解只作为暖启动候选，注入前会用当前参数重新做可行性检查。

```python
from optplatform import Constraint, ParameterAxis, ParameterSweep, Problem, Variable

problem = Problem(
    "capacity",
    [Variable("x", 0, 5)],
    {"x": 1},
    [Constraint("cap", {"x": 1}, 0)],
)
sweep = ParameterSweep(
    "runs/cap-sweep",
    problem,
    ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap"),
    [-1, 0, 3],
)
sweep.run()
sweep.summary()
# {'verified': {'infeasible': [-1], 'optimal': [0, 3]}, 'unresolved': [], 'pending': [], ...}
```

- 预算耗尽或取消时该点保持未核实并留在 `pending`，恢复 `run()` 后重求；求解器自身无法证书化的点（如目标中性无界方向）记入 `unresolved`，不会被连成确定区间。
- `summary()` 只列出逐一核实的离散点，明确不声称点与点之间的区间结论。
- `add_checkpoints(values)` 在结论变化附近追加检查点；已核实的点不会重求。
- `run(max_points=k)` 可暂停，重新构造 `ParameterSweep` 会从 `sweep.json` 清单恢复进度；清单与基准问题或参数轴不匹配会报错。
- `sweep.derive(已核实值, 新轴, 新取值, 新目录)` 从一个已核实点派生另一组参数变化，原扫描记录保持可查。

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

检查点示例见 `examples/small_example.py`。

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

测试包含小规模穷举对照、无界/有限最优/不可行三类无界变量问题、预算耗尽后的恢复、预算/取消后的非最优声明、检查点恢复、问题指纹校验和独立证明文件校验。
