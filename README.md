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

`ParameterSweep` 比较同一问题在参数区间内的结论变化。用 `ParameterAxis` 指定一个目标系数（`AxisKind.OBJECTIVE_COEFFICIENT`）或约束边界（`AxisKind.CONSTRAINT_RHS`）及一组离散参数点；每个点独立求解、独立保存 `problem.json` / `search.json` / `result.json` / `proof.json` 和本次预算，结论按 `optimal` / `infeasible` / `unbounded` 归类为已核实，`best_known` / `unsolved` 及未运行的点保留在 `summary().pending` 中。离散点不会被连成确定区间——没有逐点证明的位置一律不算有结论。

相邻参数点会把最近已求解点的可行解作为热启动候选，但候选先对当前参数问题重新做 `is_feasible` 核对，结论仍由当前参数的完整搜索和证明得出，旧证明不覆盖整段范围。`refine(...)` 可在结论变化附近追加检查点；`run(should_stop=...)` 暂停后可用 `ParameterSweep.load(dir)` 断点续跑；`derive(point_index, new_axis, dir)` 从一个已核实点派生另一维参数扫描，原扫描记录保持可查、不被修改。

```python
sweep = ParameterSweep.create(
    problem,
    ParameterAxis(AxisKind.CONSTRAINT_RHS, "cap", [-1, 0, 1, 2]),
    "runs/cap-sweep",
)
summary = sweep.run(SearchBudget(max_nodes=1000))
report = sweep.review_point(0)  # 独立复核某个参数点的证明
```

注意：若多面体在目标方向有界但存在目标中性的无界方向，对应参数点保持 `unsolved`（见“结果语义”），不会被误认为有限最优或无界。示例见 `examples/sweep_example.py`。

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
