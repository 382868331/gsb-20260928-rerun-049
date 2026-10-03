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

## 二维参数研究

`ParameterStudy2D` 同时沿两个参数轴（目标系数或约束边界）变化，在平面的整数格点上逐点求解。每个已检查位置在 `points/<x>__<y>/` 下独立保存问题、搜索状态、结果和证明；只有当前参数组合重新求解并复核通过才标记为 `verified`。

```python
from optplatform import Constraint, ParameterAxis, ParameterStudy2D, Problem, Variable

problem = Problem(
    "band",
    [Variable("x", 0, 6)],
    {"x": 1},
    [Constraint("floor", {"x": -1}, 0), Constraint("cap", {"x": 1}, 6)],
)
study = ParameterStudy2D(
    "runs/band-2d",
    problem,
    ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "floor"),
    [-1, 0],
    ParameterAxis(ParameterAxis.CONSTRAINT_RHS, "cap"),
    [1, 2, 3],
)
study.run()
study.summary()
# verified 只列出逐一核实的离散点；不暗示点与点之间或周边区域已证明
```

- `summary()` 按结论分组列出离散已核实点、未解决点和待算点，并明确不声称任何二维区域结论；显示时不能把相邻同色点连成块来暗示块内都已证明。
- `add_checkpoints([(x, y), ...])` 在结论变化附近追加检查位置；已核实的点不会重求。
- `run(max_points=k)` 可暂停；预算耗尽或取消时该位置保持未核实并留在 `pending`，重新构造 `ParameterStudy2D` 会从 `study.json` 清单恢复后继续。

### 区域结论

`request_region_conclusion(x范围, y范围, status=None)` 申请对一个闭矩形区域给出整体结论；也可以传入 `coordinates=[(x, y), ...]` 选择任意非矩形整数格点子区域。只有所选区域内**每个**整数格点都已逐一核实、且结论一致（或与申请的 `status` 相符）时才记录为区域结论；否则返回未核实报告，列出缺失位置 `missing` 和结论不符位置 `mismatched`，离散点记录保持不变。四个角点结论相同不足以推出内部；边界退化时可把一条线表示为坐标列表，把点表示为单元素列表。非矩形区域只覆盖列出的坐标，不覆盖其外接矩形内的空洞。

```python
report = study.request_region_conclusion((-1, 0), (2, 3), status="optimal")
if report.verified:
    report.conclusion.depends_on  # 该区域结论依赖的全部已核实点
```

只续算一个非矩形子区域时使用 `run_region(coordinates, max_points=None, budget=None)`。列表外的待算格点仍保留在 `pending`，不会被自动补齐；预算中断时正在计算的点也继续留在清单首位。

- 区域结论记录其依赖的每一个已核实点（`depends_on`）、覆盖的精确坐标（`coordinates`）、模型版本（`model_version`）和模型指纹（`model_hash`）。保存重开后 `review_saved()` 会重新核对每个点的证明以及区域依赖是否仍然成立。
- 区域可以相交；同模型版本中结论互相矛盾的相交区域不会被后来结果覆盖，`conflicts()` 会指出共享格点和双方各自依赖的证据点。不同模型版本不会被混为一个结论。
- `derive(x, y, 新轴, 新取值, 新目录)` 从一个已核实位置派生一维 `ParameterSweep`，原研究记录保持可查；从未核实的位置派生会被拒绝。

### 模型分支与复用

`derive_branch(新问题, 新x取值, 新y取值, 新目录, change_type, model_version)` 从已完成的二维研究派生新模型分支。分支有独立目录和模型版本；父研究的区域不会变成新版本的逐点结果。

`request_reused_region_conclusion(coordinates, status=None)` 只在父研究存在**覆盖坐标完全相同**的区域结论，且变化类型允许状态传递时，才在分支中记录一条复用结论。复用结论通过 `reused_from` 记录父目录、父模型版本、父模型指纹、区域、证据点和变化类型；这些坐标在分支中仍可保留为待检查点，不会伪装成已经在新版本重新求解。

| 变化类型 | 可安全传递的结论 | 必须重新求解 |
| --- | --- | --- |
| `ModelChange.PARAMETER_RANGE` | 相同参数坐标上的 `optimal`、`infeasible`、`unbounded` | 新加入范围 |
| `ModelChange.CONSTRAINT_TIGHTENED` | 旧模型已 `infeasible` | 旧模型 `optimal`、`unbounded` 或新约束交叉处 |
| `ModelChange.CONSTRAINT_RELAXED` | 旧模型已 `unbounded` | 旧模型 `optimal`、`infeasible` |
| `ModelChange.OBJECTIVE_CHANGED` | 旧模型已 `infeasible` | 任何旧可行点的最优值或无界性 |

例如父模型有两个独立上界 `x <= a`、`y <= b`，最大化 `x+y` 时内点 `(a,b)=(2,2)` 的最优值是 4；分支加入交叉约束 `x+y <= 2` 后，角点参数仍可能保持旧结果，但 `(2,2)` 的最优值变为 2。该区域不能从父模型复用，必须逐点重新求解。`tests/test_grid.py` 中的 `test_crossing_constraint_changes_interior_optimum_and_requires_resolve` 同时展示了这种区别和上述四类传递规则。

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
