# 优化求解平台（第一版）

带上下界的整数线性规划（ILP）求解器。核心约束是**问题描述、搜索过程、
结果证明三者分开保存**，并且以后更换搜索策略时不能改变问题含义。

## 架构

| 层 | 模块 | 职责 |
| --- | --- | --- |
| 问题描述 | `optplatform/model.py` | 不可变的 `ILP / Var / Constraint / Objective`，与算法无关 |
| 线性松弛 | `optplatform/lp.py` | 精确有理（`Fraction`）两阶段单纯形，返回最优/不可行/无界射线 |
| 搜索过程 | `optplatform/branch_bound.py` | 可恢复分支定界：`SearchState`、节点界限、剪枝统计、预算与取消 |
| 结果证明 | `optplatform/proof.py` | `OPTIMAL / INFEASIBLE / UNBOUNDED` 的证据；中断只能是 `BEST_KNOWN` |
| 检查点 | `optplatform/checkpoint.py` | `problem.json`、`search.json`、`result.json` 三个独立文件 |
| 入口 | `optplatform/runner.py`、`optplatform/__main__.py` | 运行、中断、继续 |

策略通过 `BranchingStrategy` 协议注入（内置 `MostFractional` 与
`FirstFractional`），只影响分支顺序，不影响问题数据与最终结论。

## 结论语义

- `OPTIMAL`：frontier 全部探索或被界限剪枝，松弛界限与 incumbent 重合。
- `INFEASIBLE`：第一阶段单纯形证明松弛不可行，故整数可行域为空。
- `UNBOUNDED`：给出整数可行点与整数改进射线（有理衰退方向通分放大），
  沿射线走任意非负整数步都保持可行且目标持续改进。
- `BUDGET` / `CANCELLED`：只报告已知最好可行解与开放节点数，证书为
  `best_known`，**不声称最优**；`bound` 不伪造收紧。

## 检查点与恢复

```
<dir>/problem.json   # 问题含义（含 sha256 指纹）
<dir>/search.json    # frontier（每个节点带继承界限/无界标记）、incumbent、统计
<dir>/result.json    # 仅在得到证明后写入
```

待处理节点携带其松弛界限与无界标记，恢复后这些节点可在不重新求解父节点
的情况下被继承界限直接剪枝，因此**已证明不必探索的分支不会丢失**。
恢复时用问题指纹校验：问题变了会拒绝加载旧检查点。写入采用临时文件
原子替换。

## 用法

```bash
python -m optplatform list
python -m optplatform solve knapsack --dir runs/k1
python -m optplatform solve knapsack --dir runs/k1 --max-nodes 1   # 预算耗尽
touch /tmp/STOP
python -m optplatform solve knapsack --dir runs/k1 --cancel-file /tmp/STOP
python -m optplatform solve knapsack --dir runs/k1                # 继续
```

代码内：

```python
from optplatform import ILP, Var, Constraint, ConstraintSense, ObjectiveSense
from optplatform import BranchBoundSolver
from optplatform.branch_bound import Budget

p = ILP.build(
    "demo",
    [Var("x", 0, 5), Var("y", 0, 5)],
    [Constraint({"x": 1, "y": 1}, ConstraintSense.LE, 5)],
    {"x": 5, "y": 4}, ObjectiveSense.MAX,
)
result, state = BranchBoundSolver().search(p, budget=Budget(max_nodes=10))
```

## 验证

`tests/` 对 40 个随机小问题和内置例子做**逐点穷举交叉核对**，并覆盖空
可行域、无界射线、预算/取消不冒充最优、检查点恢复不重复开节点、策略
切换结论一致、问题指纹不匹配拒绝恢复等。

```bash
python -m pytest -q
```

## 当前版本范围

- 整数变量（含 `ub=None` 的无限上界/下界）、`<=/>=/==` 约束、最大化/最小化。
- 无第三方依赖，纯标准库、精确有理运算。
- 分支策略可插拔；后续可新增割平面等策略而不改动 `model.py` 的问题含义。
