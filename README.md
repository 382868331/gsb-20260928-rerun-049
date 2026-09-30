# Python 优化求解平台

当前版本支持整数上下界（任一端可为 `None`，即无界）、线性目标和 `<=` 线性约束的整数线性问题。有限界问题走区间分支限界；含无界端的问题走 Fourier–Motzkin 预处理加倍增扩展盒，与有限问题共用同一个公开入口、结果协议和存储格式。模型定义是不可变对象；搜索状态、结果和证明分别保存，不与问题描述混写。

## 核心结构

- `Problem` / `Variable` / `Constraint`：不可变问题定义，带规范化后的 SHA-256 指纹。
- `BranchAndBound`：精确整数区间分支搜索，不使用浮点运算。
- `BranchingStrategy`：只决定分支变量；换策略不会改变问题含义或最优值。
- `SearchState`：待开节点、已关闭分支、当前最好整数解和节点计数，可序列化为检查点。
- `SolveResult`：明确区分 `optimal`、`infeasible`、`best_known`、`unsolved` 和 `unbounded`。
- `ExactSolver` / `BranchAndBound`：统一公开入口，自动识别无界变量；`unbounded` 只在同时存在可行整数见证点与严格改进的整数回收射线时给出。
- `SearchStore`：分别写入 `problem.json`、`search.json`、`result.json`、`proof.json`；问题指纹变化时旧文件整体移入 `history/`。

## 结果语义

预算耗尽或取消时，只要仍有待开分支或扩展盒尚未给出结论，结果只能是 `best_known` 或 `unsolved`，不会标记为最优或无界。每个待开节点保留其当前乐观界，扩展盒半径记录在检查点中，恢复后继续搜索，已关闭分支记录不会被重新探索。

空可行域在所有分支关闭（有限界问题）或 Fourier–Motzkin 推出 0 <= -1 的 Farkas 非负组合（连续松弛为空）后返回 `infeasible`，证书只引用命名的原始约束行，可独立复核。

含无界端的问题按以下顺序得出确定结论：连续松弛不可行 -> `infeasible`；在不断倍增的有限盒中找到整数可行点且存在满足 `A d <= 0`、目标严格改进的整数回收射线 -> `unbounded`；不存在改进射线且盒内最优解在每个人工盒面上都有整数松弛 -> `optimal`。预算不足永远不会产生后三种确定结论。

## 结论复核

`SearchStore.review()` 从磁盘重新读取 `problem.json` 与 `proof.json`（及可选的 `search.json`），返回 `ReviewReport(valid, problems)`。复核内容包括：证明指纹与当前问题是否一致（约束被修改后旧证明立即失效）、指派是否可行且目标值正确、Farkas 乘数是否非负且确实推出 0 <= -1、无界见证点是否可行、射线是否保持在可行域内并严格改进目标、最优结论的开分支/盒内性条件。也可直接调用 `review_proof(problem, search, proof)`。

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

测试包含小规模穷举对照、空可行域、预算/取消后的非最优声明、检查点恢复、问题指纹校验和独立证明文件校验。
