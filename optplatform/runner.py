"""把求解器与检查点存储连起来：每次运行都可中断、可恢复。"""

from __future__ import annotations

from pathlib import Path

from .branch_bound import BranchBoundSolver, Budget, SearchState, SolveStatus
from .checkpoint import CheckpointStore
from .model import ILP
from .proof import CertificateType, Result


def run(
    problem: ILP,
    directory: str | Path,
    solver: BranchBoundSolver | None = None,
    max_nodes: int | None = None,
    cancel_file: str | Path | None = None,
) -> tuple[Result, SearchState, bool]:
    """执行一段搜索并落盘。

    返回 ``(result, state, resumed)``。无论是否被预算/取消打断，搜索
    过程都会写入 ``search.json``；只有拿到证明时才写 ``result.json``。
    """
    solver = solver or BranchBoundSolver()
    store = CheckpointStore(directory)
    store.save_problem(problem)

    resumed = False
    if store.has_state():
        state = store.load_state(problem)
        resumed = True
        if state.finished and store.has_result():
            return store.load_result(), state, resumed
    else:
        state = SearchState.initial()

    def cancelled() -> bool:
        return cancel_file is not None and Path(cancel_file).exists()

    result, state = solver.search(
        problem, state, Budget(max_nodes=max_nodes, cancel_event=cancelled if cancel_file else None)
    )
    store.save_state(problem, state)
    if result.is_proven:
        store.save_result(problem, result)
    return result, state, resumed
