"""命令行：python -m optplatform solve <example> --dir <checkpoint-dir>。"""

from __future__ import annotations

import argparse
import json

from .branch_bound import BranchBoundSolver, FirstFractional, MostFractional
from .examples import EXAMPLES, load_example
from .runner import run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="optplatform", description="整数线性规划分支搜索")
    sub = parser.add_subparsers(dest="cmd", required=True)

    solve = sub.add_parser("solve", help="求解或从检查点继续")
    solve.add_argument("example", choices=sorted(EXAMPLES))
    solve.add_argument("--dir", required=True, help="检查点目录")
    solve.add_argument("--max-nodes", type=int, default=None, help="本次运行节点预算")
    solve.add_argument("--cancel-file", default=None, help="该文件存在时取消搜索")
    solve.add_argument(
        "--strategy", choices=("most-fractional", "first-fractional"), default="most-fractional"
    )

    ls = sub.add_parser("list", help="列出内置示例")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        for name, desc in EXAMPLES.items():
            print(f"{name:16s} {desc}")
        return 0

    problem = load_example(args.example)
    strategy = FirstFractional() if args.strategy == "first-fractional" else MostFractional()
    result, state, resumed = run(
        problem,
        args.dir,
        solver=BranchBoundSolver(strategy),
        max_nodes=args.max_nodes,
        cancel_file=args.cancel_file,
    )
    payload = {
        "resumed": resumed,
        "status": result.status.value,
        "certificate": result.certificate.value,
        "incumbent": None if result.incumbent is None else {
            k: str(v) for k, v in result.incumbent.items()
        },
        "objective": None if result.objective is None else str(result.objective),
        "bound": None if result.bound is None else str(result.bound),
        "open_nodes": len(state.frontier),
        "nodes_opened": state.stats.nodes_opened,
        "claim": result.assert_claim(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
