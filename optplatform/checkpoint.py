"""检查点：问题描述、搜索过程、结果证明分开保存。

目录布局::

    <dir>/problem.json      问题含义（与策略无关，重建求解时校验一致）
    <dir>/search.json       搜索过程（frontier、incumbent、界限、统计）
    <dir>/result.json       完成后的证明；未完成时不存在

写入采用临时文件 + 原子替换，避免半写文件损坏检查点。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from .branch_bound import Node, SearchState, SearchStats, SolveStatus
from .model import (
    Constraint,
    ConstraintSense,
    ILP,
    Objective,
    ObjectiveSense,
    Var,
    VarType,
)
from .proof import CertificateType, Result

PROBLEM_FILE = "problem.json"
SEARCH_FILE = "search.json"
RESULT_FILE = "result.json"


def _f(value: Fraction | None):
    return None if value is None else str(value)


def _point(point: dict[str, Fraction] | None):
    return None if point is None else {k: str(v) for k, v in point.items()}


def _parse_point(data) -> dict[str, Fraction] | None:
    return None if data is None else {k: Fraction(v) for k, v in data.items()}

# ---------------------------------------------------------------- 问题


def problem_to_dict(problem: ILP) -> dict:
    return {
        "name": problem.name,
        "variables": [
            {"name": v.name, "lb": str(v.lb), "ub": _f(v.ub), "vtype": v.vtype.value}
            for v in problem.variables
        ],
        "constraints": [
            {
                "name": c.name,
                "terms": {k: str(v) for k, v in c.terms.items()},
                "sense": c.sense.value,
                "rhs": str(c.rhs),
            }
            for c in problem.constraints
        ],
        "objective": {
            "sense": problem.objective.sense.value,
            "coefficients": {
                k: str(v) for k, v in problem.objective.coefficients.items()
            },
        },
    }


def problem_from_dict(data: dict) -> ILP:
    variables = tuple(
        Var(
            name=v["name"],
            lb=Fraction(v["lb"]),
            ub=None if v["ub"] is None else Fraction(v["ub"]),
            vtype=VarType(v["vtype"]),
        )
        for v in data["variables"]
    )
    constraints = tuple(
        Constraint(
            terms={k: Fraction(v) for k, v in c["terms"].items()},
            sense=ConstraintSense(c["sense"]),
            rhs=Fraction(c["rhs"]),
            name=c.get("name", ""),
        )
        for c in data["constraints"]
    )
    objective = Objective(
        coefficients={k: Fraction(v) for k, v in data["objective"]["coefficients"].items()},
        sense=ObjectiveSense(data["objective"]["sense"]),
    )
    return ILP(
        name=data["name"],
        variables=variables,
        constraints=constraints,
        objective=objective,
    )


def problem_fingerprint(problem: ILP) -> str:
    blob = json.dumps(problem_to_dict(problem), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 搜索过程


def state_to_dict(problem: ILP, state: SearchState) -> dict:
    return {
        "problem_fingerprint": problem_fingerprint(problem),
        "next_node_id": state.next_node_id,
        "finished": state.finished,
        "incumbent": _point(state.incumbent),
        "incumbent_value": _f(state.incumbent_value),
        "bound": _f(state.bound),
        "stats": state.stats.__dict__,
        "frontier": [
            {
                "node_id": n.node_id,
                "bound": _f(n.bound),
                "unbounded": n.unbounded,
                "extra": [
                    {
                        "terms": {k: str(v) for k, v in terms},
                        "sense": sense.value,
                        "rhs": str(rhs),
                    }
                    for terms, sense, rhs in n.extra
                ],
            }
            for n in state.frontier
        ],
    }


def state_from_dict(data: dict) -> SearchState:
    frontier = []
    for n in data["frontier"]:
        extra = tuple(
            (
                frozenset({k: Fraction(v) for k, v in e["terms"].items()}.items()),
                ConstraintSense(e["sense"]),
                Fraction(e["rhs"]),
            )
            for e in n["extra"]
        )
        frontier.append(
            Node(
                node_id=n["node_id"],
                extra=extra,
                bound=None if n["bound"] is None else Fraction(n["bound"]),
                unbounded=n["unbounded"],
            )
        )
    return SearchState(
        frontier=frontier,
        next_node_id=data["next_node_id"],
        incumbent=_parse_point(data["incumbent"]),
        incumbent_value=None
        if data["incumbent_value"] is None
        else Fraction(data["incumbent_value"]),
        bound=None if data["bound"] is None else Fraction(data["bound"]),
        finished=data["finished"],
        stats=SearchStats(**data["stats"]),
    )


# ---------------------------------------------------------------- 结果证明


def result_to_dict(problem: ILP, result: Result) -> dict:
    data = {
        "problem_fingerprint": problem_fingerprint(problem),
        "status": result.status.value if hasattr(result.status, "value") else str(result.status),
        "certificate": result.certificate.value,
        "incumbent": _point(result.incumbent),
        "objective": _f(result.objective),
        "bound": _f(result.bound),
        "claim": result.assert_claim(),
        "evidence": result.evidence,
    }
    return data


def result_from_dict(data: dict) -> Result:
    return Result(
        status=SolveStatus(data["status"]),
        incumbent=_parse_point(data["incumbent"]),
        objective=None if data["objective"] is None else Fraction(data["objective"]),
        bound=None if data["bound"] is None else Fraction(data["bound"]),
        certificate=CertificateType(data["certificate"]),
        evidence=data.get("evidence", {}),
    )


# ---------------------------------------------------------------- 存储


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class CheckpointStore:
    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)

    def save_problem(self, problem: ILP) -> None:
        _atomic_write(self.dir / PROBLEM_FILE, problem_to_dict(problem))

    def load_problem(self) -> ILP:
        with open(self.dir / PROBLEM_FILE, encoding="utf-8") as f:
            return problem_from_dict(json.load(f))

    def save_state(self, problem: ILP, state: SearchState) -> None:
        _atomic_write(self.dir / SEARCH_FILE, state_to_dict(problem, state))

    def load_state(self, problem: ILP) -> SearchState:
        with open(self.dir / SEARCH_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if data["problem_fingerprint"] != problem_fingerprint(problem):
            raise ValueError("检查点的问题指纹与当前问题不一致，拒绝恢复")
        return state_from_dict(data)

    def has_state(self) -> bool:
        return (self.dir / SEARCH_FILE).exists()

    def save_result(self, problem: ILP, result: Result) -> None:
        _atomic_write(self.dir / RESULT_FILE, result_to_dict(problem, result))

    def load_result(self) -> Result:
        with open(self.dir / RESULT_FILE, encoding="utf-8") as f:
            return result_from_dict(json.load(f))

    def has_result(self) -> bool:
        return (self.dir / RESULT_FILE).exists()
