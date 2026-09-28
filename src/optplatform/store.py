from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .model import Problem
from .proof import Proof, build_proof, verify_proof
from .search import SearchState
from .types import SolveResult


class SearchStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.problem_path = self.directory / "problem.json"
        self.search_path = self.directory / "search.json"
        self.result_path = self.directory / "result.json"
        self.proof_path = self.directory / "proof.json"

    def save_problem(self, problem: Problem) -> None:
        _atomic_write_json(self.problem_path, problem.to_dict())

    def load_problem(self) -> Problem:
        return Problem.from_dict(_read_json(self.problem_path))

    def save_state(self, search: SearchState) -> None:
        _atomic_write_json(self.search_path, search.to_dict())

    def load_state(self) -> SearchState:
        return SearchState.from_dict(_read_json(self.search_path))

    def save_result(self, result: SolveResult) -> None:
        _atomic_write_json(self.result_path, result.to_dict())

    def load_result(self) -> SolveResult:
        return SolveResult.from_dict(_read_json(self.result_path))

    def save_proof(self, problem: Problem, search: SearchState, result: SolveResult) -> Proof:
        proof = build_proof(problem, search, result)
        if not verify_proof(problem, search, proof):
            raise RuntimeError("internal proof verification failed")
        _atomic_write_json(self.proof_path, proof.to_dict())
        return proof

    def load_proof(self) -> Proof:
        return Proof.from_dict(_read_json(self.proof_path))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
