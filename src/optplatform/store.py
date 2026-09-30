from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model import Problem
from .proof import Proof, ReviewReport, build_proof, review_proof
from .search import SearchState
from .types import SolveResult


ARCHIVED_NAMES = ("problem.json", "search.json", "result.json", "proof.json")


class SearchStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.problem_path = self.directory / "problem.json"
        self.search_path = self.directory / "search.json"
        self.result_path = self.directory / "result.json"
        self.proof_path = self.directory / "proof.json"
        self.archive_dir = self.directory / "history"

    def save_problem(self, problem: Problem) -> None:
        self._archive_if_stale(problem.canonical_hash())
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
        report = review_proof(problem, search, proof)
        if not report.valid:
            raise RuntimeError("internal proof review failed: " + "; ".join(report.problems))
        _atomic_write_json(self.proof_path, proof.to_dict())
        return proof

    def load_proof(self) -> Proof:
        return Proof.from_dict(_read_json(self.proof_path))

    def review(self) -> ReviewReport:
        """Reload the saved problem/state/proof from disk and re-check the conclusion."""

        problem = self.load_problem()
        proof = self.load_proof()
        search = self.load_state() if self.search_path.exists() else None
        return review_proof(problem, search, proof)

    def _archive_if_stale(self, new_hash: str) -> None:
        if not self.problem_path.exists():
            return
        existing = Problem.from_dict(_read_json(self.problem_path))
        if existing.canonical_hash() == new_hash:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        destination = self.archive_dir / f"{stamp}-{existing.canonical_hash()[:12]}"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ARCHIVED_NAMES:
            source = self.directory / name
            if source.exists():
                shutil.move(str(source), str(destination / name))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
