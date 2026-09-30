from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence


Number = int | Fraction


class InvalidProblemError(ValueError):
    pass


@dataclass(frozen=True)
class Variable:
    name: str
    lower: int
    upper: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise InvalidProblemError("variable name must be a non-empty string")
        if not _is_int(self.lower) or (self.upper is not None and not _is_int(self.upper)):
            raise InvalidProblemError(f"variable {self.name!r} bounds must be integers")
        if self.upper is not None and self.lower > self.upper:
            raise InvalidProblemError(f"variable {self.name!r} has an empty domain")


@dataclass(frozen=True)
class Constraint:
    name: str
    coefficients: Mapping[str, Number]
    rhs: Number

    def __post_init__(self) -> None:
        if not self.name:
            raise InvalidProblemError("constraint name must be non-empty")


class ObjectiveSense:
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True)
class Problem:
    name: str
    variables: tuple[Variable, ...]
    objective: Mapping[str, Number]
    constraints: tuple[Constraint, ...]
    sense: str = ObjectiveSense.MAXIMIZE

    def __init__(
        self,
        name: str,
        variables: Sequence[Variable],
        objective: Mapping[str, Number],
        constraints: Sequence[Constraint] = (),
        sense: str = ObjectiveSense.MAXIMIZE,
    ) -> None:
        if not name:
            raise InvalidProblemError("problem name must be non-empty")
        if not variables:
            raise InvalidProblemError("problem must contain at least one variable")
        if len({variable.name for variable in variables}) != len(variables):
            raise InvalidProblemError("variable names must be unique")
        if len({constraint.name for constraint in constraints}) != len(constraints):
            raise InvalidProblemError("constraint names must be unique")
        if sense not in (ObjectiveSense.MAXIMIZE, ObjectiveSense.MINIMIZE):
            raise InvalidProblemError("objective sense must be maximize or minimize")

        names = {variable.name for variable in variables}
        cleaned_objective = {key: Fraction(value) for key, value in objective.items() if Fraction(value) != 0}
        unknown = set(cleaned_objective) - names
        if unknown:
            raise InvalidProblemError(f"unknown objective variables: {sorted(unknown)}")

        cleaned_constraints = []
        for constraint in constraints:
            coefficients = {
                key: Fraction(value)
                for key, value in constraint.coefficients.items()
                if Fraction(value) != 0
            }
            unknown = set(coefficients) - names
            if unknown:
                raise InvalidProblemError(
                    f"constraint {constraint.name!r} references unknown variables: {sorted(unknown)}"
                )
            cleaned_constraints.append(
                Constraint(constraint.name, coefficients, Fraction(constraint.rhs))
            )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "variables", tuple(variables))
        object.__setattr__(self, "objective", cleaned_objective)
        object.__setattr__(self, "constraints", tuple(cleaned_constraints))
        object.__setattr__(self, "sense", sense)

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(variable.name for variable in self.variables)

    def initial_bounds(self) -> dict[str, tuple[int, int]]:
        if any(variable.upper is None for variable in self.variables):
            raise InvalidProblemError("an unbounded problem must first be normalized for branch search")
        return {variable.name: (variable.lower, variable.upper) for variable in self.variables}

    def has_finite_bounds(self) -> bool:
        return all(variable.upper is not None for variable in self.variables)

    def polyhedron_rows(self) -> list[tuple[dict[str, Number], Number]]:
        rows: list[tuple[dict[str, Number], Number]] = [
            (dict(constraint.coefficients), constraint.rhs)
            for constraint in self.constraints
        ]
        for variable in self.variables:
            rows.append(({variable.name: -1}, -variable.lower))
            if variable.upper is not None:
                rows.append(({variable.name: 1}, variable.upper))
        return rows

    def canonical_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def objective_value(self, assignment: Mapping[str, int]) -> Fraction:
        return sum(
            Fraction(self.objective.get(name, 0)) * Fraction(assignment[name])
            for name in self.variable_names
        )

    def is_feasible(self, assignment: Mapping[str, int]) -> bool:
        for variable in self.variables:
            value = assignment[variable.name]
            if not _is_int(value) or value < variable.lower:
                return False
            if variable.upper is not None and value > variable.upper:
                return False
        for constraint in self.constraints:
            left = sum(
                Fraction(coefficient) * Fraction(assignment[name])
                for name, coefficient in constraint.coefficients.items()
            )
            if left > Fraction(constraint.rhs):
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sense": self.sense,
            "variables": [
                {"name": variable.name, "lower": variable.lower, "upper": variable.upper}
                for variable in self.variables
            ],
            "objective": {
                name: _number_to_string(value)
                for name, value in sorted(self.objective.items())
            },
            "constraints": [
                {
                    "name": constraint.name,
                    "coefficients": {
                        name: _number_to_string(value)
                        for name, value in sorted(constraint.coefficients.items())
                    },
                    "rhs": _number_to_string(constraint.rhs),
                }
                for constraint in self.constraints
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "Problem":
        return cls(
            data["name"],
            [
                Variable(
                    item["name"],
                    int(item["lower"]),
                    None if item.get("upper") is None else int(item["upper"]),
                )
                for item in data["variables"]
            ],
            {
                name: Fraction(str(value))
                for name, value in data.get("objective", {}).items()
            },
            [
                Constraint(
                    item["name"],
                    {name: Fraction(str(value)) for name, value in item["coefficients"].items()},
                    Fraction(str(item["rhs"])),
                )
                for item in data.get("constraints", [])
            ],
            data.get("sense", ObjectiveSense.MAXIMIZE),
        )


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number_to_string(value: Number) -> str:
    return str(Fraction(value))
