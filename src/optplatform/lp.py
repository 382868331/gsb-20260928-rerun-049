from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd
from typing import Mapping, Sequence


Number = int | Fraction


@dataclass(frozen=True)
class FarkasCertificate:
    """Non-negative multipliers deriving 0 <= -1 from the named input rows."""

    multipliers: dict[str, Fraction]

    def to_dict(self) -> dict:
        return {"multipliers": {name: str(value) for name, value in sorted(self.multipliers.items())}}

    @classmethod
    def from_dict(cls, data: Mapping) -> "FarkasCertificate":
        return cls({name: Fraction(str(value)) for name, value in data["multipliers"].items()})


@dataclass(frozen=True)
class FeasibilityWitness:
    point: dict[str, Fraction]


@dataclass
class _Row:
    coeffs: dict[str, Fraction]
    rhs: Fraction
    lineage: dict[str, Fraction] = field(default_factory=dict)


def lcm(left: int, right: int) -> int:
    return abs(left * right) // gcd(left, right) if left and right else abs(left or right)


def feasible_point(
    variables: Sequence[str],
    constraints: Sequence[tuple[Mapping[str, Number], Number]],
) -> dict[str, Fraction] | None:
    named = [(f"row_{index}", coeffs, rhs) for index, (coeffs, rhs) in enumerate(constraints)]
    decision = certify_feasibility(variables, named)
    if isinstance(decision, FeasibilityWitness):
        return dict(decision.point)
    return None


def certify_feasibility(
    variables: Sequence[str],
    constraints: Sequence[tuple[str, Mapping[str, Number], Number]],
) -> FeasibilityWitness | FarkasCertificate:
    """Exact Fourier-Motzkin feasibility check for a*x <= b systems.

    Infeasibility returns non-negative multipliers on the named rows whose
    non-negative combination reads 0 <= -1; the multipliers alone are enough
    to re-check the claim without the elimination trace.
    """

    rows = [
        _Row(
            {name: Fraction(value) for name, value in coeffs.items() if Fraction(value) != 0},
            Fraction(rhs),
            {row_name: Fraction(1)},
        )
        for row_name, coeffs, rhs in constraints
    ]
    contradiction_holder: list[_Row] = []
    point = _eliminate(rows, list(variables), contradiction_holder)
    if point is not None:
        return FeasibilityWitness(point)
    contradiction = contradiction_holder[0]
    scale = Fraction(-1) / contradiction.rhs
    multipliers = {
        name: value * scale
        for name, value in contradiction.lineage.items()
        if value * scale != 0
    }
    return FarkasCertificate(dict(sorted(multipliers.items())))


def check_farkas(
    certificate: FarkasCertificate,
    constraints: Sequence[tuple[str, Mapping[str, Number], Number]],
) -> bool:
    rows = {
        name: ({key: Fraction(value) for key, value in coeffs.items()}, Fraction(rhs))
        for name, coeffs, rhs in constraints
    }
    combined: dict[str, Fraction] = {}
    rhs = Fraction(0)
    for name, multiplier in certificate.multipliers.items():
        if multiplier < 0 or name not in rows:
            return False
        coeffs, row_rhs = rows[name]
        for key, value in coeffs.items():
            combined[key] = combined.get(key, Fraction(0)) + multiplier * value
        rhs += multiplier * row_rhs
    return all(value == 0 for value in combined.values()) and rhs <= -1


def improving_ray(
    variables: Sequence[str],
    constraints: Sequence[tuple[str, Mapping[str, Number], Number]],
    objective: Mapping[str, Number],
) -> dict[str, int] | None:
    """Return an integer recession direction d with A d <= 0 and c*d >= 1."""

    homogeneous = [(name, dict(coeffs), 0) for name, coeffs, _rhs in constraints]
    homogeneous.append(
        ("__objective_improvement__", {name: -Fraction(value) for name, value in objective.items()}, -1)
    )
    decision = certify_feasibility(variables, homogeneous)
    if not isinstance(decision, FeasibilityWitness):
        return None
    point = decision.point
    denominator = 1
    for value in point.values():
        denominator = lcm(denominator, value.denominator)
    direction = {name: int(point.get(name, Fraction(0)) * denominator) for name in variables}
    improvement = sum(Fraction(objective.get(name, 0)) * direction[name] for name in variables)
    if improvement <= 0:
        return None
    return direction


def integer_point_along_ray(
    point: Mapping[str, Fraction],
    direction: Mapping[str, int],
    variables: Sequence[str],
) -> dict[str, int]:
    denominator = 1
    for name in variables:
        denominator = lcm(denominator, Fraction(point[name]).denominator)
    return {
        name: int(Fraction(point[name]) * denominator + direction[name] * denominator)
        for name in variables
    }


def _eliminate(
    rows: list[_Row],
    variables: list[str],
    contradiction: list[_Row],
) -> dict[str, Fraction] | None:
    if not variables:
        worst = min(
            (row for row in rows if not row.coeffs and row.rhs < 0),
            key=lambda row: row.rhs,
            default=None,
        )
        if worst is not None:
            contradiction.append(worst)
            return None
        return {}

    name = variables[0]
    rest = variables[1:]
    positive: list[tuple[dict[str, Fraction], Fraction, _Row, Fraction]] = []
    negative: list[tuple[dict[str, Fraction], Fraction, _Row, Fraction]] = []
    reduced: list[_Row] = []

    for row in rows:
        coefficient = row.coeffs.get(name, Fraction(0))
        if coefficient == 0:
            reduced.append(row)
        elif coefficient > 0:
            normalized = {key: value / coefficient for key, value in row.coeffs.items() if key != name}
            positive.append((normalized, row.rhs / coefficient, row, coefficient))
        else:
            scale = -coefficient
            normalized = {key: value / scale for key, value in row.coeffs.items() if key != name}
            negative.append((normalized, row.rhs / scale, row, scale))

    for upper_row, upper_rhs, upper_raw, upper_scale in positive:
        for lower_row, lower_rhs, lower_raw, lower_scale in negative:
            coeffs = {
                key: lower_row.get(key, Fraction(0)) + upper_row.get(key, Fraction(0))
                for key in set(lower_row) | set(upper_row)
            }
            coeffs = {key: value for key, value in coeffs.items() if value != 0}
            rhs = upper_rhs + lower_rhs
            lineage: dict[str, Fraction] = {}
            for raw, scale in ((upper_raw, upper_scale), (lower_raw, lower_scale)):
                for key, value in raw.lineage.items():
                    lineage[key] = lineage.get(key, Fraction(0)) + value / scale
            reduced.append(_Row(coeffs, rhs, lineage))

    values = _eliminate(reduced, rest, contradiction)
    if values is None:
        return None

    upper_entries = [
        (rhs - sum(row.get(key, 0) * values[key] for key in rest), raw, scale)
        for row, rhs, raw, scale in positive
    ]
    lower_entries = [
        (sum(row.get(key, 0) * values[key] for key in rest) - rhs, raw, scale)
        for row, rhs, raw, scale in negative
    ]
    upper_entry = min(upper_entries, key=lambda item: item[0], default=None)
    lower_entry = max(lower_entries, key=lambda item: item[0], default=None)
    upper = upper_entry[0] if upper_entry is not None else None
    lower = lower_entry[0] if lower_entry is not None else None
    if upper is not None and lower is not None and lower > upper:
        lineage = {}
        for _value, raw, scale in (upper_entry, lower_entry):
            for key, multiplier in raw.lineage.items():
                lineage[key] = lineage.get(key, Fraction(0)) + multiplier / scale
        contradiction.append(_Row({}, upper - lower, lineage))
        return None
    if lower is not None:
        value = lower
    elif upper is not None:
        value = upper
    else:
        value = Fraction(0)
    return {name: value, **values}
