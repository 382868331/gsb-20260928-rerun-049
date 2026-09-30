from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Mapping, Sequence


Number = int | Fraction


class UnboundedPolyhedron(Exception):
    pass


def lcm(left: int, right: int) -> int:
    return abs(left * right) // gcd(left, right) if left and right else abs(left or right)


def feasible_point(
    variables: Sequence[str],
    constraints: Sequence[tuple[Mapping[str, Number], Number]],
) -> dict[str, Fraction] | None:
    """Return a rational point satisfying every ``a*x <= b`` row, or None.

    The witness is reconstructed by Fourier-Motzkin elimination using only
    exact rational arithmetic, so callers can re-check every coordinate.
    """

    system = [
        ({name: Fraction(value) for name, value in row.items()}, Fraction(rhs))
        for row, rhs in constraints
    ]
    return _solve(system, list(variables))


def improving_ray(
    variables: Sequence[str],
    constraints: Sequence[tuple[Mapping[str, Number], Number]],
    objective: Mapping[str, Number],
) -> dict[str, int] | None:
    """Return an integer direction ``d`` with ``A d <= 0`` and ``c*d >= 1``.

    Such a direction is a recession ray of the feasible polyhedron that
    strictly improves the transformed objective.  A rational cone witness is
    cleared to integers by multiplying by the denominator LCM.
    """

    homogeneous = [
        ({name: Fraction(value) for name, value in row.items()}, Fraction(0))
        for row, _ in constraints
    ]
    homogeneous.append(
        ({name: -Fraction(value) for name, value in objective.items()}, Fraction(-1))
    )
    point = _solve(
        [(dict(row), rhs) for row, rhs in homogeneous], list(variables)
    )
    if point is None:
        return None
    denominator = 1
    for value in point.values():
        denominator = lcm(denominator, value.denominator)
    direction = {name: int(point[name] * denominator) for name in variables}
    improvement = sum(Fraction(objective.get(name, 0)) * direction[name] for name in variables)
    if improvement <= 0 or any(value != int(value) for value in direction.values()):
        return None
    return direction


def integer_point_along_ray(
    point: Mapping[str, Fraction],
    direction: Mapping[str, int],
    variables: Sequence[str],
) -> dict[str, int]:
    """Move a rational feasible point far enough along an integer ray to land
    on integer coordinates while keeping feasibility (``A d <= 0``)."""

    denominator = 1
    for name in variables:
        denominator = lcm(denominator, Fraction(point[name]).denominator)
    return {
        name: int(Fraction(point[name]) * denominator + direction[name] * denominator)
        for name in variables
    }


def _solve(
    system: list[tuple[dict[str, Fraction], Fraction]],
    variables: list[str],
) -> dict[str, Fraction] | None:
    if not variables:
        if any(Fraction(0) > rhs for _, rhs in system):
            return None
        return {}

    name = variables[0]
    rest = variables[1:]
    positive: list[tuple[dict[str, Fraction], Fraction, Fraction]] = []
    negative: list[tuple[dict[str, Fraction], Fraction, Fraction]] = []
    reduced: list[tuple[dict[str, Fraction], Fraction]] = []

    for row, rhs in system:
        coefficient = row.get(name, Fraction(0))
        if coefficient == 0:
            reduced.append((row, rhs))
        elif coefficient > 0:
            normalized = {key: value / coefficient for key, value in row.items() if key != name}
            positive.append((normalized, rhs / coefficient, coefficient))
        else:
            normalized = {key: value / -coefficient for key, value in row.items() if key != name}
            normalized = {key: -value for key, value in normalized.items()}
            negative.append((normalized, -rhs / coefficient, -coefficient))

    for upper_row, upper_rhs, _ in positive:
        for lower_row, lower_rhs, _ in negative:
            row = {
                key: lower_row.get(key, Fraction(0)) - upper_row.get(key, Fraction(0))
                for key in set(lower_row) | set(upper_row)
            }
            row = {key: value for key, value in row.items() if value != 0}
            reduced.append((row, upper_rhs - lower_rhs))

    values = _solve(reduced, rest)
    if values is None:
        return None

    upper_bounds: list[Fraction] = []
    lower_bounds: list[Fraction] = []
    for row, rhs in positive:
        upper_bounds.append(rhs - sum(row.get(key, 0) * values[key] for key in rest))
    for row, rhs in negative:
        lower_bounds.append(rhs - sum(row.get(key, 0) * values[key] for key in rest))

    upper = min(upper_bounds) if upper_bounds else None
    lower = max(lower_bounds) if lower_bounds else None
    if upper is not None and lower is not None and lower > upper:
        return None
    if lower is not None:
        value = lower
    elif upper is not None:
        value = upper
    else:
        value = Fraction(0)
    return {name: value, **values}
