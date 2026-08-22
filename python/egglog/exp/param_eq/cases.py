"""Small public/project-authored cases used by bounded CI."""

from __future__ import annotations

from dataclasses import dataclass

_DEMO_SAMPLE_POINTS = ((-1.25, 0.5), (0.25, 1.5), (2.0, -0.75))


@dataclass(frozen=True)
class DemoCase:
    """A redistributable Param-Eq demonstration expression."""

    name: str
    source: str
    sample_points: tuple[tuple[float, float], ...] = _DEMO_SAMPLE_POINTS


DEMO_CASES = (
    DemoCase(
        name="paper_eq4_instantiation",
        source="(2.3 * (3.7*x0 + 5.1*x1)) / 7.9",
    ),
    DemoCase(
        name="repeated_monomial",
        source="2.0*x0*x0 + 3.0*x0*x0 + 5.0*x1",
    ),
    DemoCase(
        name="function_composition",
        source="log(exp(2.0*x0 + 3.0))",
    ),
)
