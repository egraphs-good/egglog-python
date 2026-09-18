"""Experimental parameter-reducing simplifier for symbolic-regression expressions."""

from __future__ import annotations

from .cases import DEMO_CASES, DemoCase
from .domain import (
    ContainerMonomial,
    ContainerPolynomial,
    Num,
    ParamCost,
    binary_to_containers,
    container_cost_model,
    containers_to_binary,
    exp,
    log,
    param_cost_model,
    parse_expression,
    polynomial,
    render_num,
    sqrt,
)
from .pipeline import (
    PaperPipelineReport,
    run_paper_pipeline,
    run_paper_pipeline_container,
)

__all__ = [
    "DEMO_CASES",
    "ContainerMonomial",
    "ContainerPolynomial",
    "DemoCase",
    "Num",
    "PaperPipelineReport",
    "ParamCost",
    "binary_to_containers",
    "container_cost_model",
    "containers_to_binary",
    "exp",
    "log",
    "param_cost_model",
    "parse_expression",
    "polynomial",
    "render_num",
    "run_paper_pipeline",
    "run_paper_pipeline_container",
    "sqrt",
]
