from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("param-eq")
    group.addoption(
        "--param-eq-slow",
        action="store_true",
        default=False,
        help="run slow param_eq corpus tests",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--param-eq-slow"):
        return

    skip_slow = pytest.mark.skip(reason="need --param-eq-slow to run")
    for item in items:
        if "slow" in item.keywords and "egglog/exp/param_eq" in str(item.fspath):
            item.add_marker(skip_slow)
