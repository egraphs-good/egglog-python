import copy

import egglog.runtime
import pytest


@pytest.fixture(autouse=True)
def reset_conversions():
    old_conversions = copy.copy(egglog.runtime.CONVERSIONS)
    yield
    egglog.runtime.CONVERSIONS = old_conversions
