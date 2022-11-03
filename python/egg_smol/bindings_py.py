# TODO: Figure out what these modules should be called
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Variant:
    name: str
    types: list[str]
    cost: Optional[int] = None
