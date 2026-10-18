"""Presentation names and frontend options; semantic state lives in protobuf records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

__all__ = ["Ident", "RuleEvalMode"]


@dataclass(frozen=True)
class Ident:
    name: str
    module: str | None = None

    def __str__(self) -> str:
        if self.module:
            return f"{self.module}.{self.name}"
        return self.name

    @classmethod
    def builtin(cls, name: str) -> Ident:
        return cls(name, "egglog.builtins")


RuleEvalMode: TypeAlias = Literal["seminaive", "naive", "unsafe-seminaive"]
