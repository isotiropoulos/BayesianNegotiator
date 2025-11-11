from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

IssueKey = Tuple[str, str, int]  # (leftOperand, operator, index)


@dataclass
class IssueModel:
    """Belief model container with adapters to/from rightOperand representation."""

    operand: str
    operator: str
    kind: str  # 'num' | 'enum' | 'set'
    model: Any
    to_internal: Any
    to_external: Any


class NegotiationInfeasible(Exception):
    pass
