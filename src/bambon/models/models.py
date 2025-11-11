from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple
IssueKey = Tuple[str, str]  # (leftOperand, myOperator)

@dataclass
class BaseModel:
    def observe(self, x: Any, lr: float):
        pass
    def sample(self, inertia_val: float, inertia: float):
        pass

    def accept(self, *args):
        pass


@dataclass
class Issue:
    """Belief model container with adapters to/from rightOperand representation."""
    kind: str  # 'num' | 'enum' | 'set'
    operand: str
    operator: str
    oppOperator: str | None
    lastValue: Any
    oppLastValue: Any
    model: BaseModel
    to_internal: Any
    to_external: Any
    hard_rule: Any | None
    accepted: bool = False

class HardConstraints:
    """Belief model container with adapters to/from rightOperand representation."""
    operand: str
    operator: str
    value: Any


class InternalError(Exception):
    reason: str

class NegotiationInfeasible(InternalError):
    pass

class MultipleIssuesError(InternalError):
    pass

class IssueNotFoundError(InternalError):
    pass