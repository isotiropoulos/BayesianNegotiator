from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple, List

IssueKey = Tuple[str, str]  # (leftOperand, myOperator)

@dataclass
class BaseModel:
    def observe(self, x: Any, operator: str, lr: float = 1.0) -> None:
        """Update belief from evidence x under a given operator."""
        raise NotImplementedError

    def sample(
        self,
        x: Any = None,
        operator: str = "eq",
        inertia_val: Any = None,
        inertia: float = 0.0,
        **kwargs,
    ):
        """Propose a new value (or set/label)."""
        raise NotImplementedError

    def accept(self, x: Any, operator: str = "eq", **kwargs):
        """Decide whether to accept x under given operator."""
        raise NotImplementedError

@dataclass
class Issue:
    """Belief model container with adapters to/from rightOperand representation."""
    kind: str  # 'num' | 'enum' | 'set'
    operand: str
    operator: str
    initOperator: str
    oppOperator: str | None
    lastValue: Any
    initValue: Any
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
    reason: str

class NegotiationInfeasible(Exception):
    pass

@dataclass
class ActionState:
    lr: float
    inertia: float
    tau_set: List[float, float] # [tau_in, tau_out]
    max_set_k: int
    jaccard_snap_threshold: float

    # Converge metrics
    context_score: float = None
    severity: float = None

    # rolling histories per action
    _hopeless_streak: int = 0
    _num_move_hist: List[float] = field(default_factory=list)
    _opp_set_sizes: List[int] = field(default_factory=list)

