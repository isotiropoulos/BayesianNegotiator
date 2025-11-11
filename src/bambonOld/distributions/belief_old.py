from __future__ import annotations

from typing import Any, Dict, Optional, Set

from .distributions import EnumDirichlet, NumericBeta, SetBelief
from ..models.models import IssueKey, IssueModel
from ..utils.helpers import (
    is_iso_duration,
    duration_to_seconds,
    seconds_to_duration,
    is_iso_date,
    iso_to_days,
    days_to_iso,
)


class BeliefRegistry:
    """Holds per-issue belief models and provides observe/propose APIs."""

    def __init__(self) -> None:
        self.models: Dict[IssueKey, IssueModel] = {}

    # --------------------------------------------------------------------- #
    # Model detection & creation
    # --------------------------------------------------------------------- #
    def _detect_and_make_model(self, key: IssueKey, value: Any) -> IssueModel:
        """
        Decide model kind and adapters from a rightOperand sample (operator-agnostic).
        The operator is part of the IssueKey and not used here.
        """
        left_operand, operator, _ = key

        # Numeric (int/float)
        if isinstance(value, (int, float)):
            m = NumericBeta(L=float(value) - 50.0, U=float(value) + 50.0)
            return IssueModel(
                left_operand,
                operator,
                "num",
                m,
                to_internal=lambda x: float(x),
                to_external=lambda x: float(x),
            )

        # Strings: ISO duration or ISO date -> numeric adapter; else enum
        if isinstance(value, str):
            if is_iso_duration(value):
                base = float(duration_to_seconds(value))
                m = NumericBeta(L=base - 15, U=base + 15)  # ±15 days in seconds
                return IssueModel(
                    left_operand,
                    operator,
                    "num",
                    m,
                    to_internal=lambda s: float(duration_to_seconds(s)),
                    to_external=lambda secs: seconds_to_duration(float(secs)),
                )
            if is_iso_date(value):
                base = float(iso_to_days(value))
                m = NumericBeta(L=base - 15, U=base + 15)
                return IssueModel(
                    left_operand,
                    operator,
                    "num",
                    m,
                    to_internal=lambda s: float(iso_to_days(s)),
                    to_external=lambda d: days_to_iso(int(round(d))),
                )
            # Categorical string
            m = EnumDirichlet(labels=set())
            m.observe(value, lr=0.0)  # register label without learning
            return IssueModel(
                left_operand,
                operator,
                "enum",
                m,
                to_internal=lambda s: str(s),
                to_external=lambda s: str(s),
            )

        # List[str] -> Set of strings
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            m = SetBelief()
            m.observe(set(value), lr=0.0)  # register items
            return IssueModel(
                left_operand,
                operator,
                "set",
                m,
                to_internal=lambda xs: set(xs),
                to_external=lambda S: sorted(list(S)),
            )

        # Fallback: treat unknown as categorical string
        m = EnumDirichlet(labels=set())
        m.observe(str(value), lr=0.0)
        return IssueModel(
            left_operand,
            operator,
            "enum",
            m,
            to_internal=lambda s: str(s),
            to_external=lambda s: str(s),
        )

    # --------------------------------------------------------------------- #
    # Learning / observing
    # --------------------------------------------------------------------- #
    def observe_offer(self, issues: Dict[IssueKey, Any], lr: float = 1.0) -> None:
        """Update beliefs from an offer (already parsed to IssueKey → rightOperand)."""
        for key, raw in issues.items():
            model = self.models.get(key)
            if model is None:
                model = self._detect_and_make_model(key, raw)
                self.models[key] = model

            if model.kind == "num":
                x = model.to_internal(raw)
                model.model.observe(x, lr=lr)
            elif model.kind == "enum":
                label = model.to_internal(raw)
                model.model.observe(label, lr=lr)
            elif model.kind == "set":
                values: Set[str] = model.to_internal(raw)
                for e in values:
                    model.model.ensure(e)  # ensure label is known before updating
                model.model.observe(values, lr=lr)

    # --------------------------------------------------------------------- #
    # Proposing
    # --------------------------------------------------------------------- #
    def propose(
        self,
        my_prev: Dict[IssueKey, Any],
        opp_prev: Optional[Dict[IssueKey, Any]],
        inertia: float,
        tau_set: float,
        min_overlap: int,
        max_set_k: Optional[int],
        jaccard_snap_threshold: float,
    ) -> Dict[IssueKey, Any]:
        """
        Propose values for known keys (we do not invent new issues).
        """
        prop: Dict[IssueKey, Any] = {}

        for key, model in self.models.items():
            my_prev_val = my_prev.get(key)
            opp_prev_val = opp_prev.get(key) if opp_prev else None

            if model.kind == "num":
                prev_internal = model.to_internal(my_prev_val) if my_prev_val is not None else None
                val = model.model.sample(prev_internal, inertia=inertia)
                prop[key] = model.to_external(val)

            elif model.kind == "enum":
                prev_lab = model.to_internal(my_prev_val) if my_prev_val is not None else None
                lab = model.model.sample(prev_lab, inertia=inertia)
                prop[key] = model.to_external(lab)

            elif model.kind == "set":
                prev_set = model.to_internal(my_prev_val) if my_prev_val is not None else set()
                opp_set = model.to_internal(opp_prev_val) if opp_prev_val is not None else None
                sampled = model.model.sample(
                    prev_set=prev_set,
                    tau=tau_set,
                    inertia=inertia,
                    force_overlap_from=opp_set,
                    min_overlap=min_overlap,
                    max_k=max_set_k,
                    intersection_first_with=opp_set,
                    intersection_min_jaccard=jaccard_snap_threshold,
                )
                prop[key] = model.to_external(sampled)

        return prop
