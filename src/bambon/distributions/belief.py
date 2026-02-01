from __future__ import annotations
from random import choice, sample
from typing import Any, Dict, List
from statistics import mean
from ..models.models import IssueKey, Issue, ActionState
from ..utils.helpers import (
    strip_key,
    HardRule,
    create_issue,
    clamp,
)


class BeliefRegistry:
    """Holds per-issue belief models and provides observe/propose APIs."""

    def __init__(self, constraints: List[Dict[str, Any]], hard_constraints: List[Dict[str, Any]]) -> None:
        self.issue_models: List[Issue] = self._parse_offer_constraints(constraints, hard_constraints)

    @staticmethod
    def _parse_offer_constraints(constraints: List[Dict[str, Any]], hard_constraints) -> List[Issue]:
        out: List[Issue] = []

        constraints = [{strip_key(key): val for key, val in one_constraint.items()} for one_constraint in constraints]
        if hard_constraints:
            hard_constraints = [
                {strip_key(key): val for key, val in one_constraint.items()}
                for one_constraint in hard_constraints
            ]

        for c in constraints:
            left_operand = strip_key(c["leftOperand"])
            operator = strip_key(c["operator"])
            value = c["rightOperand"]

            try:
                U = min([item["rightOperand"]
                    for item in (hard_constraints or [])
                    if strip_key(item["leftOperand"]) == left_operand and strip_key(item["operator"]) in ('lt', 'lteq')
                ])
            except:
                U = None
            try:

                L = max([item["rightOperand"]
                    for item in (hard_constraints or [])
                    if strip_key(item["leftOperand"]) == left_operand and strip_key(item["operator"]) in ('gt', 'gteq')
                ])
            except:
                L = None

            bounds = None
            if L or U:
                bounds = [L, U]
            issue = create_issue(left_operand, operator, value, bounds)

            lambdas = [
                HardRule(item["operator"], issue.to_internal(item["rightOperand"]))
                for item in (hard_constraints or [])
                if strip_key(item["leftOperand"]) == left_operand
            ]

            issue.hard_rule = lambdas
            out.append(issue)

        return out

    # --------------------------- OBSERVE ---------------------------

    def observe(self, action_state: ActionState) -> bool:
        """
        Update beliefs from the last incoming offer (stored on each Issue as
        issue.oppOperator / issue.oppLastValue).

        Returns True iff at least one hard constraint was violated
        (i.e., someone tried something outside my hard constraints).
        """
        hc_failed = False
        for issue in self.issue_models:
            opp_val = issue.oppLastValue
            if opp_val is None:
                continue

            x_internal = issue.to_internal(opp_val)

            # Only learn from offers that satisfy all hard rules
            if issue.hard_rule and all(rule(issue.oppOperator, x_internal) for rule in issue.hard_rule):
                issue.model.observe(
                    x_internal,
                    issue.oppOperator,  # opponent operator
                    action_state.lr,
                )
            else:
                if issue.hard_rule:
                    hc_failed = True
        return hc_failed

    # --------------------------- PROPOSING ---------------------------

    def _objective_from_op(self, op: str) -> str:
        """Infer objective for numeric acceptance from operator."""
        op = (op or "").lower()
        if op in ("gt", "gteq"):
            return "maximize"
        if op in ("lt", "lteq"):
            return "minimize"
        return "eq"  # equality/unknown → use credible band

    def propose(
        self,
        action_state: ActionState,
        check_accept: bool = True,

    ) -> Dict[IssueKey, Any]:
        """
        Main decision step per round.

        For each issue:
          1. If hard constraints are violated by the opponent, propose a correction.
          2. Else, if check_accept=True, ask the model:
               "Given my belief and *my* constraint, is their constraint OK for me?"
          3. If accepted → echo their operator+value.
             Else → sample a new proposal from my model under my operator.
        """
        prop: Dict[IssueKey, Any] = {}

        for issue in self.issue_models:
            # If this issue is already agreed, just repeat my last commitment.
            if issue.accepted:
                prop[(issue.operand, issue.operator)] = (
                    issue.operator,
                    issue.lastValue,
                )
                continue

            # ---------- HARD RULES ----------
            handled_hard_violation = False
            correction_value = None

            for rule in (issue.hard_rule or []):
                x_opp_internal = issue.to_internal(issue.oppLastValue) if issue.oppLastValue is not None else None

                if x_opp_internal is not None and not rule(issue.oppOperator, x_opp_internal):
                    # Opponent violated hard constraint -> compute correction
                    if issue.kind == "num":
                        # Move midway between my last and hard-rule value in internal space
                        correction_value = mean(
                            [issue.to_internal(issue.lastValue), rule.value]
                        )

                    elif issue.kind == "enum":
                        if rule.operator == "isAnyOf":
                            correction_value = choice(list(rule.value))
                        else:
                            correction_value = None  # fall back to lastValue

                    elif issue.kind == "set":
                        base = set(issue.to_internal(issue.oppLastValue))
                        if rule.operator == "isAnyOf":
                            allowed = set(rule.value)
                            pool = list(allowed - base)
                            need = int(clamp(len(issue.to_internal(issue.lastValue)), 0, len(allowed)))
                            need = max(0, need)
                            k = min(need, len(pool))
                            add = set(sample(pool, k)) if k > 0 else set()
                            correction_value = base | add
                        elif rule.operator == "isAllOf":
                            must = set(rule.value)
                            correction_value = base | (must - base)
                        elif rule.operator == "isNoneOf":
                            banned = (
                                set(rule.value)
                                if isinstance(rule.value, (set, list, tuple))
                                else {rule.value}
                            )
                            correction_value = base - banned
                        else:
                            correction_value = None

                    # Emit proposal based on correction or fallback to my last value
                    if correction_value is not None:
                        prop[(issue.operand, issue.operator)] = (
                            None,
                            issue.to_external(correction_value),
                        )
                    else:
                        prop[(issue.operand, issue.operator)] = (
                            None,
                            issue.lastValue,
                        )

                    handled_hard_violation = True
                    break  # stop checking further hard rules for this issue

            if handled_hard_violation:
                continue  # next issue

            # ---------- ACCEPTANCE CHECKS (before sampling) ----------
            if issue.oppLastValue is not None and check_accept:
                my_op = issue.operator
                my_val_internal = issue.to_internal(issue.lastValue)
                opp_op = issue.oppOperator
                opp_val_internal = issue.to_internal(issue.oppLastValue)

                if issue.kind == "num":
                    # Given my numeric constraint (my_op, my_val_internal) and belief,
                    # is their numeric constraint (opp_op, opp_val_internal) OK for me?
                    ok = issue.model.accept(
                        my_op=my_op,
                        my_val=my_val_internal,
                        opp_op=opp_op,
                        opp_val=opp_val_internal,
                        last_value=my_val_internal,
                        credible_mass=0.90,
                        risk=0.30,
                        mc=3000,
                        ctx_score=action_state.context_score,
                        severity=action_state.severity,
                    )
                    if ok:
                        prop[(issue.operand, issue.operator)] = (
                            issue.oppOperator,
                            issue.oppLastValue,
                        )
                        issue.accepted = True

                elif issue.kind == "enum":
                    # my_val_internal: my label or set of labels (internal)
                    # opp_val_internal: opponent's label or set of labels (internal)
                    ok = issue.model.accept(
                        my_op=my_op,
                        my_val=my_val_internal,
                        opp_op=opp_op,
                        opp_val=opp_val_internal,
                        risk=0.30,  # accept if ≥ 90% chance argmax lies in intersection set
                        mc=3000,
                        ctx_score=action_state.context_score,
                        severity=action_state.severity,
                    )
                    if ok:
                        prop[(issue.operand, issue.operator)] = (
                            issue.oppOperator,
                            issue.to_external(opp_val_internal),
                        )
                        issue.accepted = True

                elif issue.kind == "set":
                    # my_val_internal: set[str] I want (e.g. {"en"})
                    # opp_val_internal: set[str] in their offer
                    ok = issue.model.accept(
                        my_op=my_op,
                        my_val=my_val_internal,
                        opp_op=opp_op,
                        opp_val=opp_val_internal,
                        tau_in=max(0.5, action_state.tau_set[0] - 0.05),
                        tau_out=action_state.tau_set[1],
                        jaccard_min=action_state.jaccard_snap_threshold,
                        mc=1500,
                        ctx_score=action_state.context_score,
                        severity=action_state.severity,
                    )
                    if ok:
                        prop[(issue.operand, issue.operator)] = (
                            issue.oppOperator,
                            issue.to_external(opp_val_internal),
                        )
                        issue.accepted = True

            if issue.accepted:
                continue  # already accepted, next issue

            # ---------- SAMPLING (fallback when we don't accept) ----------
            if issue.kind == "num":
                prev_internal = issue.to_internal(issue.lastValue)
                anchor_internal = issue.to_internal(issue.oppLastValue)
                val_internal = issue.model.sample(
                    x=anchor_internal,
                    operator=issue.oppOperator,        # my numeric operator (eq/gteq/lteq/gt/lt/neq)
                    inertia_val=prev_internal,      # stick to previous with some prob
                    inertia=action_state.inertia,
                )

                prop[(issue.operand, issue.operator)] = (
                    None,
                    issue.to_external(val_internal),
                )

            elif issue.kind == "enum":
                prev_lab_internal = issue.to_internal(issue.lastValue)
                lab_internal = issue.model.sample(
                    x=prev_lab_internal,
                    operator=issue.operator,            # 'eq' / 'isAnyOf' / 'isAllOf' / 'isNoneOf' / 'neq'
                    inertia=action_state.inertia,
                    max_k=action_state.max_set_k,

                )
                prop[(issue.operand, issue.operator)] = (
                    None,
                    issue.to_external(lab_internal),
                )

            elif issue.kind == "set":
                prev_set_internal = issue.to_internal(issue.lastValue)
                sampled_internal = issue.model.sample(
                    x=prev_set_internal,
                    operator=issue.operator,            # 'isAllOf' / 'isAnyOf' / 'isNoneOf' / 'eq' / 'neq'
                    tau=action_state.tau_set,
                    inertia=action_state.inertia,
                    max_k=action_state.max_set_k,
                    required={rule.operator: rule.value for rule in issue.hard_rule}
                )
                prop[(issue.operand, issue.operator)] = (
                    None,
                    issue.to_external(sampled_internal),
                )

        return prop
