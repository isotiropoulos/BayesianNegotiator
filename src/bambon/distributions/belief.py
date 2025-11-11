from __future__ import annotations

from random import choice, sample
from typing import Any, Dict, Optional, Set, List, Tuple
from functools import partial
from statistics import mean

from numpy.ma.extras import median

from .distributions import EnumDirichlet, NumericBeta, SetBelief
from ..models.models import IssueKey, Issue
from ..utils.helpers import (strip_key, HardRule, create_issue,
                             is_iso_duration, duration_to_seconds,
                             seconds_to_duration, is_iso_date, iso_to_days, days_to_iso, clip
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
            hard_constraints = [{strip_key(key): val for key, val in one_constraint.items()} for one_constraint in hard_constraints]

        for c in constraints:
            left_operand = strip_key(c["leftOperand"])
            operator = strip_key(c["operator"])
            value = c["rightOperand"]

            issue = create_issue(left_operand, operator, value)
            lambdas = [
                HardRule(item['operator'], issue.to_internal(item['rightOperand']))
                # partial(LAMBDA_OPS.get(strip_key(item['operator']), no_hard_constraint), issue.to_internal(item['rightOperand']))
                for item in hard_constraints
                if strip_key(item["leftOperand"]) == left_operand
            ]

            issue.hard_rule = lambdas
            out.append(issue)

        return out

    def observe(self, action_state: Any) -> bool:
        hc_failed = False
        for issue in self.issue_models:
            if issue.hard_rule and all([rule(issue.to_internal(issue.oppLastValue)) for rule in issue.hard_rule]):
                issue.model.observe(issue.to_internal(issue.oppLastValue), action_state.lr)
            else:
                hc_failed = True
                pass
        return hc_failed


    # ---------------- Proposing ---------------- #
    # inside class BeliefRegistry

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
            action_state: Any,
            jaccard_snap_threshold: float,
    ) -> Dict[IssueKey, Any]:


        prop: Dict[IssueKey, Any] = {}

        for issue in self.issue_models:

            correction_value = None

            # ---------- HARD RULES ----------
            handled_hard_violation = False
            correction_value = None

            for rule in (issue.hard_rule or []):
                if not rule(issue.to_internal(issue.oppLastValue)):
                    # build a correction proposal based on kind
                    if issue.kind == 'num':
                        correction_value = mean([issue.to_internal(issue.lastValue), rule.value])

                    elif issue.kind == 'enum':
                        if rule.operator == 'isAnyOf':
                            correction_value = choice(list(rule.value))
                        else:
                            correction_value = None  # fall back to lastValue

                    elif issue.kind == 'set':
                        base = set(issue.to_internal(issue.oppLastValue))
                        if rule.operator == 'isAnyOf':
                            allowed = set(rule.value)
                            need = int(clip(len(issue.to_internal(issue.lastValue)), 0, len(allowed)))
                            add = set(sample(list(allowed - base), need)) if need > 0 else set()
                            correction_value = base | add
                        elif rule.operator == 'isAllOf':
                            must = set(rule.value)
                            correction_value = base | (must - base)
                        elif rule.operator == 'isNoneOf':
                            banned = set(rule.value) if isinstance(rule.value, (set, list, tuple)) else {rule.value}
                            correction_value = base - banned
                        else:
                            correction_value = None  # fall back to lastValue

                    # write prop with either correction or my lastValue
                    if correction_value is not None:
                        prop[(issue.operand, issue.operator)] = issue.to_external(correction_value)
                    else:
                        prop[(issue.operand, issue.operator)] = issue.lastValue

                    handled_hard_violation = True
                    break  # stop checking other hard rules for this issue

            if handled_hard_violation:
                continue
                # go to next issue
            # else:
            # ---------- ACCEPTANCE CHECKS (before sampling) ----------
            accepted_now = False
            if issue.oppLastValue is not None:
                if issue.kind == "num":
                    obj = self._objective_from_op(issue.operator)
                    x = issue.to_internal(issue.oppLastValue)
                    span = max(1e-9, (issue.model.U - issue.model.L))
                    if obj == "eq":
                        ok = issue.model.accept(
                            x,
                            use_credible_rule=True,
                            credible_mass=0.90,
                            mc=3000,
                        )
                    else:
                        ok = issue.model.accept(
                            x,
                            objective=("maximize" if obj == "maximize" else "minimize"),
                            eps=0.02 * span,  # 2% of span
                            risk=0.10,  # ≤10% chance there's a meaningfully better draw
                            mc=3000,
                        )
                    if ok:
                        prop[(issue.operand, issue.operator)] = issue.oppLastValue
                        try:
                            issue.accepted = True
                        except Exception:
                            pass
                        accepted_now = True

                elif issue.kind == "enum":
                    cand = issue.to_internal(issue.oppLastValue)  # may be str OR set[str]
                    ok, _ = issue.model.accept(
                        cand,
                        risk=0.10,  # accept if ≥90% chance the winner equals candidate (or is in set)
                        mc=3000,
                        min_margin=0.0,
                    )
                    if ok:
                        prop[(issue.operand, issue.operator)] = issue.to_external(cand)
                        try:
                            issue.accepted = True
                        except Exception:
                            pass
                        accepted_now = True

                elif issue.kind == "set":
                    opp_set = issue.to_internal(issue.oppLastValue)
                    ok, _, _ = issue.model.accept(
                        opp_set,
                        tau_in=max(0.5, action_state.tau_set - 0.05),
                        tau_out=0.6,
                        require_all_in_confident=True,
                        require_all_out_confident=False,
                        jaccard_min=jaccard_snap_threshold,
                        mc=1500,
                    )
                    if ok:
                        prop[(issue.operand, issue.operator)] = issue.to_external(opp_set)
                        try:
                            issue.accepted = True
                        except Exception:
                            pass
                        accepted_now = True

            if accepted_now:
                continue  # next issue

            # ---------- SAMPLING (fallback) ----------
            if issue.kind == "num":
                prev_internal = issue.to_internal(issue.lastValue)
                val = issue.model.sample(prev_internal, inertia=action_state.inertia)
                prop[(issue.operand, issue.operator)] = issue.to_external(val)

            elif issue.kind == "enum":
                prev_lab = issue.to_internal(issue.lastValue)
                lab = issue.model.sample(prev_lab, inertia=action_state.inertia)
                prop[(issue.operand, issue.operator)] = issue.to_external(lab)

            elif issue.kind == "set":
                prev_set = issue.to_internal(issue.lastValue)
                opp_set = issue.to_internal(issue.oppLastValue)
                sampled = issue.model.sample(
                    prev_set=prev_set,
                    tau=action_state.tau_set,
                    inertia=action_state.inertia,
                    force_overlap_from=opp_set,
                    min_overlap=action_state.min_overlap,
                    max_k=action_state.max_set_k,
                    intersection_first_with=opp_set,
                    intersection_min_jaccard=jaccard_snap_threshold,
                )
                prop[(issue.operand, issue.operator)] = issue.to_external(sampled)

        return prop
