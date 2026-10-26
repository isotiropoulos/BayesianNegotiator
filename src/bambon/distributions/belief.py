from __future__ import annotations
from random import choice, sample, random
from typing import Any, Dict, List
from statistics import mean
from ..models.models import IssueKey, Issue, ActionState
from ..utils.utility import numeric_utility, numeric_value_for_utility
from ..utils.helpers import (
    strip_key,
    HardRule,
    create_issue,
    create_merged_categorical_issue,
    create_merged_numeric_issue,
    has_numeric_range,
    merged_issue_clauses,
    split_categorical_clauses,
    is_numeric_value,
    numeric_family,
    clamp,
    satisfies,
    INCLUSION_OPS,
)

import os as _os

#: Commit-time acceptance no longer overwrites the agent's own standing demand.
#:
#: Agreement is all-or-nothing over the package: `Bambon.observe` concludes only
#: when EVERY issue of EVERY action is accepted, and `_assign_opp` clears an
#: issue's `accepted` flag as soon as the counterparty moves that term. Accepting
#: an issue used to write the counterparty's operator and value onto
#: `Issue.lastValue` / `Issue.operator`, and the reset cleared only the flag --
#: so the agent surrendered its ask on a package that was never concluded and
#: could not take it back. Measured on far_wide: 16.6 such writes per encounter,
#: 74.8% of them subsequently reset. The agreed term now lives on
#: `Issue.agreedValue` / `agreedOperator` and is what the outgoing offer echoes.


def _as_set(v) -> set:
    if v is None:
        return set()
    if isinstance(v, (list, tuple, set, frozenset)):
        return {str(x) for x in v}
    return {str(v)}
# Sensitivity hook (ESWA): scale every Monte-Carlo draw count. 1.0
# reproduces every published number.
_MC_SCALE = float(_os.environ.get("BAMBON_MC_SCALE", 1.0))
def _mc(n):
    return max(25, int(round(n * _MC_SCALE)))


def _as_label_set(value) -> set:
    """Coerce a categorical internal value into a set of labels."""
    if value is None:
        return set()
    if isinstance(value, (set, frozenset, list, tuple)):
        return set(value)
    return {value}


def _timebid_bound(mine, theirs, hard_bound, action_state):
    """Belief-free concession on one bound of a numeric range issue.

    Interpolates from the agent's own current bound toward the opponent's on the
    same aspiration schedule the full model uses, and never past the agent's own
    hard bound (the caller clamps again, so this is belt-and-braces). Reaches
    the opponent's bound only when the aspiration has fallen to zero.
    """
    if mine is None:
        return mine
    give = clamp(1.0 - clamp(action_state.aspiration, 0.0, 1.0), 0.0, 1.0)
    target = theirs if theirs is not None else (hard_bound if hard_bound is not None else mine)
    return mine + give * (target - mine)


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

        # Group constraints by operand so that several constraints on the same
        # operand (within the same action) are merged into ONE negotiable issue.
        by_operand: Dict[str, List[Dict[str, Any]]] = {}
        order: List[str] = []
        for c in constraints:
            operand = strip_key(c["leftOperand"])
            if operand not in by_operand:
                by_operand[operand] = []
                order.append(operand)
            by_operand[operand].append(c)

        for left_operand in order:
            group = by_operand[left_operand]
            numeric = all(is_numeric_value(c["rightOperand"]) for c in group)

            if len(group) == 1:
                # Single constraint on this operand: unchanged behaviour.
                out.append(BeliefRegistry._build_single_issue(group[0], left_operand, hard_constraints))
            elif numeric:
                if has_numeric_range(group):
                    # Lower + upper bound -> ONE interval issue negotiated coherently.
                    out.append(create_merged_numeric_issue(left_operand, group, hard_constraints))
                else:
                    # Same-direction numeric clauses: keep one issue per clause.
                    for c in group:
                        out.append(BeliefRegistry._build_single_issue(c, left_operand, hard_constraints))
            else:
                # Several categorical clauses: merge into a single set-membership issue
                # (allowed / required vs forbidden) backed by one belief.
                out.append(create_merged_categorical_issue(left_operand, group, hard_constraints))

        return out

    @staticmethod
    def _build_single_issue(c: Dict[str, Any], left_operand: str, hard_constraints) -> Issue:
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

        issue.hard_rule = [
            HardRule(item["operator"], issue.to_internal(item["rightOperand"]))
            for item in (hard_constraints or [])
            if strip_key(item["leftOperand"]) == left_operand
        ]
        return issue

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
            if issue.merged:
                hc_failed = self._observe_merged(issue, action_state) or hc_failed
                continue

            opp_val = issue.oppLastValue
            if opp_val is None:
                continue

            x_internal = issue.to_internal(opp_val)

            # Only learn from offers that satisfy all hard rules.
            #
            # An EMPTY `hard_rule` is falsy, so it used to fall through to the
            # else-branch and skip the update entirely: an issue with no hard
            # constraint on its operand never learned anything at all. The merged
            # sibling in `_observe_merged` below has always had this right --
            # `(not issue.hard_rule) or all(...)`. Zero impact on
            # data/generated_agentodrl (every issue there carries a hard rule),
            # but it silently disables learning on any corpus with sparse hard
            # policies. Gated on the same flag so flag-off stays byte-identical.
            if issue.hard_rule:
                _learn = all(rule(issue.oppOperator, x_internal)
                             for rule in issue.hard_rule)
            else:
                _learn = True
            if _learn:
                issue.model.observe(
                    x_internal,
                    issue.oppOperator,  # opponent operator
                    action_state.lr,
                )
            else:
                if issue.hard_rule:
                    hc_failed = True
        return hc_failed

    def _observe_merged(self, issue: Issue, action_state: ActionState) -> bool:
        """Learn from a merged opponent offer (categorical inclusion/exclusion or numeric range)."""
        hc_failed = False

        if issue.kind == "numrange":
            if not issue.oppLastValue:
                return False
            opp_lo, opp_hi = issue.oppLastValue
            if opp_lo is not None:
                issue.model.observe(issue.to_internal(opp_lo), issue.lower_op or "gteq", action_state.lr)
            if opp_hi is not None:
                issue.model_hi.observe(issue.to_internal(opp_hi), issue.upper_op or "lteq", action_state.lr)
            return False

        if issue.oppLastValue is not None:
            S = issue.to_internal(issue.oppLastValue)
            if (not issue.hard_rule) or all(rule(issue.oppOperator, S) for rule in issue.hard_rule):
                issue.model.observe(S, issue.oppOperator, action_state.lr)
            elif issue.hard_rule:
                hc_failed = True
        if issue.opp_forbidden:
            issue.model.observe(set(issue.opp_forbidden), "isNoneOf", action_state.lr)
        return hc_failed

    # --------------------------- INFEASIBILITY ---------------------------

    def reservation_deadlock(self, action_state: ActionState) -> bool:
        """Is some issue stuck with no term left that I could sign?

        True when, for at least one unaccepted issue, the opponent's standing
        term is one my own hard policy forbids me to commit to *and* I have
        nothing left to concede on that issue (my own term already sits at my
        hard bound). This is the signal behind
        ``Bambon.hopeless_mode == 'reservation'``.

        Deliberately conservative: an issue where I could still move, or where the
        opponent's term is signable, is not a deadlock. The caller adds the timing
        condition, so the two together mean "the deadline is close, the term on
        the table is illegal for me, and I have already offered my limit".
        """
        strict = getattr(action_state, "strict_categorical_commit", False)
        for issue in self.issue_models:
            if issue.accepted or issue.oppLastValue is None:
                continue

            if issue.kind == "numrange":
                # A two-sided window: stuck when the intersection of the two
                # windows is empty even after clipping to my own hard bounds.
                lo, hi = self._hard_numeric_bounds(issue)
                try:
                    opp_lo, opp_hi = issue.oppLastValue
                    o_lo = issue.to_internal(opp_lo) if opp_lo is not None else None
                    o_hi = issue.to_internal(opp_hi) if opp_hi is not None else None
                except Exception:
                    continue
                L = max([x for x in (lo, o_lo) if x is not None], default=None)
                U = min([x for x in (hi, o_hi) if x is not None], default=None)
                if L is not None and U is not None and L > U:
                    return True
                continue

            try:
                opp_val = issue.to_internal(issue.oppLastValue)
            except Exception:
                continue
            if not self._hard_commit_violation(issue, issue.oppOperator, opp_val, strict):
                continue                      # I could sign it -> not a deadlock

            if issue.kind == "num":
                lo, hi = self._hard_numeric_bounds(issue)
                step = self._resolution(issue)
                try:
                    mine = issue.to_internal(issue.lastValue)
                except Exception:
                    continue
                if mine is None:
                    continue
                at_limit = ((lo is not None and mine <= lo + step)
                            or (hi is not None and mine >= hi - step))
                if not at_limit:
                    continue                  # still room to concede
            return True
        return False

    # --------------------------- PROPOSING ---------------------------

    def _objective_from_op(self, op: str) -> str:
        """Infer objective for numeric acceptance from operator."""
        op = (op or "").lower()
        if op in ("gt", "gteq"):
            return "maximize"
        if op in ("lt", "lteq"):
            return "minimize"
        return "eq"  # equality/unknown → use credible band

    # --------------------------- HARD-RULE GUARDS ---------------------------
    # A reached agreement records *concrete* committed values. Hard rules are
    # otherwise tested with operator-ray overlap (`satisfies`), which is correct
    # for "are these two constraints jointly satisfiable" but too permissive for a
    # committed point: e.g. an opponent clause `gteq 897` overlaps a `gteq 900`
    # hard floor as a ray, yet the concrete 897 still breaches it. These guards
    # evaluate the committed value as an exact point so BAMBON can never sign or
    # propose a value outside its own non-negotiable hard policy.

    @staticmethod
    def _hard_point_violation(issue: Issue, x_internal) -> bool:
        """True if a concrete numeric value (a committed point) breaks a hard rule."""
        if x_internal is None or issue.kind != "num":
            return False
        for rule in (issue.hard_rule or []):
            if rule("eq", x_internal) is False:
                return True
        return False

    @staticmethod
    def _resolution(issue: Issue) -> float:
        """Smallest representable increment for this issue, in internal space.

        Internal space is days for dates, seconds for durations and the identity
        for plain numbers, so an ISO-valued operand steps by 1 and an integer
        operand by 1. A float-valued operand has no natural quantum, so a small
        epsilon is used: enough to clear a strict bound, small enough not to move
        the value perceptibly.
        """
        v = issue.initValue
        if isinstance(v, (list, tuple)):                 # numrange: [lo, hi]
            v = next((x for x in v if x is not None), None)
        if isinstance(v, str) or isinstance(v, bool) or isinstance(v, int):
            return 1.0
        return 1e-9

    @classmethod
    def _hard_numeric_bounds(cls, issue: Issue):
        """Hard-feasible ``[lo, hi]`` in internal space, made **inclusive**.

        A strict bound is moved inward by one representable step
        (:meth:`_resolution`), because a committed numeric term is evaluated as a
        *point*: a hard rule ``gt 10`` admits 11, not 10. Returning ``rule.value``
        unchanged for ``gt``/``lt`` -- as this method used to -- made
        :meth:`_clamp_to_hard` clamp straight onto the excluded endpoint, so the
        agent emitted an offer infeasible for *itself* and Proposition 1 failed.
        The 50 hand-authored scenarios never triggered it because their hard
        policies use only ``lteq``, ``gteq`` and ``isAnyOf``; on the generated
        out-of-distribution corpus it accounted for 4.6% of emitted offers.
        """
        step = cls._resolution(issue)
        lo = hi = None
        for rule in (issue.hard_rule or []):
            if rule.operator in ("gteq", "gt"):
                v = rule.value + (step if rule.operator == "gt" else 0.0)
                lo = v if lo is None else max(lo, v)
            elif rule.operator in ("lteq", "lt"):
                v = rule.value - (step if rule.operator == "lt" else 0.0)
                hi = v if hi is None else min(hi, v)
        return lo, hi

    @staticmethod
    def _hard_numeric_excluded(issue: Issue):
        """Points a `neq` hard rule forbids this numeric issue from committing to."""
        return [rule.value for rule in (issue.hard_rule or [])
                if rule.operator == "neq" and isinstance(rule.value, (int, float))]

    @classmethod
    def _clamp_to_hard(cls, issue: Issue, x_internal):
        """Project a numeric proposal into the agent's hard-feasible region.

        Two parts. Interval clamping handles ``gteq``/``gt``/``lteq``/``lt``. A
        ``neq`` rule excludes a *point*, which clamping cannot express: both
        parties can converge on exactly the forbidden value, each then refuses to
        sign it (correctly), and because the repair leaves the value untouched
        neither can move off it -- the encounter deadlocks at a single point. It
        is a Proposition 1 violation as well as a livelock: the emitted offer is
        infeasible for the emitter. Nudging off the excluded point by one
        resolution step, in whichever direction stays inside the interval, fixes
        both. (Found by the operator-coverage corpus; no earlier corpus put `neq`
        in a hard policy.)
        """
        if x_internal is None:
            return x_internal
        lo, hi = cls._hard_numeric_bounds(issue)
        if lo is not None and x_internal < lo:
            x_internal = lo
        if hi is not None and x_internal > hi:
            x_internal = hi

        banned = cls._hard_numeric_excluded(issue)
        if not banned:
            return x_internal
        step = cls._resolution(issue)
        tol = step / 2.0

        def excluded(v):
            return any(abs(v - b) < tol for b in banned)

        def inside(v):
            return ((lo is None or v >= lo - tol) and (hi is None or v <= hi + tol))

        if not excluded(x_internal):
            return x_internal
        for k in range(1, 8):
            for cand in (x_internal + k * step, x_internal - k * step):
                if inside(cand) and not excluded(cand):
                    return cand
        return x_internal          # nothing feasible nearby; caller reports it

    # ---- universal committed-term hard gate (numeric AND categorical) ----

    @classmethod
    def _hard_commit_violation(cls, issue: Issue, op, value,
                               strict_categorical: bool = False) -> bool:
        """True if committing ``(op, value)`` would breach one of my hard rules.

        Previously only ``kind == "num"`` was gated (``_hard_point_violation``);
        enum/set/merged-categorical acceptance had no hard check at all, so
        BAMBON could sign e.g. ``language isAnyOf {en, fr}`` while its own hard
        policy required ``language eq en``.

        The test is on the *committed* clause: an agreement records concrete
        terms, so a numeric point is re-tested with ``eq`` (a ``gteq 897``
        clause overlaps a ``gteq 900`` hard floor as a ray, yet the concrete
        897 still breaches it), and a categorical term is tested with its own
        operator against the hard rule.

        ``strict_categorical`` swaps the categorical test from **overlap** to
        **containment**. ``sat`` treats an inclusion requirement as an overlap
        test, so a committed ``isAnyOf {en, fr}`` satisfies a hard
        ``isAnyOf {en, de}`` merely by mentioning ``en`` -- the agent can sign a
        term broader than its own allow-list even though every offer it *emits*
        is projected into that list. Turning this on requires the committed label
        set to sit inside the allow-set and outside the forbidden set, which is
        what the hard policy arguably means. It is off by default because it
        changes the notion of violation used throughout the paper (Def. 2), so
        enabling it changes the reported compliance numbers as well as behaviour.
        """
        if value is None:
            return False
        for rule in (issue.hard_rule or []):
            if issue.kind == "num":
                if rule("eq", value) is False:
                    return True
            elif rule(op or issue.operator, value) is False:
                return True
        if strict_categorical and issue.kind in ("set", "enum") and issue.hard_rule:
            allowed, required, banned = cls._hard_categorical_sets(issue)
            S = set(value) if isinstance(value, (set, frozenset, list, tuple)) else {value}
            if allowed is not None and not S.issubset(allowed):
                return True
            if S & banned:
                return True
            if required and not required.issubset(S):
                return True
        return False

    @staticmethod
    def _hard_categorical_sets(issue: Issue):
        """(allowed, required, banned) label sets implied by my hard rules.

        ``allowed`` is ``None`` when no rule bounds the universe. ``eq``/
        ``isAnyOf`` bound what a committed term may contain; ``isAllOf``/``eq``
        state what it must contain; ``isNoneOf``/``neq`` what it may not.
        """
        allowed = None
        required: set = set()
        banned: set = set()
        for rule in (issue.hard_rule or []):
            v = rule.value
            V = set(v) if isinstance(v, (set, frozenset, list, tuple)) else {v}
            if rule.operator == "isAnyOf":
                allowed = V if allowed is None else (allowed & V)
            elif rule.operator == "eq":
                allowed = V if allowed is None else (allowed & V)
                required |= V
            elif rule.operator == "isAllOf":
                required |= V
            elif rule.operator in ("isNoneOf", "neq"):
                banned |= V
        return allowed, required, banned

    @classmethod
    def _project_to_hard_categorical(cls, issue: Issue, value):
        """Project a categorical proposal into my hard-feasible region.

        ``SetBelief.sample`` treats a ``required["isAnyOf"]`` entry as labels to
        *add*, never as a whitelist bounding the sample, so a belief-driven set
        could grow past its own hard allow-list. That is how BAMBON came to
        propose ``purpose isAllOf {analytics, open-source, publication,
        research}`` against a hard ``purpose isAnyOf {analytics, publication}``
        — an offer it could then never accept back, deadlocking the whole
        negotiation on one issue. Clamping the proposal keeps BAMBON's own
        offers signable by BAMBON.
        """
        if value is None or not issue.hard_rule:
            return value
        scalar = not isinstance(value, (set, frozenset, list, tuple))
        T = {value} if scalar else set(value)
        allowed, required, banned = cls._hard_categorical_sets(issue)
        if allowed is not None:
            T &= allowed
        T |= required
        T -= banned
        if not T:
            # Never emit an empty categorical term: fall back to the hard core.
            core = ((allowed or set()) | required) - banned
            if not core:
                return value
            T = {sorted(core)[0]}
        if scalar:
            return sorted(T)[0] if len(T) > 1 else next(iter(T))
        return T

    @staticmethod
    def _satisfied_by(my_op, my_val, their_op, their_val, numeric: bool) -> bool:
        """Does a committed ``(their_op, their_val)`` term satisfy ``(my_op, my_val)``?

        A committed numeric term is a point, not a ray, so it is tested with
        ``eq`` — otherwise a ``lteq 70`` clause would 'satisfy' a ``gteq 60``
        demand purely because the two rays overlap.
        """
        if my_val is None or their_val is None:
            return False
        try:
            return satisfies(my_op, my_val, "eq" if numeric else (their_op or my_op), their_val) is True
        except Exception:
            return False

    @classmethod
    def _meets_my_demand(cls, issue: Issue, opp_op, opp_val_internal) -> bool:
        """Deterministic accept: belief sampling must not veto a term I cannot
        rationally hold out against. Fires in exactly two situations.

        1. **They met my opening ask.** Their term satisfies my *initial*
           demand, so no amount of further haggling can improve it for me.
        2. **It is my own offer coming back.** Their term and my current term
           satisfy each other, i.e. we are proposing the same thing. For an
           ``isAllOf`` set of k labels the belief confidence is a product of k
           posterior means (``_conf_op``), so even a byte-identical echo of
           BAMBON's own offer scored below the acceptance bar and the
           negotiation ran to the deadline — 0 agreements in 1500 tries.

        Deliberately NOT "their term satisfies my *current* demand": my current
        demand has already drifted toward the opponent, so accepting on it
        concedes utility the belief model would still have fought for. That
        looser rule closed every negotiation but cost the first mover ~0.22
        ODRL utility in self-play. Restricting to the two cases above keeps the
        deadlock fix without the giveaway.
        """
        numeric = issue.kind == "num"
        my_op = issue.operator or issue.initOperator
        my_val = issue.to_internal(issue.lastValue) if issue.lastValue is not None else None

        # (1) their term clears the demand I opened with
        init_op = issue.initOperator or my_op
        init_val = issue.to_internal(issue.initValue) if issue.initValue is not None else None
        if cls._satisfied_by(init_op, init_val, opp_op, opp_val_internal, numeric):
            return True

        # (2) mutual satisfaction == we are asking for the same thing
        return (cls._satisfied_by(my_op, my_val, opp_op, opp_val_internal, numeric)
                and cls._satisfied_by(opp_op, opp_val_internal, my_op, my_val, numeric))

    @classmethod
    def _blind_accept(cls, issue: Issue, opp_op, opp_val_internal) -> bool:
        """Belief-free acceptance: the BAMBON-OM(acc) ablation arm.

        Replaces ``issue.model.accept(...)`` -- the *only* consumer of the
        opponent posterior anywhere in the acceptance path -- with the classic
        time-dependent rule from the automated-negotiation literature: accept a
        term that is at least as good as the one I am currently asking for.

        This is a *time-dependent* rule, not a static one, because my current
        term rides the aspiration schedule A(t): early on ``issue.lastValue``
        still sits near my opening ask so almost nothing clears the bar, and as
        t grows the bar falls with my concessions. So the arm is a fair control
        for "negotiate without modelling the opponent" rather than a strawman
        that accepts anything.

        Not consulted: ``issue.model`` (the posterior), ``context_score``,
        ``severity``, ``accept_probs``. Still enforced by the caller and
        deliberately left intact, because they are not opponent modelling:

        * ``_hard_commit_violation`` -- the hard-policy veto, so Proposition 1
          holds identically in this arm;
        * ``_meets_my_demand``       -- own opening ask / own offer echoed back.
        """
        numeric = issue.kind == "num"
        my_op = issue.operator or issue.initOperator
        my_val = (issue.to_internal(issue.lastValue)
                  if issue.lastValue is not None else None)
        return cls._satisfied_by(my_op, my_val, opp_op, opp_val_internal, numeric)

    @staticmethod
    def _accelerate_concession(issue: Issue, action_state, val_internal):
        """Speed up concession on a numeric issue that is the lone holdout.

        When the rest of the action has essentially converged (context >=
        concede_ctx_min) but THIS issue is still far apart (issue-severity
        s_i = 1 - sigma_i >= concede_sev_hot), nudge the proposed value toward the
        opponent's last value by an extra clamp(gain * s_i * round_frac, 0,
        step_cap) fraction of the remaining gap. Ramped by round_frac (hold early,
        push late). The caller still clamps the result to hard bounds, so we never
        concede past our reservation. Disabled when concede_gain == 0.
        """
        if (val_internal is None
                or issue.oppLastValue is None
                or action_state.context_score is None
                or getattr(action_state, "concede_gain", 0.0) <= 0.0):
            return val_internal
        s_i = clamp(1.0 - clamp(getattr(issue, "similarity", 1.0), 0.0, 1.0), 0.0, 1.0)
        if (action_state.context_score < action_state.concede_ctx_min
                or s_i < action_state.concede_sev_hot):
            return val_internal
        try:
            opp_i = issue.to_internal(issue.oppLastValue)
        except Exception:
            return val_internal
        if opp_i is None:
            return val_internal
        step = clamp(action_state.concede_gain * s_i * action_state.round_frac,
                     0.0, action_state.concede_step_cap)
        return val_internal + step * (opp_i - val_internal)

    # ===================== UTILITY-AWARE POLICY (opt-in) =====================
    def _numeric_anchor_res(self, issue, reservation_span):
        """(anchor, reservation, op) in internal value space for a numeric issue."""
        op = (issue.operator or issue.initOperator or "eq").lower()
        anchor = issue.to_internal(issue.initValue)
        lo, hi = self._hard_numeric_bounds(issue)
        try:
            span = abs(issue.model.U - issue.model.L)
        except Exception:
            span = 0.0
        if not span:
            span = abs(anchor) or 1.0
        off = max(reservation_span, 0.0) * span
        if op in ("gteq", "gt"):
            res = lo if (lo is not None and lo < anchor) else anchor - off
        elif op in ("lteq", "lt"):
            res = hi if (hi is not None and hi > anchor) else anchor + off
        else:
            res = anchor + off
        return anchor, res, op

    def _utility_bid_numeric(self, issue, action_state):
        """#2 + #3: best-for-us value on the opponent acceptance frontier
        (internal value).

        The search is TWO-SIDED. Pass 1 runs from the agent's opening ask
        (u = 1) down to the round's aspiration A(t) and takes the FIRST
        candidate the posterior says the counterparty is likely enough to
        accept, which is the greediest point on the estimated frontier. When
        nothing in that range clears the floor, pass 2 continues below A(t),
        down to the agent's own hard bound, and takes the greediest point there
        that does clear.

        Pass 2 is what makes the posterior an operator in both directions. With
        pass 1 alone the belief could ask for more than the concession schedule
        but never less, and a miss collapsed the bid back to A(t) -- exactly
        what the belief-free ``_timebid_numeric`` would have proposed. Against a
        counterparty whose acceptance rides its own clock that asymmetry could
        only ever delay a crossing, never hasten one, so belief-driven bidding
        lost agreements to a plain schedule. It no longer can.

        Conceding in pass 2 spends self-utility, never compliance: u = 0 maps to
        ``res``, the agent's own hard bound where one exists, so no candidate
        lies outside the feasible region, and the caller applies
        ``_clamp_to_hard`` afterwards regardless. Proposition 1 is untouched.
        """
        anchor, res, op = self._numeric_anchor_res(issue, action_state.reservation_span)
        u_target = clamp(action_state.aspiration, 0.0, 1.0)
        p_floor = clamp(action_state.p_floor, 0.0, 1.0)
        N = 21

        def _first_clearing(us):
            """Greediest candidate in ``us`` the posterior clears, else None.

            ``us`` must be ordered best-for-us first, so the first hit is the most
            favourable acceptable point rather than merely an acceptable one.
            """
            vs = [numeric_value_for_utility(u, anchor, res, op) for u in us]
            try:
                ps = issue.model.accept_probs(vs, operator=op, mc=_mc(2000))
            except Exception:
                ps = [1.0] * len(vs)
            for u, v, pacc in zip(us, vs, ps):
                if pacc >= p_floor:
                    return u, v
            return None

        # Pass 1: [A(t), 1]. Whenever the posterior clears the floor somewhere
        # in this range the bid is the greediest point on the estimated frontier
        # and pass 2 is never reached.
        us = [1.0 - (1.0 - u_target) * (i / (N - 1)) for i in range(N)] if N > 1 else [u_target]
        hit = _first_clearing(us)

        # Pass 2: (A(t), 0], scanned downward. Reached only when nothing at or
        # above the schedule looks acceptable -- precisely the case a one-sided
        # bidder answers by re-proposing A(t) and stalling. The agent instead
        # concedes to the greediest point below the schedule that the posterior
        # does clear, so the belief can hasten a crossing as well as delay one.
        #
        # Safe for Proposition 1: u = 0 maps to `res`, the agent's own hard bound
        # where one exists (see `_numeric_anchor_res`), so no candidate lies
        # outside the feasible region, and the caller applies `_clamp_to_hard`
        # afterwards regardless. Conceding here spends self-utility, never
        # compliance.
        if hit is None and N > 1 and u_target > 1e-9:
            lo = [u_target - u_target * (i / (N - 1)) for i in range(1, N)]
            hit = _first_clearing(lo)

        if hit is None:
            # Nothing the agent may lawfully offer looks acceptable to the
            # counterparty. Hold the schedule rather than pay utility for a
            # crossing the posterior does not predict.
            chosen_u = u_target
            v_target = numeric_value_for_utility(u_target, anchor, res, op)
        else:
            chosen_u, v_target = hit

        # The posterior may concede, never hold out. `accept_probs` answers
        # "does this value satisfy the opponent's inferred constraint?", not
        # "will the opponent accept this offer?" -- the panel decides on an
        # aggregate utility over every term, so a per-issue marginal saturates
        # near 1.0 as soon as the posterior sharpens. With the floor no longer
        # binding, pass 1 returned its first candidate, which is u = 1.0 by
        # construction: the agent proposed its opening ask on 59% of numeric
        # terms while its schedule sat at 0.36, i.e. it never conceded at all.
        # Measured against outcomes, that probability was anti-predictive of
        # agreement (AUC 0.22, and 0.25 restricted to the opening 20% of rounds,
        # so it is not a duration artifact).
        #
        # Clamping to A(t) keeps the belief as a one-way operator in the safe
        # direction: it may bring the bid BELOW the concession schedule through
        # pass 2 when the schedule does not look acceptable, and may never push
        # it above. Until `accept_probs` is calibrated against observed
        # accept/reject events, this is the only direction its output can be
        # trusted in.
        if chosen_u > u_target:
            chosen_u = u_target
            v_target = numeric_value_for_utility(u_target, anchor, res, op)

        prev = issue.to_internal(issue.lastValue) if issue.lastValue is not None else None
        if prev is not None and action_state.inertia > 0.0:
            if numeric_utility(prev, anchor, res, op) >= chosen_u and random() < action_state.inertia:
                v_target = prev
        return v_target

    # ===================== ABLATION: BELIEF-FREE BIDDING =====================
    # These two methods are the `bidding_mode == "time"` counterparts of the
    # belief-driven bid. They are reachable only through that switch and are
    # never used by the full model.
    #
    # What is removed: the opponent posterior. `_utility_bid_numeric` walks a
    # utility grid until `issue.model.accept_probs` says the opponent is likely
    # enough to take the bid; the set/enum samplers draw labels at posterior
    # inclusion rates under tau_in/tau_out. Neither is consulted here.
    #
    # What is kept, identically: the self-utility model (Eq. 13), the aspiration
    # schedule A(t) (Eq. 15), inertia, the accelerated-concession nudge, the
    # hard-feasibility projection, and the whole acceptance path. The bid is
    # therefore a *memoryless* time-dependent concession -- it may look at the
    # offer currently on the table, but not at any model accumulated over the
    # history of the encounter.

    def _timebid_numeric(self, issue, action_state):
        """Belief-free numeric bid: the value whose self-utility equals A(t)."""
        anchor, res, op = self._numeric_anchor_res(issue, action_state.reservation_span)
        u_target = clamp(action_state.aspiration, 0.0, 1.0)
        v_target = numeric_value_for_utility(u_target, anchor, res, op)
        if (op or "").lower() in ("eq", "neq"):
            # `numeric_value_for_utility` pins eq/neq at the anchor, which would
            # make an eq issue a hardliner. Concede along the anchor->reservation
            # segment on the same schedule instead.
            v_target = anchor + (1.0 - u_target) * (res - anchor)
        prev = issue.to_internal(issue.lastValue) if issue.lastValue is not None else None
        if prev is not None and action_state.inertia > 0.0:
            if numeric_utility(prev, anchor, res, op) >= u_target and random() < action_state.inertia:
                v_target = prev
        return v_target

    def _timebid_labels(self, issue, action_state, my_labels, opp_labels, incl_op):
        """Belief-free categorical bid: a scheduled walk toward the last offer.

        Concede a ``1 - A(t)`` fraction of the symmetric difference between the
        agent's opening label set and the set on the table -- adding their labels
        and dropping the agent's own -- in a deterministic (sorted) order, so the
        bid is reproducible and carries no accumulated opponent model. The result
        is capped at ``max_set_k`` and projected into the feasible region by the
        caller.
        """
        from math import ceil

        opening = set(issue.to_internal(issue.initValue)) if issue.initValue is not None else set(my_labels)
        target = set(opp_labels or set())
        give = clamp(1.0 - clamp(action_state.aspiration, 0.0, 1.0), 0.0, 1.0)

        add_pool = sorted(target - opening)
        drop_pool = sorted(opening - target)
        n_add = int(ceil(give * len(add_pool)))
        n_drop = int(ceil(give * len(drop_pool))) if incl_op in ("isAnyOf", "isAllOf") else 0

        S = (opening | set(add_pool[:n_add])) - set(drop_pool[:n_drop])
        if not S:
            S = set(opening) or set(target)
        cap = max(1, int(action_state.max_set_k))
        if len(S) > cap:
            keep = sorted(S & target) + sorted(S - target)
            S = set(keep[:cap])
        return S

    def _utility_accept_gate(self, issue, action_state, opp_val_internal):
        """#7: decision-theoretic accept-vs-wait, layered on the belief accept()."""
        anchor, res, op = self._numeric_anchor_res(issue, action_state.reservation_span)
        u_opp = numeric_utility(opp_val_internal, anchor, res, op)
        if u_opp >= action_state.aspiration:
            return True
        prev = getattr(issue, "_u_prev_opp", None)
        improving = 1.0 if (prev is not None and
                            u_opp > numeric_utility(prev, anchor, res, op) + 1e-9) else 0.0
        issue._u_prev_opp = opp_val_internal
        remaining = clamp(1.0 - action_state.round_frac, 0.0, 1.0)
        p_concede = (0.25 + 0.5 * improving) * remaining
        expected_gain = p_concede * max(0.0, 1.0 - u_opp)
        return expected_gain <= action_state.time_cost


    # ---------------- continuation-value acceptance (opt-in) ----------------

    def _posterior_label_means(self, model, labels):
        """pi_e for each label, from whichever categorical posterior this is."""
        if hasattr(model, "_posterior_means"):
            try:
                return model._posterior_means(set(labels))
            except Exception:
                pass
        a = getattr(model, "alpha", None)
        if isinstance(a, dict) and a:
            tot = float(sum(a.values())) or 1.0
            return {e: float(a.get(e, 1.0)) / tot for e in labels}
        return {}

    def _issue_utilities(self, issue, action_state, mc: int = 64, q: float = 0.5):
        """(u_in, u_lim) for one issue, both in [0,1] and both MY utility.

        u_in  : what the counterparty's standing term is worth to me.
        u_lim : what it would be worth if they conceded to the limit the
                posterior puts on them. Never below u_in.
        Returns (None, None) for an issue that cannot be scored.
        """
        from ..utils.helpers import beta_sample
        kind = issue.kind
        if kind in ("num", "numrange"):
            try:
                anchor, res, op = self._numeric_anchor_res(
                    issue, getattr(action_state, "reservation_span", 0.5))
            except Exception:
                return None, None
            try:
                if kind == "numrange":
                    vals = [v for v in (issue.oppLastValue or []) if v is not None]
                    if not vals:
                        return None, None
                    xs = [issue.to_internal(v) for v in vals]
                else:
                    x = issue.to_internal(issue.oppLastValue)
                    if x is None:
                        return None, None
                    xs = [x]
                u_in = sum(numeric_utility(x, anchor, res, op) for x in xs) / len(xs)
            except Exception:
                return None, None
            u_lim = u_in
            model = issue.model
            try:
                draws = sorted(
                    numeric_utility(model._from_unit(beta_sample(model.alpha, model.beta)),
                                    anchor, res, op)
                    for _ in range(mc))
                u_lim = draws[min(len(draws) - 1, int(q * len(draws)))]
            except Exception:
                pass
            return u_in, max(u_in, u_lim)

        mine = _as_set(issue.initValue)
        opp = _as_set(issue.oppLastValue)
        if not mine:
            return None, None
        u_in = len(mine & opp) / float(len(mine))
        pi = self._posterior_label_means(issue.model, mine | opp)
        u_lim = (sum(pi.get(e, 0.0) for e in mine) / float(len(mine))) if pi else u_in
        return u_in, max(u_in, u_lim)


    def _combi_accept(self, issue, action_state, opp_op, opp_val_internal) -> bool:
        """AC_combi-style acceptance: the time rule, with the belief lowering the bar.

        Accept the term if EITHER
          (a) it already satisfies the demand I am currently standing on -- the
              time-dependent rule of the BAMBON-OM(acc) arm; my standing term
              relaxes as the deadline approaches, so this loosens over time; or
          (b) the posterior says holding out is not worth another round: the
              utility I would gain if the counterparty conceded all the way to
              the limit its posterior allows is no more than one round's waiting
              cost.

        The bar can therefore only ever move DOWN relative to the time rule, so
        this is never more stubborn than BAMBON-OM(acc) -- which is the defect
        that made the shipped belief test look harmful.

        Why not the aspiration schedule A(t) as the time half: with
        concede_horizon = 0.3, A(t) reaches its floor at 30% of the deadline and
        is 0.000 for the remaining 70% (measured: 42.7% of all per-issue
        decisions see A(t) <= 0.01), so "u >= A(t)" is vacuously true for most of
        the encounter. The agent's actual standing term is the meaningful bar.
        """
        if self._blind_accept(issue, opp_op, opp_val_internal):
            return True
        u_in, u_lim = self._issue_utilities(issue, action_state)
        if u_in is None:
            return False
        cost = float(getattr(action_state, "time_cost", 0.02) or 0.0)
        return (u_lim - u_in) <= cost

    def _continuation_accept(self, action_state) -> bool:
        """Accept-vs-wait on the package. See accept_mode='continuation'."""
        u_ins, u_lims = [], []
        for issue in self.issue_models:
            if issue.oppLastValue is None:
                return False                      # nothing on the table yet
            ui, ul = self._issue_utilities(issue, action_state)
            if ui is None:
                continue
            u_ins.append(ui)
            u_lims.append(ul)
        if not u_ins:
            return False
        u_in = sum(u_ins) / len(u_ins)
        u_lim = sum(u_lims) / len(u_lims)

        rf = clamp(float(getattr(action_state, "round_frac", 0.0) or 0.0), 0.0, 1.0)
        remaining = 1.0 - rf
        T = float(getattr(action_state, "deadline_rounds", 0.0) or 0.0)
        if T <= 0.0:
            T = 50.0
        # Last move: a lawful offer is better than timing out. The per-issue veto
        # upstream still decides whether this offer is lawful at all.
        if remaining * T <= 1.0:
            return True
        c = clamp(float(getattr(action_state, "time_cost", 0.02) or 0.0), 0.0, 0.99)
        delta = 1.0 - c
        w = 0.5 * remaining * T
        u_exp_best = u_in + max(0.0, u_lim - u_in) * remaining
        V = (delta ** w) * u_exp_best
        return u_in >= V

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

        # One accept-vs-wait decision for the whole package, taken before the
        # per-issue loop and reused by every issue's stage-3 test. `blind` arms
        # never consult it, so the ablation stays a clean contrast.
        pkg = None
        if (check_accept
                and getattr(action_state, "accept_mode", "belief") == "continuation"):
            pkg = self._continuation_accept(action_state)
        self._pkg_accept = pkg

        for issue in self.issue_models:
            if issue.merged:
                self._propose_merged(issue, action_state, check_accept, prop)
                continue

            # If this issue is already agreed, repeat the agreed term.
            #
            # This used to emit (issue.operator, issue.lastValue), which was the
            # agreed term only because accepting an issue OVERWROTE lastValue
            # with the counterparty's value. Under BAMBON_NONDESTRUCTIVE_ACCEPT
            # that overwrite is gone, so lastValue is still the agent's own
            # unconceded demand -- and echoing it silently RETRACTED the
            # concession in the outgoing offer while the agent went on believing
            # the issue was settled. The more an arm accepts per issue, the more
            # incoherent its offers became. Emit what was actually agreed.
            if issue.accepted:
                if issue.agreedValue is not None:
                    prop[(issue.operand, issue.operator)] = (
                        issue.agreedOperator or issue.operator,
                        issue.agreedValue,
                    )
                else:
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
                            # Drop out-of-allow-list labels FIRST. The old code only
                            # *added* missing allowed labels, so when the offending
                            # set already contained every allowed label the pool was
                            # empty and the "correction" reproduced the violating set
                            # verbatim — a fixed point that hung the negotiation.
                            base &= allowed
                            pool = list(allowed - base)
                            need = int(clamp(len(issue.to_internal(issue.lastValue)), 0, len(allowed)))
                            need = max(0, need - len(base))
                            k = min(need, len(pool))
                            add = set(sample(pool, k)) if k > 0 else set()
                            correction_value = base | add
                            if not correction_value and allowed:
                                correction_value = {sorted(allowed)[0]}
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

                    # Emit proposal based on correction or fallback to my last value.
                    # Both are projected into my hard-feasible region: the fallback
                    # `lastValue` is belief-driven and may itself have drifted past
                    # my own hard policy.
                    if correction_value is not None:
                        if issue.kind in ("set", "enum"):
                            correction_value = self._project_to_hard_categorical(issue, correction_value)
                        elif issue.kind == "num":
                            correction_value = self._clamp_to_hard(issue, correction_value)
                        prop[(issue.operand, issue.operator)] = (
                            None,
                            issue.to_external(correction_value),
                        )
                    else:
                        fallback = issue.to_internal(issue.lastValue)
                        if issue.kind in ("set", "enum"):
                            fallback = self._project_to_hard_categorical(issue, fallback)
                        elif issue.kind == "num":
                            fallback = self._clamp_to_hard(issue, fallback)
                        issue.lastValue = issue.to_external(fallback)
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

                # Two gates that wrap EVERY kind-specific acceptance test below.
                #   hard_ok  - veto: a term breaching my own hard policy is never
                #              signed, no matter what the belief model returns.
                #   meets    - shortcut: a term that already satisfies my current
                #              demand is accepted without consulting the belief.
                # Order matters: the shortcut can only fire inside the veto.
                # BAMBON-NoVeto removes exactly this gate and nothing else.
                hard_ok = (not getattr(action_state, "hard_veto", True)) or (
                    not self._hard_commit_violation(
                        issue, opp_op, opp_val_internal,
                        getattr(action_state, 'strict_categorical_commit', False)))
                meets = hard_ok and self._meets_my_demand(issue, opp_op, opp_val_internal)
                # BAMBON-OM(acc) ablation: strip the posterior out of acceptance
                # while leaving the Bayesian bidding path untouched. `blind` is
                # read once here so all three kind-specific branches below
                # divert to the same belief-free rule.
                blind = getattr(action_state, "accept_mode", "belief") == "blind"
                # BAMBON-OM(combi): the time rule with the posterior allowed to
                # lower the bar. See _combi_accept.
                combi = getattr(action_state, "accept_mode", "belief") == "combi"

                if issue.kind == "num":
                    # Given my numeric constraint (my_op, my_val_internal) and belief,
                    # is their numeric constraint (opp_op, opp_val_internal) OK for me?
                    if not hard_ok:
                        ok = False
                    elif meets:
                        # Their value already clears my own numeric demand.
                        ok = True
                    elif blind:
                        ok = self._blind_accept(issue, opp_op, opp_val_internal)
                    elif combi:
                        ok = self._combi_accept(issue, action_state, opp_op,
                                                opp_val_internal)
                    elif pkg is not None:
                        ok = pkg                     # continuation rule
                    else:
                        ok = issue.model.accept(
                            my_op=my_op,
                            my_val=my_val_internal,
                            opp_op=opp_op,
                            opp_val=opp_val_internal,
                            last_value=my_val_internal,
                            credible_mass=0.90,
                            risk=0.30,
                            mc=_mc(3000),
                            ctx_score=action_state.context_score,
                            severity=action_state.severity,
                            round_frac=action_state.round_frac,
                            deadline_beta=action_state.deadline_beta,
                        )
                        if ok and action_state.utility_mode and (issue.operator or "").lower() in ("gteq", "gt", "lteq", "lt"):
                            ok = self._utility_accept_gate(issue, action_state, opp_val_internal)
                    if ok:
                        prop[(issue.operand, issue.operator)] = (
                            issue.oppOperator,
                            issue.oppLastValue,
                        )
                        issue.agreedOperator = issue.oppOperator
                        issue.agreedValue = issue.oppLastValue
                        issue.accepted = True

                elif issue.kind == "enum":
                    # my_val_internal: my label or set of labels (internal)
                    # opp_val_internal: opponent's label or set of labels (internal)
                    if not hard_ok:
                        ok = False
                    elif meets:
                        ok = True
                    elif blind:
                        ok = self._blind_accept(issue, opp_op, opp_val_internal)
                    elif combi:
                        ok = self._combi_accept(issue, action_state, opp_op,
                                                opp_val_internal)
                    elif pkg is not None:
                        ok = pkg                     # continuation rule
                    else:
                        ok = issue.model.accept(
                            my_op=my_op,
                            my_val=my_val_internal,
                            opp_op=opp_op,
                            opp_val=opp_val_internal,
                            risk=0.30,  # accept if ≥ 90% chance argmax lies in intersection set
                            mc=_mc(3000),
                            ctx_score=action_state.context_score,
                            severity=action_state.severity,
                            round_frac=action_state.round_frac,
                            deadline_beta=action_state.deadline_beta,
                        )
                    if ok:
                        prop[(issue.operand, issue.operator)] = (
                            issue.oppOperator,
                            issue.to_external(opp_val_internal),
                        )
                        issue.agreedOperator = issue.oppOperator
                        issue.agreedValue = issue.to_external(opp_val_internal)
                        issue.accepted = True

                elif issue.kind == "set":
                    # my_val_internal: set[str] I want (e.g. {"en"})
                    # opp_val_internal: set[str] in their offer
                    if not hard_ok:
                        ok = False
                    elif meets:
                        ok = True
                    elif blind:
                        ok = self._blind_accept(issue, opp_op, opp_val_internal)
                    elif combi:
                        ok = self._combi_accept(issue, action_state, opp_op,
                                                opp_val_internal)
                    elif pkg is not None:
                        ok = pkg                     # continuation rule
                    else:
                        ok = issue.model.accept(
                            my_op=my_op,
                            my_val=my_val_internal,
                            opp_op=opp_op,
                            opp_val=opp_val_internal,
                            tau_in=max(0.5, action_state.tau_set[0] - 0.05),
                            tau_out=action_state.tau_set[1],
                            jaccard_min=action_state.jaccard_snap_threshold,
                            mc=_mc(1500),
                            ctx_score=action_state.context_score,
                            severity=action_state.severity,
                            round_frac=action_state.round_frac,
                            deadline_beta=action_state.deadline_beta,
                        )
                    if ok:
                        prop[(issue.operand, issue.operator)] = (
                            issue.oppOperator,
                            issue.to_external(opp_val_internal),
                        )
                        issue.agreedOperator = issue.oppOperator
                        issue.agreedValue = issue.to_external(opp_val_internal)
                        issue.accepted = True

            if issue.accepted:
                continue  # already accepted, next issue

            # ---------- SAMPLING (fallback when we don't accept) ----------
            if issue.kind == "num":
                if getattr(action_state, "bidding_mode", "belief") == "time":
                    val_internal = self._timebid_numeric(issue, action_state)
                # The frontier search runs for the whole encounter, not only
                # while the concession schedule is still moving. Gating it on
                # `utility_mode` confined it to the first `concede_horizon` of
                # the deadline and handed the rest of every negotiation to the
                # raw posterior sampler, which is where the two-sided bid is
                # most needed. See _utility_bid_numeric.
                elif (issue.operator or "").lower() in ("gteq", "gt", "lteq", "lt"):
                    val_internal = self._utility_bid_numeric(issue, action_state)
                else:
                    prev_internal = issue.to_internal(issue.lastValue)
                    anchor_internal = issue.to_internal(issue.oppLastValue)
                    val_internal = issue.model.sample(
                        x=anchor_internal,
                        operator=issue.oppOperator,        # my numeric operator (eq/gteq/lteq/gt/lt/neq)
                        inertia_val=prev_internal,      # stick to previous with some prob
                        inertia=action_state.inertia,
                    )

                # Accelerate concession on a lone holdout issue (gated on context
                # + issue-severity + time), then never offer past my hard limit.
                val_internal = self._accelerate_concession(issue, action_state, val_internal)
                val_internal = self._clamp_to_hard(issue, val_internal)

                prop[(issue.operand, issue.operator)] = (
                    None,
                    issue.to_external(val_internal),
                )

            elif issue.kind == "enum":
                prev_lab_internal = issue.to_internal(issue.lastValue)
                if getattr(action_state, "bidding_mode", "belief") == "time":
                    opp_lab = (issue.to_internal(issue.oppLastValue)
                               if issue.oppLastValue is not None else None)
                    S = self._timebid_labels(
                        issue, action_state,
                        _as_label_set(prev_lab_internal), _as_label_set(opp_lab),
                        issue.operator,
                    )
                    lab_internal = (sorted(S)[0] if not isinstance(prev_lab_internal, (set, frozenset, list, tuple))
                                    else S)
                else:
                    lab_internal = issue.model.sample(
                        x=prev_lab_internal,
                        operator=issue.operator,        # 'eq' / 'isAnyOf' / 'isAllOf' / 'isNoneOf' / 'neq'
                        inertia=action_state.inertia,
                        max_k=action_state.max_set_k,
                    )
                # The enum sampler is given no hard-rule `required` mapping at
                # all, so project its output into my hard-feasible label set.
                lab_internal = self._project_to_hard_categorical(issue, lab_internal)
                prop[(issue.operand, issue.operator)] = (
                    None,
                    issue.to_external(lab_internal),
                )

            elif issue.kind == "set":
                prev_set_internal = issue.to_internal(issue.lastValue)
                if getattr(action_state, "bidding_mode", "belief") == "time":
                    opp_set = (issue.to_internal(issue.oppLastValue)
                               if issue.oppLastValue is not None else None)
                    sampled_internal = self._timebid_labels(
                        issue, action_state,
                        _as_label_set(prev_set_internal), _as_label_set(opp_set),
                        issue.operator,
                    )
                else:
                    sampled_internal = issue.model.sample(
                        x=prev_set_internal,
                        operator=issue.operator,        # 'isAllOf' / 'isAnyOf' / 'isNoneOf' / 'eq' / 'neq'
                        tau=action_state.tau_set,
                        inertia=action_state.inertia,
                        max_k=action_state.max_set_k,
                        required={rule.operator: rule.value for rule in issue.hard_rule}
                    )
                # `required` only ever *adds* labels inside SetBelief.sample, so an
                # `isAnyOf` hard rule does not bound the sampled set. Project.
                sampled_internal = self._project_to_hard_categorical(issue, sampled_internal)
                prop[(issue.operand, issue.operator)] = (
                    None,
                    issue.to_external(sampled_internal),
                )

        self._coordinate_numeric_ranges(prop)
        return prop

    def _coordinate_numeric_ranges(self, prop: Dict[IssueKey, Any]) -> None:
        """Keep merged numeric ranges valid: a same-operand lower bound must never
        exceed the upper bound. Several constraints on one numeric operand (e.g. a
        gteq + lteq window) form a single negotiable interval; this pass guarantees
        every emitted offer describes a non-empty interval."""
        by_operand: Dict[str, List[Issue]] = {}
        for issue in self.issue_models:
            if issue.kind == "num":
                by_operand.setdefault(issue.operand, []).append(issue)

        for operand, issues in by_operand.items():
            lowers = [i for i in issues if numeric_family(i.operator) == "lower"]
            uppers = [i for i in issues if numeric_family(i.operator) == "upper"]
            if not lowers or not uppers:
                continue
            lo_issue, hi_issue = lowers[0], uppers[0]
            klo = (operand, lo_issue.operator)
            khi = (operand, hi_issue.operator)
            if klo not in prop or khi not in prop:
                continue

            lo_i = lo_issue.to_internal(prop[klo][1])
            hi_i = hi_issue.to_internal(prop[khi][1])
            if lo_i <= hi_i:
                continue

            # Crossing bounds -> pull the not-yet-agreed side(s) back to a valid value.
            if lo_issue.accepted and not hi_issue.accepted:
                prop[khi] = (prop[khi][0], hi_issue.to_external(lo_i))
            elif hi_issue.accepted and not lo_issue.accepted:
                prop[klo] = (prop[klo][0], lo_issue.to_external(hi_i))
            else:
                mid = (lo_i + hi_i) / 2.0
                prop[klo] = (prop[klo][0], lo_issue.to_external(mid))
                prop[khi] = (prop[khi][0], hi_issue.to_external(mid))

    # --------------------------- MERGED CATEGORICAL ---------------------------

    @staticmethod
    def _emit_merged(prop: Dict[IssueKey, Any], operand: str, incl_op: str,
                     allowed_value, forbidden) -> None:
        for op, val in merged_issue_clauses(incl_op, allowed_value, forbidden):
            prop[(operand, op)] = (op, val)

    def _emit_range(self, prop: Dict[IssueKey, Any], issue: Issue, lo_ext, hi_ext) -> None:
        if lo_ext is not None and issue.lower_op:
            prop[(issue.operand, issue.lower_op)] = (issue.lower_op, lo_ext)
        if hi_ext is not None and issue.upper_op:
            prop[(issue.operand, issue.upper_op)] = (issue.upper_op, hi_ext)

    def _propose_numrange(self, issue: Issue, action_state: ActionState,
                          check_accept: bool, prop: Dict[IssueKey, Any]) -> None:
        INF = float("inf")
        my_lo = issue.to_internal(issue.lastValue[0]) if issue.lastValue[0] is not None else None
        my_hi = issue.to_internal(issue.lastValue[1]) if issue.lastValue[1] is not None else None

        if issue.accepted:
            self._emit_range(prop, issue, issue.lastValue[0], issue.lastValue[1])
            return

        opp_lo = opp_hi = None
        if issue.oppLastValue:
            opp_lo = issue.to_internal(issue.oppLastValue[0]) if issue.oppLastValue[0] is not None else None
            opp_hi = issue.to_internal(issue.oppLastValue[1]) if issue.oppLastValue[1] is not None else None

        strict = (issue.lower_op == "gt") or (issue.upper_op == "lt")

        # Hard-feasible bounds: the committed window may never cross them.
        hard_lo, hard_hi = self._hard_numeric_bounds(issue)

        # ---------- ACCEPTANCE: agree on the intersection of both windows ----------
        if issue.oppLastValue is not None and check_accept:
            L = max(x for x in (my_lo, opp_lo, -INF) if x is not None)
            U = min(x for x in (my_hi, opp_hi, INF) if x is not None)
            # Never commit a window outside my own hard policy.
            if hard_lo is not None:
                L = max(L, hard_lo)
            if hard_hi is not None:
                U = min(U, hard_hi)
            feasible = (L < U) if strict else (L <= U)
            if feasible and getattr(self, "_pkg_accept", True) is not False:
                lo_ext = issue.to_external(L) if L != -INF else None
                hi_ext = issue.to_external(U) if U != INF else None
                # Committing the intersection is what the OFFER says; it is not a
                # change to what I am asking for. See _NONDESTRUCTIVE_ACCEPT.
                issue.accepted = True
                self._emit_range(prop, issue, lo_ext, hi_ext)
                return

        # ---------- COUNTER: relax my window toward the opponent's ----------
        new_lo, new_hi = my_lo, my_hi
        belief_free = getattr(action_state, "bidding_mode", "belief") == "time"
        if issue.lower_op:
            if belief_free:
                new_lo = _timebid_bound(my_lo, opp_lo, hard_lo, action_state)
            else:
                anchor = opp_lo if opp_lo is not None else my_lo
                sampled = issue.model.sample(
                    x=anchor, operator=issue.lower_op,
                    inertia_val=my_lo, inertia=action_state.inertia,
                )
                # A counter-offer on a two-sided window must RELAX it: the
                # committed term is the intersection of the two windows, so
                # narrowing can only ever destroy an overlap that already
                # exists. The posterior still chooses how far to move; it may
                # not choose the direction. (Without this the belief branch
                # samples freely and both agents walk away from each other --
                # measured on the generated corpus, where two-sided ranges are
                # common; the 50 hand-authored scenarios contain none, so the
                # path is never exercised there.)
                new_lo = sampled if my_lo is None else min(my_lo, sampled)
        if issue.upper_op:
            if belief_free:
                new_hi = _timebid_bound(my_hi, opp_hi, hard_hi, action_state)
            else:
                anchor = opp_hi if opp_hi is not None else my_hi
                sampled = issue.model_hi.sample(
                    x=anchor, operator=issue.upper_op,
                    inertia_val=my_hi, inertia=action_state.inertia,
                )
                new_hi = sampled if my_hi is None else max(my_hi, sampled)

        # Never counter-offer a window past my own hard limits.
        if new_lo is not None and hard_lo is not None:
            new_lo = max(new_lo, hard_lo)
        if new_hi is not None and hard_hi is not None:
            new_hi = min(new_hi, hard_hi)

        # never let the interval collapse/invert
        if new_lo is not None and new_hi is not None and new_lo >= new_hi:
            new_lo, new_hi = my_lo, my_hi

        lo_ext = issue.to_external(new_lo) if new_lo is not None else None
        hi_ext = issue.to_external(new_hi) if new_hi is not None else None
        issue.lastValue = [lo_ext, hi_ext]
        self._emit_range(prop, issue, lo_ext, hi_ext)

    def _propose_merged(self, issue: Issue, action_state: ActionState,
                        check_accept: bool, prop: Dict[IssueKey, Any]) -> None:
        if issue.kind == "numrange":
            self._propose_numrange(issue, action_state, check_accept, prop)
            return

        operand = issue.operand
        incl_op = issue.operator
        my_allowed = set(issue.to_internal(issue.lastValue)) if issue.lastValue else set()
        my_forbidden = set(issue.forbidden)

        # Already agreed -> echo the agreed (opponent) combined values.
        if issue.accepted:
            agreed_allowed = issue.oppLastValue if issue.oppLastValue is not None else sorted(my_allowed)
            agreed_forbidden = issue.opp_forbidden if issue.opp_forbidden else my_forbidden
            self._emit_merged(prop, operand, issue.oppOperator or incl_op, agreed_allowed, agreed_forbidden)
            return

        opp_allowed = set(issue.to_internal(issue.oppLastValue)) if issue.oppLastValue is not None else set()
        opp_forbidden = set(issue.opp_forbidden)
        opp_op = issue.oppOperator

        # ---------- ACCEPTANCE ----------
        if issue.oppLastValue is not None and check_accept:
            my_required = my_allowed if incl_op in ("eq", "isAllOf") else set()
            opp_required = opp_allowed if opp_op in ("eq", "isAllOf") else set()
            forbidden_ok = my_forbidden.isdisjoint(opp_required) and opp_forbidden.isdisjoint(my_required)

            # Same two gates as the single-issue path: never sign a term that
            # breaches my hard policy; accept outright one that already meets my
            # demand (the belief product-confidence cannot veto it).
            hard_ok = (not getattr(action_state, "hard_veto", True)) or (
                not self._hard_commit_violation(
                    issue, opp_op, opp_allowed,
                    getattr(action_state, 'strict_categorical_commit', False)))
            if not hard_ok:
                forbidden_ok = False
            belief_ok = True
            if hard_ok and self._meets_my_demand(issue, opp_op, opp_allowed):
                pass
            elif getattr(action_state, "accept_mode", "belief") == "blind":
                # BAMBON-OM(acc): merged (inclusion + exclusion) issues take the
                # same belief-free route as the single-issue branches above.
                belief_ok = self._blind_accept(issue, opp_op, opp_allowed)
            elif getattr(action_state, "accept_mode", "belief") == "combi":
                belief_ok = self._combi_accept(issue, action_state, opp_op,
                                               opp_allowed)
            elif getattr(self, "_pkg_accept", None) is not None:
                belief_ok = self._pkg_accept         # continuation rule
            elif opp_allowed and incl_op in INCLUSION_OPS:
                belief_ok = issue.model.accept(
                    my_op=incl_op,
                    my_val=my_allowed,
                    opp_op=opp_op,
                    opp_val=opp_allowed,
                    tau_in=max(0.5, action_state.tau_set[0] - 0.05),
                    tau_out=action_state.tau_set[1],
                    jaccard_min=action_state.jaccard_snap_threshold,
                    mc=_mc(1500),
                    ctx_score=action_state.context_score,
                    severity=action_state.severity,
                    round_frac=action_state.round_frac,
                    deadline_beta=action_state.deadline_beta,
                )

            if forbidden_ok and belief_ok:
                issue.accepted = True
                self._emit_merged(prop, operand, opp_op or incl_op,
                                  sorted(opp_allowed) if opp_allowed else sorted(my_allowed),
                                  opp_forbidden if opp_forbidden else my_forbidden)
                return

        # ---------- SAMPLING (counter) ----------
        if incl_op in INCLUSION_OPS:
            required: Dict[str, set] = {}
            if my_forbidden:
                required["isNoneOf"] = set(my_forbidden)
            if incl_op in ("isAllOf", "eq") and my_allowed:
                required["isAllOf"] = set(my_allowed)

            if getattr(action_state, "bidding_mode", "belief") == "time":
                sampled = self._timebid_labels(
                    issue, action_state, my_allowed, opp_allowed, incl_op)
            else:
                sampled = issue.model.sample(
                    x=my_allowed,
                    operator=incl_op,
                    tau=action_state.tau_set,
                    inertia=action_state.inertia,
                    max_k=action_state.max_set_k,
                    required=required or None,
                )
            sampled = set(sampled) if sampled else set(my_allowed)
            sampled -= my_forbidden
            if not sampled:
                sampled = set(my_allowed)
            sampled = set(self._project_to_hard_categorical(issue, sampled))
            issue.lastValue = issue.to_external(sampled)
            self._emit_merged(prop, operand, incl_op, sorted(sampled), my_forbidden)
        else:
            # exclusion-only merged issue: keep my forbidden set
            self._emit_merged(prop, operand, "isNoneOf", sorted(my_forbidden), my_forbidden)
