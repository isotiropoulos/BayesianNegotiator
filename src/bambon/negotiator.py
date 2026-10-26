from __future__ import annotations
from dataclasses import dataclass, field
from statistics import median, pstdev
from typing import Any, Dict, List, Optional, Tuple
import uuid
from .distributions.belief import BeliefRegistry
from .models.models import NegotiationInfeasible, Issue, ActionState
from .utils.utility import aspiration as aspiration_curve, accept_prob_floor
from .utils.helpers import (
    strip_key,
    create_issue,
    clamp,
    format_operator_roperand,
    update_inertia,
    numeric_family,
    split_categorical_clauses,
    merged_issue_clauses,
)


# ODRL rule types that are negotiated with identical logic.
RULE_TYPES: Tuple[str, ...] = ("permission", "obligation", "prohibition")

# A negotiable unit is identified by its rule type and action, e.g. ("permission", "use").
RegistryKey = Tuple[str, str]


@dataclass
class Bambon:
    """ Adaptive negotiator with beliefs, set snapping, and patient feasibility checks. """

    name: str
    # GLOBAL DEFAULTS (used to init each action's state)
    lr: float = 0.4
    inertia: float = 0.4
    tau_in: float = 0.5
    tau_out: float = 0.6
    max_set_k: int = 3

    # Policy / offers
    hard_policy: Optional[Dict[str, Any]] = None
    init_offer: Optional[Dict[str, Any]] = None
    allow_unseen: bool = True

    # Misc knobs
    no_agreement_until:int = 3
    hopeless_window: int = 10
    jaccard_snap_threshold: float = 0.60
    max_rounds: int = 60
    tau_cooling_threshold: int = 0.5
    tau_in_cooled: float = 0.65
    tau_out_cooled: float = 0.65
    rounds_seen: int = 0
    lr_min: float = 0.2
    lr_max: float = 1.0
    vol_window: int = 5
    max_param_delta: float = 0.15
    hopeless_severity: float = 0.95
    hopeless_patience: int = 8
    hopeless_context: float = 0.15

    # Which infeasibility rule to use.
    #   'severity'    (default, as evaluated in the paper) -- decline when the two
    #       sides have stayed far apart for `hopeless_patience` rounds. Simple, but
    #       it measures "are we far apart *now*", which is true early in every hard
    #       negotiation, so on a corpus where a lawful contract always exists it
    #       produces only FALSE declines (measured: 0.38 of encounters, and no
    #       (window, patience) setting separates true declines from false ones).
    #   'reservation' -- decline only when no signable term remains: the standing
    #       term is forbidden by my own hard policy AND I am already at my
    #       reservation on that issue AND the deadline is close
    #       (`hopeless_min_round_frac`). Sound in the sense that it never fires
    #       while the agent still has something to concede.
    #
    # ADOPTED DEFAULT (changed from 'severity'): measured on 50 feasible + 40
    # infeasible scenarios x 3 opponents, 'reservation' strictly dominates
    # 'severity' on every axis -- false declines 0.380 -> 0.000, Nash|agreed
    # 0.339 -> 0.347, true declines on infeasible scenarios 0.133 -> 0.600.
    # Set hopeless_mode='severity' to reproduce the originally submitted numbers.
    hopeless_mode: str = "reservation"
    hopeless_min_round_frac: float = 0.5    # only used by hopeless_mode='reservation'

    # --- utility-aware policy (proposals #1/#2/#3/#7) ---
    # Master switch. Set utility_mode=False for the original belief-only behaviour.
    utility_mode: bool = True
    concession_beta: float = 1.0      # 1 linear; >1 Boulware (concede late, greedier); <1 Conceder (more deals)
    aspiration_start: float = 1.0     # open near our initial offer (utility 1.0)
    reservation_utility: float = 0.0  # least self-utility we will ever settle for
    p_accept_min: float = 0.60        # accept-prob floor early (moderate bids; avoids tripping opponent hopelessness)
    p_accept_max: float = 0.85        # accept-prob floor near deadline (play safe)
    reservation_span: float = 0.5     # reservation width when no hard bound exists
    time_cost: float = 0.02           # per-round impatience used in accept-vs-wait
    deadline: Optional[int] = None    # rounds to plan concession over (default max_rounds)
    concede_horizon: float = 0.3      # fully conceded (and revert to original) by this fraction of the deadline

    # --- ablation switch: where the bid comes from ---
    # 'belief' is the full model. 'time' removes the Bayesian opponent model from
    # *bidding only*: posteriors are still updated and still drive acceptance,
    # but the proposal becomes a memoryless time-scheduled concession computed
    # from this agent's own aspiration and reservation. Used for the
    # BAMBON-OM(bid) arm of the ablation; see ActionState.bidding_mode.
    bidding_mode: str = "belief"

    # --- ablation switch: where ACCEPTANCE comes from (complement of the above) ---
    # 'belief' is the full model. 'blind' removes the Bayesian opponent model from
    # *acceptance only*: posteriors are still updated and still drive the bid, but
    # the accept test becomes the classic time-dependent rule (accept a term at
    # least as good as my current ask, which itself concedes along A(t)). The
    # hard-policy veto and the repair projection stay in force, so this arm is not
    # a compliance experiment. Used for the BAMBON-OM(acc) arm; see
    # ActionState.accept_mode.
    accept_mode: str = "combi"


    # Require a committed categorical term to be *contained* in this agent's own
    # allow-list rather than merely to overlap it. Off by default: it is a
    # stricter reading of the hard policy than Def. 2 scores, so turning it on
    # changes both behaviour and the compliance numbers. See ActionState.
    strict_categorical_commit: bool = False

    # Commit-time hard veto. Off only for the BAMBON-NoVeto ablation; see
    # ActionState.hard_veto.
    hard_veto: bool = True

    # --- deadline-pressure on acceptance (always active, independent of utility_mode) ---
    # rho_eff ramps toward rho_max as the deadline nears so two patient agents
    # (e.g. Bambon-vs-Bambon, wide ZOPA) close instead of timing out. beta > 1
    # keeps the pressure mild until the late game (Boulware-style).
    deadline_beta: float = 2.0

    # --- accelerated concession on a lone holdout numeric issue ---
    # Speed up convergence on a still-contested issue once the rest of the action
    # has aligned, so wide-gap negotiations cross before the round cap instead of
    # timing out. Conservative defaults; set concede_gain=0.0 to disable.
    concede_gain: float = 0.5         # 0 disables; higher = faster concession on laggard issues
    concede_step_cap: float = 0.34    # max fraction of the remaining gap closed per round
    concede_ctx_min: float = 0.6      # only when context (rest of deal) >= this
    concede_sev_hot: float = 0.5      # only when this issue's severity (1 - sigma_i) >= this

    # runtime containers
    # Keyed by (rule_type, action), e.g. ("permission", "use"), ("obligation", "delete").
    beliefs: Dict[RegistryKey, BeliefRegistry] = field(init=False, default_factory=dict)
    action_states: Dict[RegistryKey, ActionState] = field(init=False, default_factory=dict)



    def __post_init__(self) -> None:
        # Build per-(rule_type, action) beliefs and states.
        if not self.init_offer:
            raise ValueError("init_offer is required")

        # Permissions, obligations and prohibitions are negotiated with identical logic.
        for rule_type in RULE_TYPES:
            # Safe access to hard constraints per action for this rule type
            hard_by_action: Dict[str, Any] = {}
            if self.hard_policy:
                for p in self.hard_policy.get(rule_type, []) or []:
                    hard_by_action[strip_key(str(p.get("action")))] = p

            for item in self.init_offer.get(rule_type, []) or []:
                action = strip_key(str(item.get("action")))
                constraints = item.get("constraint", [])
                hcs = hard_by_action.get(action, None)
                hard_constraints = (hcs or {}).get("constraint", [])

                key: RegistryKey = (rule_type, action)
                # Belief registry per (rule_type, action)
                self.beliefs[key] = BeliefRegistry(constraints, hard_constraints)

                # Initialize per-action adaptive state from GLOBAL defaults
                self.action_states[key] = ActionState(
                    lr=self.lr,
                    inertia=self.inertia,
                    tau_set= [self.tau_in, self.tau_out], #self.tau_set,
                    max_set_k=self.max_set_k,
                    jaccard_snap_threshold=self.jaccard_snap_threshold,
                    bidding_mode=self.bidding_mode,
                    accept_mode=self.accept_mode,
                    strict_categorical_commit=self.strict_categorical_commit,
                    hard_veto=self.hard_veto,
                )


    def _update_opponent_offer(self, offer):
        for rule_type in RULE_TYPES:
            for item in offer.get(rule_type, []) or []:
                key: RegistryKey = (rule_type, strip_key(str(item['action'])))
                current_beliefs = self.beliefs.get(key, None)
                if not current_beliefs:
                    # Such (rule_type, action) can not be negotiated, since the Agent has no initial offer
                    # TODO: add in the hard constraints allowed actions
                    continue

                self._update_beliefs_from_constraints(current_beliefs, item.get('constraint', []))

    def _update_beliefs_from_constraints(self, current_beliefs, constraints):
        # Group the opponent's constraints by operand so that several constraints on
        # the same operand are routed together (merged categorical, or numeric range).
        opp_by_operand: Dict[str, List[Dict[str, Any]]] = {}
        order: List[str] = []
        for offer_constraint in constraints:
            operand = strip_key(offer_constraint["leftOperand"])
            rec = {
                "leftOperand": operand,
                "operator": strip_key(offer_constraint["operator"]),
                "rightOperand": offer_constraint["rightOperand"],
            }
            if operand not in opp_by_operand:
                opp_by_operand[operand] = []
                order.append(operand)
            opp_by_operand[operand].append(rec)

        for operand in order:
            self._route_operand(current_beliefs, operand, opp_by_operand[operand])

    def _route_operand(self, current_beliefs, operand, opp_clauses):
        my_issues = [i for i in current_beliefs.issue_models if i.operand == operand]

        if not my_issues:
            if self.allow_unseen:
                self._add_unseen_operand(current_beliefs, operand, opp_clauses)
                my_issues = [i for i in current_beliefs.issue_models if i.operand == operand]
            else:
                return

        if not my_issues:
            return

        if len(my_issues) == 1 and my_issues[0].kind == "numrange":
            self._route_numrange(my_issues[0], opp_clauses)
        elif len(my_issues) == 1 and (my_issues[0].merged or len(opp_clauses) > 1) and my_issues[0].kind not in ("num",):
            # One merged categorical issue, or a single categorical issue facing a multi-clause operand.
            self._route_merged(my_issues[0], opp_clauses)
        elif all(i.kind == "num" for i in my_issues):
            self._route_numeric(my_issues, opp_clauses)
        elif len(my_issues) == 1 and len(opp_clauses) == 1:
            c = opp_clauses[0]
            self._assign_opp(my_issues[0], c["operator"], c["rightOperand"])
        else:
            self._route_merged(my_issues[0], opp_clauses)

    def _route_numrange(self, issue, opp_clauses):
        lower_vals, upper_vals = [], []
        for c in opp_clauses:
            op = c["operator"]
            if op in ("gt", "gteq"):
                lower_vals.append(c["rightOperand"])
            elif op in ("lt", "lteq"):
                upper_vals.append(c["rightOperand"])
        opp_lo = max(lower_vals, key=issue.to_internal) if lower_vals else None
        opp_hi = min(upper_vals, key=issue.to_internal) if upper_vals else None
        new_val = [opp_lo, opp_hi]
        if issue.accepted and issue.oppLastValue != new_val:
            issue.accepted = False
            issue.agreedValue = issue.agreedOperator = None
        issue.oppLastValue = new_val

    def _assign_opp(self, issue, operator, value):
        if issue.accepted and (issue.oppLastValue != value or issue.oppOperator != operator):
            issue.accepted = False
            issue.agreedValue = issue.agreedOperator = None
        issue.oppOperator = operator
        issue.oppLastValue = value

    def _route_numeric(self, my_issues, opp_clauses):
        opp_by_family: Dict[str, List[Dict[str, Any]]] = {}
        for c in opp_clauses:
            opp_by_family.setdefault(numeric_family(c["operator"]), []).append(c)
        for issue in my_issues:
            fam = numeric_family(issue.operator)
            candidates = opp_by_family.get(fam)
            if not candidates and len(opp_clauses) == 1:
                candidates = opp_clauses
            if candidates:
                c = candidates[0]
                self._assign_opp(issue, c["operator"], c["rightOperand"])

    def _route_merged(self, issue, opp_clauses):
        allowed, forbidden, incl_op = split_categorical_clauses(opp_clauses)
        new_val = sorted(allowed) if allowed else None
        changed = (
            issue.oppLastValue != new_val
            or issue.oppOperator != incl_op
            or set(issue.opp_forbidden) != set(forbidden)
        )
        if issue.accepted and changed:
            issue.accepted = False
            issue.agreedValue = issue.agreedOperator = None
        issue.oppOperator = incl_op
        issue.oppLastValue = new_val
        issue.opp_forbidden = set(forbidden)

    def _add_unseen_operand(self, current_beliefs, operand, opp_clauses):
        from .distributions.belief import BeliefRegistry
        new_issues = BeliefRegistry._parse_offer_constraints(opp_clauses, None)
        current_beliefs.issue_models.extend(new_issues)



    def observe(self, opp_offer: Dict[str, Any]) -> Tuple[Dict[str, Tuple[float, float]] | None, Dict[str, Any]]:
        self.rounds_seen += 1
        if self.rounds_seen > self.max_rounds:
            raise NegotiationInfeasible(f"{self.name}: Maximum allowed rounds exceeded.")

        # Parse opponents offer
        self._update_opponent_offer(opp_offer)

        # Update Context and Severity scores
        self._context_and_severity()

        # Patient hopelessness check (tracked independently per (rule_type, action))
        if self.hopeless_window and self.hopeless_mode == "severity":
            for (rule_type, action), state in self.action_states.items():
                if (
                        self.rounds_seen >= self.hopeless_window and
                        state.context_score is not None and
                        state.severity is not None and
                        state.context_score < self.hopeless_context and
                        state.severity >= self.hopeless_severity
                ):
                    state._hopeless_streak += 1
                else:
                    state._hopeless_streak = 0
                if state._hopeless_streak >= self.hopeless_patience:
                    raise NegotiationInfeasible(
                        f"{self.name}: negotiation appears infeasible for {rule_type} action '{action}' "
                        f"(severity={state.severity:.2f} for {state._hopeless_streak} consecutive rounds, "
                        f"context={state.context_score:.2f})"
                    )
        elif self.hopeless_window and self.hopeless_mode == "reservation":
            # Reservation-based infeasibility. The severity rule above asks "are
            # we far apart?", which is true early in *every* hard negotiation --
            # so its declines and its false declines come from the same signal
            # and no setting of (window, patience) separates them. This rule asks
            # the question that actually matters: "is there a term left that I
            # could sign?" It fires only when (a) the deadline is close enough
            # that waiting cannot help, (b) the opponent's standing term is one my
            # own hard policy forbids me to sign, and (c) I have nothing left to
            # concede on that issue -- sustained for `hopeless_patience` rounds.
            _frac = clamp(self.rounds_seen / max(1, self.deadline or self.max_rounds), 0.0, 1.0)
            for (rule_type, action), state in self.action_states.items():
                belief = self.beliefs.get((rule_type, action))
                stuck = (
                    self.rounds_seen >= self.hopeless_window
                    and _frac >= self.hopeless_min_round_frac
                    and belief is not None
                    and belief.reservation_deadlock(state)
                )
                state._hopeless_streak = state._hopeless_streak + 1 if stuck else 0
                if state._hopeless_streak >= self.hopeless_patience:
                    raise NegotiationInfeasible(
                        f"{self.name}: no signable term remains for {rule_type} action "
                        f"'{action}' -- the standing term is outside my hard policy and I am "
                        f"at my reservation ({state._hopeless_streak} consecutive rounds, "
                        f"t={_frac:.2f})"
                    )

        # Check for hard constraint violation
        for action, belief in self.beliefs.items():
            conf = self._compute_signals(action, belief.issue_models)
            if conf:
                self._adapt(action, conf)

        # Time-aware acceptance (always on, independent of utility_mode): expose
        # the deadline fraction and its shape on every action state so the belief
        # accept() can ramp rho_eff toward rho_max as the deadline approaches.
        _deadline = self.deadline or self.max_rounds
        _round_frac = clamp(self.rounds_seen / max(1, _deadline), 0.0, 1.0)
        for st in self.action_states.values():
            st.round_frac = _round_frac
            st.deadline_rounds = _deadline
            st.bidding_mode = self.bidding_mode
            st.accept_mode = self.accept_mode
            st.strict_categorical_commit = self.strict_categorical_commit
            st.hard_veto = self.hard_veto
            st.deadline_beta = self.deadline_beta
            st.concede_gain = self.concede_gain
            st.concede_step_cap = self.concede_step_cap
            st.concede_ctx_min = self.concede_ctx_min
            st.concede_sev_hot = self.concede_sev_hot

        # Concession schedule: refresh the per-round aspiration A(t) and the
        # acceptance-probability floor, then push them onto every action state so
        # propose()/accept() can read them.
        #
        # This runs unconditionally. Both bidders read `st.aspiration` -- the
        # belief bidder as the boundary between its two search passes, the
        # belief-free bidder of the BAMBON-OM(bid) ablation as the bid itself --
        # so skipping the refresh would leave A(t) pinned at its ActionState
        # default of 1.0 and produce an agent that never concedes. `utility_mode`
        # still switches off the utility-aware ACCEPTANCE gate, which is the only
        # thing it ever ought to have governed.
        deadline = self.deadline or self.max_rounds
        round_frac = clamp(self.rounds_seen / max(1, deadline), 0.0, 1.0)
        # Reach full concession by `concede_horizon` of the deadline so there
        # are rounds left to actually converge (protects agreement rate).
        eff_frac = clamp(round_frac / max(1e-6, self.concede_horizon), 0.0, 1.0)
        asp = aspiration_curve(eff_frac, self.aspiration_start,
                               self.reservation_utility, self.concession_beta)
        p_floor = accept_prob_floor(eff_frac, self.p_accept_min, self.p_accept_max)
        # The utility-aware acceptance gate stays confined to the conceding
        # phase; past the horizon acceptance reverts to plain BAMBON.
        active = round_frac < self.concede_horizon
        for st in self.action_states.values():
            st.utility_mode = active and self.utility_mode
            st.aspiration = asp
            st.p_floor = p_floor
            st.round_frac = round_frac
            st.time_cost = self.time_cost
            st.reservation_span = self.reservation_span

        # Train & Predict
        predictions = {}
        compliant_actions = []
        for action_str, belief in self.beliefs.items():
            action = self.action_states.get(action_str, None)
            if action:
                # Train and check if any hard constraint failed
                hc_failed = belief.observe(action)
                if not hc_failed:
                    compliant_actions.append(action_str)

                # Make predictions
                if self.rounds_seen >= self.no_agreement_until:
                    predictions[action_str] = belief.propose(action)
                else:
                    predictions[action_str] = belief.propose(action, False)


        # An agreement is only reached when EVERY negotiated unit (each action across
        # permissions, obligations and prohibitions) is fully accepted. Otherwise we keep
        # counter-offering on all units. This prevents prematurely agreeing on a subset of
        # actions/rules while others are still being negotiated (which previously blocked
        # multi-action/multi-rule negotiations from ever concluding).
        #
        # ``no_agreement_until`` gates this too, and must. It used to gate only
        # the ``propose()`` call above -- but ``predictions`` is discarded
        # whenever ``all_accepted`` is true, so the knob could not delay an
        # agreement by a single round. "No agreement until round N" silently
        # meant nothing, for every value.
        #
        # It also decides the degenerate case. ``all(... for ... in [])`` is
        # vacuously true, so an offer whose rules carry no constraints --
        # nothing to negotiate -- produced an immediate agreement on round 1
        # regardless of configuration. That is defensible on its own (there is
        # genuinely nothing to argue about), but it meant a constraint-free
        # offer could never be counter-offered. See
        # tests/test_no_agreement_until.py.
        all_accepted = (
            self.rounds_seen >= self.no_agreement_until
            and bool(self.beliefs)
            and all(
                all(issue.accepted for issue in beliefs.issue_models)
                for beliefs in self.beliefs.values()
            )
        )

        if all_accepted:
            self._context_and_severity(True)
            accepted = {key: beliefs.issue_models for key, beliefs in self.beliefs.items()}
            return {key: (val.context_score, val.severity) for key, val in self.action_states.items()}, self._construct_agreement(accepted)

        else:
            self._update_last_values(predictions)
            return None, self._construct_counter_offer(predictions)

    def _update_last_values(self, predictions):
        for action_key, constraints in predictions.items():
            belief = self.beliefs.get(action_key)
            for issue in belief.issue_models:
                if issue.merged:
                    # merged issues update their own lastValue during propose()
                    continue
                if issue.accepted:
                    # The prediction for an accepted issue IS the counterparty's
                    # term -- that is what the emitted offer commits to. Copying
                    # it onto my own standing demand would concede the issue
                    # outright, and `_assign_opp` only ever clears `accepted`,
                    # never this write, so the concession could not be undone
                    # when the counterparty moved. No agreement has been
                    # concluded yet, so my ask does not move.
                    continue
                try:
                    pred_operator, pred_value = constraints[ (issue.operand, issue.operator) ]
                except:
                    raise
                if isinstance(pred_value, list) and issue.operator == 'eq':
                    if len(pred_value) == 1:
                        pred_value = pred_value[0]
                    else:
                        issue.operator = 'isAllOf'
                if isinstance(pred_value, list) and issue.operator == 'neq':
                    if len(pred_value) == 1:
                        pred_value = pred_value[0]
                    else:
                        issue.operator = 'isNoneOf'
                if not isinstance(pred_value, list) and issue.operator in ('isAllOf', 'isAnyOf', 'isNoneOf'):
                    if issue.operator == 'isNoneOf':
                        issue.operator = 'neq'
                    else:
                        issue.operator = 'eq'
                issue.operator = pred_operator if pred_operator else issue.operator
                issue.lastValue = pred_value


    def _context_and_severity(self, init=False): # -> Dict[str, Tuple[float, float]]:
        """Weighted similarity across issues (0..1)."""
        from .utils.context import _issue_similarity
        for action, constraints in self.beliefs.items():

            current_action = self.action_states.get(action, None)
            if not current_action:
                continue

            sims = []
            for it in constraints.issue_models:
                sim = _issue_similarity(it, init)
                it.similarity = sim          # stash per-issue alignment for accelerated concession
                sims.append(sim)
            severities = [1 - sim for sim in sims]

            if not sims:
                ctx_score = 0.0
            else:
                ctx_score = sum(sims) / len(sims)

            if not severities:
                severity = 0.0
            else:
                ssorted = sorted(severities)
                q95 = ssorted[int(0.95 * (len(ssorted) - 1))]
                wmax = max(severities)
                severity = float(clamp(0.7 * wmax + 0.3 * q95, 0.0, 1.0))

            current_action.severity = severity
            current_action.context_score = ctx_score
        # return out

    def _construct_agreement(self, accepted: Dict[RegistryKey, List[Issue]]):
        resp_offer = self.init_offer.copy()
        resp_offer['@id'] = f'urn:uid:{uuid.uuid4()}'
        resp_offer['@type'] = 'odrl:Agreement'

        # Group accepted units back into their rule type (permission/obligation/prohibition).
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for (rule_type, action), issues in accepted.items():
            constraint_list: List[Dict[str, Any]] = []
            for issue in issues:
                if issue.kind == "numrange":
                    lo_ext, hi_ext = issue.lastValue
                    if lo_ext is not None and issue.lower_op:
                        constraint_list.append({
                            'leftOperand': f'odrl:{issue.operand}',
                            'operator': f'odrl:{issue.lower_op}',
                            'rightOperand': lo_ext,
                        })
                    if hi_ext is not None and issue.upper_op:
                        constraint_list.append({
                            'leftOperand': f'odrl:{issue.operand}',
                            'operator': f'odrl:{issue.upper_op}',
                            'rightOperand': hi_ext,
                        })
                elif issue.merged:
                    for op, val in merged_issue_clauses(issue.oppOperator, issue.oppLastValue, issue.opp_forbidden):
                        constraint_list.append({
                            'leftOperand': f'odrl:{issue.operand}',
                            'operator': f'odrl:{op}',
                            'rightOperand': val,
                        })
                else:
                    constraint_list.append({
                        'leftOperand': f'odrl:{issue.operand}',
                        'operator': f'odrl:{issue.oppOperator}',
                        'rightOperand': issue.oppLastValue,
                    })
            grouped.setdefault(rule_type, []).append({
                'action': action,
                'constraint': constraint_list,
            })

        for rule_type in RULE_TYPES:
            if rule_type in grouped:
                resp_offer[rule_type] = grouped[rule_type]

        return resp_offer

    def _construct_counter_offer(self, predictions: Dict[RegistryKey, Dict[Tuple[str, str], Any]]):
        resp_offer = self.init_offer.copy()
        resp_offer['@id'] = f'urn:uid:{uuid.uuid4()}'

        # Group predicted units back into their rule type (permission/obligation/prohibition).
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for (rule_type, action), constraints in predictions.items():
            grouped.setdefault(rule_type, []).append({
                'action': action,
                'constraint': [
                    {
                        'leftOperand': f'odrl:{operand}',
                        'operator': f'odrl:{format_operator_roperand(value[0],value[1])[0] if value[0] else format_operator_roperand(operator,value[1])[0]}',
                        'rightOperand': format_operator_roperand(operator,value[1])[1]
                    }
                    for (operand, operator), value in constraints.items()
                ]
            })

        for rule_type in RULE_TYPES:
            if rule_type in grouped:
                resp_offer[rule_type] = grouped[rule_type]

        return resp_offer

    # # ---------------- Signals & adaptation ---------------- #
    def _compute_signals(self, state:str, issues: List[Issue]) -> Dict[str, float] | None:
        jaccs: List[float] = []
        num_diffs: List[float] = []
        opp_set_sizes_round: List[int] = []

        current_action = self.action_states.get(state, None)
        if not current_action:
            return None
        for issue in issues:
            if issue.oppLastValue is None:
                continue
            if issue.kind == "enum":
                continue
            elif issue.kind == "set":
                A = issue.to_internal(issue.lastValue); B = issue.to_internal(issue.oppLastValue)
                j = (len(A & B) / len(A | B)) if (A or B) else 1.0
                jaccs.append(j)
                opp_set_sizes_round.append(len(B))
            elif issue.kind == "num":
                ai = issue.to_internal(issue.lastValue); bi = issue.to_internal(issue.oppLastValue)
                num_diffs.append(abs(ai - bi))

        jacc = (sum(jaccs) / len(jaccs)) if jaccs else 0.0
        num_move = median(num_diffs) if num_diffs else 0.0

        current_action._num_move_hist.append(num_move)
        if len(current_action._num_move_hist) > self.vol_window:
            current_action._num_move_hist.pop(0)
        if len(current_action._num_move_hist) >= 2:
            m = sum(current_action._num_move_hist) / len(current_action._num_move_hist)
            s = pstdev(current_action._num_move_hist)
            vol = s / (m + 1e-6)
        else:
            vol = 0.0
        vol = clamp(vol, 0.0, 1.0)

        if opp_set_sizes_round:
            current_action._opp_set_sizes.append(int(median(opp_set_sizes_round)))
            if len(current_action._opp_set_sizes) > self.vol_window:
                current_action._opp_set_sizes.pop(0)
        opp_set_size_median = int(median(current_action._opp_set_sizes)) if current_action._opp_set_sizes else 1

        return dict(jacc=jacc, vol=vol, opp_set_size_median=opp_set_size_median)


    def _adapt(self, state:str, conf: Dict[str, float]) -> None:
        r = self.rounds_seen
        R = max(1, self.max_rounds)
        t = r / R
        current_state = self.action_states.get(state, None)
        if not current_state:
            return

        # for key, value in conf.items():
        jacc = clamp(conf.get("jacc", 0.0), 0.0, 1.0)
        vol = clamp(conf.get("vol", 0.0), 0.0, 1.0)
        opp_k = max(1, int(conf.get("opp_set_size_median", 1)))

        def bounded_update(cur: float, target: float, lo: float, hi: float) -> float:
            step = clamp(target - cur, -self.max_param_delta, self.max_param_delta)
            return clamp(cur + step, lo, hi)

        lr_target = self.lr_max * (1 - 0.7 * t) / (1 + 2 * vol)
        lr_target = clamp(lr_target, self.lr_min, self.lr_max)
        current_state.lr = bounded_update(current_state.lr, lr_target, self.lr_min, self.lr_max)

        tau_in_target = 0.55 + 0.25 * t + 0.2 * (jacc - 0.5)
        tau_out_target = 0.65 + 0.10 * t - 0.15 * (jacc - 0.5)

        if jacc < 0.30:
            tau_in_target -= 0.10
        tau_in_target = clamp(tau_in_target, 0.35, 0.9)
        if t >= self.tau_cooling_threshold:
            tau_in_target = max(tau_in_target, self.tau_in_cooled)
            tau_out_target = max(tau_out_target, self.tau_out_cooled)

        tau_out_target = clamp(tau_out_target, 0.4, 0.9)
        current_state.tau_set = [
            bounded_update(current_state.tau_set[0], tau_in_target, 0.35, 0.9),
            bounded_update(current_state.tau_set[1], tau_out_target, 0.35, 0.9),
        ]

        current_state.max_set_k = int(clamp(max(3, min(6, opp_k + 1)), 2, 8))

        current_state.inertia = update_inertia(
            ctx=current_state.context_score,
            sev=current_state.severity,
            iota_min=0.05,
            iota_max=0.65,
            ctx_switch=0.70,
            use_severity=True,
        )
