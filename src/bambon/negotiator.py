from __future__ import annotations
from dataclasses import dataclass, field
from statistics import median, pstdev
from typing import Any, Dict, List, Optional, Tuple
import uuid
from .distributions.belief import BeliefRegistry
from .models.models import NegotiationInfeasible, Issue, ActionState
from .utils.helpers import strip_key, create_issue, clamp, format_operator_roperand, update_inertia



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
    _hopeless_streak: int = 0

    # runtime containers
    beliefs: Dict[str, BeliefRegistry] = field(init=False, default_factory=dict)
    action_states: Dict[str, ActionState] = field(init=False, default_factory=dict)



    def __post_init__(self) -> None:
        # Build per-action beliefs and states
        if not self.init_offer:
            raise ValueError("init_offer is required")

        # Safe access to hard constraints per action
        hard_by_action: Dict[str, Any] = {}
        if self.hard_policy:
            for p in self.hard_policy.get("permission", []):
                hard_by_action[str(p.get("action"))] = p

        for item in self.init_offer.get("permission", []):
            action = str(item.get("action"))
            constraints = item.get("constraint", [])
            hcs = hard_by_action.get(action, None)
            hard_constraints = (hcs or {}).get("constraint", [])

            key = strip_key(action)  # your helper
            # Belief registry per action
            self.beliefs[key] = BeliefRegistry(constraints, hard_constraints)

            # Initialize per-action adaptive state from GLOBAL defaults
            self.action_states[key] = ActionState(
                lr=self.lr,
                inertia=self.inertia,
                tau_set= [self.tau_in, self.tau_out], #self.tau_set,
                max_set_k=self.max_set_k,
                jaccard_snap_threshold=self.jaccard_snap_threshold
            )


    def _update_opponent_offer(self, offer):
        for item in offer['permission']:
            # TODO: Manage many same actions
            current_beliefs = self.beliefs.get(item['action'], None)
            if not current_beliefs:
                # Such action can not be negotiated, since the Agent has no initial offer
                # TODO: add in the hard constraints allowed actions
                continue

            for offer_constraint in item['constraint']:
                left_operand = strip_key(offer_constraint["leftOperand"])
                operator = strip_key(offer_constraint["operator"])
                value = offer_constraint["rightOperand"]
                issues_models = [i for i in current_beliefs.issue_models if i.operand == left_operand]


                if len(issues_models) == 0 and self.allow_unseen:
                    current_beliefs.issue_models.append(create_issue(left_operand, operator, value))
                elif len(issues_models) == 1:
                    if issues_models[0].accepted and \
                            (issues_models[0].oppLastValue != value or
                        issues_models[0].oppOperator != operator):
                        issues_models[0].accepted = False
                    issues_models[0].oppOperator = operator
                    issues_models[0].oppLastValue = value
                else:
                    for im in issues_models:
                        if im.operator == operator:
                            if im.accepted and \
                                    (im.oppLastValue != value or
                                     im.oppOperator != operator):
                                im.accepted = False
                            im.oppOperator = operator
                            im.oppLastValue = value
                            break



    def observe(self, opp_offer: Dict[str, Any]) -> Tuple[Dict[str, Tuple[float, float]] | None, Dict[str, Any]]:
        self.rounds_seen += 1
        if self.rounds_seen > self.max_rounds:
            raise NegotiationInfeasible(f"{self.name}: Maximum allowed rounds exceeded.")

        # Parse opponents offer
        self._update_opponent_offer(opp_offer)

        # Update Context and Severity scores
        self._context_and_severity()

        # Patient hopelessness check
        if self.hopeless_window:
            for action, state in self.action_states.items():
                if (
                        self.rounds_seen >= self.hopeless_window and 
                        state.context_score < self.hopeless_context and
                        state.severity >= self.hopeless_severity
                ):
                    self._hopeless_streak += 1
                else:
                    self._hopeless_streak = 0
                if self._hopeless_streak >= self.hopeless_patience:
                    raise NegotiationInfeasible(
                        f"{self.name}: negotiation appears infeasible for action {action}"
                        f"(severity={state.severity:.2f} for {self._hopeless_streak} consecutive rounds, "
                        f"context={state.context_score:.2f})"
                    )
                else:
                    self._hopeless_streak = 0

        # Check for hard constraint violation
        for action, belief in self.beliefs.items():
            conf = self._compute_signals(action, belief.issue_models)
            if conf:
                self._adapt(action, conf)

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


        accepted = {
            action: beliefs.issue_models
            for action, beliefs in self.beliefs.items()
            if all([issue.accepted for issue in beliefs.issue_models])
        }

        if accepted:
            self._context_and_severity(True)
            return {key: (val.context_score, val.severity) for key,val in self.action_states.items() if key in accepted.keys()}, self._construct_agreement(accepted)

        else:
            self._update_last_values(predictions)
            return None, self._construct_counter_offer(predictions)

    def _update_last_values(self, predictions):
        for action_key, constraints in predictions.items():
            belief = self.beliefs.get(action_key)
            for issue in belief.issue_models:
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

            sims = [_issue_similarity(it, init) for it in constraints.issue_models]
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

    def _construct_agreement(self, permissions: Dict[str, List[Issue]]):

        resp_offer = self.init_offer.copy()
        resp_offer['@id'] = f'urn:uid:{uuid.uuid4()}'
        resp_offer['@type'] = 'odrl:Agreement'
        resp_offer['permission'] = [
            {
                'action': action,
                'constraint': [
                    {
                        'leftOperand': f'odrl:{issue.operand}',
                        'operator': f'odrl:{issue.oppOperator}',
                        'rightOperand': issue.oppLastValue
                    }
                    for issue in issues
                ]
            }
            for action, issues in permissions.items()
        ]

        return resp_offer

    def _construct_counter_offer(self, permissions: Dict[str, Dict[Tuple[str, str], Any]]):
        resp_offer = self.init_offer.copy()
        resp_offer['@id'] = f'urn:uid:{uuid.uuid4()}'
        resp_offer['permission'] = [
            {
                'action': action,
                'constraint': [
                    {
                        'leftOperand': f'odrl:{operand}',
                        'operator': f'odrl:{format_operator_roperand(value[0],value[1])[0] if value[0] else format_operator_roperand(operator,value[1])[0]}',
                        'rightOperand': format_operator_roperand(operator,value[1])[1]
                    }
                    for (operand, operator), value in constraints.items()
                ]
            }
            for action, constraints in permissions.items()
        ]

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
