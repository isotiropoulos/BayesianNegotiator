import copy
import os
import sys
import random
import json
from collections import defaultdict
from typing import Dict, Optional, List, Tuple, Any
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# 0. BAMBON imports
# ----------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.bambon.negotiator import Bambon
from src.bambon.models.models import NegotiationInfeasible
from src.bambon.utils.helpers import (
    is_iso_date, iso_to_days, is_iso_duration, duration_to_seconds, satisfies,
)
from utility import UtilityConfig, odrl_utility
from helpers import nash_bargaining_analysis

# Single shared ODRL-utility configuration (deontic rule weights, quadratic
# numeric penalty, weighted set overlap, exp hard-constraint penalty H=exp(-λV)).
ODRL_UTILITY_CONFIG = UtilityConfig()

NUM_OPS = ('lt', 'lteq', 'gt', 'gteq', 'eq', 'neq')

# A bilateral negotiation has no natural upper bound for stubborn baseline
# agents (only BAMBON raises NegotiationInfeasible). Without a deadline two
# baselines that never reach a mutual accept loop forever -- this is what made
# QLearning-as-provider "take forever". MAX_TURNS acts as the negotiation
# deadline: if no agreement is reached in time, it counts as a failed deal.
MAX_TURNS = 200


def fetch_data(case: str):
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', case)
    with open(f'{base}/provider.json', 'r', encoding='utf-8') as f:
        provider = json.load(f)
    with open(f'{base}/consumer.json', 'r', encoding='utf-8') as f:
        consumer = json.load(f)
    with open(f'{base}/hard_provider.json', 'r', encoding='utf-8') as f:
        hard_provider = json.load(f)
    with open(f'{base}/hard_consumer.json', 'r', encoding='utf-8') as f:
        hard_consumer = json.load(f)
    return provider, hard_provider, consumer, hard_consumer

# ----------------------------------------------------------------------
# 1. Generic negotiation loop (works with any agent that has observe())
# ----------------------------------------------------------------------
RULE_TYPES = ['permission', 'obligation', 'prohibition']


def trajectory_keys(*offers) -> List[str]:
    """Stable list of ``{rule_type}:{action}:{operand}`` keys across the offers.

    These are the per-round series tracked for the dynamics (``T*``) plots, e.g.
    ``permission:use:percentage`` or ``obligation:anonymize:industry``.
    """
    keys: List[str] = []
    for offer in offers:
        if not offer:
            continue
        for rule_type in RULE_TYPES:
            for rule in offer.get(rule_type, []) or []:
                action = rule.get('action')
                for c in rule.get('constraint', []) or []:
                    op = c['leftOperand'].replace('odrl:', '')
                    k = f"{rule_type}:{action}:{op}"
                    if k not in keys:
                        keys.append(k)
    return keys


def _extract_offer_values(offer: Dict) -> Dict[str, Any]:
    """{key -> rightOperand} for one offer (first clause wins per operand)."""
    out: Dict[str, Any] = {}
    if not offer:
        return out
    for rule_type in RULE_TYPES:
        for rule in offer.get(rule_type, []) or []:
            action = rule.get('action')
            for c in rule.get('constraint', []) or []:
                op = c['leftOperand'].replace('odrl:', '')
                k = f"{rule_type}:{action}:{op}"
                if k not in out:
                    out[k] = c['rightOperand']
    return out


def _record_round(run: Dict, offer: Dict, round_idx: int) -> None:
    """Append the offer's values (aligned across all tracked keys) at a round."""
    run['rounds'].append(round_idx)
    vals = _extract_offer_values(offer)
    for k in run['data']:
        run['data'][k].append(vals.get(k))


def negotiate_generic(agent1, agent2, max_turns: int = MAX_TURNS,
                      traj_keys: Optional[List[str]] = None):
    """Runs a bilateral negotiation.

    Returns ``(agreed_offer_or_None, turns_taken, run)`` where ``run`` is the
    per-round trajectory record (or ``None`` if ``traj_keys`` is not supplied).
    The negotiation stops with no agreement once ``max_turns`` is exceeded
    (deadline) so non-converging baseline pairings cannot loop forever.

    Round convention for the trajectory (matches ``full_trajectory_analysis``):
    round ``-1`` = provider's initial offer, round ``0`` = consumer's initial
    offer, then odd rounds = provider counter-offers, even rounds = consumer
    counter-offers.
    """
    from time import perf_counter
    # Start with agent2's initial offer (as in the original code)
    received = copy.deepcopy(agent2.init_offer)
    turn = 0
    agents = [agent1, agent2]
    accepted = 0
    final = None

    run = None
    if traj_keys is not None:
        run = {'rounds': [], 'data': {k: [] for k in traj_keys},
               'durations': {'provider': [], 'consumer': []}}
        _record_round(run, agent1.init_offer, -1)   # provider init
        _record_round(run, agent2.init_offer, 0)    # consumer init

    while turn < max_turns:
        current = agents[turn % 2]
        side = 'provider' if turn % 2 == 0 else 'consumer'
        t0 = perf_counter()
        try:
            accept, replied = current.observe(received)
        except NegotiationInfeasible:
            return None, turn, run
        dt = perf_counter() - t0
        received = replied
        if run is not None:
            run['durations'][side].append(dt)
            _record_round(run, received, turn + 1)
        if accept:
            accepted += 1
            final = received
        else:
            accepted = 0
        if accepted == 2:
            return final, turn, run
        turn += 1
    # Deadline reached without agreement.
    return None, turn, run

# ----------------------------------------------------------------------
# 2. Utility evaluation (per-operand, direction-aware, normalised to [0,1])
# ----------------------------------------------------------------------
def _to_num(v):
    """Convert a rightOperand to a comparable scalar (number / ISO date / ISO duration)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            if is_iso_date(v):
                return float(iso_to_days(v))
            if is_iso_duration(v):
                return float(duration_to_seconds(v))
            return float(v)
        except Exception:
            return None
    return None


def _setish(v):
    if isinstance(v, (list, tuple, set)):
        return {str(x) for x in v}
    return {str(v)}


def build_operand_ranges(*offers) -> Dict[str, Tuple[float, float]]:
    """Per-operand [min, max] over every numeric value seen across the given offers.

    These ranges define the negotiation span used to normalise utility, so that a
    large-magnitude operand (count, payAmount) is not penalised more than a small one.
    """
    vals: Dict[str, List[float]] = defaultdict(list)
    for offer in offers:
        if not offer:
            continue
        for rule_type in RULE_TYPES:
            for rule in offer.get(rule_type, []):
                for constr in rule.get('constraint', []):
                    op = constr['leftOperand'].replace('odrl:', '')
                    n = _to_num(constr['rightOperand'])
                    if n is not None:
                        vals[op].append(n)
    return {op: (min(xs), max(xs)) for op, xs in vals.items() if xs}


def _find_actual(offer, rule_type, action, op):
    for orule in offer.get(rule_type, []):
        if action is not None and orule.get('action') != action:
            continue
        for oconstr in orule.get('constraint', []):
            if oconstr['leftOperand'].replace('odrl:', '') == op:
                return oconstr['rightOperand']
    return None


def _reservation_clause(reservation, rule_type, action, op):
    """(operator, value) of a party's hard/reservation bound for an operand, or (None, None)."""
    if not reservation:
        return None, None
    for rule in reservation.get(rule_type, []):
        if action is not None and rule.get('action') != action:
            continue
        for c in rule.get('constraint', []):
            if c['leftOperand'].replace('odrl:', '') == op:
                return c['operator'].replace('odrl:', ''), c['rightOperand']
    return None, None


def _operand_utility(a_op, a_val, agreed, h_op, h_val, op, ranges) -> float:
    """Aspiration→reservation utility in [0,1] for one operand.

    ``1.0`` = the agreed term is at the party's aspiration (its init value) or
    better; ``0.0`` = it is at the party's reservation (its hard limit); linear in
    between. A term pushed *past* the reservation (a hard-policy violation) scores
    ``0`` — i.e. an illegal concession yields no utility.
    """
    an, aa, hn = _to_num(agreed), _to_num(a_val), _to_num(h_val)

    # ---- numeric / temporal / duration ----
    if an is not None and aa is not None:
        direction = h_op or a_op  # the hard rule fixes the feasible direction
        if direction in ('lteq', 'lt'):              # smaller is better
            r = hn if hn is not None else (ranges.get(op, (aa, aa))[1])
            if r is None or r <= aa:
                return 1.0 if an <= aa else 0.0
            return max(0.0, min(1.0, (r - an) / (r - aa)))
        if direction in ('gteq', 'gt'):              # larger is better
            r = hn if hn is not None else (ranges.get(op, (aa, aa))[0])
            if r is None or r >= aa:
                return 1.0 if an >= aa else 0.0
            return max(0.0, min(1.0, (an - r) / (aa - r)))
        # eq/neq numeric: closeness to aspiration, tolerance = distance to reservation
        if hn is not None:
            tol = abs(hn - aa) or 1.0
        else:
            lo, hi = ranges.get(op, (aa, aa))
            tol = (hi - lo) or 1.0
        return max(0.0, min(1.0, 1.0 - abs(an - aa) / tol))

    # ---- categorical / set ----
    A, G = _setish(a_val), _setish(agreed)
    H = _setish(h_val) if h_val is not None else None
    # Outside the reservation allow-set => violation => 0.
    if H is not None and h_op in ('isAnyOf', 'isAllOf', 'eq') and not G.issubset(H):
        return 0.0
    if not (A | G):
        return 1.0
    return len(A & G) / len(A | G)


def compute_utility(offer: Dict, aspiration: Dict, reservation: Dict = None,
                    ranges: Dict[str, Tuple[float, float]] = None) -> float:
    """Aspiration→reservation utility in [0,1] for the party whose ``aspiration``
    (init offer) and ``reservation`` (hard policy) are given.

    For every operand the party cares about we score the agreed term on the
    party's own concession axis: ``1`` at its aspiration (or beyond), ``0`` at its
    reservation, linear between, and ``0`` for anything past the reservation (an
    illegal term). A dropped operand scores ``0``. The party's utility is the mean
    over operands. This is the standard negotiation-utility model and makes the
    metric directly meaningful: a high score means the party got close to its ideal
    while staying inside its feasible region.
    """
    if offer is None:
        return 0.0
    ranges = ranges or {}
    scores: List[float] = []
    for rule_type in RULE_TYPES:
        for rule in aspiration.get(rule_type, []):
            action = rule.get('action')
            for constr in rule.get('constraint', []):
                op = constr['leftOperand'].replace('odrl:', '')
                a_op = constr['operator'].replace('odrl:', '')
                a_val = constr['rightOperand']
                actual = _find_actual(offer, rule_type, action, op)
                if actual is None:
                    scores.append(0.0)
                    continue
                h_op, h_val = _reservation_clause(reservation, rule_type, action, op)
                scores.append(_operand_utility(a_op, a_val, actual, h_op, h_val, op, ranges))
    return sum(scores) / max(1, len(scores))

# ----------------------------------------------------------------------
# 2b. Hard-constraint (policy-compliance) violation check
# ----------------------------------------------------------------------
# An agent's `hard_policy` is its non-negotiable reservation policy (the worst
# terms it may accept). A reached agreement *violates* that policy if any agreed
# clause crosses a hard limit. We test each agreed clause against the hard rule
# with the same `satisfies` overlap test BAMBON enforces internally, so the
# notion of "violation" is identical to BAMBON's own compliance guard. Baselines
# do not enforce this and can therefore concede past their own hard limits.


def _agreed_clauses(offer, rule_type, action, op):
    """All (operator, rightOperand) clauses for an operand in the agreement.

    A numeric operand may appear twice (a range: a lower and an upper bound), so
    we return every matching clause and test them all against the hard rule.
    """
    out = []
    for orule in offer.get(rule_type, []):
        if action is not None and orule.get('action') != action:
            continue
        for oconstr in orule.get('constraint', []):
            if oconstr['leftOperand'].replace('odrl:', '') == op:
                out.append((oconstr['operator'].replace('odrl:', ''), oconstr['rightOperand']))
    return out


def _clause_ok(hard_op, hard_val, agreed_op, agreed_val) -> bool:
    """True if an agreed clause is compatible with (does not violate) a hard rule."""
    hn, an = _to_num(hard_val), _to_num(agreed_val)
    if hn is not None and an is not None:
        # Numeric / temporal / duration: compare on the real line. The agreed
        # right-operand is the settled value, treated as a concrete point.
        res = satisfies(hard_op if hard_op in NUM_OPS else 'eq', hn, 'eq', an)
    else:
        res = satisfies(hard_op, hard_val, agreed_op, agreed_val)
    # `satisfies` returns None for incomparable operand pairs -> not a violation.
    return res is None or bool(res)


def count_hard_violations(agreement: Dict, hard_policy: Dict) -> int:
    """Number of hard-policy constraints the agreed offer violates.

    A hard constraint counts as violated if (a) the operand is absent from the
    agreement (the contract drops the agent's non-negotiable limit), or (b) an
    agreed clause for that operand is incompatible with the hard rule.
    """
    if agreement is None or not hard_policy:
        return 0
    violations = 0
    for rule_type in RULE_TYPES:
        for hrule in hard_policy.get(rule_type, []):
            action = hrule.get('action')
            for hconstr in hrule.get('constraint', []):
                op = hconstr['leftOperand'].replace('odrl:', '')
                hard_op = hconstr['operator'].replace('odrl:', '')
                hard_val = hconstr['rightOperand']
                clauses = _agreed_clauses(agreement, rule_type, action, op)
                if not clauses:
                    violations += 1
                    continue
                if any(not _clause_ok(hard_op, hard_val, a_op, a_val) for a_op, a_val in clauses):
                    violations += 1
    return violations

# ----------------------------------------------------------------------
# 3. Agent definitions (all implement observe)
# ----------------------------------------------------------------------

# 3.1 AspirationNegotiator (classical conceder)
class AspirationNegotiator:
    # Historical calibration: the constants below were chosen for MAX_TURNS=200,
    # i.e. a budget of 100 moves per agent. They are stated here as *fractions*
    # of that budget so a change of deadline rescales the strategy instead of
    # silently truncating it. `budget=100` reproduces the original agent exactly.
    ACCEPT_FRAC = 0.20          # accepted from its own move 20 of 100
    BASE_RATE = 0.05            # geometric concession per move, over 100 moves

    def __init__(self, init_offer: Dict, hard_policy: Dict, concession_rate=None,
                 budget: int = 100):
        self.init_offer = copy.deepcopy(init_offer)
        self.hard_policy = hard_policy
        self.budget = max(1, int(budget))
        # Keep the concession *trajectory* invariant under a change of budget:
        # (1-r')^budget == (1-r)^100, so the agent travels the same fraction of
        # the distance by the time its deadline arrives.
        self.concession_rate = (concession_rate if concession_rate is not None
                                else 1.0 - (1.0 - self.BASE_RATE) ** (100.0 / self.budget))
        self.accept_at = max(1, round(self.ACCEPT_FRAC * self.budget))
        self.current_offer = copy.deepcopy(init_offer)
        self.round = 0
    def observe(self, received_offer: Dict):
        self.round += 1
        # Concede: move each numeric constraint towards opponent's value
        new_offer = copy.deepcopy(self.current_offer)
        for rule_type in ['permission', 'obligation', 'prohibition']:
            for rule in new_offer.get(rule_type, []):
                for constr in rule.get('constraint', []):
                    op = constr['leftOperand'].replace('odrl:', '')
                    opp_val = None
                    for orule in received_offer.get(rule_type, []):
                        for oconstr in orule.get('constraint', []):
                            if oconstr['leftOperand'].replace('odrl:', '') == op:
                                opp_val = oconstr['rightOperand']
                                break
                        if opp_val: break
                    if opp_val is not None:
                        try:
                            cur = float(constr['rightOperand'])
                            opp = float(opp_val)
                            new_val = cur + self.concession_rate * (opp - cur)
                            constr['rightOperand'] = new_val
                        except: pass
        self.current_offer = new_offer
        # Accept once a fixed fraction of its own move budget has elapsed.
        if self.round >= self.accept_at:
            return True, received_offer
        return False, new_offer

# 3.2 QLearningNegotiator (tabular Q-learning)
class QLearningNegotiator:
    ACCEPT_FRAC = 0.30          # accepted from its own move 30 of 100
    SLOW, FAST = 0.05, 0.25     # concession rates per move, over 100 moves
    N_STATES = 20               # round buckets; 5 moves per bucket at budget 100

    def __init__(self, init_offer: Dict, hard_policy: Dict, epsilon=0.2, alpha=0.1,
                 gamma=0.9, budget: int = 100):
        self.init_offer = copy.deepcopy(init_offer)
        self.hard_policy = hard_policy
        self.current_offer = copy.deepcopy(init_offer)
        self.budget = max(1, int(budget))
        self.accept_at = max(1, round(self.ACCEPT_FRAC * self.budget))
        self.bucket = max(1, round(self.budget / self.N_STATES))
        _k = 100.0 / self.budget
        self.slow = 1.0 - (1.0 - self.SLOW) ** _k
        self.fast = 1.0 - (1.0 - self.FAST) ** _k
        self.round = 0
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = defaultdict(float)  # state-action -> value
        self.last_state = None
        self.last_action = None
        # Discretize offers: we treat each constraint value as part of state
        # Simpler: state = round index (discrete), action = which constraint to change
        self.actions = ['concede_slow', 'concede_fast', 'stay']
    def _get_state(self):
        # Same number of buckets whatever the budget, so the learned table has
        # the same shape at every deadline.
        return min(self.round // self.bucket, self.N_STATES)
    def _apply_action(self, action, received_offer):
        new_offer = copy.deepcopy(self.current_offer)
        if action == 'concede_slow':
            rate = self.slow
        elif action == 'concede_fast':
            rate = self.fast
        else:
            rate = 0.0
        for rule_type in ['permission', 'obligation', 'prohibition']:
            for rule in new_offer.get(rule_type, []):
                for constr in rule.get('constraint', []):
                    op = constr['leftOperand'].replace('odrl:', '')
                    opp_val = None
                    for orule in received_offer.get(rule_type, []):
                        for oconstr in orule.get('constraint', []):
                            if oconstr['leftOperand'].replace('odrl:', '') == op:
                                opp_val = oconstr['rightOperand']
                                break
                        if opp_val: break
                    if opp_val is not None:
                        try:
                            cur = float(constr['rightOperand'])
                            opp = float(opp_val)
                            new_val = cur + rate * (opp - cur)
                            constr['rightOperand'] = new_val
                        except: pass
        return new_offer
    def observe(self, received_offer: Dict):
        self.round += 1
        state = self._get_state()
        # Choose action epsilon-greedy
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_vals = [self.q_table[(state, a)] for a in self.actions]
            max_q = max(q_vals)
            best_actions = [a for a, q in zip(self.actions, q_vals) if q == max_q]
            action = random.choice(best_actions)
        # Apply action to produce new offer
        new_offer = self._apply_action(action, received_offer)
        # Reward: improvement in utility compared to own last offer (measured
        # against its own aspiration, with its hard policy as the reservation).
        old_util = compute_utility(self.current_offer, self.init_offer, self.hard_policy)
        new_util = compute_utility(new_offer, self.init_offer, self.hard_policy)
        reward = new_util - old_util
        # Q-learning update
        if self.last_state is not None and self.last_action is not None:
            old_q = self.q_table[(self.last_state, self.last_action)]
            next_max = max([self.q_table[(state, a)] for a in self.actions]) if self.round > 1 else 0
            self.q_table[(self.last_state, self.last_action)] = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
        self.last_state = state
        self.last_action = action
        self.current_offer = new_offer
        # Accept once a fixed fraction of its own move budget has elapsed.
        if self.round >= self.accept_at:
            return True, received_offer
        return False, new_offer

# 3.3 TitForTat (mirroring)
def _offer_signature(offer: Dict) -> tuple:
    """A comparable fingerprint of an offer's *negotiable content* only.

    Offers carry bookkeeping metadata that changes every round even when the
    negotiated terms are identical (BAMBON stamps a fresh ``@id = uuid4()`` on
    every counter-offer and an extra ``@type`` on agreements). Comparing whole
    dicts therefore never matches, which is why TitForTat — whose acceptance test
    was raw ``==`` — could never close a deal with BAMBON. This signature ignores
    the metadata and compares only (rule_type, action, operand, operator, value).
    """
    sig = []
    for rule_type in RULE_TYPES:
        for rule in offer.get(rule_type, []) or []:
            action = rule.get('action')
            clauses = []
            for c in rule.get('constraint', []) or []:
                op = c['leftOperand'].replace('odrl:', '')
                operator = c['operator'].replace('odrl:', '')
                val = c['rightOperand']
                if isinstance(val, list):
                    val = tuple(sorted(map(str, val)))
                else:
                    val = str(val)
                clauses.append((op, operator, val))
            sig.append((rule_type, str(action), tuple(sorted(clauses))))
    return tuple(sorted(sig))


class TitForTatNegotiator:
    # Mirrors the opponent and accepts when they stop moving; no round constant,
    # so nothing to rescale. `budget` is accepted for signature uniformity.
    def __init__(self, init_offer: Dict, hard_policy: Dict, budget: int = 100):
        self.init_offer = copy.deepcopy(init_offer)
        self.hard_policy = hard_policy
        self.last_sent = None
        self.round = 0
    def observe(self, received_offer: Dict):
        self.round += 1
        if self.last_sent is None:
            self.last_sent = self.init_offer
            return False, self.init_offer
        # Accept once the opponent stops moving — compared on negotiable content,
        # not on volatile @id/@type metadata.
        if _offer_signature(received_offer) == _offer_signature(self.last_sent):
            return True, received_offer
        self.last_sent = received_offer
        return False, received_offer

# 3.4 Hardliner (rarely concedes)
class HardlinerNegotiator:
    ACCEPT_FRAC = 0.50          # accepted from its own move 50 of 100

    def __init__(self, init_offer: Dict, hard_policy: Dict, hard_rounds=None,
                 budget: int = 100):
        self.init_offer = copy.deepcopy(init_offer)
        self.hard_policy = hard_policy
        self.budget = max(1, int(budget))
        self.hard_rounds = (hard_rounds if hard_rounds is not None
                            else max(1, round(self.ACCEPT_FRAC * self.budget)))
        self.round = 0
    def observe(self, received_offer: Dict):
        self.round += 1
        if self.round >= self.hard_rounds:
            return True, received_offer
        return False, self.init_offer

# 3.5 RandomNegotiator
class RandomNegotiator:
    # accept_prob is a per-move probability, not a round index, so it needs no
    # rescaling; `budget` is accepted only so every baseline has one signature.
    def __init__(self, init_offer: Dict, hard_policy: Dict, accept_prob=0.3,
                 budget: int = 100):
        self.init_offer = copy.deepcopy(init_offer)
        self.hard_policy = hard_policy
        self.accept_prob = accept_prob
        self.round = 0
    def observe(self, received_offer: Dict):
        self.round += 1
        if self.round > 1 and random.random() < self.accept_prob:
            return True, received_offer
        return False, self.init_offer

# ----------------------------------------------------------------------
# 4. Benchmark runner (providers × consumers)
# ----------------------------------------------------------------------
def run_full_benchmark(provider_factories: Dict[str, callable],
                       consumer_factories: Dict[str, callable],
                       ranges: Dict[str, Tuple[float, float]],
                       n_runs=5,
                       traj_keys: Optional[List[str]] = None):
    """Run every provider×consumer pairing ``n_runs`` times.

    Returns ``(df, trajectories)`` where ``trajectories`` maps
    ``(provider_name, consumer_name)`` to the list of per-round trajectory
    records of the runs that reached agreement (used for the per-pairing ``T*``
    dynamics plots).
    """
    from time import perf_counter
    results = []
    trajectories: Dict[Tuple[str, str], List[Dict]] = {}
    all_trajectories: List[Dict] = []  # every negotiation's full per-round trajectory
    for prov_name, prov_factory in provider_factories.items():
        for cons_name, cons_factory in consumer_factories.items():
            pair_runs: List[Dict] = []
            for run in range(n_runs):
                random.seed(run)
                np.random.seed(run)  # BAMBON's NumericBeta draws from numpy
                provider = prov_factory()
                consumer = cons_factory()
                t0 = perf_counter()
                agreement, turns, run_traj = negotiate_generic(
                    provider, consumer, traj_keys=traj_keys)
                wall = perf_counter() - t0
                if agreement is not None and run_traj is not None:
                    pair_runs.append(run_traj)
                # Keep the FULL per-round trajectory of EVERY negotiation (agreed or
                # not), same record shape as out_test/full_results.json plus the
                # identifying provider/consumer/run/outcome fields.
                if run_traj is not None:
                    all_trajectories.append({
                        "provider": prov_name,
                        "consumer": cons_name,
                        "run": run,
                        "agreed": agreement is not None,
                        "turns": turns,
                        "rounds": run_traj["rounds"],
                        "data": run_traj["data"],
                        "durations": run_traj["durations"],
                    })
                prov_hard = getattr(provider, 'hard_policy', None)
                cons_hard = getattr(consumer, 'hard_policy', None)
                if agreement is None:
                    u_prov = u_cons = 0.0
                    agreed = False
                    prov_viol = cons_viol = 0
                    odrl_prov = odrl_utility(None, provider.init_offer, prov_hard, ranges,
                                             ODRL_UTILITY_CONFIG)
                    odrl_cons = odrl_utility(None, consumer.init_offer, cons_hard, ranges,
                                             ODRL_UTILITY_CONFIG)
                else:
                    # Each party's utility: 1 at its aspiration (init offer), 0 at its
                    # reservation (hard policy), 0 for any term pushed beyond it.
                    u_prov = compute_utility(agreement, provider.init_offer, prov_hard, ranges)
                    u_cons = compute_utility(agreement, consumer.init_offer, cons_hard, ranges)
                    agreed = True
                    # Did either party agree to terms breaching its own hard policy?
                    prov_viol = count_hard_violations(agreement, prov_hard)
                    cons_viol = count_hard_violations(agreement, cons_hard)
                    # ODRL-semantic utility (rule-weighted satisfaction × exp(-λV)).
                    odrl_prov = odrl_utility(agreement, provider.init_offer, prov_hard, ranges,
                                             ODRL_UTILITY_CONFIG)
                    odrl_cons = odrl_utility(agreement, consumer.init_offer, cons_hard, ranges,
                                             ODRL_UTILITY_CONFIG)
                ou_prov, ou_cons = odrl_prov["utility"], odrl_cons["utility"]
                # An ODRL contract is *enforceable* only if BOTH parties can
                # legally comply, i.e. neither agreed to terms breaching its own
                # hard policy. A contract that voids either party yields no
                # realised utility for anyone (it cannot be executed).
                enforceable = bool(agreed and prov_viol == 0 and cons_viol == 0)
                eu_prov = ou_prov if enforceable else 0.0
                eu_cons = ou_cons if enforceable else 0.0
                results.append({
                    "provider": prov_name,
                    "consumer": cons_name,
                    "run": run,
                    "agreement": agreed,
                    "turns": turns,
                    "wall_time_s": wall,
                    "utility_provider": u_prov,
                    "utility_consumer": u_cons,
                    "social_welfare": u_prov + u_cons,
                    "nash_product": u_prov * u_cons,
                    "fairness": 1.0 - abs(u_prov - u_cons),
                    "provider_hard_violations": prov_viol,
                    "consumer_hard_violations": cons_viol,
                    "provider_violated": agreed and prov_viol > 0,
                    "consumer_violated": agreed and cons_viol > 0,
                    # --- ODRL-semantic utility (request 3/4) ---
                    "odrl_utility_provider": ou_prov,
                    "odrl_utility_consumer": ou_cons,
                    "odrl_satisfaction_provider": odrl_prov["satisfaction"],
                    "odrl_satisfaction_consumer": odrl_cons["satisfaction"],
                    "odrl_violation_provider": odrl_prov["violation"],
                    "odrl_violation_consumer": odrl_cons["violation"],
                    "odrl_hard_penalty_provider": odrl_prov["hard_penalty"],
                    "odrl_hard_penalty_consumer": odrl_cons["hard_penalty"],
                    "odrl_nash_product": ou_prov * ou_cons,
                    "odrl_social_welfare": ou_prov + ou_cons,
                    "odrl_fairness": 1.0 - abs(ou_prov - ou_cons),
                    # --- enforceable-contract utility (void if either party breaches) ---
                    "contract_enforceable": enforceable,
                    "enforceable_utility_provider": eu_prov,
                    "enforceable_utility_consumer": eu_cons,
                    "enforceable_nash_product": eu_prov * eu_cons,
                })
            trajectories[(prov_name, cons_name)] = pair_runs
            # Compact per-pair progress line.
            pair = [r for r in results if r["provider"] == prov_name and r["consumer"] == cons_name]
            rate = sum(r["agreement"] for r in pair) / len(pair)
            avg_turns = sum(r["turns"] for r in pair) / len(pair)
            print(f"{prov_name:28s} vs {cons_name:28s}: agree={rate:5.0%}  avg_turns={avg_turns:5.1f}")
    return pd.DataFrame(results), trajectories, all_trajectories

# ----------------------------------------------------------------------
# 4b. Results aggregation (JSON only — no plots)
# ----------------------------------------------------------------------
def _stem(name: str) -> str:
    return name.replace('_provider', '').replace('_consumer', '')


def _is_bambon(name: str) -> bool:
    return name.startswith('bambon')


def _nash_block(sub: pd.DataFrame) -> dict:
    """Nash-bargaining analysis over the agreed deals in ``sub`` using the
    ODRL-semantic utilities of the two parties (provider, consumer)."""
    agreed = sub[sub['agreement']]
    profiles = list(zip(agreed['odrl_utility_provider'], agreed['odrl_utility_consumer']))
    return nash_bargaining_analysis(
        profiles, disagreement=[0.0, 0.0],
        agreements=list(sub['agreement']), n_agents=2)


def _by_agent_summary(results_df: pd.DataFrame) -> dict:
    """Per-agent summary pooling BOTH roles (the agent as provider and as
    consumer). This is the headline comparison table: agreement / compliance
    rate, ODRL-semantic self-utility, enforceable-contract utility, Nash product,
    fairness and the absolute count of own hard-policy violations.
    """
    df = results_df.copy()
    df['ps'] = df['provider'].map(_stem)
    df['cs'] = df['consumer'].map(_stem)
    out: dict = {}
    for agent in sorted(set(df['ps']) | set(df['cs'])):
        gp = df[df['ps'] == agent]
        gc = df[df['cs'] == agent]
        agp = gp[gp['agreement']]
        agc = gc[gc['agreement']]
        self_u = pd.concat([agp['odrl_utility_provider'], agc['odrl_utility_consumer']])
        enf_u = pd.concat([gp['enforceable_utility_provider'], gc['enforceable_utility_consumer']])
        nash = pd.concat([agp['odrl_nash_product'], agc['odrl_nash_product']])
        fair = pd.concat([agp['odrl_fairness'], agc['odrl_fairness']])
        agree = pd.concat([gp['agreement'], gc['agreement']])
        enforceable = pd.concat([gp['contract_enforceable'], gc['contract_enforceable']])
        viol = int(agp['provider_violated'].sum() + agc['consumer_violated'].sum())
        out[agent] = {
            "is_bambon": _is_bambon(agent),
            "n_negotiations": int(len(agree)),
            "agreement_rate": float(agree.mean()),
            "enforceable_contract_rate": float(enforceable.mean()),
            "mean_odrl_self_utility": float(self_u.mean()) if len(self_u) else 0.0,
            "mean_enforceable_utility": float(enf_u.mean()) if len(enf_u) else 0.0,
            "mean_nash_product": float(nash.mean()) if len(nash) else 0.0,
            "mean_fairness": float(fair.mean()) if len(fair) else 0.0,
            "own_hard_violations": viol,
        }
    return out


def write_full_results(results_df: pd.DataFrame, out_dir: str, n_runs: int) -> str:
    """Write the complete benchmark to ``full_results.json``.

    Contains run metadata, the ODRL-utility configuration, every per-negotiation
    record (including the new ODRL-semantic utility, satisfaction, violation and
    hard penalty for both parties), and Nash-bargaining analysis (Nash product,
    Jain's fairness index, mean utility / social welfare and agreement rate)
    computed overall and per BAMBON profile / baseline as provider.
    """
    cfg = ODRL_UTILITY_CONFIG
    agreed = results_df[results_df['agreement']]
    payload = {
        "scenario": "full",
        "n_runs_per_pairing": n_runs,
        "n_negotiations": int(len(results_df)),
        "n_pairings": int(results_df.groupby(['provider', 'consumer']).ngroups),
        "utility_config": {
            "rule_weights": cfg.rule_weights,
            "violation_rule_weights": cfg.violation_rule_weights,
            "numeric_penalty_power": cfg.numeric_penalty_power,
            "numeric_directional": cfg.numeric_directional,
            "set_preservation_weight": cfg.set_preservation_weight,
            "set_compactness_weight": cfg.set_compactness_weight,
            "hard_lambda": cfg.hard_lambda,
            "use_semantic": cfg.use_semantic,
            "semantic_model_name": cfg.semantic_model_name,
        },
        "overall": {
            "agreement_rate": float(results_df['agreement'].mean()),
            "enforceable_contract_rate": float(results_df['contract_enforceable'].mean()),
            "mean_odrl_utility_provider": float(agreed['odrl_utility_provider'].mean()) if len(agreed) else 0.0,
            "mean_odrl_utility_consumer": float(agreed['odrl_utility_consumer'].mean()) if len(agreed) else 0.0,
            "nash_bargaining": _nash_block(results_df),
        },
        "by_agent": _by_agent_summary(results_df),
        "nash_by_provider": {
            prov: _nash_block(grp) for prov, grp in results_df.groupby('provider')
        },
        "results": json.loads(results_df.to_json(orient='records')),
    }
    path = os.path.join(out_dir, 'full_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return path


# ----------------------------------------------------------------------
# 5. Main: build all agents (BAMBON profiles + representatives)
# ----------------------------------------------------------------------
def main():
    # Load data: the FULL scenario exercises permission (use, transform),
    # obligation (anonymize) and prohibition (redistribute) rules, i.e. multiple
    # rule types and multiple actions, instead of the single-action `use` case.
    init_provider, hard_provider, init_consumer, hard_consumer = fetch_data('full')

    # BAMBON profiles.
    #
    # Tuning goals (compliance is *always* guaranteed by the hard-feasibility
    # clamps in belief.py, independent of these knobs, so we can tune freely for
    # agreement quality without ever risking a violation):
    #   * High agreement rate, INCLUDING BAMBON-vs-BAMBON self-play — the previous
    #     "hold firm at all costs" tuning made two firm BAMBONs deadlock (e.g.
    #     aggressive-vs-exploratory agreed only ~31% of the time). The fix here is
    #     to keep BAMBON firm *early* (so it does not cave and lose utility) but to
    #     guarantee that any *feasible* deal still converges before the deadline.
    #   * Balanced, higher-utility deals — moderate `inertia` keeps BAMBON from
    #     collapsing straight onto the opponent's value, so agreements land nearer
    #     the middle (better Nash/fairness) while still converging.
    #   * Three distinct, paper-meaningful personalities: balanced (the workhorse),
    #     aggressive (holds firm for self-utility) and exploratory (fast, fair).
    #
    # Why the earlier profiles deadlocked, and what changed:
    #   The acceptance threshold in negotiator._adapt RISES over time
    #   (tau_in_target = 0.55 + 0.25*t + ...), and `inertia` is recomputed from
    #   context each round, so when two firm BAMBONs start far apart their offers
    #   stay dissimilar -> high severity -> high inertia -> neither moves -> the
    #   pair runs to the MAX_TURNS deadline with no deal. Profile knobs cannot lower
    #   that rising threshold, but they CAN keep beliefs *drifting together*:
    #     - Raise the learning-rate floor `lr_min` (0.14/0.20/0.30 -> ~0.32/0.36):
    #       beliefs keep tracking the opponent even when inertia is high, so offers
    #       converge instead of oscillating. This is the single biggest fix.
    #     - Lower `jaccard_snap_threshold` (0.66/0.55/0.50 -> ~0.46/0.45/0.42): lets
    #       BAMBON snap-accept a *similar* categorical/set offer (purpose, industry,
    #       fileFormat are the sticky, conflicting issues) instead of holding out.
    #     - Lower the cooled acceptance floors `tau_in_cooled`/`tau_out_cooled` and
    #       the `tau_cooling_threshold` so the endgame does not ratchet the bar UP.
    #     - Slightly lower `inertia` and `no_agreement_until`; raise `max_rounds`.
    #   Effect (10-run focused tournament): BAMBON-vs-BAMBON matrix agreement
    #   ~81% -> ~93% with essentially NO loss of self-utility/Nash, and BAMBON-vs-
    #   baseline pairings stay at 100%. Compliance is unaffected (hard clamps in
    #   src), so firmness is still pushed as hard as convergence allows.
    BAMBON_PROFILES = {
        # Balanced (firm workhorse): strong hold, reliable agreement. Faster belief
        # tracking (higher lr floor) + easier set-snap so it never self-deadlocks.
        "bambon_balanced": {"lr": 0.50, "inertia": 0.55, "tau_in": 0.60, "tau_out": 0.74,
                            "max_set_k": 3, "no_agreement_until": 3, "hopeless_window": 16,
                            "jaccard_snap_threshold": 0.45, "max_rounds": 150,
                            "tau_cooling_threshold": 0.55, "tau_in_cooled": 0.60, "tau_out_cooled": 0.56,
                            "lr_min": 0.32, "lr_max": 0.95, "vol_window": 5, "max_param_delta": 0.12,
                            "hopeless_severity": 0.98, "hopeless_patience": 24, "hopeless_context": 0.06},
        # Aggressive: firmest of all — highest self-utility, holds longest, but now
        # still converges (lr floor keeps it drifting toward any feasible deal).
        "bambon_aggressive": {"lr": 0.42, "inertia": 0.60, "tau_in": 0.72, "tau_out": 0.84,
                              "max_set_k": 3, "no_agreement_until": 4, "hopeless_window": 20,
                              "jaccard_snap_threshold": 0.46, "max_rounds": 160,
                              "tau_cooling_threshold": 0.55, "tau_in_cooled": 0.62,
                              "tau_out_cooled": 0.58, "lr_min": 0.32, "lr_max": 0.95, "vol_window": 5,
                              "max_param_delta": 0.09, "hopeless_severity": 0.99, "hopeless_patience": 30,
                              "hopeless_context": 0.04},
        # Exploratory: explores the categorical/set space fast (high lr, larger
        # steps, bigger set budget); firm on acceptance but converges quickest.
        "bambon_exploratory": {"lr": 0.85, "inertia": 0.46, "tau_in": 0.58, "tau_out": 0.70,
                               "max_set_k": 4, "no_agreement_until": 3, "hopeless_window": 14,
                               "jaccard_snap_threshold": 0.42, "max_rounds": 150,
                               "tau_cooling_threshold": 0.50, "tau_in_cooled": 0.58, "tau_out_cooled": 0.54,
                               "lr_min": 0.36, "lr_max": 1.1, "vol_window": 4, "max_param_delta": 0.20,
                               "hopeless_severity": 0.97, "hopeless_patience": 26, "hopeless_context": 0.07}
    }

    def make_bambon_factory(profile, role, init_offer, hard_policy):
        def factory():
            kwargs = {k:v for k,v in profile.items() if k != "lr"}
            agent = Bambon(name=role, lr=profile.get("lr",1.0),
                           init_offer=copy.deepcopy(init_offer),
                           hard_policy=hard_policy, **kwargs)
            agent.init_offer = init_offer
            agent.hard_policy = hard_policy
            return agent
        return factory

    # Provider factories (using provider data)
    provider_factories = {}
    for name, profile in BAMBON_PROFILES.items():
        provider_factories[f"{name}_provider"] = make_bambon_factory(profile, "provider", init_provider, hard_provider)
    # Add representatives
    provider_factories["Aspiration_provider"] = lambda: AspirationNegotiator(init_provider, hard_provider)
    provider_factories["QLearning_provider"] = lambda: QLearningNegotiator(init_provider, hard_provider)
    provider_factories["TitForTat_provider"] = lambda: TitForTatNegotiator(init_provider, hard_provider)
    provider_factories["Hardliner_provider"] = lambda: HardlinerNegotiator(init_provider, hard_provider)
    provider_factories["Random_provider"] = lambda: RandomNegotiator(init_provider, hard_provider)

    # Consumer factories (using consumer data)
    consumer_factories = {}
    for name, profile in BAMBON_PROFILES.items():
        consumer_factories[f"{name}_consumer"] = make_bambon_factory(profile, "consumer", init_consumer, hard_consumer)
    consumer_factories["Aspiration_consumer"] = lambda: AspirationNegotiator(init_consumer, hard_consumer)
    consumer_factories["QLearning_consumer"] = lambda: QLearningNegotiator(init_consumer, hard_consumer)
    consumer_factories["TitForTat_consumer"] = lambda: TitForTatNegotiator(init_consumer, hard_consumer)
    consumer_factories["Hardliner_consumer"] = lambda: HardlinerNegotiator(init_consumer, hard_consumer)
    consumer_factories["Random_consumer"] = lambda: RandomNegotiator(init_consumer, hard_consumer)

    # Per-operand negotiation ranges used to normalise utility to [0,1].
    ranges = build_operand_ranges(init_provider, hard_provider, init_consumer, hard_consumer)

    # Per-round trajectory keys (every rule:action:operand), tracked for the T-plots.
    traj_keys = trajectory_keys(init_provider, init_consumer)

    n_runs = 100
    print("Running FULL-scenario benchmark: BAMBON vs. 5 representative agent types")
    results_df, trajectories, all_trajectories = run_full_benchmark(
        provider_factories, consumer_factories, ranges, n_runs=n_runs, traj_keys=traj_keys)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_full")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "bambon_vs_all_categories.csv")
    results_df.to_csv(out_csv, index=False)

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Overall agreement rate: {results_df['agreement'].mean():.2%}")
    print("\nAgreement rate per provider:")
    print(results_df.groupby("provider")["agreement"].mean().sort_values(ascending=False))
    print("\nAverage utility as provider:")
    print(results_df.groupby("provider")["utility_provider"].mean().sort_values(ascending=False))
    print("\nMean ODRL Nash product over agreed deals (per provider):")
    agreed = results_df[results_df["agreement"]]
    print(agreed.groupby("provider")["odrl_nash_product"].mean().sort_values(ascending=False))
    print("\nMean ODRL utility as provider (agreed deals):")
    print(agreed.groupby("provider")["odrl_utility_provider"].mean().sort_values(ascending=False))

    print("\nHard-constraint violations in agreed deals (count of agreements breaching own hard policy):")
    prov_v = agreed.groupby("provider")["provider_violated"].sum()
    cons_v = agreed.groupby("consumer")["consumer_violated"].sum()
    by_agent = {}
    for name, v in prov_v.items():
        by_agent[_stem(name)] = by_agent.get(_stem(name), 0) + int(v)
    for name, v in cons_v.items():
        by_agent[_stem(name)] = by_agent.get(_stem(name), 0) + int(v)
    for ag, v in sorted(by_agent.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {ag:28s}: {v}")
    print(f"\nDetailed CSV saved: {out_csv}")

    # Full per-round trajectories of ALL negotiations (same record shape as
    # out_test/full_results.json, plus provider/consumer/run/outcome fields).
    negotiations_path = os.path.join(out_dir, "full_negotiations.json")
    with open(negotiations_path, "w", encoding="utf-8") as f:
        json.dump(all_trajectories, f)
    print(f"\nAll full negotiations (per-round trajectories) saved: {negotiations_path} "
          f"({len(all_trajectories)} negotiations)")

    # Full results (every negotiation + ODRL utility + Nash-bargaining analysis) → JSON.
    json_path = write_full_results(results_df, out_dir, n_runs)
    nb = json.load(open(json_path, "r", encoding="utf-8"))["overall"]["nash_bargaining"]
    print("\nNash-bargaining analysis (overall, ODRL utilities):")
    print(f"  agreement rate           : {nb['agreement_rate']:.2%}")
    print(f"  mean Nash product        : {nb['nash_product_mean']:.4f}")
    print(f"  Jain's fairness index    : {nb['jains_fairness_index_mean']:.4f}")
    print(f"  mean utility / agent     : {[round(x, 4) for x in nb['mean_utility_per_agent']]}")
    print(f"\nFull results JSON saved: {json_path}")


if __name__ == "__main__":
    main()
