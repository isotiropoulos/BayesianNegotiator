"""A generic, strategy-agnostic hard-policy safety mechanism for ODRL agents.

Why this module exists
---------------------
In the BAMBON evaluation the five reference baselines (Aspiration, Q-learning,
Tit-for-Tat, Hardliner, Random) treat their hard policy as a *scalar
reservation used to score outcomes* -- the standard treatment in the automated
negotiation literature -- rather than as a *constraint on commitments*. BAMBON
treats it as a constraint, through two mechanisms:

* **hard-feasibility repair** -- every emitted offer is projected into the
  agent's own feasible region (paper, Eq. 12 / Prop. 1);
* **commit-time veto** -- a term that is infeasible on the committed point is
  never signed, whatever the belief model says (paper, Prop. 2).

A compliance gap measured against baselines that lack those two mechanisms is
therefore not attributable to BAMBON's *Bayesian* machinery: it is attributable
to the presence of a safety mechanism at all. This module removes that
confound by supplying the same two mechanisms in a form that can be wrapped
around **any** agent, so that every arm of the benchmark can be run with an
*identical* hard-policy guard and the Bayesian contribution can be isolated.

Design constraints (what makes it a fair control)
-------------------------------------------------
1. **Strategy-agnostic.** :class:`HardPolicyVeto` never inspects, alters or
   replaces the wrapped agent's concession behaviour. It only (a) blocks an
   accept and (b) rewrites the *values* of an emitted offer that lie outside the
   agent's own feasible region.
2. **Belief-free.** No posterior, no opponent model, no learning, no utility
   function, no state carried between rounds other than a stall counter used by
   the optional walk-away rule. The repair is a deterministic projection.
3. **No BAMBON code.** The only project imports are the ODRL satisfaction
   predicate ``satisfies`` and the ISO-8601 <-> scalar conversions, both of which
   belong to the problem formulation (Sec. 3) rather than to the BAMBON model
   (Sec. 4).
4. **Identical semantics of "violation".** Feasibility is decided with exactly
   the predicate used to score compliance in the results tables, including the
   rule that a *missing* operand counts as a violation (the contract dropped a
   non-negotiable limit) and that a committed numeric term is tested as a point
   rather than as a half-line.

Usage
-----
    from src.bambon.baselines import HardPolicyVeto

    agent = HardPolicyVeto(AspirationNegotiator(init_offer, hard_policy),
                           hard_policy=hard_policy)
    agreement, turns, _ = negotiate_generic(agent, other)

or, for a whole factory dictionary::

    from src.bambon.baselines import veto_wrap
    VetoAspiration = veto_wrap(AspirationNegotiator)
    agent = VetoAspiration(init_offer, hard_policy)      # same signature

Guarantee
---------
If both parties are wrapped, every reachable agreement is feasible for both, so
compliance is 1.0 by construction for either denominator -- which is the point:
it turns compliance from a headline result into a *control*, and moves the
comparison onto agreement rate, Nash product, fairness and utility.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..models.models import NegotiationInfeasible
from ..utils.helpers import (
    days_to_iso,
    duration_to_seconds,
    is_iso_date,
    is_iso_duration,
    iso_to_days,
    satisfies,
    seconds_to_duration,
    strip_key,
)

RULE_TYPES: Tuple[str, ...] = ("permission", "obligation", "prohibition")

# Operators that constrain a scalar (numeric / temporal / duration) operand.
NUM_OPS: Tuple[str, ...] = ("lt", "lteq", "gt", "gteq", "eq", "neq")
# Operators that bound what a committed categorical term may contain.
ALLOW_OPS: Tuple[str, ...] = ("isAnyOf", "eq")
# Operators that state what a committed categorical term must contain.
REQUIRE_OPS: Tuple[str, ...] = ("isAllOf", "eq")
# Operators that state what a committed categorical term may not contain.
BAN_OPS: Tuple[str, ...] = ("isNoneOf", "neq")


# ======================================================================
# 0. scalar / set coercions (shared with the problem formulation)
# ======================================================================
def _to_num(value: Any) -> Optional[float]:
    """phi: numeric / ISO-date / ISO-duration -> a common scalar space.

    Dates map to days since the epoch, durations to seconds, plain numbers to
    themselves. Returns ``None`` for anything categorical.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            if is_iso_date(value):
                return float(iso_to_days(value))
            if is_iso_duration(value):
                return float(duration_to_seconds(value))
        except Exception:
            return None
    return None


def _render_like(template: Any, scalar: float) -> Any:
    """phi^-1: put a scalar back into the lexical form of ``template``."""
    if isinstance(template, str):
        if is_iso_date(template):
            return days_to_iso(int(round(scalar)))
        if is_iso_duration(template):
            return seconds_to_duration(max(0.0, float(scalar)))
        return template
    if isinstance(template, bool):
        return template
    if isinstance(template, int):
        return int(round(scalar))
    return float(scalar)


def _labels(value: Any) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(v) for v in value}
    return {str(value)}


def _emit_labels(template: Any, labels: Set[str]) -> Any:
    """Render a label set in the shape the original right operand had."""
    ordered = sorted(labels)
    if isinstance(template, (list, tuple, set, frozenset)):
        return ordered
    return ordered[0] if len(ordered) == 1 else ordered


def _step(template: Any) -> float:
    """Smallest meaningful increment for a strict (gt / lt) repair."""
    if isinstance(template, str):
        return 1.0                      # one day, or one second
    if isinstance(template, int):
        return 1.0
    return 1e-9


# ======================================================================
# 1. feasibility, stated exactly as the results tables score it
# ======================================================================
def _clause_ok(hard_op: str, hard_val: Any, agreed_op: str, agreed_val: Any) -> bool:
    """Is a committed clause compatible with one hard-policy constraint?

    Mirrors ``count_hard_violations`` in the benchmark: a committed numeric term
    is a *point*, so it is re-tested with ``eq`` rather than as a half-line
    (a ``gteq 897`` clause overlaps a ``gteq 900`` floor as a ray, yet the
    concrete 897 still breaches it). ``satisfies`` returns ``None`` for
    incomparable operator pairs, which is not counted as a violation.
    """
    hn, an = _to_num(hard_val), _to_num(agreed_val)
    if hn is not None and an is not None:
        res = satisfies(hard_op if hard_op in NUM_OPS else "eq", hn, "eq", an)
    else:
        res = satisfies(hard_op, hard_val, agreed_op, agreed_val)
    return res is None or bool(res)


def _iter_hard_constraints(hard_policy: Optional[Dict]):
    """Yield ``(rule_type, action, operand, operator, right_operand)``."""
    if not hard_policy:
        return
    for rule_type in RULE_TYPES:
        for rule in hard_policy.get(rule_type, []) or []:
            action = rule.get("action")
            for c in rule.get("constraint", []) or []:
                yield (
                    rule_type,
                    action,
                    strip_key(str(c["leftOperand"])),
                    strip_key(str(c["operator"])),
                    c["rightOperand"],
                )


def _offer_clauses(offer: Dict, rule_type: str, action: Any, operand: str) -> List[Dict]:
    """Every constraint dict in ``offer`` for one (rule_type, action, operand)."""
    out: List[Dict] = []
    for rule in offer.get(rule_type, []) or []:
        if action is not None and rule.get("action") != action:
            continue
        for c in rule.get("constraint", []) or []:
            if strip_key(str(c["leftOperand"])) == operand:
                out.append(c)
    return out


def feasibility_report(offer: Optional[Dict], hard_policy: Optional[Dict]) -> List[str]:
    """Human-readable list of the hard constraints ``offer`` violates."""
    if offer is None or not hard_policy:
        return []
    bad: List[str] = []
    for rule_type, action, operand, hop, hval in _iter_hard_constraints(hard_policy):
        clauses = _offer_clauses(offer, rule_type, action, operand)
        if not clauses:
            bad.append(f"{rule_type}:{action}:{operand} missing (hard {hop} {hval})")
            continue
        for c in clauses:
            aop = strip_key(str(c["operator"]))
            if not _clause_ok(hop, hval, aop, c["rightOperand"]):
                bad.append(
                    f"{rule_type}:{action}:{operand} {aop} {c['rightOperand']} "
                    f"violates hard {hop} {hval}"
                )
    return bad


def is_feasible(offer: Optional[Dict], hard_policy: Optional[Dict]) -> bool:
    """``True`` iff ``offer`` violates no constraint of ``hard_policy``."""
    return not feasibility_report(offer, hard_policy)


# ======================================================================
# 2. the projection Pi, lifted to a whole ODRL document
# ======================================================================
def _hard_scalar_bounds(constraints: Sequence[Tuple[str, Any]]):
    """(lo, hi, eq, strict_lo, strict_hi) implied by scalar hard constraints."""
    lo = hi = eq = None
    strict_lo = strict_hi = False
    for hop, hval in constraints:
        x = _to_num(hval)
        if x is None:
            continue
        if hop in ("gteq", "gt"):
            if lo is None or x > lo:
                lo, strict_lo = x, hop == "gt"
        elif hop in ("lteq", "lt"):
            if hi is None or x < hi:
                hi, strict_hi = x, hop == "lt"
        elif hop == "eq":
            eq = x
    return lo, hi, eq, strict_lo, strict_hi


def _hard_label_sets(constraints: Sequence[Tuple[str, Any]]):
    """(allowed, required, banned) label sets implied by categorical hard rules.

    ``allowed`` is ``None`` when no rule bounds the universe. This is Eq. 12 of
    the paper read off an ODRL document instead of an internal issue model.
    """
    allowed: Optional[Set[str]] = None
    required: Set[str] = set()
    banned: Set[str] = set()
    for hop, hval in constraints:
        V = _labels(hval)
        if hop == "isAnyOf":
            allowed = V if allowed is None else (allowed & V)
        elif hop == "eq":
            allowed = V if allowed is None else (allowed & V)
            required |= V
        elif hop == "isAllOf":
            required |= V
        elif hop in ("isNoneOf", "neq"):
            banned |= V
    return allowed, required, banned


def _project_scalar(value: Any, constraints: Sequence[Tuple[str, Any]]) -> Any:
    """Clamp a scalar right operand into the hard-feasible interval."""
    x = _to_num(value)
    if x is None:
        return value
    lo, hi, eq, strict_lo, strict_hi = _hard_scalar_bounds(constraints)
    if eq is not None:
        return value if x == eq else _render_like(value, eq)
    step = _step(value)
    x0 = x
    if lo is not None:
        floor = lo + step if strict_lo else lo
        if x < floor:
            x = floor
    if hi is not None:
        ceil = hi - step if strict_hi else hi
        if x > ceil:
            x = ceil
    # Return the original right operand untouched when it is already in range.
    # Re-rendering would otherwise renormalise an equivalent lexical form
    # (`PT240H` -> `P10D`), which is a spurious change to the offer.
    return value if x == x0 else _render_like(value, x)


INCLUSION_TERM_OPS: Tuple[str, ...] = ("eq", "isAnyOf", "isAllOf")
EXCLUSION_TERM_OPS: Tuple[str, ...] = ("isNoneOf", "neq")


def _project_labels(value: Any, constraints: Sequence[Tuple[str, Any]],
                    agreed_op: str) -> Tuple[str, Any]:
    """Project a categorical clause into the feasible region, operator and all.

    Eq. 12 of the paper is ``Pi(S) = ((S & A) | R) \\ F``, stated for a clause
    that *includes* labels. A clause may also *exclude* them, and then the same
    algebra does not apply -- which matters because Prop. 1 has to hold for every
    clause an agent can emit, not just the inclusion ones. Two adjustments:

    * ``required`` is intersected with ``allowed``, so the operator is
      **idempotent**: the result is inside ``allowed`` and disjoint from
      ``banned``, hence a fixed point of a second application.
    * an **exclusion** clause is projected by removing from the exclusion set
      everything the hard policy allows or requires (excluding a permitted label
      is what breaches an ``isAnyOf`` rule). If the hard policy states a positive
      requirement, no exclusion clause can satisfy it at all -- under the ODRL
      predicate ``sat(isAllOf, R, isNoneOf, B)`` holds only for empty ``R`` -- so
      the clause is *rewritten* as the corresponding inclusion clause. This is
      why the function returns an operator as well as a value.

    A hard policy whose required set is not inside its own allow-set is
    unsatisfiable on that operand. The fallback below keeps the function total,
    and :func:`is_feasible` still reports the residual rather than hiding it.
    """
    allowed, required, banned = _hard_label_sets(constraints)
    if allowed is not None:
        required = required & allowed

    if agreed_op in EXCLUSION_TERM_OPS:
        if required:
            # Positive requirement: flip to an inclusion clause carrying it.
            keep = required - banned
            if keep:
                op = "eq" if len(keep) == 1 else "isAllOf"
                return op, _emit_labels(value, keep)
        B = _labels(value) - (allowed or set())
        if not B:
            # Nothing may be excluded. Say so with a term that satisfies every
            # remaining rule: an inclusion clause over what is permitted.
            core = ((allowed if allowed is not None else set()) - banned)
            if core:
                return ("eq" if len(core) == 1 else "isAnyOf"), _emit_labels(value, core)
            B = _labels(value)
        return agreed_op, _emit_labels(value, B)

    S = _labels(value)
    if allowed is not None:
        S = S & allowed
    S = (S | required) - banned
    if not S:
        core = ((allowed if allowed is not None else set()) | required) - banned
        # Deterministic, so the repair is reproducible across seeds.
        S = {sorted(core)[0]} if core else _labels(value)
    if agreed_op == "eq" and len(S) > 1:
        # An `eq` clause commits a single label, but an `isAllOf` hard rule may
        # demand several; keep the clause satisfiable by widening the operator
        # rather than dropping part of the requirement.
        if len(required) > 1:
            return "isAllOf", _emit_labels(value, S)
        S = {sorted(S)[0]}
    return agreed_op, _emit_labels(value, S)


def _ensure_rule(offer: Dict, rule_type: str, action: Any) -> Dict:
    for rule in offer.setdefault(rule_type, []):
        if rule.get("action") == action:
            return rule
    rule = {"action": action, "constraint": []}
    offer[rule_type].append(rule)
    return rule


def project_offer(offer: Optional[Dict], hard_policy: Optional[Dict]) -> Optional[Dict]:
    """Project an ODRL offer into the agent's own feasible region.

    The three repair paths, in the order the paper states them:

    * **value repair** -- a scalar right operand is clamped into
      ``[lo, hi]``; a categorical one is projected by ``Pi``;
    * **restoration** -- a hard constraint whose operand is absent from the
      offer is re-inserted verbatim, because dropping the clause is itself a
      violation under Def. 2;
    * **fallback** -- when no feasible term exists for an operand at all the
      original value is left in place, so the function is total and the caller
      can detect the residual with :func:`is_feasible`.

    The operation is idempotent and depends on nothing but ``offer`` and
    ``hard_policy``.
    """
    if offer is None:
        return None
    if not hard_policy:
        return copy.deepcopy(offer)

    out = copy.deepcopy(offer)

    # group the hard constraints by the term they bound
    grouped: Dict[Tuple[str, Any, str], List[Tuple[str, Any]]] = {}
    for rule_type, action, operand, hop, hval in _iter_hard_constraints(hard_policy):
        grouped.setdefault((rule_type, action, operand), []).append((hop, hval))

    for (rule_type, action, operand), constraints in grouped.items():
        clauses = _offer_clauses(out, rule_type, action, operand)
        if not clauses:
            rule = _ensure_rule(out, rule_type, action)
            for hop, hval in constraints:
                # Restore the dropped limit at its own bound, projected so that
                # a *strict* hard operator is restored at a value that actually
                # satisfies it (a `gt 5` rule restored as `gt 5` would still be
                # a violation, because a committed term is tested as a point).
                if _to_num(hval) is not None:
                    r_op, restored = hop, _project_scalar(hval, constraints)
                else:
                    r_op, restored = _project_labels(hval, constraints, hop)
                rule["constraint"].append({
                    "leftOperand": f"odrl:{operand}",
                    "operator": f"odrl:{r_op}",
                    "rightOperand": restored,
                })
            continue
        for c in clauses:
            aop = strip_key(str(c["operator"]))
            if _to_num(c["rightOperand"]) is not None:
                c["rightOperand"] = _project_scalar(c["rightOperand"], constraints)
            else:
                new_op, new_val = _project_labels(c["rightOperand"], constraints, aop)
                c["operator"] = f"odrl:{new_op}"
                c["rightOperand"] = new_val
    return out


# ======================================================================
# 3. the wrapper
# ======================================================================
class HardPolicyVeto:
    """Give any ``observe``-protocol agent BAMBON's hard-policy guard.

    The wrapped agent decides *what to concede and when*; this object only
    decides *what may be signed* and *what may be emitted*.

    Parameters
    ----------
    agent:
        Any object exposing ``observe(offer) -> (accept: bool, counter: dict)``
        and an ``init_offer`` attribute.
    hard_policy:
        The agent's own hard policy. Defaults to ``agent.hard_policy``.
    veto:
        Block accepts of infeasible offers (paper Prop. 2). Default ``True``.
    repair:
        Project every emitted offer into the feasible region (paper Prop. 1).
        Default ``True``. Set ``repair=False, veto=True`` for a veto-only arm.
    walk_away_after:
        If a positive integer ``n``, raise :class:`NegotiationInfeasible` once
        ``n`` consecutive incoming offers have been infeasible *and* unchanged in
        their negotiable content. This is a deliberately naive stall detector,
        included so that BAMBON's belief-driven infeasibility detection has a
        belief-free counterpart to be compared against on infeasible scenarios.
        ``0`` (the default) disables it, in which case an impossible encounter
        ends in a timeout rather than a walk-away.

    Attributes
    ----------
    n_vetoed:
        How many accepts the veto blocked.
    n_repaired:
        How many emitted offers the repair actually changed.
    """

    def __init__(
        self,
        agent: Any,
        hard_policy: Optional[Dict] = None,
        *,
        veto: bool = True,
        repair: bool = True,
        walk_away_after: int = 0,
    ) -> None:
        self.agent = agent
        self.hard_policy = hard_policy if hard_policy is not None else getattr(
            agent, "hard_policy", None)
        self.veto = veto
        self.repair = repair
        self.walk_away_after = int(walk_away_after)

        self.n_vetoed = 0
        self.n_repaired = 0
        self.rounds_seen = 0
        self._stall = 0
        self._last_sig: Optional[tuple] = None

    # -- transparency: look like the agent we wrap ------------------------
    @property
    def init_offer(self) -> Dict:
        return self.agent.init_offer

    @property
    def name(self) -> str:
        return f"veto({getattr(self.agent, 'name', type(self.agent).__name__)})"

    def __getattr__(self, item):                       # pragma: no cover - proxy
        # Only reached for attributes this wrapper does not define itself.
        return getattr(self.__dict__["agent"], item)

    # -- protocol ---------------------------------------------------------
    def observe(self, received_offer: Dict) -> Tuple[bool, Dict]:
        self.rounds_seen += 1
        accept, counter = self.agent.observe(received_offer)

        incoming_ok = is_feasible(received_offer, self.hard_policy)

        # Stage 1 -- commit-time veto (Prop. 2). A term that is infeasible on the
        # committed point is never signed, whatever the strategy decided.
        if accept and self.veto and not incoming_ok:
            self.n_vetoed += 1
            accept = False
            # Answer with the nearest feasible version of what is on the table:
            # the maximally cooperative belief-free reply, and the analogue of
            # BAMBON's correction path.
            counter = received_offer

        # Optional belief-free stall detector.
        if self.walk_away_after > 0:
            sig = _content_signature(received_offer)
            if (not incoming_ok) and sig == self._last_sig:
                self._stall += 1
            else:
                self._stall = 0
            self._last_sig = sig
            if self._stall >= self.walk_away_after:
                raise NegotiationInfeasible(
                    f"{self.name}: {self._stall} consecutive infeasible, unchanged offers"
                )

        # Stage 2 -- hard-feasibility repair (Prop. 1). Only *proposals* are
        # repaired. An accept is a signature on the terms already on the table:
        # rewriting them would change the contract under the opponent's feet,
        # and it is unnecessary because Stage 1 has just established that those
        # terms are feasible.
        if accept:
            return True, counter
        if self.repair and counter is not None:
            projected = project_offer(counter, self.hard_policy)
            if _content_signature(projected) != _content_signature(counter):
                self.n_repaired += 1
            counter = projected

        return False, counter


def _content_signature(offer: Optional[Dict]) -> tuple:
    """Fingerprint of an offer's negotiable content, ignoring metadata.

    Offers carry bookkeeping that changes every round (``@id`` is a fresh uuid on
    every BAMBON counter-offer), so whole-dict comparison never matches.
    """
    if offer is None:
        return ()
    sig = []
    for rule_type in RULE_TYPES:
        for rule in offer.get(rule_type, []) or []:
            clauses = []
            for c in rule.get("constraint", []) or []:
                v = c["rightOperand"]
                v = tuple(sorted(map(str, v))) if isinstance(v, (list, tuple, set)) else str(v)
                clauses.append((strip_key(str(c["leftOperand"])),
                                strip_key(str(c["operator"])), v))
            sig.append((rule_type, str(rule.get("action")), tuple(sorted(clauses))))
    return tuple(sorted(sig))


def veto_wrap(agent_cls, **veto_kwargs):
    """Return a factory with the baselines' ``(init_offer, hard_policy)`` signature.

    ``veto_wrap(AspirationNegotiator)(init, hard)`` is a drop-in replacement for
    ``AspirationNegotiator(init, hard)`` in any tournament harness.
    """
    def factory(init_offer, hard_policy, *args, **kwargs):
        inner = agent_cls(init_offer, hard_policy, *args, **kwargs)
        return HardPolicyVeto(inner, hard_policy, **veto_kwargs)

    factory.__name__ = f"Veto{getattr(agent_cls, '__name__', 'Agent')}"
    factory.wrapped = agent_cls
    return factory


__all__ = [
    "HardPolicyVeto",
    "feasibility_report",
    "is_feasible",
    "project_offer",
    "veto_wrap",
]
