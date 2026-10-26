"""NegMAS-compatible negotiator wrapping the BAMBON Bayesian negotiator.

This module exposes :class:`Bambon` (see :mod:`src.bambon.negotiator`) as a
first-class `NegMAS <https://negmas.readthedocs.io>`_ negotiator so it can take
part in any NegMAS ``SAOMechanism`` / tournament alongside the ANL/ANAC agents,
Boulware/Conceder, GENIUS agents, etc.

Design goals
------------
* **Follow the NegMAS negotiator contract in full.** ``BambonNegotiator`` is a
  proper :class:`negmas.sao.SAONegotiator`
  (https://negmas.readthedocs.io/en/latest/modules/negotiators.html#negmas.negotiators.Negotiator)
  and implements the whole documented callback lifecycle:
  ``join`` -> ``on_preferences_changed`` -> ``on_negotiation_start`` ->
  ``propose`` / ``respond`` (per round) -> ``on_negotiation_end`` -> ``on_leave``.
  ``super()`` is always called where the docs require it.
* **Lose no BAMBON functionality.** The whole decision logic still lives in the
  unmodified :meth:`Bambon.observe` loop (beliefs, set snapping, utility-aware
  concession, deadline pressure, hopelessness/infeasibility detection). This
  adapter only maps NegMAS outcomes <-> ODRL offers and surfaces BAMBON's
  ``(accept?, counter)`` decision through the SAO ``propose``/``respond`` API.
* **Be self-contained.** It depends only on the ``src.bambon`` package and
  ``negmas`` -- not on the test tree. A default ODRL outcome space and a default
  ODRL utility function are provided, but both can be overridden.

Quick start
-----------
    from src.bambon.negmas_negotiator import make_bambon_session

    prov, cons, mech = make_bambon_session(
        init_provider, init_consumer,
        hard_provider=hard_provider, hard_consumer=hard_consumer,
        n_steps=200,
    )
    mech.run()
    print(mech.agreement)      # NegMAS outcome (or None)

Lower level::

    from negmas.sao import SAOMechanism
    from src.bambon.negmas_negotiator import (
        BambonNegotiator, ODRLOutcomeSpace, ODRLUtilityFunction,
    )
    from src.bambon.negotiator import Bambon

    space = ODRLOutcomeSpace(init_provider, init_consumer,
                             hard_provider, hard_consumer)
    prov = BambonNegotiator(
        name="provider",
        bambon=Bambon(name="provider", init_offer=init_provider,
                      hard_policy=hard_provider),
        space=space,
        ufun=ODRLUtilityFunction(space, init_provider, hard_provider),
    )
    ...

Requires ``negmas`` (``pip install negmas``); it is an optional dependency of
this project.
"""
from __future__ import annotations

import copy
import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple

try:  # NegMAS is an optional dependency of this project.
    from negmas import ResponseType
    from negmas.outcomes import Outcome, make_issue
    from negmas.preferences import UtilityFunction
    from negmas.sao import SAOMechanism, SAONegotiator
except Exception as exc:  # pragma: no cover - only hit when negmas is missing
    raise ImportError(
        "src.bambon.negmas_negotiator requires the optional dependency 'negmas'. "
        "Install it with `pip install negmas`."
    ) from exc

from .negotiator import Bambon, RULE_TYPES
from .models.models import NegotiationInfeasible
from .utils.helpers import (
    is_iso_date,
    is_iso_duration,
    iso_to_days,
    duration_to_seconds,
    days_to_iso,
    seconds_to_duration,
    satisfies,
    strip_key,
)
from .utils.utility import numeric_utility

import os as _os
import warnings as _warnings


# Discretisation granularity for numeric/temporal issues in the outcome space.
N_NUMERIC_LEVELS = 12


# ----------------------------------------------------------------------
# Sender-aware ODRL rendering.
#
# `to_odrl` used to rebuild EVERY offer from one template -- the provider's
# opening document -- replacing only `rightOperand`. The operator was therefore
# always the provider's, whoever actually sent the offer. Measured on
# data/generated_agentodrl (300 scenarios x 5 corpora, identical each time): the
# consumer's real operator differs from the provider's on 31.9% of negotiable
# terms, and on the provider side `Issue.oppOperator` equalled the agent's OWN
# operator on 100% of issues. That field drives the belief update, the
# acceptance test and the hard-commit veto, so the opponent model was being
# trained and queried on a mirror of the agent's own preferences.
#
# Each offer is now rendered from its SENDER's own opening document, so that
# sender's operators survive the round-trip. Note that the feasibility/veto
# checks in the harness deliberately do NOT use this: they must read the
# document exactly as the scorer builds it. See odrl_tournament/adapter.py.
# ----------------------------------------------------------------------


# ======================================================================
# small value helpers (ODRL rightOperand normalisation)
# ======================================================================
def _to_num(value: Any) -> Optional[float]:
    """Return a numeric view of an ODRL rightOperand, or None if not numeric.

    Numbers pass through; ISO-8601 dates map to *days since BASE_DATE* and ISO
    durations map to *seconds*, exactly like BAMBON's internal representation.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if is_iso_date(value):
            return float(iso_to_days(value))
        if is_iso_duration(value):
            return float(duration_to_seconds(value))
    return None


def _setish(value: Any) -> set:
    """Coerce any categorical/set rightOperand into a ``set`` of labels."""
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return set(value)
    return {value}


def _subsets(items) -> "itertools.chain":
    items = list(items)
    return itertools.chain.from_iterable(
        itertools.combinations(items, r) for r in range(1, len(items) + 1)
    )


def _jaccard_distance(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def _find_clause(offer: Dict, rt: str, act: str, operand: str) -> Any:
    """Return the rightOperand for (rule_type, action, operand) in an ODRL offer."""
    for rule in offer.get(rt, []) or []:
        if strip_key(str(rule.get("action"))) != act:
            continue
        for c in rule.get("constraint", []) or []:
            if strip_key(c["leftOperand"]) == operand:
                return c["rightOperand"]
    return None


def _find_operator(offer: Dict, rt: str, act: str, operand: str) -> Optional[str]:
    for rule in offer.get(rt, []) or []:
        if strip_key(str(rule.get("action"))) != act:
            continue
        for c in rule.get("constraint", []) or []:
            if strip_key(c["leftOperand"]) == operand:
                return strip_key(c["operator"])
    return None


# ======================================================================
# 1. Outcome space:  ODRL offers  <->  NegMAS discrete outcomes
# ======================================================================
class ODRLOutcomeSpace:
    """Build a NegMAS discrete issue set from the two parties' ODRL offers.

    Each ``(rule_type, action, leftOperand)`` triple present in the provider's
    initial offer becomes one NegMAS issue. Numeric/temporal operands are
    discretised into :data:`N_NUMERIC_LEVELS` levels spanning both parties'
    offers (and, when supplied, their hard policies); categorical/set operands
    enumerate the relevant candidate label-tuples.

    The object also provides the bidirectional mapping used by the negotiator
    and utility function:

    * :meth:`to_odrl`  -- NegMAS outcome tuple  -> ODRL offer dict
    * :meth:`to_outcome` -- ODRL offer dict -> nearest outcome tuple
    """

    def __init__(
        self,
        init_provider: Dict,
        init_consumer: Dict,
        hard_provider: Optional[Dict] = None,
        hard_consumer: Optional[Dict] = None,
        n_numeric_levels: int = N_NUMERIC_LEVELS,
    ):
        self.template = copy.deepcopy(init_provider)
        # One template per side. `self.template` stays the provider's document so
        # every existing caller of `to_odrl(outcome)` is untouched.
        self.templates = {
            "provider": self.template,
            "consumer": (copy.deepcopy(init_consumer) if init_consumer
                         else self.template),
        }
        self.n_numeric_levels = max(2, int(n_numeric_levels))
        self.meta: List[Dict[str, Any]] = []
        self.issues = []

        seen = set()
        for rt in RULE_TYPES:
            for rule in init_provider.get(rt, []) or []:
                act = strip_key(str(rule.get("action")))
                for c in rule.get("constraint", []) or []:
                    operand = strip_key(c["leftOperand"])
                    key = (rt, act, operand)
                    if key in seen:
                        continue
                    seen.add(key)
                    pv = c["rightOperand"]
                    cv = _find_clause(init_consumer, rt, act, operand)
                    hpv = _find_clause(hard_provider, rt, act, operand) if hard_provider else None
                    hcv = _find_clause(hard_consumer, rt, act, operand) if hard_consumer else None
                    values, kind = self._candidate_values(pv, cv, hpv, hcv)
                    name = f"{rt}:{act}:{operand}"
                    self.issues.append(make_issue(values=values, name=name))
                    self.meta.append(
                        {
                            "rt": rt,
                            "act": act,
                            "operand": operand,
                            "operator": strip_key(c["operator"]),
                            # The same term as each side actually stated it.
                            # The provider's copy is the legacy "operator" above.
                            "operator_by_side": {
                                "provider": strip_key(c["operator"]),
                                "consumer": (
                                    _find_operator(init_consumer, rt, act, operand)
                                    or strip_key(c["operator"])),
                            },
                            "kind": kind,
                            "template_iso": pv if isinstance(pv, str) else None,
                        }
                    )

        if not self.issues:
            raise ValueError(
                "ODRLOutcomeSpace: no negotiable constraints found in the provider "
                "initial offer (need permission/obligation/prohibition rules with "
                "constraints)."
            )

        # The issue set is enumerated from the PROVIDER's opening offer alone, so
        # a term the consumer opens on that the provider never mentions is not in
        # the outcome space and cannot be negotiated by either side. Measured at
        # zero on data/generated_agentodrl; recorded (and warned about under the
        # fix flag) so it can never go unnoticed on another corpus.
        self.orphan_keys = []
        if init_consumer:
            for rt in RULE_TYPES:
                for rule in init_consumer.get(rt, []) or []:
                    act = strip_key(str(rule.get("action")))
                    for c in rule.get("constraint", []) or []:
                        key = (rt, act, strip_key(c["leftOperand"]))
                        if key not in seen:
                            self.orphan_keys.append(key)
        if self.orphan_keys:
            _warnings.warn(
                f"ODRLOutcomeSpace: {len(self.orphan_keys)} consumer term(s) are "
                f"absent from the provider's opening offer and therefore not "
                f"negotiable: {self.orphan_keys[:5]}",
                RuntimeWarning, stacklevel=2)

    def _candidate_values(self, pv, cv, hpv, hcv) -> Tuple[List[Any], str]:
        """Return ``(candidate_values, kind)`` for one operand.

        Numeric/temporal -> a sorted list of discretised numbers; categorical/set
        -> a sorted list of label-tuples (each a hashable NegMAS outcome value).
        """
        if _to_num(pv) is not None:  # numeric or ISO-temporal
            nums = [x for x in (_to_num(pv), _to_num(cv), _to_num(hpv), _to_num(hcv)) if x is not None]
            lo, hi = (min(nums), max(nums)) if nums else (0.0, 1.0)
            if hi <= lo:
                hi = lo + 1.0
            n = self.n_numeric_levels
            vals = [round(lo + (hi - lo) * i / (n - 1), 4) for i in range(n)]
            return sorted(set(vals)), "numeric"

        # categorical / set: enumerate candidate label-sets as sorted tuples
        cands = set()
        for v in (pv, cv, hpv, hcv):
            for s in _subsets(_setish(v)):
                cands.add(tuple(sorted(map(str, s))))
        cands.discard(())
        if not cands:  # degenerate operand with no labels anywhere
            cands.add((str(pv),))
        return sorted(cands), "set"

    # ---- mapping ----------------------------------------------------------
    def to_odrl(self, outcome: Tuple, sender: Optional[str] = None) -> Dict:
        """Rebuild a complete ODRL offer dict from a NegMAS outcome tuple.

        ``sender`` names the party this outcome is being expressed as --
        "provider" or "consumer". The offer is rendered from that party's own
        opening document, so its operators -- and hence the direction of every
        constraint -- survive the round-trip. With ``sender=None`` the provider's
        template is used, which is what the feasibility checks want because it is
        how the scorer builds the contract.
        """
        tmpl = self.template
        if sender in getattr(self, "templates", {}):
            tmpl = self.templates[sender]
        offer = copy.deepcopy(tmpl)
        by_key = {
            (m["rt"], m["act"], m["operand"]): outcome[i] for i, m in enumerate(self.meta)
        }
        for rt in RULE_TYPES:
            for rule in offer.get(rt, []) or []:
                act = strip_key(str(rule.get("action")))
                for c in rule.get("constraint", []) or []:
                    operand = strip_key(c["leftOperand"])
                    key = (rt, act, operand)
                    if key not in by_key:
                        continue
                    m = next(mm for mm in self.meta
                             if (mm["rt"], mm["act"], mm["operand"]) == key)
                    c["rightOperand"] = self._render(by_key[key], m)
        return offer

    def to_outcome(self, offer: Dict) -> Tuple:
        """Snap an arbitrary ODRL offer to the nearest point of the outcome space."""
        out: List[Any] = []
        for i, m in enumerate(self.meta):
            raw = _find_clause(offer, m["rt"], m["act"], m["operand"])
            vals = self.issues[i].values
            if m["kind"] == "numeric":
                x = _to_num(raw)
                snapped = (
                    min(vals, key=lambda v: abs(v - x)) if x is not None
                    else vals[len(vals) // 2]
                )
            else:
                tgt = tuple(sorted(map(str, _setish(raw))))
                snapped = min(vals, key=lambda v: _jaccard_distance(set(v), set(tgt)))
            out.append(snapped)
        return tuple(out)

    def _render(self, raw, m) -> Any:
        """Turn a stored issue value back into an ODRL rightOperand."""
        if m["kind"] != "numeric":
            lst = list(raw)
            return lst if len(lst) != 1 else lst[0]
        iso = m["template_iso"]
        if iso and is_iso_date(iso):
            return days_to_iso(int(round(raw)))
        if iso and is_iso_duration(iso):
            return seconds_to_duration(max(0.0, float(raw)))
        return raw


# ======================================================================
# 2. Utility function:  a self-contained ODRL utility as a NegMAS ufun
# ======================================================================
class ODRLUtilityFunction(UtilityFunction):
    """Anchored ODRL self-utility, exposed as a NegMAS :class:`UtilityFunction`.

    The utility is anchored on the agent's own initial offer (utility ``1.0``)
    and gated by its hard policy (any hard-policy breach -> utility ``0.0`` so a
    non-compliant contract is never preferred over no-deal). This mirrors the
    normative view used by BAMBON's own utility-aware concession
    (:mod:`src.bambon.utils.utility`).

    Pass ``utility_fn`` to override the per-offer scoring with your own callable
    ``utility_fn(offer: dict) -> float in [0, 1]`` (e.g. the project benchmark's
    ``compute_utility``) while still using this class for NegMAS integration and
    hard-policy gating.
    """

    def __init__(
        self,
        space: ODRLOutcomeSpace,
        init_offer: Dict,
        hard_policy: Optional[Dict] = None,
        *,
        utility_fn: Optional[Callable[[Dict], float]] = None,
        **kwargs,
    ):
        kwargs.setdefault("issues", space.issues)
        kwargs.setdefault("outcome_space", None)
        super().__init__(**kwargs)
        self._space = space
        self._init = init_offer
        self._hard = hard_policy
        self._utility_fn = utility_fn
        # Never accept an outcome worse than no-deal (hard-policy breach == 0).
        self.reserved_value = 0.0
        # Precompute per-issue scoring/gating descriptors so eval() is O(#issues)
        # with no dict traversal or deep copy -- essential because enumerating
        # NegMAS agents (Boulware/Conceder/...) call eval() over the *whole*
        # outcome space, potentially every round.
        self._prepare()

    def _prepare(self) -> None:
        """Precompute per-issue anchors/reservations and hard-policy checks."""
        meta = self._space.meta
        # Offer operator per issue is fixed by the provider template (to_odrl only
        # rewrites rightOperand), so we can gate/score straight from the outcome.
        self._issue_op: List[str] = [m["operator"] for m in meta]
        self._issue: List[Dict[str, Any]] = []
        for m in meta:
            rt, act, operand = m["rt"], m["act"], m["operand"]
            entry: Dict[str, Any] = {"kind": m["kind"]}
            if m["kind"] == "numeric":
                anchor = _to_num(_find_clause(self._init, rt, act, operand))
                entry["anchor"] = anchor
                entry["op"] = _find_operator(self._init, rt, act, operand) or m["operator"]
                entry["reservation"] = self._numeric_reservation(
                    rt, act, operand, anchor if anchor is not None else 0.0, entry["op"]
                )
            else:
                entry["init_set"] = set(map(str, _setish(_find_clause(self._init, rt, act, operand))))
            self._issue.append(entry)

        # Hard-policy checks mapped to the issue index they constrain.
        index = {(m["rt"], m["act"], m["operand"]): i for i, m in enumerate(meta)}
        self._hard_checks: List[Tuple[int, str, Any]] = []
        if self._hard:
            for rt in RULE_TYPES:
                for rule in self._hard.get(rt, []) or []:
                    act = strip_key(str(rule.get("action")))
                    for c in rule.get("constraint", []) or []:
                        operand = strip_key(c["leftOperand"])
                        i = index.get((rt, act, operand))
                        if i is None:
                            # Operand is not part of the negotiated outcome space;
                            # it cannot be scored from an outcome tuple, so skip it.
                            continue
                        self._hard_checks.append(
                            (i, strip_key(c["operator"]), self._coerce(c["rightOperand"]))
                        )

    @staticmethod
    def _outcome_value(kind: str, raw: Any) -> Any:
        """Coerce a single outcome-tuple element for scoring/gating."""
        if kind == "numeric":
            return float(raw)
        return set(map(str, raw)) if isinstance(raw, (list, tuple, set)) else {str(raw)}

    # NegMAS calls eval() to score an outcome.
    def eval(self, outcome: Optional[Tuple]) -> float:
        if outcome is None:
            return self.reserved_value
        # Fast outcome-based hard-policy gate: any breach -> utility 0.
        for i, hop, hval in self._hard_checks:
            ov = self._outcome_value(self._issue[i]["kind"], outcome[i])
            if satisfies(hop, hval, self._issue_op[i], ov) is False:
                return 0.0
        if self._utility_fn is not None:
            return float(self._utility_fn(self._space.to_odrl(outcome)))
        # Default anchored scorer, straight from the outcome tuple.
        scores: List[float] = []
        for i, d in enumerate(self._issue):
            raw = outcome[i]
            if d["kind"] == "numeric":
                if d["anchor"] is None:
                    continue
                scores.append(numeric_utility(float(raw), d["anchor"], d["reservation"], d["op"]))
            else:
                a = d["init_set"]
                b = set(map(str, raw)) if isinstance(raw, (list, tuple, set)) else {str(raw)}
                if not a and not b:
                    continue
                scores.append(1.0 - _jaccard_distance(a, b))
        return sum(scores) / len(scores) if scores else 0.0

    # -- offer-based scorers (kept for external inspection / custom use) ----
    def _default_utility(self, offer: Dict) -> float:
        """Mean per-issue self-utility of ``offer`` (1.0 == our initial offer)."""
        scores: List[float] = []
        for m in self._space.meta:
            rt, act, operand = m["rt"], m["act"], m["operand"]
            init_val = _find_clause(self._init, rt, act, operand)
            off_val = _find_clause(offer, rt, act, operand)
            if m["kind"] == "numeric":
                anchor = _to_num(init_val)
                v = _to_num(off_val)
                if anchor is None or v is None:
                    continue
                res = self._numeric_reservation(rt, act, operand, anchor, m["operator"])
                scores.append(numeric_utility(v, anchor, res, m["operator"]))
            else:
                a = set(map(str, _setish(init_val)))
                b = set(map(str, _setish(off_val)))
                if not a and not b:
                    continue
                scores.append(1.0 - _jaccard_distance(a, b))
        return sum(scores) / len(scores) if scores else 0.0

    def _numeric_reservation(self, rt, act, operand, anchor, op) -> float:
        """Reservation (utility-0) point for a numeric issue.

        Uses the hard-policy bound when present; otherwise falls back to a
        symmetric span around the anchor so utility stays well-defined.
        """
        if self._hard is not None:
            hv = _to_num(_find_clause(self._hard, rt, act, operand))
            if hv is not None and hv != anchor:
                return hv
        span = abs(anchor) * 0.5 or 1.0
        op = (op or "").lower()
        if op in ("gteq", "gt"):
            return anchor - span
        if op in ("lteq", "lt"):
            return anchor + span
        return anchor + span

    # -- hard-policy gate ---------------------------------------------------
    def _hard_violations(self, offer: Dict) -> int:
        """Count hard-policy constraints that ``offer`` fails to satisfy."""
        if not self._hard:
            return 0
        violations = 0
        for rt in RULE_TYPES:
            for rule in self._hard.get(rt, []) or []:
                act = strip_key(str(rule.get("action")))
                for c in rule.get("constraint", []) or []:
                    operand = strip_key(c["leftOperand"])
                    hard_op = strip_key(c["operator"])
                    hard_val = c["rightOperand"]
                    off_op = _find_operator(offer, rt, act, operand)
                    off_val = _find_clause(offer, rt, act, operand)
                    if off_op is None:
                        violations += 1  # required operand missing from the offer
                        continue
                    hv = self._coerce(hard_val)
                    ov = self._coerce(off_val)
                    ok = satisfies(hard_op, hv, off_op, ov)
                    if ok is False:
                        violations += 1
        return violations

    @staticmethod
    def _coerce(value: Any) -> Any:
        """Map an ODRL rightOperand into the space ``satisfies`` expects."""
        num = _to_num(value)
        if num is not None:
            return num
        if isinstance(value, (list, tuple, set)):
            return set(map(str, value))
        return value


# ======================================================================
# 3. BAMBON as a NegMAS SAONegotiator
# ======================================================================
class BambonNegotiator(SAONegotiator):
    """Drive an unmodified :class:`Bambon` inside a NegMAS ``SAOMechanism``.

    The negotiator maps NegMAS outcomes <-> ODRL offers and forwards every
    received offer to :meth:`Bambon.observe`, which returns ``(signals, offer)``
    where ``signals`` is ``None`` while still negotiating and a non-empty dict
    once BAMBON is willing to sign. That decision is surfaced through the SAO
    ``respond``/``propose`` interface:

    * ``respond`` -> ``ACCEPT_OFFER`` when BAMBON agrees, ``END_NEGOTIATION`` when
      BAMBON deems the negotiation infeasible, otherwise ``REJECT_OFFER`` (and it
      stashes BAMBON's counter-offer for the next ``propose``).
    * ``propose`` -> BAMBON's opening offer on the first move, then whatever
      counter-offer was produced by the last ``respond``.

    A **fresh** BAMBON instance is created from the supplied one at the start of
    every negotiation (via deep copy), so the same :class:`BambonNegotiator` can
    be reused across a tournament without leaking belief state between runs.

    Parameters
    ----------
    bambon:
        A configured :class:`Bambon` used as the template for each negotiation.
    space:
        The :class:`ODRLOutcomeSpace` shared by the mechanism.
    sync_deadline:
        When True (default), align BAMBON's concession deadline (and, if needed,
        its ``max_rounds`` cap) with the mechanism's ``n_steps`` on negotiation
        start, so BAMBON's time-aware concession matches the actual horizon.
    """

    def __init__(
        self,
        *args,
        bambon: Bambon,
        space: ODRLOutcomeSpace,
        sync_deadline: bool = True,
        side: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._template = bambon
        self._space = space
        self._sync_deadline = sync_deadline
        # Which of the two opening documents is mine. Falls back to the
        # negotiator's name, which the tournament sets to the role.
        self._side = side
        # per-negotiation state (reset in on_negotiation_start)
        self._b: Optional[Bambon] = None
        self._pending: Optional[Tuple] = None
        self._dead = False

    # -- lifecycle (see the NegMAS callback-order guarantees) ---------------
    def on_preferences_changed(self, changes):
        # ALWAYS called before on_negotiation_start. BAMBON keeps its own
        # (belief-driven) notion of value, so there is nothing to recompute
        # here; we simply honour the documented contract and call super().
        super().on_preferences_changed(changes)

    def on_negotiation_start(self, state) -> None:
        # Required by NegMAS to call super(). Start every negotiation from a
        # pristine BAMBON so no belief state leaks across runs.
        super().on_negotiation_start(state)
        self._b = copy.deepcopy(self._template)
        self._pending = None
        self._dead = False
        if self._sync_deadline:
            n_steps = self._negotiation_n_steps()
            if n_steps:
                self._b.deadline = n_steps
                # Never let BAMBON's own round cap end the negotiation before
                # the mechanism's deadline (would forfeit reachable agreements).
                if self._b.max_rounds < n_steps + 1:
                    self._b.max_rounds = n_steps + 1

    def on_negotiation_end(self, state) -> None:
        super().on_negotiation_end(state)

    def on_leave(self, state) -> None:
        # Docs: MUST call the base implementation when overriding on_leave.
        super().on_leave(state)

    # -- SAO protocol -------------------------------------------------------
    def propose(self, state, dest: Optional[str] = None) -> Optional["Outcome"]:
        if self._dead or self._b is None:
            return None
        if self._pending is not None:
            out, self._pending = self._pending, None
            return out
        # Opening move: BAMBON's initial offer, snapped to the outcome space.
        return self._space.to_outcome(self._b.init_offer)

    def respond(self, state, source: Optional[str] = None) -> "ResponseType":
        if self._dead or self._b is None:
            return ResponseType.END_NEGOTIATION
        offer = getattr(state, "current_offer", None)
        if offer is None:
            return ResponseType.REJECT_OFFER
        try:
            signals, counter = self._b.observe(
                self._space.to_odrl(offer, sender=self._opp_side()))
        except NegotiationInfeasible:
            # BAMBON judged the negotiation hopeless / past its round cap.
            self._dead = True
            self._pending = None
            return ResponseType.END_NEGOTIATION
        if signals is not None:  # BAMBON reached full agreement on this offer
            # An acceptance under SAO binds the outcome ON THE TABLE, so BAMBON
            # must be agreeing to exactly that and nothing else. `observe`
            # returns the agreement document it constructed from its own
            # per-issue `accepted` state; if that ever maps to a different
            # outcome -- a stale term, an issue the counterparty dropped from
            # its last offer -- then the two sides disagree about what is being
            # signed. The safe move is not to accept a document BAMBON never
            # vetted, but to put BAMBON's own version on the table as a
            # proposal, which the counterparty is then free to accept or refuse.
            #
            # Measured on far_wide (45 scenarios x 8 published opponents x both
            # roles): 40 acceptances, 40 exact matches, 0 mismatches. This is an
            # invariant guard, not a behaviour change -- see the regression test
            # that flags-off stays bit-identical.
            try:
                mine = tuple(self._space.to_outcome(counter))
            except Exception:
                mine = tuple(offer)
            if mine != tuple(offer):
                self._accept_mismatches = getattr(self, "_accept_mismatches", 0) + 1
                self._pending = mine
                return ResponseType.REJECT_OFFER
            return ResponseType.ACCEPT_OFFER
        # Reject and remember BAMBON's counter-offer for the next propose().
        self._pending = self._space.to_outcome(counter)
        return ResponseType.REJECT_OFFER

    # -- helpers ------------------------------------------------------------
    def _own_side(self) -> str:
        """"provider" or "consumer": whose opening document is mine."""
        if self._side in ("provider", "consumer"):
            return self._side
        return "consumer" if str(self.name) == "consumer" else "provider"

    def _opp_side(self) -> str:
        return "consumer" if self._own_side() == "provider" else "provider"

    def _negotiation_n_steps(self) -> Optional[int]:
        nmi = getattr(self, "nmi", None)
        n_steps = getattr(nmi, "n_steps", None) if nmi is not None else None
        try:
            return int(n_steps) if n_steps is not None else None
        except (TypeError, ValueError):
            return None


# ======================================================================
# 4. Convenience factory
# ======================================================================
def make_bambon_negotiator(
    name: str,
    init_offer: Dict,
    space: ODRLOutcomeSpace,
    hard_policy: Optional[Dict] = None,
    *,
    utility_fn: Optional[Callable[[Dict], float]] = None,
    bambon: Optional[Bambon] = None,
    sync_deadline: bool = True,
    side: Optional[str] = None,
    **bambon_kwargs,
) -> BambonNegotiator:
    """Build a ready-to-add :class:`BambonNegotiator` with its ODRL ufun.

    Any extra keyword arguments are forwarded to the :class:`Bambon` constructor
    (``lr``, ``inertia``, ``tau_in``, ``utility_mode``, ...), so no BAMBON
    tuning knob is lost. Pass a pre-built ``bambon`` to use it verbatim instead.
    """
    if bambon is None:
        bambon = Bambon(
            name=name,
            init_offer=init_offer,
            hard_policy=hard_policy,
            **bambon_kwargs,
        )
    ufun = ODRLUtilityFunction(space, init_offer, hard_policy, utility_fn=utility_fn)
    return BambonNegotiator(
        name=name,
        bambon=bambon,
        space=space,
        ufun=ufun,
        sync_deadline=sync_deadline,
        side=side,
    )


def make_bambon_session(
    init_provider: Dict,
    init_consumer: Dict,
    *,
    hard_provider: Optional[Dict] = None,
    hard_consumer: Optional[Dict] = None,
    n_steps: int = 200,
    provider_kwargs: Optional[Dict[str, Any]] = None,
    consumer_kwargs: Optional[Dict[str, Any]] = None,
    provider_utility_fn: Optional[Callable[[Dict], float]] = None,
    consumer_utility_fn: Optional[Callable[[Dict], float]] = None,
) -> Tuple[BambonNegotiator, BambonNegotiator, "SAOMechanism"]:
    """Wire up a full provider-vs-consumer BAMBON negotiation in one call.

    Returns ``(provider, consumer, mechanism)`` with both negotiators already
    added to a shared :class:`negmas.sao.SAOMechanism`. Call ``mechanism.run()``.
    """
    space = ODRLOutcomeSpace(init_provider, init_consumer, hard_provider, hard_consumer)
    provider = make_bambon_negotiator(
        "provider", init_provider, space, hard_provider,
        utility_fn=provider_utility_fn, side="provider", **(provider_kwargs or {}),
    )
    consumer = make_bambon_negotiator(
        "consumer", init_consumer, space, hard_consumer,
        utility_fn=consumer_utility_fn, side="consumer", **(consumer_kwargs or {}),
    )
    mechanism = SAOMechanism(issues=space.issues, n_steps=n_steps)
    mechanism.add(provider, ufun=provider.ufun)
    mechanism.add(consumer, ufun=consumer.ufun)
    return provider, consumer, mechanism


__all__ = [
    "ODRLOutcomeSpace",
    "ODRLUtilityFunction",
    "BambonNegotiator",
    "make_bambon_negotiator",
    "make_bambon_session",
    "N_NUMERIC_LEVELS",
]
