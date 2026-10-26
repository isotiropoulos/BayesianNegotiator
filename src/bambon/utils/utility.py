"""
Utility-aware negotiation policy (opt-in via ``Bambon.utility_mode``).

This module is intentionally separate from the *belief* layer. The belief models
(``NumericBeta`` / ``EnumDirichlet`` / ``SetBelief``) stay purely Bayesian and
*descriptive* of the opponent. The functions here are *normative* for us: they
turn the agent's own initial offer into an explicit self-utility and decide how
much of that utility we are willing to trade for an agreement.

Implements proposals:
  #1  explicit self-utility anchored on the initial offer
  #3  time-dependent aspiration (concession schedule)
  (#2 frontier bidding and #7 accept-vs-wait are built on these in belief.py)

Everything is defined for NUMERIC issues, where "closer to our initial offer" is
well defined. Enum/set issues keep their original behaviour.
"""
from __future__ import annotations

from .helpers import clamp


# --------------------------------------------------------------------------- #
#  #1  Self-utility, anchored on the initial offer
# --------------------------------------------------------------------------- #
#
# A numeric issue has two reference points in INTERNAL value space:
#   anchor      = our initial offered value (initValue)  -> utility 1.0 (best)
#   reservation = the worst value we will still accept    -> utility 0.0
#
# Utility increases monotonically from the reservation toward the anchor along
# the direction implied by our operator:
#   gteq/gt : we impose a floor; higher threshold is better for us (more
#             protection). Conceding means lowering it toward the reservation.
#   lteq/lt : we impose a cap; lower threshold is better for us. Conceding
#             means raising it toward the reservation.
#   eq/neq  : utility peaks at the anchor and decays with distance from it.

def numeric_utility(v: float, anchor: float, reservation: float, op: str) -> float:
    """Self-utility in [0,1] for committing numeric value ``v`` (internal space)."""
    op = (op or "").lower()
    if op in ("gteq", "gt"):
        denom = (anchor - reservation) or 1e-9
        return clamp((v - reservation) / denom, 0.0, 1.0)
    if op in ("lteq", "lt"):
        denom = (reservation - anchor) or 1e-9
        return clamp((reservation - v) / denom, 0.0, 1.0)
    # eq / neq: symmetric tent around the anchor
    span = abs(reservation - anchor) or 1e-9
    return clamp(1.0 - abs(v - anchor) / span, 0.0, 1.0)


def numeric_value_for_utility(u: float, anchor: float, reservation: float, op: str) -> float:
    """Inverse of :func:`numeric_utility`: the value achieving target utility ``u``."""
    u = clamp(u, 0.0, 1.0)
    op = (op or "").lower()
    if op in ("gteq", "gt"):
        return reservation + u * (anchor - reservation)
    if op in ("lteq", "lt"):
        return reservation - u * (reservation - anchor)
    # eq / neq: anchor is the only utility-maximising point we propose
    return anchor


# --------------------------------------------------------------------------- #
#  #3  Concession schedule (time-dependent aspiration)
# --------------------------------------------------------------------------- #

def aspiration(round_frac: float, u_start: float = 1.0,
               u_floor: float = 0.0, beta: float = 2.0) -> float:
    """
    Target self-utility for the current round (the bar we hold out for).

    round_frac : rounds_seen / deadline, clamped to [0,1].
    u_start    : aspiration at t=0 (~1.0 -> open near our initial offer).
    u_floor    : reservation utility (the least we will settle for, ~0.0).
    beta       : concession shape.
                 beta > 1  -> Boulware  (hold firm early, concede late)
                 beta = 1  -> linear
                 beta < 1  -> Conceder   (give ground early)
    """
    rf = clamp(round_frac, 0.0, 1.0)
    return u_floor + (u_start - u_floor) * (1.0 - rf ** beta)


def accept_prob_floor(round_frac: float, p_min: float = 0.30,
                      p_max: float = 0.85) -> float:
    """
    Minimum opponent-acceptance probability a *demanding* bid must clear.

    Rises toward the deadline: early on we will float ambitious offers the
    opponent may reject; near the deadline we insist on offers the belief says
    are likely to be accepted, so we actually close.
    """
    rf = clamp(round_frac, 0.0, 1.0)
    return p_min + (p_max - p_min) * rf
