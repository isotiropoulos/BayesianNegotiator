"""Unit evidence for the hard-constraint penalty swept in the paper's
sensitivity analysis.

The paper states three properties of ``adapter.standard_odrl_ufun(..., penalty)``
in print, and a reader is entitled to see them checked rather than asserted:

1. **Inert at zero.** ``penalty=0`` reproduces the scalar-reservation utility
   used for the headline tables *exactly*, so the ``lambda = 0`` column of the
   sweep is a control and not a re-measurement.
2. **The charge is the one the scorer applies.** The penalty is folded into the
   per-issue lookup tables, which is only sound if the per-issue violation count
   equals the document-level ``count_hard_violations`` the benchmark grades with.
   It does, because a hard clause names one ``(rule_type, action, operand)``
   triple and each such triple is one issue.
3. **The arithmetic is the stated one**, namely
   ``u'(w) = u(w) - penalty * nu(w) / k``.

Note what these tests do *not* touch: no NegMAS strategy is modified anywhere.
The penalty lives entirely in the preference object handed to a published agent.

Run with::

    python -m pytest tests/test_penalty_ufun.py -q
"""
from __future__ import annotations

import os
import random
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "odrl-tournament")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from odrl_tournament.adapter import (                                  # noqa: E402
    _hard_index, build_space, standard_odrl_ufun,
)
from odrl_tournament.profile import load_scenario, scenario_names      # noqa: E402
from odrl_tournament.scoring import count_hard_violations              # noqa: E402
from src.bambon.baselines.veto import _clause_ok                       # noqa: E402

CORPUS = os.path.join(_ROOT, "data", "generated_agentodrl", "far_wide")
RESERVATION = 0.35        # the value the published runs use
LEVELS = 41               # the published value-grid resolution
N_SCENARIOS = 4
N_DRAWS = 120


def _scenarios():
    if not os.path.isdir(CORPUS):
        pytest.skip(f"corpus not present: {CORPUS}")
    for name in scenario_names(CORPUS)[:N_SCENARIOS]:
        ip, hp, ic, hc = load_scenario(os.path.join(CORPUS, name))
        yield name, ip, hp, ic, hc


def _outcomes(space, seed=0):
    rng = random.Random(seed)
    grids = [list(i.all) for i in space.issues]
    return [tuple(rng.choice(g) for g in grids) for _ in range(N_DRAWS)]


def test_penalty_zero_is_the_published_utility():
    """penalty=0 must be bit-identical, not merely close, to the old default."""
    for name, ip, hp, ic, hc in _scenarios():
        space = build_space(ip, ic, n_levels=LEVELS)
        for init, hard in ((ip, hp), (ic, hc)):
            plain = standard_odrl_ufun(space, init, hard, RESERVATION)
            zero = standard_odrl_ufun(space, init, hard, RESERVATION, penalty=0.0)
            assert plain.reserved_value == zero.reserved_value
            for o in _outcomes(space):
                assert float(plain(o)) == float(zero(o)), name


def test_per_issue_charge_matches_the_scorer():
    """The per-issue violation count must equal the document-level predicate.

    If these ever diverge the penalty would charge for something other than what
    the compliance column grades, and the sweep would measure nothing.
    """
    for name, ip, hp, ic, hc in _scenarios():
        space = build_space(ip, ic, n_levels=LEVELS)
        for hard in (hp, hc):
            idx = _hard_index(hard)
            for o in _outcomes(space, seed=1):
                per_issue = sum(
                    1
                    for i, m in enumerate(space.meta)
                    for hop, hval in idx.get((m["rt"], m["act"], m["operand"]), [])
                    if not _clause_ok(hop, hval, m["operator"], o[i])
                )
                assert per_issue == count_hard_violations(space.to_odrl(o), hard), name


@pytest.mark.parametrize("lam", [1.0, 4.0, 8.0, 32.0])
def test_penalty_arithmetic(lam):
    """u'(w) == u(w) - lam * nu(w) / k, for every outcome."""
    for name, ip, hp, ic, hc in _scenarios():
        space = build_space(ip, ic, n_levels=LEVELS)
        k = len(space.meta)
        for init, hard in ((ip, hp), (ic, hc)):
            u = standard_odrl_ufun(space, init, hard, RESERVATION)
            up = standard_odrl_ufun(space, init, hard, RESERVATION, penalty=lam)
            for o in _outcomes(space, seed=2):
                nu = count_hard_violations(space.to_odrl(o), hard)
                assert float(up(o)) == pytest.approx(
                    float(u(o)) - lam * nu / k, abs=1e-12), (name, lam)


def test_reserved_value_is_untouched_by_the_penalty():
    """The penalty is added to the scalar floor, it does not replace it."""
    for name, ip, hp, ic, hc in _scenarios():
        space = build_space(ip, ic, n_levels=LEVELS)
        for init, hard in ((ip, hp), (ic, hc)):
            for lam in (0.0, 8.0, 32.0):
                u = standard_odrl_ufun(space, init, hard, RESERVATION, penalty=lam)
                assert u.reserved_value == pytest.approx(RESERVATION), name
