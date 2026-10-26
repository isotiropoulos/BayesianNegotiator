"""A peer that stays silent on one of our issues must not crash the round.

``_num_similarity`` had its None-guard one line too late::

    ai = issue.to_internal(issue.lastValue)
    bi = issue.to_internal(issue.oppLastValue)   # to_internal is float(x)
    if ai is None or bi is None:                 # never reached
        return 0.0

``oppLastValue`` is None whenever the counterparty has not spoken to that issue
— the normal state on round one for any constraint our offer names and theirs
does not. So ``float(None)`` raised ``TypeError`` out of ``_context_and_severity``
and straight out of ``observe()``.

That shape is not exotic. The Eclipse DSP TCK sends
``permission: [{action: use, constraint: []}]``, so any exposed offer carrying a
real constraint met it on the first message. ``_range_similarity`` already
guarded correctly; only the scalar path did not.

Run with::

    python -m pytest tests/test_partial_offer_similarity.py -q
"""
from __future__ import annotations

import copy
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.bambon.negotiator import Bambon  # noqa: E402


def _offer(constraints):
    return {"permission": [{"action": "use", "constraint": list(constraints)}]}


SILENT = _offer([])
SCALAR = _offer([{"leftOperand": "count", "operator": "lteq", "rightOperand": 5}])
RANGE = _offer([
    {"leftOperand": "count", "operator": "gteq", "rightOperand": 1},
    {"leftOperand": "count", "operator": "lteq", "rightOperand": 9},
])
CATEGORICAL = _offer([
    {"leftOperand": "purpose", "operator": "isAnyOf", "rightOperand": ["research", "audit"]},
])


@pytest.mark.parametrize("ours", [SCALAR, RANGE, CATEGORICAL],
                         ids=["scalar", "numrange", "categorical"])
@pytest.mark.parametrize("mode", ["severity", "reservation"])
def test_peer_silent_on_our_issue_does_not_raise(ours, mode):
    agent = Bambon(
        name="t",
        init_offer=copy.deepcopy(ours),
        hopeless_mode=mode,
        hopeless_window=0,
        max_rounds=500,
    )
    accepted, response = agent.observe(copy.deepcopy(SILENT))
    assert response is not None or accepted


@pytest.mark.parametrize("mode", ["severity", "reservation"])
def test_survives_several_silent_rounds(mode):
    """Not just round one — ``lastValue`` moves, ``oppLastValue`` stays None."""
    agent = Bambon(
        name="t",
        init_offer=copy.deepcopy(SCALAR),
        hopeless_mode=mode,
        hopeless_window=0,
        no_agreement_until=999,
        max_rounds=500,
    )
    for _ in range(5):
        agent.observe(copy.deepcopy(SILENT))
