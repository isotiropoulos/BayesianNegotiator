# src/bambon/utils/agreement.py
from __future__ import annotations
from typing import Any, Dict
from ..models.models import IssueKey
from ..utils.helpers import (
    _interp, is_iso_date, is_iso_duration,
    iso_to_days, duration_to_seconds,
)

def offers_equal_generic(a: Dict[IssueKey, Any], b: Dict[IssueKey, Any]) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a.keys():
        va, vb = a[k], b[k]
        if isinstance(va, list):
            va = set(va)
        if isinstance(vb, list):
            vb = set(vb)
        if va != vb:
            return False
    return True

def offers_close_generic(
    a: Dict[IssueKey, Any],
    b: Dict[IssueKey, Any],
    round_idx: int = 0,
    max_rounds: int = 0,
) -> bool:
    t = (round_idx / max_rounds) if max_rounds else 0.0
    eps_num_by_operand = {
        "odrl:payAmount": _interp(50.0, 200.0, t),      # relaxed
        "odrl:delayPeriod": _interp(86400.0, 3*86400.0, t),
        "odrl:dateTime": _interp(1.0, 2.0, t),
    }
    jaccard_min = _interp(0.75, 0.50, t)               # relaxed

    if a.keys() != b.keys():
        return False
    for k in a.keys():
        left_operand, _, _ = k
        va, vb = a[k], b[k]

        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            eps = eps_num_by_operand.get(left_operand)
            if eps is None:
                avg = max(1.0, (abs(va) + abs(vb)) / 2.0)
                eps = max(1.0, 0.02 * avg)
            if abs(float(va) - float(vb)) > eps:
                return False
        elif isinstance(va, str) and isinstance(vb, str):
            if is_iso_duration(va) and is_iso_duration(vb):
                if abs(duration_to_seconds(va) - duration_to_seconds(vb)) > eps_num_by_operand.get(left_operand, 86400.0):
                    return False
            elif is_iso_date(va) and is_iso_date(vb):
                if abs(iso_to_days(va) - iso_to_days(vb)) > eps_num_by_operand.get(left_operand, 1.0):
                    return False
            else:
                if va != vb:
                    return False
        elif isinstance(va, list) and isinstance(vb, list):
            A, B = set(va), set(vb)
            j = 1.0 if (A.issubset(B) or B.issubset(A)) else (len(A & B) / len(A | B) if (A or B) else 1.0)
            if j < jaccard_min:
                return False
        else:
            if va != vb:
                return False
    return True
