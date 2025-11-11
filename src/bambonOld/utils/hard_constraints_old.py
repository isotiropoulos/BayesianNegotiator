# bambon/utils/hard_constraints.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from .helpers import parse_offer_generic, is_iso_duration, duration_to_seconds, is_iso_date, iso_to_days

# We interpret "hard" as: the incoming offer must satisfy these constraints.
# Supported operators for hard checks: eq, neq, gteq, lteq, isAnyOf, isAllOf, isNoneOf, isWithin (date → interval)

def _to_num(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        if is_iso_date(v):
            return float(iso_to_days(v))
        if is_iso_duration(v):
            return float(duration_to_seconds(v))
        try:
            return float(v)
        except Exception:
            return None
    return None

def _normalize_date_interval(ro: Any) -> Tuple[Optional[float], Optional[float]]:
    # ro is ["2025-..T..Z", "2026-..T..Z"] or xsd:date list already normalized by parse_offer_generic
    if not isinstance(ro, list) or len(ro) != 2:
        return None, None
    a, b = ro[0], ro[1]
    if not (isinstance(a, str) and isinstance(b, str)):
        return None, None
    if not (is_iso_date(a) and is_iso_date(b)):
        return None, None
    lo = float(iso_to_days(a))
    hi = float(iso_to_days(b))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi

def _setify(x: Any) -> set:
    if isinstance(x, list):
        return set(x)
    return set([x])

def check_against(my_hard_offer: Dict[str, Any], incoming_offer: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Return (ok, reasons). 'ok' is True if incoming_offer satisfies my hard rules.
    We compare the already-parsed constraints (generic keys) for both offers.
    Only my_hard_offer constraints are enforced; we do NOT assume anything about the other side.
    """
    if not my_hard_offer:
        return True, []

    # Parse both; we only *use* my_hard parsed constraints to check incoming
    hard_issues = parse_offer_generic(my_hard_offer)
    inc_issues  = parse_offer_generic(incoming_offer)

    reasons: List[str] = []
    # For each hard constraint, check the corresponding (lo, op, idx) presence and satisfaction
    # If incoming does not have that (lo, op) at all, we still try best-effort validation by operand.
    for (lo, op, idx), hard_ro in hard_issues.items():
        # Grab *all* incoming values for this operand (any operator)
        inc_for_operand = [(k, v) for k, v in inc_issues.items() if k[0] == lo]
        if not inc_for_operand:
            reasons.append(f"missing operand {lo} in incoming")
            continue

        # Now enforce depending on operator family
        if op in ("odrl:eq", "odrl:neq"):
            satisfied = False
            for (k2, v2) in inc_for_operand:
                if k2[1] == "odrl:eq":
                    if op == "odrl:eq":
                        satisfied |= (v2 == hard_ro)
                    else:
                        satisfied |= (v2 != hard_ro)
            if not satisfied:
                reasons.append(f"{lo} {op} hard not met")
        elif op in ("odrl:gteq", "odrl:lteq", "odrl:gt", "odrl:lt"):
            # Numeric-like check. Find any numeric-like incoming constraint or eq.
            hnum = _to_num(hard_ro)
            if hnum is None:
                reasons.append(f"{lo} {op}: non-numeric hard value")
                continue
            satisfied = False
            for (k2, v2) in inc_for_operand:
                if k2[1] in ("odrl:eq","odrl:gteq","odrl:lteq","odrl:gt","odrl:lt"):
                    vnum = _to_num(v2)
                    if vnum is None:
                        continue
                    if op in ("odrl:gteq","odrl:gt"):
                        satisfied |= (vnum >= hnum)
                    else:  # lteq/lt
                        satisfied |= (vnum <= hnum)
            if not satisfied:
                reasons.append(f"{lo} {op} hard not met")
        # elif op == "odrl:isWithin":
        #     # Hard date interval: incoming must be inside [lo,hi] (as eq or another within)
        #     hlo, hhi = _normalize_date_interval(hard_ro)
        #     if hlo is None:
        #         reasons.append(f"{lo} isWithin: invalid hard interval")
        #         continue
        #     satisfied = False
        #     for (k2, v2) in inc_for_operand:
        #         if k2[1] == "odrl:eq":
        #             v = _to_num(v2)
        #             if v is not None:
        #                 satisfied |= (hlo <= v <= hhi)
        #         elif k2[1] in ("odrl:gteq","odrl:lteq"):
        #             # Weak check: if incoming has bounds, ensure they intersect the hard window
        #             vnum = _to_num(v2)
        #             if vnum is None:
        #                 continue
        #             if k2[1] == "odrl:gteq":
        #                 satisfied |= (vnum <= hhi)
        #             else:  # lteq
        #                 satisfied |= (vnum >= hlo)
        #         elif k2[1] == "odrl:isWithin":
        #             lo2, hi2 = _normalize_date_interval(v2)
        #             if lo2 is not None:
        #                 # require inc interval to be inside hard interval
        #                 satisfied |= (lo2 >= hlo and hi2 <= hhi)
        #     if not satisfied:
        #         reasons.append(f"{lo} isWithin hard not met")
        elif op == "odrl:isWithin":
            # Hard date interval: incoming must be fully inside [hlo, hhi]
            hlo, hhi = _normalize_date_interval(hard_ro)
            if hlo is None:
                reasons.append(f"{lo} isWithin: invalid hard interval")
                continue

            # Reconstruct a single incoming interval [ilo, ihi]
            ilo = None
            ihi = None

            for (k2, v2) in inc_for_operand:
                if k2[1] == "odrl:isWithin":
                    lo2, hi2 = _normalize_date_interval(v2)
                    if lo2 is not None:
                        ilo = lo2 if ilo is None else min(ilo, lo2)
                        ihi = hi2 if ihi is None else max(ihi, hi2)

                elif k2[1] == "odrl:eq":
                    v = _to_num(v2)
                    if v is not None:
                        # eq is a degenerate interval [v, v]
                        ilo = v if ilo is None else min(ilo, v)
                        ihi = v if ihi is None else max(ihi, v)

                elif k2[1] == "odrl:gteq":
                    v = _to_num(v2)
                    if v is not None:
                        ilo = v if ilo is None else max(ilo, v)

                elif k2[1] == "odrl:lteq":
                    v = _to_num(v2)
                    if v is not None:
                        ihi = v if ihi is None else min(ihi, v)

            satisfied = (ilo is not None and ihi is not None and ilo <= ihi and ilo >= hlo and ihi <= hhi)
            if not satisfied:
                reasons.append(f"{lo} isWithin hard not met")

        elif op in ("odrl:isAnyOf","odrl:isNoneOf","odrl:isAllOf"):
            # Set-like checks
            H = _setify(hard_ro)
            satisfied = False
            for (k2, v2) in inc_for_operand:
                if not isinstance(v2, list) and k2[1] != "odrl:eq":
                    continue
                V = _setify(v2)
                if op == "odrl:isAllOf":
                    satisfied |= H.issubset(V)
                elif op == "odrl:isAnyOf":
                    satisfied |= bool(H & V)
                else:  # isNoneOf
                    satisfied |= not bool(H & V)
            if not satisfied:
                reasons.append(f"{lo} {op} hard not met")
        else:
            # Any other operator: require exact equality if present; otherwise fail softly
            satisfied = False
            for (k2, v2) in inc_for_operand:
                if k2[1] == op:
                    satisfied |= (v2 == hard_ro)
            if not satisfied:
                reasons.append(f"{lo} {op} hard not met (unsupported op - exact check)")
    return (len(reasons) == 0), reasons
