# bambon/utils/hard_constraints.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from .helpers import parse_offer_generic, is_iso_duration, duration_to_seconds, is_iso_date, iso_to_days

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
    if not isinstance(ro, list) or len(ro) != 2:
        return None, None
    a, b = ro[0], ro[1]
    if not (isinstance(a, str) and isinstance(b, str)):
        return None, None
    if not (is_iso_date(a) and is_iso_date(b)):
        return None, None
    lo = float(iso_to_days(a)); hi = float(iso_to_days(b))
    if lo > hi: lo, hi = hi, lo
    return lo, hi

def _setify(x: Any) -> set:
    if isinstance(x, list):
        return set(x)
    return {x}

def check_against(my_hard_offer: Dict[str, Any], incoming_offer: Dict[str, Any]) -> Tuple[bool, List[str]]:
    ok, reasons, _ = check_against_ex(my_hard_offer, incoming_offer)
    return ok, reasons

def check_against_ex(my_hard_offer: Dict[str, Any], incoming_offer: Dict[str, Any], allow_unseen: bool) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
    """
    Verbose checker.
    Returns: (ok, reasons, fixes)
      - ok: bool
      - reasons: human-readable strings (like before)
      - fixes: list of structured hints for list-type constraints you can act on:
          {
            "leftOperand": str,
            "hard_op": "odrl:isAnyOf" | "odrl:isAllOf" | "odrl:isNoneOf",
            "hard_set": list,              # my allowed/required/forbidden set
            "incoming_sets": [list,...],   # sets seen on incoming side (each candidate)
            "incoming_ops": [str,...],     # ops corresponding to incoming_sets
            "suggest": {                   # minimal, local suggestion using MY hard only
              "action": "set_eq" | "add_items" | "remove_items",
              "value": str | list          # see below
            }
          }
    """
    if not my_hard_offer:
        return True, [], []

    hard_issues = parse_offer_generic(my_hard_offer)
    inc_issues  = parse_offer_generic(incoming_offer)

    reasons: List[str] = []
    fixes:   List[Dict[str, Any]] = []

    for (lo, op, idx), hard_ro in hard_issues.items():
        inc_for_operand = [(k, v) for k, v in inc_issues.items() if k[0] == lo]
        if not inc_for_operand:
            if allow_unseen:
                suggest = {"action": "add_constraint", "value": hard_ro}

                fixes.append({
                    "leftOperand": lo,
                    "hard_op": op,
                    "incoming_sets": None,
                    "incoming_ops": None,
                    "suggest": suggest
                })
            else:
                reasons.append(f"missing operand {lo} in incoming")
            continue

        # ---- equality / inequality
        if op in ("odrl:eq", "odrl:neq"):
            satisfied = False
            for (k2, v2) in inc_for_operand:
                if k2[1] == "odrl:eq":
                    if op == "odrl:eq":
                        if v2 == hard_ro:
                            satisfied = True; break
                    else:
                        if v2 != hard_ro:
                            satisfied = True; break
            if not satisfied:
                reasons.append(f"{lo} {op} hard not met")

        # ---- numeric family
        elif op in ("odrl:gteq", "odrl:lteq", "odrl:gt", "odrl:lt"):
            hnum = _to_num(hard_ro)
            if hnum is None:
                reasons.append(f"{lo} {op}: non-numeric hard value")
                continue
            satisfied = False
            for (k2, v2) in inc_for_operand:
                if k2[1] in ("odrl:eq","odrl:gteq","odrl:lteq","odrl:gt","odrl:lt"):
                    vnum = _to_num(v2)
                    if vnum is None: continue
                    if op in ("odrl:gteq","odrl:gt"):
                        if vnum >= hnum: satisfied = True; break
                    else:
                        if vnum <= hnum: satisfied = True; break
            if not satisfied:
                reasons.append(f"{lo} {op} hard not met")

        # ---- date interval
        elif op == "odrl:isWithin":
            hlo, hhi = _normalize_date_interval(hard_ro)
            if hlo is None:
                reasons.append(f"{lo} isWithin: invalid hard interval")
                continue
            ilo = None; ihi = None
            for (k2, v2) in inc_for_operand:
                if k2[1] == "odrl:isWithin":
                    lo2, hi2 = _normalize_date_interval(v2)
                    if lo2 is not None:
                        ilo = lo2 if ilo is None else min(ilo, lo2)
                        ihi = hi2 if ihi is None else max(ihi, hi2)
                elif k2[1] == "odrl:eq":
                    v = _to_num(v2)
                    if v is not None:
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
            ok_within = (ilo is not None and ihi is not None and ilo <= ihi and ilo >= hlo and ihi <= hhi)
            if not ok_within:
                reasons.append(f"{lo} isWithin hard not met")

        # ---- set family (list-type) → produce a fix hint
        elif op in ("odrl:isAnyOf","odrl:isAllOf","odrl:isNoneOf"):
            H = _setify(hard_ro)
            satisfied = False
            incoming_sets = []
            incoming_ops  = []
            for (k2, v2) in inc_for_operand:
                if isinstance(v2, list) or k2[1] == "odrl:eq":
                    V = _setify(v2)
                    incoming_sets.append(list(V))
                    incoming_ops.append(k2[1])
                    if op == "odrl:isAllOf":
                        if H.issubset(V): satisfied = True; break
                    elif op == "odrl:isAnyOf":
                        if H & V: satisfied = True; break
                    else:  # isNoneOf
                        if not (H & V): satisfied = True; break
            if not satisfied:
                reasons.append(f"{lo} {op} hard not met")

                # Build a local suggestion based on MY hard only:
                suggest = None
                if op == "odrl:isAnyOf":
                    # choose any element from my allowed set (if opponent used a single eq not in H)
                    # prefer a deterministic choice
                    pick = sorted(H)
                    if pick:
                        suggest = {"action": "set_eq", "value": pick[0]}
                elif op == "odrl:isAllOf":
                    # request the missing items
                    cur = set(incoming_sets[0]) if incoming_sets else set()
                    missing = sorted(H - cur)
                    if missing:
                        suggest = {"action": "add_items", "value": missing}
                elif op == "odrl:isNoneOf":
                    # request removal of forbidden items
                    cur = set(incoming_sets[0]) if incoming_sets else set()
                    banned = sorted(H & cur)
                    if banned:
                        suggest = {"action": "remove_items", "value": banned}

                fixes.append({
                    "leftOperand": lo,
                    "hard_op": op,
                    "hard_set": sorted(H),
                    "incoming_sets": incoming_sets,
                    "incoming_ops": incoming_ops,
                    "suggest": suggest
                })

        # ---- fallback
        else:
            satisfied = False
            for (k2, v2) in inc_for_operand:
                if k2[1] == op and v2 == hard_ro:
                    satisfied = True; break
            if not satisfied:
                reasons.append(f"{lo} {op} hard not met (unsupported op - exact check)")

    ok = (len(reasons) == 0)
    return ok, reasons, fixes
