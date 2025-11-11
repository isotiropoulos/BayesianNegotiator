# src/bambon/utils/semantics.py
# Lenient merge that never fails (as provided; unchanged behavior)
from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from ..models.models import IssueKey
from ..utils.helpers import (
    is_iso_date, iso_to_days, days_to_iso,
    is_iso_duration, duration_to_seconds, seconds_to_duration,
)

class _Adapter:
    def to_num(self, v: Any) -> float: raise NotImplementedError
    def from_num(self, x: float) -> Any: raise NotImplementedError

class _FloatAdapter(_Adapter):
    def to_num(self, v: Any) -> float: return float(v)
    def from_num(self, x: float) -> Any: return float(x)

class _DateAdapter(_Adapter):
    def to_num(self, v: Any) -> float: return float(iso_to_days(str(v)))
    def from_num(self, x: float) -> Any: return days_to_iso(int(round(x)))

class _DurationAdapter(_Adapter):
    def to_num(self, v: Any) -> float: return float(duration_to_seconds(str(v)))
    def from_num(self, x: float) -> Any: return seconds_to_duration(max(0.0, float(x)))

def _pick_adapter(sample: Any) -> Optional[_Adapter]:
    if isinstance(sample, (int, float)): return _FloatAdapter()
    if isinstance(sample, str) and is_iso_date(sample): return _DateAdapter()
    if isinstance(sample, str) and is_iso_duration(sample): return _DurationAdapter()
    return None

def merge(my_issues: Dict[IssueKey, Any], opp_issues: Dict[IssueKey, Any]) -> Dict[IssueKey, Any]:
    grouped: Dict[str, List[Tuple[IssueKey, Any]]] = {}
    for d in (my_issues, opp_issues):
        for k, v in d.items():
            grouped.setdefault(k[0], []).append((k, v))
    out: Dict[IssueKey, Any] = {}

    for lo, pairs in grouped.items():
        sample = pairs[0][1]
        adapter = _pick_adapter(sample)

        if adapter is not None:
            lowers: List[float] = []
            uppers: List[float] = []
            eq_vals: List[float] = []
            neq_vals: List[float] = []
            for (k, v) in pairs:
                op = k[1]
                x = adapter.to_num(v)
                if op in ("odrl:gteq", "odrl:gt"): lowers.append(x)
                elif op in ("odrl:lteq", "odrl:lt"): uppers.append(x)
                elif op == "odrl:eq": eq_vals.append(x)
                elif op == "odrl:neq": neq_vals.append(x)
            if isinstance(adapter, _DateAdapter) or lowers or uppers:
                if lowers: L = min(lowers)
                elif uppers: L = min(uppers)
                else: L = eq_vals[0] if eq_vals else 0.0
                if uppers: H = max(uppers)
                elif lowers: H = max(lowers)
                else: H = eq_vals[0] if eq_vals else L
                if L > H: L, H = H, L
                out[(lo, "odrl:gteq", 0)] = adapter.from_num(L)
                out[(lo, "odrl:lteq", 0)] = adapter.from_num(H)
            else:
                val = eq_vals[0] if eq_vals else 0.0
                out[(lo, "odrl:eq", 0)] = adapter.from_num(val)
            for i, v in enumerate(neq_vals):
                out[(lo, "odrl:neq", i)] = adapter.from_num(v)
            continue

        if isinstance(sample, list):
            req_all: Set[str] = set()
            any_of: Set[str] = set()
            none_of: Set[str] = set()
            eq_set: Optional[List[str]] = None
            for (k, v) in pairs:
                op = k[1]
                if op == "odrl:isAllOf": req_all |= set(v if isinstance(v, list) else [v])
                elif op == "odrl:isAnyOf": any_of |= set(v if isinstance(v, list) else [v])
                elif op == "odrl:isNoneOf": none_of |= set(v if isinstance(v, list) else [v])
                elif op == "odrl:eq":
                    if eq_set is None: eq_set = list(v if isinstance(v, list) else [v])
            idx = 0
            if eq_set is not None: out[(lo, "odrl:eq", idx)] = sorted(eq_set); idx += 1
            if req_all: out[(lo, "odrl:isAllOf", idx)] = sorted(req_all); idx += 1
            if any_of: out[(lo, "odrl:isAnyOf", idx)] = sorted(any_of); idx += 1
            if none_of: out[(lo, "odrl:isNoneOf", idx)] = sorted(none_of); idx += 1
            continue

        eq_val: Optional[str] = None
        any_of: Set[str] = set()
        none_of: Set[str] = set()
        neq_vals: Set[str] = set()
        isa_vals: Set[str] = set()
        has_part_vals: Set[str] = set()
        part_of_vals: Set[str] = set()
        for (k, v) in pairs:
            op = k[1]
            val = v if isinstance(v, str) else str(v)
            if op == "odrl:eq":
                if eq_val is None: eq_val = val
            elif op == "odrl:isAnyOf": any_of |= set(v if isinstance(v, list) else [val])
            elif op == "odrl:isNoneOf": none_of |= set(v if isinstance(v, list) else [val])
            elif op == "odrl:neq": neq_vals.add(val)
            elif op == "odrl:isA": isa_vals.add(val)
            elif op == "odrl:hasPart": has_part_vals.add(val)
            elif op == "odrl:isPartOf": part_of_vals.add(val)
        idx = 0
        if eq_val is not None: out[(lo, "odrl:eq", idx)] = eq_val; idx += 1
        if any_of: out[(lo, "odrl:isAnyOf", idx)] = sorted(any_of); idx += 1
        if none_of: out[(lo, "odrl:isNoneOf", idx)] = sorted(none_of); idx += 1
        if neq_vals:
            for j, v in enumerate(sorted(neq_vals)):
                out[(lo, "odrl:neq", idx + j)] = v
            idx += len(neq_vals)
        if isa_vals:
            for j, v in enumerate(sorted(isa_vals)):
                out[(lo, "odrl:isA", idx + j)] = v
            idx += len(isa_vals)
        if has_part_vals:
            for j, v in enumerate(sorted(has_part_vals)):
                out[(lo, "odrl:hasPart", idx + j)] = v
            idx += len(has_part_vals)
        if part_of_vals:
            for j, v in enumerate(sorted(part_of_vals)):
                out[(lo, "odrl:isPartOf", idx + j)] = v
            idx += len(part_of_vals)

    grouped2: Dict[Tuple[str, str], List[Tuple[IssueKey, Any]]] = {}
    for k, v in out.items():
        lo, op, idx = k
        grouped2.setdefault((lo, op), []).append((k, v))
    out2: Dict[IssueKey, Any] = {}
    for (lo, op), items in grouped2.items():
        items_sorted = sorted(items, key=lambda kv: (kv[0][2], str(kv[1])))
        for i, (_, val) in enumerate(items_sorted):
            out2[(lo, op, i)] = val
    return out2
