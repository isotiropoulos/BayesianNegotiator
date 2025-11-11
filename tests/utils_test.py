from typing import Iterable, Dict, Any, List, Union

from pathlib import Path
from datetime import datetime
import json


def save_log_text(data, path="negotiation.log", *, actor_filter=None, include_pair_view=True):
    text = render_negotiation_log(data, actor_filter=actor_filter, include_pair_view=include_pair_view)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with p.open("a", encoding="utf-8") as f:
        f.write(f"\n===== NEGOTIATION SNAPSHOT @ {stamp} =====\n")
        f.write(text)
        f.write("\n")


def fetch_data(case:str):
    with open(f'./tests/data/{case}/provider.json', 'r', encoding='utf-8') as f:
        provider = json.load(f)

    with open(f'./tests/data/{case}/consumer.json', 'r', encoding='utf-8') as f:
        consumer = json.load(f)

    with open(f'./tests/data/{case}/hard_provider.json', 'r', encoding='utf-8') as f:
        hard_provider = json.load(f)

    with open(f'./tests/data/{case}/hard_consumer.json', 'r', encoding='utf-8') as f:
        hard_consumer = json.load(f)

    return provider, hard_provider, consumer, hard_consumer


def sort_odrl_constraints(data):
    """
    Accepts either:
      - a dict with key 'constraint' -> list[dict]
      - a list[dict] of constraints directly
    Returns a new object with the constraints sorted by leftOperand.
    """
    def normalize_item(item):
        # Fix common typo: lefrOperand -> leftOperand
        if "lefrOperand" in item and "leftOperand" not in item:
            item = dict(item)
            item["leftOperand"] = item.pop("lefrOperand")
        return item

    def sort_list(lst):
        lst = [normalize_item(x) for x in lst]
        return sorted(lst, key=lambda x: str(x.get("leftOperand", "")))

    # Case 1: {'constraint': [...]}
    if isinstance(data, dict) and "constraint" in data:
        new_data = dict(data)
        new_data["constraint"] = sort_list(data["constraint"])
        return new_data

    # Case 2: [...]  (list of constraints directly)
    if isinstance(data, list):
        # If it looks like [{'constraint': [...]}] (your example)
        if data and isinstance(data[0], dict) and "constraint" in data[0]:
            first = dict(data[0])
            first["constraint"] = sort_list(first["constraint"])
            return [first, *data[1:]]
        else:
            return sort_list(data)

    raise TypeError("Expected a dict with 'constraint' or a list of constraint dicts.")


def add_block(store, actor, round_no, received, replied):
    store.append({"actor": actor, "round": round_no, "side": "Received", "constraints": received})
    store.append({"actor": actor, "round": round_no, "side": "Replied", "constraints": replied})

# -------- Helpers --------
def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        # compact float formatting
        return f"{v:.3f}".rstrip('0').rstrip('.')
    if isinstance(v, list):
        return "[" + ", ".join(map(_fmt_value, v)) + "]"
    return str(v)


def _sort_constraints(constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # sort by leftOperand then operator for consistent, readable output
    return sorted(
        constraints,
        key=lambda d: (str(d.get("leftOperand")), str(d.get("operator")), str(d.get("rightOperand")))
    )

def _pair_by_operand(received: List[Dict[str, Any]], replied: List[Dict[str, Any]]):
    """
    Create quick pairs by (leftOperand, operator) to show 'A -> B' lines.
    If a key exists on one side only, show the single value.
    """
    key = lambda c: (c.get("leftOperand"), c.get("operator"))
    rcv_map = {key(c): c for c in received}
    rep_map = {key(c): c for c in replied}
    all_keys = sorted(set(rcv_map) | set(rep_map))
    pairs = []
    for k in all_keys:
        rcv = rcv_map.get(k)
        rep = rep_map.get(k)
        pairs.append((k, rcv, rep))
    return pairs

# -------- Public API --------
def render_negotiation_log(
    blocks: Iterable[Dict[str, Any]],
    *,
    actor_filter: str = None,
    include_pair_view: bool = True
) -> str:
    """
    Pretty-print the log grouped by actor -> round -> (Received, Replied).
    Optional compact pair view shows 'Received -> Replied' per constraint type.
    """

    # group by actor, round, side
    groups: Dict[str, Dict[int, Dict[str, List[Dict[str, Any]]]]] = {}
    for row in blocks:
        actor = row["actor"]
        rnd = int(row["round"])
        side = row["side"]
        groups.setdefault(actor, {}).setdefault(rnd, {}).setdefault(side, [])
        groups[actor][rnd][side].extend(row["constraints"])

    lines: List[str] = []

    for actor in sorted(groups.keys()):
        if actor_filter and actor != actor_filter:
            continue
        lines.append(f"{actor.upper()} LOG")
        lines.append("-" * (len(actor) + 4))
        for rnd in sorted(groups[actor].keys()):
            lines.append(f"Round {rnd}:")
            rcv = _sort_constraints(groups[actor][rnd].get("Received", []))
            rep = _sort_constraints(groups[actor][rnd].get("Replied", []))

            # Received
            lines.append("  • Received:")
            if not rcv:
                lines.append("      (none)")
            else:
                for c in rcv:
                    lines.append(f"      - {c.get('leftOperand')} {c.get('operator')} {_fmt_value(c.get('rightOperand'))}")

            # Replied
            lines.append("  • Replied:")
            if not rep:
                lines.append("      (none)")
            else:
                for c in rep:
                    lines.append(f"      - {c.get('leftOperand')} {c.get('operator')} {_fmt_value(c.get('rightOperand'))}")

            # Optional compact pair view
            if include_pair_view:
                pairs = _pair_by_operand(rcv, rep)
                if pairs:
                    lines.append("  • Change summary:")
                    for (lop, op), rcv_c, rep_c in pairs:
                        l = f"      - {lop} {op}: "
                        if rcv_c and rep_c:
                            l += f"{_fmt_value(rcv_c['rightOperand'])} → {_fmt_value(rep_c['rightOperand'])}"
                        elif rcv_c and not rep_c:
                            l += f"{_fmt_value(rcv_c['rightOperand'])} → (no reply)"
                        elif rep_c and not rcv_c:
                            l += f"(no receive) → {_fmt_value(rep_c['rightOperand'])}"
                        lines.append(l)

            lines.append("")  # blank line between rounds
        lines.append("")  # blank line between actors

    return "\n".join(lines).rstrip()





from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
import math

def _parse_iso(s: str) -> datetime:
    # tolerate trailing 'Z'
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def _duration_seconds(x: str) -> Optional[float]:
    # Try isodate if available; otherwise a tiny fallback (D/H/M/S only).
    try:
        import isodate
        return isodate.parse_duration(x).total_seconds()
    except Exception:
        import re
        m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", x)
        if not m:
            return None
        d, h, mn, sc = (int(g) if g else 0 for g in m.groups())
        return d * 86400 + h * 3600 + mn * 60 + sc

def _numeric_similarity(a: float, b: float) -> float:
    base = max(1.0, abs(a), abs(b))
    return max(0.0, 1.0 - min(1.0, abs(a - b) / base))

def _list_jaccard(A: List[Any], B: List[Any]) -> float:
    SA, SB = set(A), set(B)
    if not (SA or SB):
        return 1.0
    return len(SA & SB) / len(SA | SB)

def _range_overlap_ratio(a0: str, a1: str, b0: str, b1: str) -> float:
    A0, A1 = _parse_iso(a0), _parse_iso(a1)
    B0, B1 = _parse_iso(b0), _parse_iso(b1)
    lo = max(min(A0, A1), min(B0, B1))
    hi = min(max(A0, A1), max(B0, B1))
    inter = max(0.0, (hi - lo).total_seconds())
    union = (max(A0, A1) - min(A0, A1)).total_seconds() + \
            (max(B0, B1) - min(B0, B1)).total_seconds() - inter
    return inter / union if union > 0 else 1.0

def odrl_similarity_vector(initial_offer: Dict[str, Any],
                           agreement: Dict[str, Any],
                           weights: Optional[Dict[str, float]] = None
                          ) -> Tuple[List[float], Dict[str, float]]:
    """
    Returns (vector_of_similarities, details_by_leftOperand).
    Matches constraints by 'leftOperand' in the first permission block.
    """

    def _constraints(offer):
        return (offer or {}).get("permission", [{}])[0].get("constraint", [])

    init_cs = _constraints(initial_offer)
    agre_cs = _constraints(agreement)

    # Map by leftOperand (assumes one per leftOperand)
    _by_left = lambda cs: {c.get("leftOperand"): c for c in cs if "leftOperand" in c}
    I = _by_left(init_cs)
    A = _by_left(agre_cs)

    sims: List[float] = []
    details: Dict[str, float] = {}

    for left, c0 in I.items():
        c1 = A.get(left)
        if not c1:
            sims.append(0.0)
            details[left] = 0.0
            continue

        v0 = c0.get("rightOperand")
        v1 = c1.get("rightOperand")

        # --- IMPORTANT: handle date ranges before generic list logic ---
        if (isinstance(v0, list) and isinstance(v1, list)
            and len(v0) == 2 and len(v1) == 2
            and all(isinstance(x, str) for x in v0 + v1)
            and ("T" in v0[0] or "T" in v0[1] or "T" in v1[0] or "T" in v1[1])):
            s = _range_overlap_ratio(v0[0], v0[1], v1[0], v1[1])

        # Lists / sets (e.g., isAnyOf)
        elif isinstance(v0, list) and isinstance(v1, list):
            s = _list_jaccard(v0, v1)

        # Numbers
        elif isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
            s = _numeric_similarity(float(v0), float(v1))

        # Durations (xsd:duration)
        elif isinstance(v0, str) and isinstance(v1, str) and v0.startswith("P") and v1.startswith("P"):
            t0 = _duration_seconds(v0)
            t1 = _duration_seconds(v1)
            s = _numeric_similarity(t0, t1) if (t0 is not None and t1 is not None) else 0.0

        # Enums / plain strings
        else:
            s = 1.0 if v0 == v1 else 0.0

        s = float(max(0.0, min(1.0, s)))
        sims.append(s)
        details[left] = s

    # Optional weighted mean
    if weights:
        wsum = sum(weights.get(k, 1.0) for k in details.keys()) or 1.0
        weighted_mean = sum(details[k] * weights.get(k, 1.0) for k in details.keys()) / wsum
    else:
        weighted_mean = sum(sims) / len(sims) if sims else 0.0

    # Cosine to ones
    l2 = math.sqrt(sum(x * x for x in sims))
    cosine_to_ones = (sum(sims) / (l2 * math.sqrt(len(sims)))) if (l2 > 0 and sims) else 0.0

    return sims, {
        "mean": sum(sims) / len(sims) if sims else 0.0,
        "weighted_mean": weighted_mean,
        "cosine_to_ones": cosine_to_ones
    }
