# negotiator.py
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median, pstdev
from typing import Any, Dict, List, Optional, Tuple

from .distributions.belief import BeliefRegistry
from .models.models import IssueKey, NegotiationInfeasible
from .utils.agreement import offers_close_generic, offers_equal_generic
from .utils.helpers import (
    build_offer_from_issues,
    is_numeric_like,
    parse_offer_generic,
)

_NUMERIC_OPS = {"odrl:eq", "odrl:gteq", "odrl:lteq", "odrl:gt", "odrl:lt"}


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class Bambon:
    """
    Adaptive negotiator using learned beliefs and behavioral heuristics.
    - Bayesian updating per-issue (numeric/enum/set)
    - Cooling schedule for inertia/tau
    - Deterministic, unified-context handshake for numerics (no randomness)
    - Interval policy (dates always interval; other numerics interval only if both sides started with both bounds)
    - Generic "hopelessness" detector (no operand-specific logic)
    """

    def __init__(
        self,
        name: str,
        lr: float,
        inertia: float = 0.4,
        tau_set: float = 0.5,
        min_overlap: int = 1,
        max_set_k: int = 3,
        # kept for backwards compatibility but NOT used anymore (handshake is deterministic now)
        handshake_prob: float = 0.6,
        # reused as the unified-context threshold for handshake
        jaccard_snap_threshold: float = 0.70,
        max_rounds: int = 60,
        cooling_round: int = 12,
        cooled_inertia: float = 0.7,
        cooled_tau_set: float = 0.65,
        init_offer: Optional[Dict[str, Any]] = None,
        rounds_seen: int = 0,
        lr_min: float = 0.2,
        lr_max: float = 1.0,
        vol_window: int = 5,
        max_param_delta: float = 0.15,
        # --- generic hopelessness guardrails (no operand-specific rules)
        hopeless_threshold: float = 0.85,   # severity in [0,1]
        hopeless_patience: int = 3,         # consecutive rounds above threshold
        # --- unified context weights (all in [0,1], sum doesn't have to be 1)
        ctx_w_num: float = 0.5,
        ctx_w_set: float = 0.3,
        ctx_w_enum: float = 0.2,
    ) -> None:
        self.name = name
        self.beliefs = BeliefRegistry()

        # Base params (adaptive)
        self.lr = lr
        self.inertia = inertia
        self.tau_set = tau_set
        self.min_overlap = min_overlap
        self.max_set_k = max_set_k
        # NOTE: handshake_prob is ignored now (kept for compatibility)
        self.handshake_prob = handshake_prob
        self.jaccard_snap_threshold = jaccard_snap_threshold
        self.max_rounds = max_rounds
        self.cooling_round = cooling_round
        self.cooled_inertia = cooled_inertia
        self.cooled_tau_set = cooled_tau_set
        self.rounds_seen = rounds_seen

        # Adaptation config
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.vol_window = max(3, vol_window)
        self.max_param_delta = max_param_delta

        # Internal state
        self.last_issues = parse_offer_generic(init_offer or {})
        self.beliefs.observe_offer(self.last_issues, lr=0.0)
        self._num_move_hist: List[float] = []
        self._opp_set_sizes: List[int] = []

        # Interval policy bookkeeping (for non-date numerics)
        self._interval_ok: Dict[str, bool] = {}
        self._my_init_bounds = self._extract_bounds(self.last_issues)
        self._opp_init_bounds: Optional[Dict[str, set]] = None

        # Generic hopelessness
        self.hopeless_threshold = float(hopeless_threshold)
        self.hopeless_patience = int(hopeless_patience)
        self._hopeless_streak = 0

        # Unified context weights
        self._ctx_w_num = float(ctx_w_num)
        self._ctx_w_set = float(ctx_w_set)
        self._ctx_w_enum = float(ctx_w_enum)

    # ------------------------------------------------------------------ #
    # Interval policy helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_bounds(issues: Dict[IssueKey, Any]) -> Dict[str, set]:
        """
        Return per leftOperand the set of bound ops present (only for non-date numerics).
        We treat "numeric-like" as: numbers, ISO dates, or ISO durations; but dates are excluded here.
        """
        by_lo: Dict[str, set] = {}
        for (lo, op, _), ro in issues.items():
            if lo == "odrl:dateTime":
                continue
            if not is_numeric_like(ro):
                continue
            if op in ("odrl:gteq", "odrl:lteq"):
                by_lo.setdefault(lo, set()).add(op)
        return by_lo

    def _maybe_update_interval_policy_from_opp_init(self, opp_issues: Dict[IssueKey, Any]) -> None:
        """On first observe(), capture opponent's initial bounds and set interval policy."""
        if self._opp_init_bounds is not None:
            return
        self._opp_init_bounds = self._extract_bounds(opp_issues)
        for lo in set(self._my_init_bounds.keys()) | set(self._opp_init_bounds.keys()):
            my_b = self._my_init_bounds.get(lo, set())
            op_b = self._opp_init_bounds.get(lo, set())
            self._interval_ok[lo] = (
                ("odrl:gteq" in my_b and "odrl:lteq" in my_b)
                and ("odrl:gteq" in op_b and "odrl:lteq" in op_b)
            )

    # ------------------------------------------------------------------ #
    # Unified context score (deterministic handshake gate)
    # ------------------------------------------------------------------ #
    def _context_score(
        self,
        mine: Dict[IssueKey, Any],
        opp: Dict[IssueKey, Any],
    ) -> float:
        """
        Unified similarity in [0,1] across overlapping issues:
          - numerics: 1 - normalized diff (by model span if available; else relative to magnitude)
          - sets: Jaccard
          - enums/atoms: 1 if equal else 0
        Weights: self._ctx_w_num, self._ctx_w_set, self._ctx_w_enum.
        Returns 0 if no overlap.
        """
        sims: List[float] = []
        weights: List[float] = []

        for k, model in self.beliefs.models.items():
            if k not in mine or k not in opp:
                continue
            a, b = mine[k], opp[k]

            if model.kind == "num":
                ai = float(model.to_internal(a))
                bi = float(model.to_internal(b))
                L = getattr(model.model, "L", None)
                U = getattr(model.model, "U", None)

                if isinstance(L, (int, float)) and isinstance(U, (int, float)) and U > L:
                    span = max(1e-6, U - L)
                    nd = abs(ai - bi) / span
                else:
                    base = max(1.0, abs(ai), abs(bi))
                    nd = abs(ai - bi) / base

                sim = max(0.0, 1.0 - min(1.0, nd))
                sims.append(sim); weights.append(self._ctx_w_num)

            elif model.kind == "set":
                A, B = set(a), set(b)
                if not (A or B):
                    sim = 1.0
                else:
                    sim = len(A & B) / len(A | B)
                sims.append(sim); weights.append(self._ctx_w_set)

            else:  # enum/atom
                sim = 1.0 if a == b else 0.0
                sims.append(sim); weights.append(self._ctx_w_enum)

        if not sims:
            return 0.0

        wsum = sum(weights) or 1.0
        return sum(s * w for s, w in zip(sims, weights)) / wsum

    # ------------------------------------------------------------------ #
    # Hopelessness (generic; no operand-specific rules)
    # ------------------------------------------------------------------ #
    def _issue_severity(self, k: IssueKey, a: Any, b: Any) -> float:
        """
        Generic severity in [0,1] for ANY ODRL constraint:
          - Numeric-like (incl. dates/durations): |ai-bi| normalized by adaptive span (U-L),
            fallback to relative difference; use the max of the two measures.
          - Sets: 1 - Jaccard(A,B)
          - Enums/atoms: 0 if equal else 1
          - Missing on one side: 0.8
        """
        model = self.beliefs.models.get(k)
        if model is None:
            return 0.8 if (a is None) ^ (b is None) else (0.0 if a == b else 0.5)

        if a is None or b is None:
            return 0.8

        if model.kind == "num":
            ai = float(model.to_internal(a))
            bi = float(model.to_internal(b))

            L = getattr(model.model, "L", None)
            U = getattr(model.model, "U", None)
            if isinstance(L, (int, float)) and isinstance(U, (int, float)) and U > L:
                span = max(1e-6, U - L)
            else:
                span = max(1.0, abs(ai), abs(bi))

            diff = abs(ai - bi)
            sev_norm = min(1.0, diff / span)

            base = max(1.0, abs(ai), abs(bi))
            sev_rel = min(1.0, diff / base)

            return float(max(sev_norm, sev_rel))

        if model.kind == "set":
            A, B = set(a), set(b)
            if not (A or B):
                return 0.0
            j = len(A & B) / len(A | B)
            return float(1.0 - j)

        if model.kind == "enum":
            return 0.0 if a == b else 1.0

        return 0.8 if a != b else 0.0

    def _hopeless_score(self, my_issues: Dict[IssueKey, Any], opp_issues: Dict[IssueKey, Any]) -> float:
        """Max severity across overlapping issues (strong early warning)."""
        keys = [k for k in self.beliefs.models.keys() if k in my_issues and k in opp_issues]
        if not keys:
            return 0.0
        return max(self._issue_severity(k, my_issues[k], opp_issues[k]) for k in keys)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rekey_opp_to_self(opp_issues: Dict[IssueKey, Any]) -> Dict[IssueKey, Any]:
        """Pass-through; keys are already (leftOperand, operator, idx)."""
        return dict(opp_issues)

    def observe(self, opp_offer: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Observe opponent offer; return (done, offer) where done=True if agreement/near-agreement."""
        self.rounds_seen += 1
        if self.rounds_seen > self.max_rounds:
            raise NegotiationInfeasible(f"{self.name}: Maximum allowed rounds exceeded.")

        opp_issues_raw = parse_offer_generic(opp_offer)
        opp_issues = self._rekey_opp_to_self(opp_issues_raw)

        self._maybe_update_interval_policy_from_opp_init(opp_issues)

        # ---- generic hopelessness check
        hop = self._hopeless_score(self.last_issues, opp_issues)
        if hop >= self.hopeless_threshold:
            self._hopeless_streak += 1
        else:
            self._hopeless_streak = 0
        if self._hopeless_streak >= self.hopeless_patience:
            raise NegotiationInfeasible(
                f"{self.name}: negotiation appears infeasible "
                f"(severity={hop:.2f} for {self._hopeless_streak} consecutive rounds)"
            )

        signals = self._compute_signals(self.last_issues, opp_issues)
        self._adapt(signals)

        self.beliefs.observe_offer(opp_issues, lr=self.lr)

        if offers_equal_generic(opp_issues, self.last_issues):
            return True, opp_offer

        if offers_close_generic(opp_issues, self.last_issues, self.rounds_seen, self.max_rounds):
            return True, opp_offer

        return False, self.propose(opp_issues)

    def propose(self, opp_last_issues: Optional[Dict[IssueKey, Any]]) -> Dict[str, Any]:
        """Produce a counter-offer based on beliefs, context, and interval policy."""
        inertia, tau = self._current_params()

        # Base proposal
        prop_issues = self.beliefs.propose(
            my_prev=self.last_issues,
            opp_prev=opp_last_issues,
            inertia=inertia,
            tau_set=tau,
            min_overlap=self.min_overlap,
            max_set_k=self.max_set_k,
            jaccard_snap_threshold=self.jaccard_snap_threshold,
        )

        # Subset-snap for sets (unchanged)
        if opp_last_issues:
            for k, model in self.beliefs.models.items():
                if model.kind != "set":
                    continue
                if k not in opp_last_issues or k not in prop_issues:
                    continue
                A = set(prop_issues[k])
                B = set(opp_last_issues[k])
                if not A or not B:
                    continue
                if A.issubset(B) or B.issubset(A):
                    enums_ok = True
                    nums_close = True
                    for kk, mm in self.beliefs.models.items():
                        if kk == k:
                            continue
                        if kk in prop_issues and kk in opp_last_issues:
                            a = prop_issues[kk]
                            b = opp_last_issues[kk]
                            if mm.kind == "enum" and a != b:
                                enums_ok = False
                            elif mm.kind == "num":
                                ai = mm.to_internal(a)
                                bi = mm.to_internal(b)
                                if abs(ai - bi) > 2.5:
                                    nums_close = False
                    if enums_ok and nums_close:
                        prop_issues[k] = sorted(list(A if len(A) <= len(B) else B))

        # Deterministic unified-context handshake for numerics
        if opp_last_issues:
            context_score = self._context_score(prop_issues, opp_last_issues)
            if context_score >= self.jaccard_snap_threshold:
                for k, model in self.beliefs.models.items():
                    if model.kind != "num":
                        continue
                    if k in prop_issues and k in opp_last_issues:
                        ai = model.to_internal(prop_issues[k])
                        bi = model.to_internal(opp_last_issues[k])
                        mid = (ai + bi) / 2.0
                        prop_issues[k] = model.to_external(mid)

        # Normalize numeric bands per leftOperand (respecting interval policy)
        prop_issues = self._normalize_numeric_groups(prop_issues)

        self.last_issues = prop_issues
        return build_offer_from_issues(prop_issues)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _current_params(self) -> Tuple[float, float]:
        """Cooling: inertia/tau move toward cooled values after cooling_round."""
        inertia_target = self.inertia if self.rounds_seen < self.cooling_round else self.cooled_inertia
        tau_target = self.tau_set if self.rounds_seen < self.cooling_round else self.cooled_tau_set
        return inertia_target, tau_target

    def _normalize_numeric_groups(self, issues: Dict[IssueKey, Any]) -> Dict[IssueKey, Any]:
        """
        Ensure consistent emission:
          - odrl:dateTime: ALWAYS closed interval (gteq + lteq), duplicating single-bound cases.
          - Other numerics: closed interval only if self._interval_ok[operand] is True and both bounds present;
            otherwise collapse to a single odrl:eq (median(eq) or midpoint or existing bound).
        """
        by_lo: Dict[str, Dict[str, List[Tuple[IssueKey, Any]]]] = {}
        numeric_key_models: Dict[IssueKey, Any] = {}

        for k, v in issues.items():
            model = self.beliefs.models.get(k)
            if not model or model.kind != "num":
                continue
            lo, op, _ = k
            by_lo.setdefault(lo, {}).setdefault(op, []).append((k, v))
            numeric_key_models[k] = model

        out: Dict[IssueKey, Any] = dict(issues)

        for lo, ops in by_lo.items():
            ops_present = set(ops.keys())
            if not (ops_present & _NUMERIC_OPS):
                continue

            any_key = next(iter(ops[next(iter(ops_present))]))[0]  # representative key
            model = numeric_key_models[any_key]

            # Collect values
            eq_vals_int: List[float] = []
            if "odrl:eq" in ops:
                for _, ro in ops["odrl:eq"]:
                    eq_vals_int.append(model.to_internal(ro))

            lowers: List[float] = []
            uppers: List[float] = []
            if "odrl:gteq" in ops:
                for _, ro in ops["odrl:gteq"]:
                    lowers.append(model.to_internal(ro))
            if "odrl:lteq" in ops:
                for _, ro in ops["odrl:lteq"]:
                    uppers.append(model.to_internal(ro))

            # Remove existing entries for this operand
            for op2 in list(ops_present):
                for k2, _ in ops.get(op2, []):
                    out.pop(k2, None)

            have_lo = bool(lowers)
            have_hi = bool(uppers)
            is_date = (lo == "odrl:dateTime")

            if is_date:
                # Always closed interval
                if eq_vals_int:
                    v = median(eq_vals_int)
                    out[(lo, "odrl:gteq", 0)] = model.to_external(v)
                    out[(lo, "odrl:lteq", 0)] = model.to_external(v)
                else:
                    L = max(lowers) if have_lo else (min(uppers) if have_hi else None)
                    H = min(uppers) if have_hi else (max(lowers) if have_lo else None)
                    if L is None and H is None:
                        continue
                    if L is None:
                        L = H
                    if H is None:
                        H = L
                    if L > H:
                        L, H = H, L
                    out[(lo, "odrl:gteq", 0)] = model.to_external(L)
                    out[(lo, "odrl:lteq", 0)] = model.to_external(H)
                continue

            # Non-date numerics
            interval_allowed = self._interval_ok.get(lo, False)

            if eq_vals_int:
                v = median(eq_vals_int)
                out[(lo, "odrl:eq", 0)] = model.to_external(v)
            else:
                if interval_allowed and have_lo and have_hi:
                    lo_int = max(lowers)
                    hi_int = min(uppers)
                    if lo_int <= hi_int:
                        out[(lo, "odrl:gteq", 0)] = model.to_external(lo_int)
                        out[(lo, "odrl:lteq", 0)] = model.to_external(hi_int)
                    else:
                        mid = (lo_int + hi_int) / 2.0
                        out[(lo, "odrl:eq", 0)] = model.to_external(mid)
                else:
                    # Collapse to eq
                    if have_lo and have_hi:
                        val = (max(lowers) + min(uppers)) / 2.0
                    elif have_lo:
                        val = max(lowers)
                    elif have_hi:
                        val = min(uppers)
                    else:
                        continue
                    out[(lo, "odrl:eq", 0)] = model.to_external(val)

        return self._reindex_per_pair(out)

    @staticmethod
    def _reindex_per_pair(issues: Dict[IssueKey, Any]) -> Dict[IssueKey, Any]:
        """Reassign indices per (leftOperand, operator), starting at 0 in key order."""
        grouped: Dict[Tuple[str, str], List[Tuple[IssueKey, Any]]] = {}
        for k, v in issues.items():
            lo, op, idx = k
            grouped.setdefault((lo, op), []).append((k, v))

        out: Dict[IssueKey, Any] = {}
        for (lo, op), items in grouped.items():
            items_sorted = sorted(items, key=lambda kv: (kv[0][2], str(kv[1])))
            for new_idx, (_, val) in enumerate(items_sorted):
                out[(lo, op, new_idx)] = val
        return out

    # ------------------ Signals & adaptation ------------------ #
    def _compute_signals(self, my_prev: Dict[IssueKey, Any], opp_prev: Dict[IssueKey, Any]) -> Dict[str, float]:
        enum_total = enum_match = 0
        jaccs: List[float] = []
        num_diffs: List[float] = []
        opp_set_sizes_round: List[int] = []

        for k, model in self.beliefs.models.items():
            if k not in opp_prev or k not in my_prev:
                continue

            if model.kind == "enum":
                enum_total += 1
                if opp_prev[k] == my_prev[k]:
                    enum_match += 1

            elif model.kind == "set":
                A = set(my_prev[k])
                B = set(opp_prev[k])
                j = (len(A & B) / len(A | B)) if (A or B) else 1.0
                jaccs.append(j)
                opp_set_sizes_round.append(len(B))

            elif model.kind == "num":
                ai = model.to_internal(my_prev[k])
                bi = model.to_internal(opp_prev[k])
                num_diffs.append(abs(ai - bi))

        enum_match_score = (enum_match / enum_total) if enum_total > 0 else 0.0
        jacc = (sum(jaccs) / len(jaccs)) if jaccs else 0.0
        num_move = median(num_diffs) if num_diffs else 0.0

        self._num_move_hist.append(num_move)
        if len(self._num_move_hist) > self.vol_window:
            self._num_move_hist.pop(0)

        if len(self._num_move_hist) >= 2:
            m = sum(self._num_move_hist) / len(self._num_move_hist)
            s = pstdev(self._num_move_hist)
            vol = s / (m + 1e-6)
        else:
            vol = 0.0
        vol = _clip(vol, 0.0, 1.0)

        if opp_set_sizes_round:
            self._opp_set_sizes.append(int(median(opp_set_sizes_round)))
            if len(self._opp_set_sizes) > self.vol_window:
                self._opp_set_sizes.pop(0)

        opp_set_size_median = int(median(self._opp_set_sizes)) if self._opp_set_sizes else 1

        return dict(enum_match=enum_match_score, jacc=jacc, num_move=num_move, vol=vol, opp_set_size_median=opp_set_size_median)

    def _adapt(self, s: Dict[str, float]) -> None:
        r = self.rounds_seen
        R = max(1, self.max_rounds)
        t = r / R

        enum_match = _clip(s.get("enum_match", 0.0), 0.0, 1.0)
        jacc = _clip(s.get("jacc", 0.0), 0.0, 1.0)
        vol = _clip(s.get("vol", 0.0), 0.0, 1.0)
        opp_k = max(1, int(s.get("opp_set_size_median", 1)))

        def bounded_update(cur: float, target: float, lo: float, hi: float) -> float:
            step = _clip(target - cur, -self.max_param_delta, self.max_param_delta)
            return _clip(cur + step, lo, hi)

        # learning rate
        lr_target = self.lr_max * (1 - 0.7 * t) / (1 + 2 * vol)
        lr_target = _clip(lr_target, self.lr_min, self.lr_max)
        self.lr = bounded_update(self.lr, lr_target, self.lr_min, self.lr_max)

        # inertia
        inertia_target = 0.2 + 0.5 * t + 0.3 * ((enum_match + jacc) * 0.5) - 0.3 * vol
        inertia_target = _clip(inertia_target, 0.2, 0.95)
        if (enum_match + jacc) * 0.5 >= 0.7 or t >= 0.5:
            inertia_target = min(0.95, max(inertia_target, self.cooled_inertia))
        self.inertia = bounded_update(self.inertia, inertia_target, 0.2, 0.95)

        # tau_set
        tau_target = 0.55 + 0.25 * t + 0.2 * (jacc - 0.5)
        if jacc < 0.30:
            tau_target -= 0.10
        tau_target = _clip(tau_target, 0.35, 0.9)
        if t >= 0.5:
            tau_target = max(tau_target, self.cooled_tau_set)
        self.tau_set = bounded_update(self.tau_set, tau_target, 0.35, 0.9)

        # min_overlap for sets
        if jacc < 0.25:
            min_ol_target = max(1, math.ceil(0.3 * opp_k))
        elif jacc > 0.8:
            min_ol_target = 1
        else:
            min_ol_target = self.min_overlap
        self.min_overlap = int(_clip(min_ol_target, 1, 8))

        # max_set_k
        self.max_set_k = int(_clip(max(3, min(6, opp_k + 1)), 2, 8))

        # handshake_prob retained for compatibility, but not used
        # we keep adaptation logic minimal to avoid breaking callers.
