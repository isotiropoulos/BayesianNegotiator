"""Tournament helpers: Nash analysis, data loading, negotiation loop, benchmark runner."""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import types
from collections import defaultdict
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.bambon.models.models import NegotiationInfeasible
from src.bambon.utils.helpers import (
    duration_to_seconds,
    iso_to_days,
    is_iso_date,
    is_iso_duration,
    satisfies,
)

from utility import odrl_utility
from tournament_constants import (
    MAX_TURNS,
    NUM_OPS,
    ODRL_UTILITY_CONFIG,
    RULE_TYPES,
    SCENARIO,
    TESTS_DIR,
)

# ---------------------------------------------------------------------------
# Nash bargaining (from bargaining.py)
# ---------------------------------------------------------------------------

def nash_product(utilities: Sequence[float],
                 disagreement: Optional[Sequence[float]] = None) -> float:
    if not utilities:
        return 0.0
    if disagreement is None:
        disagreement = [0.0] * len(utilities)
    if len(disagreement) != len(utilities):
        raise ValueError("disagreement must match the number of agents")
    product = 1.0
    for u, d in zip(utilities, disagreement):
        product *= max(0.0, float(u) - float(d))
    return product


def jains_fairness_index(utilities: Sequence[float]) -> float:
    xs = [float(x) for x in utilities]
    if not xs:
        return 0.0
    s = sum(xs)
    s2 = sum(x * x for x in xs)
    if s2 == 0.0:
        return 1.0
    return (s * s) / (len(xs) * s2)


def mean_utility(utilities: Sequence[float]) -> float:
    xs = [float(x) for x in utilities]
    return sum(xs) / len(xs) if xs else 0.0


def social_welfare(utilities: Sequence[float]) -> float:
    return sum(float(x) for x in utilities)


def agreement_rate(agreements: Sequence[bool]) -> float:
    flags = list(agreements)
    return sum(1 for a in flags if a) / len(flags) if flags else 0.0


def nash_bargaining_analysis(
    utility_profiles: Sequence[Sequence[float]],
    disagreement: Optional[Sequence[float]] = None,
    agreements: Optional[Sequence[bool]] = None,
    n_agents: Optional[int] = None,
) -> Dict[str, object]:
    profiles = [list(map(float, p)) for p in utility_profiles if p is not None]
    if n_agents is None:
        n_agents = len(profiles[0]) if profiles else 0
    if disagreement is None:
        disagreement = [0.0] * n_agents

    nash_values = [nash_product(p, disagreement) for p in profiles]
    jain_values = [jains_fairness_index(p) for p in profiles]
    welfare_values = [social_welfare(p) for p in profiles]
    per_agent_mean = [
        (sum(p[i] for p in profiles) / len(profiles)) if profiles else 0.0
        for i in range(n_agents)
    ]
    all_values = [u for p in profiles for u in p]
    rate = agreement_rate(agreements) if agreements is not None else (
        1.0 if profiles else 0.0)

    return {
        "n_negotiations": len(profiles),
        "n_agents": n_agents,
        "disagreement": list(map(float, disagreement)),
        "agreement_rate": rate,
        "mean_utility": mean_utility(all_values),
        "mean_utility_per_agent": per_agent_mean,
        "mean_social_welfare": mean_utility(welfare_values),
        "nash_product_mean": mean_utility(nash_values),
        "nash_product_at_mean": nash_product(per_agent_mean, disagreement),
        "jains_fairness_index_mean": mean_utility(jain_values),
        "jains_fairness_index_at_mean": jains_fairness_index(per_agent_mean),
    }


# ---------------------------------------------------------------------------
# Scenario data & negotiation loop
# ---------------------------------------------------------------------------

def fetch_data(case: str):
    base = os.path.join(TESTS_DIR, "data", case)
    with open(os.path.join(base, "provider.json"), encoding="utf-8") as f:
        provider = json.load(f)
    with open(os.path.join(base, "consumer.json"), encoding="utf-8") as f:
        consumer = json.load(f)
    with open(os.path.join(base, "hard_provider.json"), encoding="utf-8") as f:
        hard_provider = json.load(f)
    with open(os.path.join(base, "hard_consumer.json"), encoding="utf-8") as f:
        hard_consumer = json.load(f)
    return provider, hard_provider, consumer, hard_consumer


def trajectory_keys(*offers) -> List[str]:
    keys: List[str] = []
    for offer in offers:
        if not offer:
            continue
        for rule_type in RULE_TYPES:
            for rule in offer.get(rule_type, []) or []:
                action = rule.get("action")
                for c in rule.get("constraint", []) or []:
                    op = c["leftOperand"].replace("odrl:", "")
                    k = f"{rule_type}:{action}:{op}"
                    if k not in keys:
                        keys.append(k)
    return keys


def _extract_offer_values(offer: Dict) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not offer:
        return out
    for rule_type in RULE_TYPES:
        for rule in offer.get(rule_type, []) or []:
            action = rule.get("action")
            for c in rule.get("constraint", []) or []:
                op = c["leftOperand"].replace("odrl:", "")
                k = f"{rule_type}:{action}:{op}"
                if k not in out:
                    out[k] = c["rightOperand"]
    return out


def _record_round(run: Dict, offer: Dict, round_idx: int) -> None:
    run["rounds"].append(round_idx)
    vals = _extract_offer_values(offer)
    for k in run["data"]:
        run["data"][k].append(vals.get(k))


def negotiate_generic(agent1, agent2, max_turns: int = MAX_TURNS,
                      traj_keys: Optional[List[str]] = None):
    received = copy.deepcopy(agent2.init_offer)
    turn = 0
    agents = [agent1, agent2]
    accepted = 0
    final = None

    run = None
    if traj_keys is not None:
        run = {"rounds": [], "data": {k: [] for k in traj_keys},
               "durations": {"provider": [], "consumer": []}}
        _record_round(run, agent1.init_offer, -1)
        _record_round(run, agent2.init_offer, 0)

    while turn < max_turns:
        current = agents[turn % 2]
        side = "provider" if turn % 2 == 0 else "consumer"
        t0 = perf_counter()
        try:
            accept, replied = current.observe(received)
        except NegotiationInfeasible:
            return None, turn, run
        dt = perf_counter() - t0
        received = replied
        if run is not None:
            run["durations"][side].append(dt)
            _record_round(run, received, turn + 1)
        if accept:
            accepted += 1
            final = received
        else:
            accepted = 0
        if accepted == 2:
            return final, turn, run
        turn += 1
    return None, turn, run


# ---------------------------------------------------------------------------
# Aspiration→reservation utility & hard-constraint checks
# ---------------------------------------------------------------------------

def to_num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            if is_iso_date(v):
                return float(iso_to_days(v))
            if is_iso_duration(v):
                return float(duration_to_seconds(v))
            return float(v)
        except Exception:
            return None
    return None


def setish(v):
    if isinstance(v, (list, tuple, set)):
        return {str(x) for x in v}
    return {str(v)}


def build_operand_ranges(*offers) -> Dict[str, Tuple[float, float]]:
    vals: Dict[str, List[float]] = defaultdict(list)
    for offer in offers:
        if not offer:
            continue
        for rule_type in RULE_TYPES:
            for rule in offer.get(rule_type, []):
                for constr in rule.get("constraint", []):
                    op = constr["leftOperand"].replace("odrl:", "")
                    n = to_num(constr["rightOperand"])
                    if n is not None:
                        vals[op].append(n)
    return {op: (min(xs), max(xs)) for op, xs in vals.items() if xs}


def build_init_operand_ranges(init_provider: Dict, init_consumer: Dict) -> Dict[str, Tuple[float, float]]:
    """Utility normalisation spans from opening offers only (no hard policies).

    Use for incomplete-information benchmarks: agents may know both parties'
    initial offers, but not secret reservation / hard constraints.
    """
    return build_operand_ranges(init_provider, init_consumer)


def _find_actual(offer, rule_type, action, op):
    for orule in offer.get(rule_type, []):
        if action is not None and orule.get("action") != action:
            continue
        for oconstr in orule.get("constraint", []):
            if oconstr["leftOperand"].replace("odrl:", "") == op:
                return oconstr["rightOperand"]
    return None


def _reservation_clause(reservation, rule_type, action, op):
    if not reservation:
        return None, None
    for rule in reservation.get(rule_type, []):
        if action is not None and rule.get("action") != action:
            continue
        for c in rule.get("constraint", []):
            if c["leftOperand"].replace("odrl:", "") == op:
                return c["operator"].replace("odrl:", ""), c["rightOperand"]
    return None, None


def _operand_utility(a_op, a_val, agreed, h_op, h_val, op, ranges) -> float:
    an, aa, hn = to_num(agreed), to_num(a_val), to_num(h_val)

    if an is not None and aa is not None:
        direction = h_op or a_op
        if direction in ("lteq", "lt"):
            r = hn if hn is not None else (ranges.get(op, (aa, aa))[1])
            if r is None or r <= aa:
                return 1.0 if an <= aa else 0.0
            return max(0.0, min(1.0, (r - an) / (r - aa)))
        if direction in ("gteq", "gt"):
            r = hn if hn is not None else (ranges.get(op, (aa, aa))[0])
            if r is None or r >= aa:
                return 1.0 if an >= aa else 0.0
            return max(0.0, min(1.0, (an - r) / (aa - r)))
        if hn is not None:
            tol = abs(hn - aa) or 1.0
        else:
            lo, hi = ranges.get(op, (aa, aa))
            tol = (hi - lo) or 1.0
        return max(0.0, min(1.0, 1.0 - abs(an - aa) / tol))

    A, G = setish(a_val), setish(agreed)
    H = setish(h_val) if h_val is not None else None
    if H is not None and h_op in ("isAnyOf", "isAllOf", "eq") and not G.issubset(H):
        return 0.0
    if not (A | G):
        return 1.0
    return len(A & G) / len(A | G)


def compute_utility(offer: Dict, aspiration: Dict, reservation: Dict = None,
                    ranges: Dict[str, Tuple[float, float]] = None) -> float:
    if offer is None:
        return 0.0
    ranges = ranges or {}
    scores: List[float] = []
    for rule_type in RULE_TYPES:
        for rule in aspiration.get(rule_type, []):
            action = rule.get("action")
            for constr in rule.get("constraint", []):
                op = constr["leftOperand"].replace("odrl:", "")
                a_op = constr["operator"].replace("odrl:", "")
                a_val = constr["rightOperand"]
                actual = _find_actual(offer, rule_type, action, op)
                if actual is None:
                    scores.append(0.0)
                    continue
                h_op, h_val = _reservation_clause(reservation, rule_type, action, op)
                scores.append(_operand_utility(a_op, a_val, actual, h_op, h_val, op, ranges))
    return sum(scores) / max(1, len(scores))


def _agreed_clauses(offer, rule_type, action, op):
    out = []
    for orule in offer.get(rule_type, []):
        if action is not None and orule.get("action") != action:
            continue
        for oconstr in orule.get("constraint", []):
            if oconstr["leftOperand"].replace("odrl:", "") == op:
                out.append((oconstr["operator"].replace("odrl:", ""), oconstr["rightOperand"]))
    return out


def _clause_ok(hard_op, hard_val, agreed_op, agreed_val) -> bool:
    hn, an = to_num(hard_val), to_num(agreed_val)
    if hn is not None and an is not None:
        res = satisfies(hard_op if hard_op in NUM_OPS else "eq", hn, "eq", an)
    else:
        res = satisfies(hard_op, hard_val, agreed_op, agreed_val)
    return res is None or bool(res)


def count_hard_violations(agreement: Dict, hard_policy: Dict) -> int:
    if agreement is None or not hard_policy:
        return 0
    violations = 0
    for rule_type in RULE_TYPES:
        for hrule in hard_policy.get(rule_type, []):
            action = hrule.get("action")
            for hconstr in hrule.get("constraint", []):
                op = hconstr["leftOperand"].replace("odrl:", "")
                hard_op = hconstr["operator"].replace("odrl:", "")
                hard_val = hconstr["rightOperand"]
                clauses = _agreed_clauses(agreement, rule_type, action, op)
                if not clauses:
                    violations += 1
                    continue
                if any(not _clause_ok(hard_op, hard_val, a_op, a_val) for a_op, a_val in clauses):
                    violations += 1
    return violations


def feasible_agreement(agreed: bool, prov_viol: int, cons_viol: int) -> bool:
    """Agreement satisfies every hard constraint for both parties."""
    return bool(agreed and prov_viol == 0 and cons_viol == 0)


def feasible_agreement_rate(results_df: pd.DataFrame) -> float:
    return float(results_df["feasible_agreement"].mean())


# ---------------------------------------------------------------------------
# ANL package bootstrap (shared by anl_2024 / anl_2025 loaders)
# ---------------------------------------------------------------------------

def ensure_anl_agents_path() -> None:
    try:
        import anl_agents  # noqa: F401
        return
    except ImportError:
        pass
    candidates = [
        os.environ.get("ANL_AGENTS_SRC"),
        os.path.join(os.environ.get("TEMP", ""), "anl-agents", "src"),
        os.path.join(TESTS_DIR, "vendor", "anl-agents", "src"),
        os.path.join(os.path.expanduser("~"), "anl-agents", "src"),
    ]
    for p in candidates:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def stub_tensorflow_if_missing() -> None:
    import importlib.util

    if "tensorflow" in sys.modules:
        tf = sys.modules["tensorflow"]
        if getattr(tf, "__spec__", None) is None:
            tf.__spec__ = importlib.util.spec_from_loader("tensorflow", loader=None)
        return
    try:
        import tensorflow  # noqa: F401
        return
    except ImportError:
        pass

    def _pkg(name: str):
        m = types.ModuleType(name)
        m.__path__ = []  # type: ignore[attr-defined]
        if getattr(m, "__spec__", None) is None:
            m.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        sys.modules[name] = m
        return m

    tf = _pkg("tensorflow")
    tf.__version__ = "0.0-stub"
    keras = _pkg("tensorflow.keras")
    models = _pkg("tensorflow.keras.models")
    models.load_model = lambda *a, **k: None
    pre = _pkg("tensorflow.keras.preprocessing")
    seq = _pkg("tensorflow.keras.preprocessing.sequence")
    seq.pad_sequences = lambda x, **k: x


# ---------------------------------------------------------------------------
# Benchmark runner & results
# ---------------------------------------------------------------------------

def _stem(name: str) -> str:
    return name.replace("_provider", "").replace("_consumer", "")


def _is_bambon(name: str) -> bool:
    return name.startswith("bambon")


def _nash_block(sub: pd.DataFrame) -> dict:
    agreed = sub[sub["agreement"]]
    profiles = list(zip(agreed["odrl_utility_provider"], agreed["odrl_utility_consumer"]))
    return nash_bargaining_analysis(
        profiles, disagreement=[0.0, 0.0],
        agreements=list(sub["agreement"]), n_agents=2,
    )


def _by_agent_summary(results_df: pd.DataFrame) -> dict:
    df = results_df.copy()
    df["ps"] = df["provider"].map(_stem)
    df["cs"] = df["consumer"].map(_stem)
    out: dict = {}
    for agent in sorted(set(df["ps"]) | set(df["cs"])):
        gp = df[df["ps"] == agent]
        gc = df[df["cs"] == agent]
        agp = gp[gp["agreement"]]
        agc = gc[gc["agreement"]]
        self_u = pd.concat([agp["odrl_utility_provider"], agc["odrl_utility_consumer"]])
        nash = pd.concat([agp["odrl_nash_product"], agc["odrl_nash_product"]])
        fair = pd.concat([agp["odrl_fairness"], agc["odrl_fairness"]])
        agree = pd.concat([gp["agreement"], gc["agreement"]])
        feasible = pd.concat([gp["feasible_agreement"], gc["feasible_agreement"]])
        viol = int(agp["provider_violated"].sum() + agc["consumer_violated"].sum())
        out[agent] = {
            "is_bambon": _is_bambon(agent),
            "n_negotiations": int(len(agree)),
            "agreement_rate": float(agree.mean()),
            "feasible_agreement_rate": float(feasible.mean()),
            "mean_odrl_self_utility": float(self_u.mean()) if len(self_u) else 0.0,
            "mean_nash_product": float(nash.mean()) if len(nash) else 0.0,
            "mean_fairness": float(fair.mean()) if len(fair) else 0.0,
            "own_hard_violations": viol,
        }
    return out


def run_full_benchmark(provider_factories: Dict[str, callable],
                       consumer_factories: Dict[str, callable],
                       ranges: Dict[str, Tuple[float, float]],
                       n_runs: int = 5,
                       traj_keys: Optional[List[str]] = None):
    results = []
    trajectories: Dict[Tuple[str, str], List[Dict]] = {}
    all_trajectories: List[Dict] = []

    for prov_name, prov_factory in provider_factories.items():
        for cons_name, cons_factory in consumer_factories.items():
            pair_runs: List[Dict] = []
            for run in range(n_runs):
                random.seed(run)
                np.random.seed(run)
                provider = prov_factory()
                consumer = cons_factory()
                t0 = perf_counter()
                agreement, turns, run_traj = negotiate_generic(
                    provider, consumer, traj_keys=traj_keys)
                wall = perf_counter() - t0

                if agreement is not None and run_traj is not None:
                    pair_runs.append(run_traj)
                if run_traj is not None:
                    all_trajectories.append({
                        "provider": prov_name,
                        "consumer": cons_name,
                        "run": run,
                        "agreed": agreement is not None,
                        "turns": turns,
                        "rounds": run_traj["rounds"],
                        "data": run_traj["data"],
                        "durations": run_traj["durations"],
                    })

                prov_hard = getattr(provider, "hard_policy", None)
                cons_hard = getattr(consumer, "hard_policy", None)
                if agreement is None:
                    u_prov = u_cons = 0.0
                    agreed = False
                    prov_viol = cons_viol = 0
                    odrl_prov = odrl_utility(None, provider.init_offer, prov_hard, ranges,
                                             ODRL_UTILITY_CONFIG)
                    odrl_cons = odrl_utility(None, consumer.init_offer, cons_hard, ranges,
                                             ODRL_UTILITY_CONFIG)
                else:
                    u_prov = compute_utility(agreement, provider.init_offer, prov_hard, ranges)
                    u_cons = compute_utility(agreement, consumer.init_offer, cons_hard, ranges)
                    agreed = True
                    prov_viol = count_hard_violations(agreement, prov_hard)
                    cons_viol = count_hard_violations(agreement, cons_hard)
                    odrl_prov = odrl_utility(agreement, provider.init_offer, prov_hard, ranges,
                                             ODRL_UTILITY_CONFIG)
                    odrl_cons = odrl_utility(agreement, consumer.init_offer, cons_hard, ranges,
                                             ODRL_UTILITY_CONFIG)

                ou_prov, ou_cons = odrl_prov["utility"], odrl_cons["utility"]
                prov_feasible = bool(agreed and prov_viol == 0)
                cons_feasible = bool(agreed and cons_viol == 0)
                joint_feasible = feasible_agreement(agreed, prov_viol, cons_viol)

                results.append({
                    "provider": prov_name,
                    "consumer": cons_name,
                    "run": run,
                    "agreement": agreed,
                    "turns": turns,
                    "wall_time_s": wall,
                    "utility_provider": u_prov,
                    "utility_consumer": u_cons,
                    "social_welfare": u_prov + u_cons,
                    "nash_product": u_prov * u_cons,
                    "fairness": 1.0 - abs(u_prov - u_cons),
                    "provider_hard_violations": prov_viol,
                    "consumer_hard_violations": cons_viol,
                    "provider_violated": agreed and prov_viol > 0,
                    "consumer_violated": agreed and cons_viol > 0,
                    "feasible_agreement": joint_feasible,
                    "provider_feasible_agreement": prov_feasible,
                    "consumer_feasible_agreement": cons_feasible,
                    "odrl_utility_provider": ou_prov,
                    "odrl_utility_consumer": ou_cons,
                    "odrl_satisfaction_provider": odrl_prov["satisfaction"],
                    "odrl_satisfaction_consumer": odrl_cons["satisfaction"],
                    "odrl_violation_provider": odrl_prov["violation"],
                    "odrl_violation_consumer": odrl_cons["violation"],
                    "odrl_hard_penalty_provider": odrl_prov["hard_penalty"],
                    "odrl_hard_penalty_consumer": odrl_cons["hard_penalty"],
                    "odrl_nash_product": ou_prov * ou_cons,
                    "odrl_social_welfare": ou_prov + ou_cons,
                    "odrl_fairness": 1.0 - abs(ou_prov - ou_cons),
                })

            trajectories[(prov_name, cons_name)] = pair_runs
            pair = [r for r in results if r["provider"] == prov_name and r["consumer"] == cons_name]
            rate = sum(r["agreement"] for r in pair) / len(pair)
            feas = sum(r["feasible_agreement"] for r in pair) / len(pair)
            avg_turns = sum(r["turns"] for r in pair) / len(pair)
            print(f"{prov_name:28s} vs {cons_name:28s}: agree={rate:5.0%}  "
                  f"feasible={feas:5.0%}  avg_turns={avg_turns:5.1f}")

    return pd.DataFrame(results), trajectories, all_trajectories


def write_full_results(results_df: pd.DataFrame, out_dir: str, n_runs: int) -> str:
    cfg = ODRL_UTILITY_CONFIG
    agreed = results_df[results_df["agreement"]]
    payload = {
        "scenario": SCENARIO,
        "n_runs_per_pairing": n_runs,
        "n_negotiations": int(len(results_df)),
        "n_pairings": int(results_df.groupby(["provider", "consumer"]).ngroups),
        "utility_config": {
            "rule_weights": cfg.rule_weights,
            "violation_rule_weights": cfg.violation_rule_weights,
            "numeric_penalty_power": cfg.numeric_penalty_power,
            "numeric_directional": cfg.numeric_directional,
            "set_preservation_weight": cfg.set_preservation_weight,
            "set_compactness_weight": cfg.set_compactness_weight,
            "hard_lambda": cfg.hard_lambda,
            "use_semantic": cfg.use_semantic,
            "semantic_model_name": cfg.semantic_model_name,
        },
        "overall": {
            "agreement_rate": float(results_df["agreement"].mean()),
            "feasible_agreement_rate": feasible_agreement_rate(results_df),
            "provider_feasible_agreement_rate": float(
                results_df["provider_feasible_agreement"].mean()),
            "consumer_feasible_agreement_rate": float(
                results_df["consumer_feasible_agreement"].mean()),
            "mean_odrl_utility_provider": float(agreed["odrl_utility_provider"].mean())
            if len(agreed) else 0.0,
            "mean_odrl_utility_consumer": float(agreed["odrl_utility_consumer"].mean())
            if len(agreed) else 0.0,
            "nash_bargaining": _nash_block(results_df),
        },
        "by_agent": _by_agent_summary(results_df),
        "nash_by_provider": {
            prov: _nash_block(grp) for prov, grp in results_df.groupby("provider")
        },
        "results": json.loads(results_df.to_json(orient="records")),
    }
    path = os.path.join(out_dir, "full_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
