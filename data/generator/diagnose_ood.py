"""Which out-of-distribution feature breaks convergence? A one-knob-at-a-time probe.

BAMBON's self-play agreement rate collapses on the generated out-of-distribution
corpus while it is unaffected by simply having *more* issues. This script
isolates the cause by generating small corpora that differ from the
in-distribution family in exactly one respect, and reporting self-play agreement
on each. The knobs are the ones the generator exposes:

    actions      how many actions per rule type (the conjunction rule requires
                 every issue of every action to be accepted before signing)
    constraints  how many constraints per rule
    strict_ops   strict inequalities and exclusion operators in the hard policies
    ranges       two-sided numeric interval issues on one operand
    unseen       left operands absent from the hand-authored 50, including
                 float-valued and sub-unit ones
    window       width of the legal interval [h_c, h_p]

Usage
-----
    python data/generator/diagnose_ood.py                 # all knobs, 15 scenarios each
    python data/generator/diagnose_ood.py --n 40 --seeds 2
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import statistics as st
import sys
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                          # noqa: E402

import benchmark_baselines_full as B                         # noqa: E402
from generate_scenarios import generate_scenario             # noqa: E402
from src.bambon.models.models import NegotiationInfeasible   # noqa: E402
from src.bambon.negotiator import Bambon                     # noqa: E402

PROFILE: Dict[str, Any] = {
    "inertia": 0.55, "tau_in": 0.60, "tau_out": 0.74, "max_set_k": 3,
    "no_agreement_until": 3, "hopeless_window": 16, "jaccard_snap_threshold": 0.45,
    "max_rounds": 150, "tau_cooling_threshold": 0.55, "tau_in_cooled": 0.60,
    "tau_out_cooled": 0.56, "lr_min": 0.32, "lr_max": 0.95, "vol_window": 5,
    "max_param_delta": 0.12, "hopeless_severity": 0.98, "hopeless_patience": 24,
    "hopeless_context": 0.06,
}

# One knob changed at a time, relative to the in-distribution family.
CONDITIONS: List[Dict[str, Any]] = [
    dict(label="ID baseline", overrides={}),
    dict(label="+ more actions (1-3)", overrides=dict(actions=(1, 3))),
    dict(label="+ more constraints (2-6)", overrides=dict(constraints=(2, 6))),
    dict(label="+ strict/exclusion hard ops", overrides=dict(strict_ops=True)),
    dict(label="   ... numeric only (lt/gt)",
         overrides=dict(strict_num_ops=True, strict_cat_ops=False)),
    dict(label="   ... categorical only",
         overrides=dict(strict_num_ops=False, strict_cat_ops=True)),
    dict(label="+ two-sided range issues", overrides=dict(ranges=True)),
    dict(label="+ unseen operands", overrides=dict(seen_only=False, unseen_bias=0.75)),
    dict(label="+ narrow legal window", overrides=dict(window=0.2)),
    dict(label="ALL (= the OOD family)",
         overrides=dict(actions=(1, 3), constraints=(2, 6), strict_ops=True,
                        ranges=True, seen_only=False, unseen_bias=0.75)),
]


def make(role: str, init: Dict, hard: Dict, mode: str = "belief") -> Bambon:
    agent = Bambon(name=role, lr=0.50, init_offer=copy.deepcopy(init),
                   hard_policy=hard, bidding_mode=mode, **PROFILE)
    agent.init_offer = init
    agent.hard_policy = hard
    return agent


def probe(label: str, overrides: Dict[str, Any], n: int, seeds: int,
          base_seed: int, mode: str) -> Dict[str, Any]:
    rng = random.Random(base_seed)
    agreed = 0
    total = 0
    turns: List[int] = []
    issues: List[int] = []
    actions: List[int] = []
    walked = 0
    for i in range(n):
        sc = generate_scenario(rng, "id", i, **overrides)
        if not sc["meta"]["witness_verified"]:
            continue                       # only score verified-feasible scenarios
        d = sc["docs"]
        ip, hp, ic, hc = d["provider"], d["hard_provider"], d["consumer"], d["hard_consumer"]
        issues.append(sc["meta"]["n_issues_spec"])
        actions.append(len(sc["meta"]["actions"]))
        for seed in range(seeds):
            random.seed(seed)
            np.random.seed(seed)
            total += 1
            try:
                agreement, t, _ = B.negotiate_generic(
                    make("provider", ip, hp, mode), make("consumer", ic, hc, mode))
            except NegotiationInfeasible:
                agreement, t = None, 0
            turns.append(t)
            if agreement is not None:
                agreed += 1
            elif t < B.MAX_TURNS:
                walked += 1
    return {
        "condition": label,
        "n": total,
        "issues": st.mean(issues) if issues else 0.0,
        "actions": st.mean(actions) if actions else 0.0,
        "agreement": agreed / total if total else float("nan"),
        "walked_away": walked / total if total else float("nan"),
        "turns": st.mean(turns) if turns else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=15, help="scenarios per condition")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--mode", default="belief", choices=("belief", "time"),
                    help="BAMBON bidding mode")
    args = ap.parse_args()

    print(f"BAMBON self-play, bidding_mode={args.mode}, "
          f"{args.n} scenarios x {args.seeds} seed(s) per condition\n")
    header = f"{'condition':30s} {'n':>4s} {'issues':>7s} {'actions':>8s} " \
             f"{'agree':>6s} {'walk':>6s} {'turns':>6s}"
    print(header)
    print("-" * len(header))
    for cond in CONDITIONS:
        r = probe(cond["label"], cond["overrides"], args.n, args.seeds,
                  args.seed, args.mode)
        print(f"{r['condition']:30s} {r['n']:4d} {r['issues']:7.1f} "
              f"{r['actions']:8.1f} {r['agreement']:6.2f} {r['walked_away']:6.2f} "
              f"{r['turns']:6.0f}")


if __name__ == "__main__":
    main()
