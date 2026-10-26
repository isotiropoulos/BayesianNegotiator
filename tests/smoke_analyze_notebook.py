"""Smoke-test notebook logic without executing the full IPython notebook."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

BAM_PROFILES = ["bambon_aggressive", "bambon_balanced", "bambon_exploratory"]


def load_json_lenient(p: str):
    raw = open(p, "r", encoding="utf-8", errors="replace").read()
    return json.JSONDecoder().raw_decode(raw)[0]


def short(s):
    return str(s).replace("_provider", "").replace("_consumer", "")


def is_bam(s):
    return str(s).startswith(("bambon", "Bayesian"))


def agent_table(df):
    rows = []
    for a in sorted(set(df.pcat) | set(df.ccat)):
        gp = df[df.pcat == a]
        gc = df[df.ccat == a]
        util = pd.concat([gp.util_prov, gc.util_cons])
        nash = pd.concat([gp.util_nash, gc.util_nash])
        agr = pd.concat([gp.agreement, gc.agreement])
        enf = pd.concat([gp.contract_enforceable, gc.contract_enforceable])
        viol = (
            int(gp.provider_violated.sum() + gc.consumer_violated.sum())
            if "provider_violated" in df.columns
            else 0
        )
        rows.append(
            [
                a,
                is_bam(a + "_provider"),
                len(gp),
                len(gc),
                agr.mean(),
                enf.mean(),
                util.mean(),
                nash.mean(),
                viol,
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            "agent",
            "is_bambon",
            "n_as_provider",
            "n_as_consumer",
            "agreement_rate",
            "enforceable_rate",
            "utility",
            "nash",
            "violations",
        ],
    ).set_index("agent")


def load_benchmark(data_dir: Path):
    fr = data_dir / "full_results.json"
    csv_path = data_dir / "bambon_vs_all_categories.csv"
    agg = recs = traj_misnamed = None
    if fr.is_file():
        res = load_json_lenient(str(fr))
        if isinstance(res, dict):
            recs = res.get("results")
            if recs and _row_is_negotiation(recs[0]):
                agg = res
            else:
                recs = None
        elif isinstance(res, list) and res:
            if _row_is_negotiation(res[0]):
                recs = res
            elif _row_is_trajectory(res[0]):
                traj_misnamed = res
    if recs is None:
        if not csv_path.is_file():
            raise FileNotFoundError(f"No valid results in {data_dir}")
        df = normalize_results_df(pd.read_csv(csv_path))
        source = "csv"
    else:
        df = normalize_results_df(pd.DataFrame(recs))
        source = "json"
    ng = data_dir / "full_negotiations.json"
    negs = load_json_lenient(str(ng)) if ng.is_file() else (traj_misnamed or [])
    return df, negs, source


def _row_is_negotiation(row):
    return isinstance(row, dict) and {"provider", "consumer", "agreement"} <= set(row)


def _row_is_trajectory(row):
    return isinstance(row, dict) and "rounds" in row and "data" in row


def normalize_results_df(df):
    if "odrl_utility_provider" not in df.columns and "utility_provider" in df.columns:
        df["odrl_utility_provider"] = df["utility_provider"]
    if "odrl_utility_consumer" not in df.columns and "utility_consumer" in df.columns:
        df["odrl_utility_consumer"] = df["utility_consumer"]
    for c in ("provider_hard_violations", "consumer_hard_violations"):
        if c not in df.columns:
            df[c] = 0
    for c in ("provider_violated", "consumer_violated"):
        if c not in df.columns:
            df[c] = False
    if "contract_enforceable" not in df.columns:
        df["contract_enforceable"] = df.agreement & ~df.provider_violated & ~df.consumer_violated
    return df


def run_dir(name: str, traj_limit: int | None = 50) -> None:
    data_dir = ROOT / name
    if not data_dir.is_dir():
        print(f"SKIP {name}: missing dir")
        return

    print(f"\n{'=' * 60}\n{name}")
    df, negs, source = load_benchmark(data_dir)
    if traj_limit is not None and len(negs) > traj_limit:
        negs = negs[:traj_limit]

    df["pcat"] = df.provider.map(short)
    df["ccat"] = df.consumer.map(short)
    df["util_prov"] = np.where(
        df.agreement & (df.provider_hard_violations == 0), df.odrl_utility_provider, 0.0
    )
    df["util_cons"] = np.where(
        df.agreement & (df.consumer_hard_violations == 0), df.odrl_utility_consumer, 0.0
    )
    df["util_nash"] = df.util_prov * df.util_cons

    agents = sorted(set(df.pcat) | set(df.ccat))
    bam_profiles_present = [p for p in BAM_PROFILES if p in agents]
    has_traj = len(negs) > 0

    print(f"negotiations={len(df)} agents={len(agents)} trajectories={len(negs)} source={source}")
    print(f"bambon_profiles_present={bam_profiles_present}")

    at = agent_table(df).sort_values("utility", ascending=False)
    print("top agents by utility:")
    print(at.head(5)[["utility", "enforceable_rate", "violations"]].round(3).to_string())

    # profile differentiation guard
    bp = at[at.is_bambon]
    profiles_present = [p for p in BAM_PROFILES if p in bp.index]
    if len(profiles_present) >= 2:
        print(f"profile comparison OK ({len(profiles_present)} profiles)")
    elif len(profiles_present) == 1:
        print(f"single profile only: {profiles_present[0]}")
    else:
        print("no BAMBON profiles")

    if has_traj:
        sample = next((t for t in negs if t.get("data")), None)
        assert sample is not None, "trajectories but no data payload"
        agreed = sum(1 for t in negs if t.get("agreed"))
        print(f"trajectory agreed={agreed}/{len(negs)} issues={len(sample.get('data', {}))}")
    else:
        print("HAS_TRAJ=False — sections 9-10 skipped")

    # utility sanity
    print(
        "utility sanity:",
        "non-agreement",
        round(df[~df.agreement].util_prov.mean(), 3),
        "own-violation",
        round(df[df.provider_violated].util_prov.mean(), 3)
        if "provider_violated" in df.columns
        else "n/a",
        "clean-deal",
        round(df[df.agreement & ~df.provider_violated].util_prov.mean(), 3)
        if "provider_violated" in df.columns
        else round(df[df.agreement].util_prov.mean(), 3),
    )
    print("OK")


def main():
    dirs = ["out_test", "out_anl", "out_full"]
    failed = []
    for d in dirs:
        try:
            run_dir(d, traj_limit=30 if d == "out_full" else None)
        except Exception:
            failed.append(d)
            traceback.print_exc()
    if failed:
        print("\nFAILED:", failed)
        sys.exit(1)
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
