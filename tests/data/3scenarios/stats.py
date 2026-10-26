"""Statistical analysis layer for the ODRL negotiation benchmark.

Produces publication-ready tables and test results for a high-IF submission.
All tests are non-parametric (Wilcoxon signed-rank or Mann-Whitney U) with
rank-biserial correlation as the effect size, and Benjamini-Hochberg (BH)
multiple-comparison correction across all pairwise agent comparisons.

Public API
----------
agent_summary_stats(results_df, ranges) -> pd.DataFrame
    Per-agent aggregate table (mean ± std, N) for every primary metric.

pairwise_stats(results_df, reference_agents, target_metric) -> pd.DataFrame
    Wilcoxon signed-rank test of each reference agent vs every other agent
    on the chosen metric (one row per comparison), with p-value, BH-adjusted
    p-value, effect size r, and significance stars.

full_stats_report(results_df, ranges, out_dir) -> None
    Writes ``stats_summary.csv``, ``pairwise_wilcoxon.csv``, and
    ``stats_report.json`` to *out_dir* and prints a condensed console summary.

primary_metrics() -> List[str]
    Canonical ordered list of metrics used throughout.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Metric catalogue
# ---------------------------------------------------------------------------

def primary_metrics() -> List[str]:
    """Ordered list of primary outcome metrics for the benchmark."""
    return [
        "agreement_rate",
        "compliance_rate",
        "feasible_agreement_rate",
        "odrl_utility_self",          # computed by agent_summary_stats
        "odrl_nash_product",
        "odrl_fairness",
        "turns_to_agreement",         # conditional on agreement
        "own_hard_violations",
    ]


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def rank_biserial_r(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank-biserial correlation r for two independent samples (Mann-Whitney U).

    r = 1 - 2U / (n1 * n2)   in [-1, 1].
    Positive r means x tends to be larger than y.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    u_stat, _ = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(1.0 - 2.0 * u_stat / (n1 * n2))


def wilcoxon_paired_r(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank-biserial r for paired Wilcoxon signed-rank test.

    r = Z / sqrt(N)  where N = number of non-zero difference pairs.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    diff = x - y
    nz = diff[diff != 0]
    if len(nz) < 4:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sp_stats.wilcoxon(nz, alternative="two-sided")
    z = sp_stats.norm.ppf(result.pvalue / 2)  # approximate Z from p
    return float(abs(z) / np.sqrt(len(nz)))


# ---------------------------------------------------------------------------
# BH correction
# ---------------------------------------------------------------------------

def bh_correct(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Returns (adjusted_p_values, rejected_mask).
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p.copy(), np.array([], dtype=bool)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adj = np.minimum(1.0, p * n / ranks)
    # enforce monotonicity (take cumulative minimum from the right)
    adj_sorted = adj[order]
    for i in range(n - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adj[order] = adj_sorted
    return adj, adj <= alpha


def _stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Per-agent aggregate
# ---------------------------------------------------------------------------

def _stem(name: str) -> str:
    return name.replace("_provider", "").replace("_consumer", "")


def agent_summary_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-agent summary DataFrame with mean, std, and N for every
    primary metric.  ``odrl_utility_self`` is derived by taking the provider
    utility when the agent was provider and the consumer utility when consumer.
    """
    df = results_df.copy()
    df["agent_provider"] = df["provider"].map(_stem)
    df["agent_consumer"] = df["consumer"].map(_stem)

    records: List[Dict] = []

    all_agents = sorted(set(df["agent_provider"]) | set(df["agent_consumer"]))
    for agent in all_agents:
        as_prov = df[df["agent_provider"] == agent]
        as_cons = df[df["agent_consumer"] == agent]

        # agreement
        agree_all = pd.concat([as_prov["agreement"], as_cons["agreement"]])
        # feasible agreement
        feas_all = pd.concat([
            as_prov["feasible_agreement"], as_cons["feasible_agreement"]])
        # self utility (ODRL)
        self_u = pd.concat([
            as_prov[as_prov["agreement"]]["odrl_utility_provider"],
            as_cons[as_cons["agreement"]]["odrl_utility_consumer"],
        ])
        # nash product
        nash = pd.concat([
            as_prov[as_prov["agreement"]]["odrl_nash_product"],
            as_cons[as_cons["agreement"]]["odrl_nash_product"],
        ])
        # fairness
        fair = pd.concat([
            as_prov[as_prov["agreement"]]["odrl_fairness"],
            as_cons[as_cons["agreement"]]["odrl_fairness"],
        ])
        # turns (conditional on agreement)
        turns_p = as_prov[as_prov["agreement"]]["turns"]
        turns_c = as_cons[as_cons["agreement"]]["turns"]
        turns = pd.concat([turns_p, turns_c])
        # violations
        viol_p = as_prov[as_prov["agreement"]]["provider_hard_violations"]
        viol_c = as_cons[as_cons["agreement"]]["consumer_hard_violations"]
        viols = pd.concat([viol_p, viol_c])

        def _ms(s: pd.Series) -> Tuple[float, float]:
            return (float(s.mean()) if len(s) else float("nan"),
                    float(s.std(ddof=1)) if len(s) > 1 else float("nan"))

        ar_m, ar_s = _ms(agree_all.astype(float))
        fr_m, fr_s = _ms(feas_all.astype(float))
        su_m, su_s = _ms(self_u)
        na_m, na_s = _ms(nash)
        fa_m, fa_s = _ms(fair)
        tu_m, tu_s = _ms(turns.astype(float))
        vi_m, vi_s = _ms(viols.astype(float))

        # compliance rate: fraction of OWN agreements where OWN hard
        # constraints are satisfied — the correct per-agent metric.
        # Denominator = agreements reached (not all negotiations).
        agree_p = as_prov[as_prov["agreement"]]
        agree_c = as_cons[as_cons["agreement"]]
        own_ok_p = (agree_p["provider_hard_violations"] == 0).astype(float)
        own_ok_c = (agree_c["consumer_hard_violations"] == 0).astype(float)
        own_ok   = pd.concat([own_ok_p, own_ok_c])
        n_agreements = int(len(own_ok))
        cr_m = float(own_ok.mean()) if n_agreements else float("nan")
        cr_s = float(own_ok.std(ddof=1)) if n_agreements > 1 else float("nan")

        records.append({
            "agent": agent,
            "N_negotiations": int(len(agree_all)),
            "N_agreements": n_agreements,
            "agreement_rate_mean": ar_m, "agreement_rate_std": ar_s,
            # compliance_rate = P(own constraints satisfied | agreement reached)
            "compliance_rate_mean": cr_m, "compliance_rate_std": cr_s,
            # feasible_agreement_rate kept for backward compat (tournament-level)
            "feasible_agreement_rate_mean": fr_m, "feasible_agreement_rate_std": fr_s,
            "odrl_utility_self_mean": su_m, "odrl_utility_self_std": su_s,
            "odrl_nash_product_mean": na_m, "odrl_nash_product_std": na_s,
            "odrl_fairness_mean": fa_m, "odrl_fairness_std": fa_s,
            "turns_to_agreement_mean": tu_m, "turns_to_agreement_std": tu_s,
            "own_hard_violations_total": int(viols.sum()) if len(viols) else 0,
        })

    return pd.DataFrame(records).set_index("agent")


# ---------------------------------------------------------------------------
# Pairwise statistical tests
# ---------------------------------------------------------------------------

def pairwise_stats(
    results_df: pd.DataFrame,
    reference_agents: Optional[List[str]] = None,
    target_metric: str = "odrl_utility_self",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Pairwise Mann-Whitney U tests between every pair of agents on *target_metric*.

    If *reference_agents* is given, only rows where the reference agent is on
    one side are included; otherwise all unique pairs are tested.

    The per-run metric for each agent is the mean across all negotiations that
    run was part of (as provider or consumer), so runs are the unit of
    observation — this keeps the sample size honest.

    Returns a DataFrame with columns:
        agent_a, agent_b, n_a, n_b, median_a, median_b,
        U_stat, p_value, p_adjusted (BH), effect_r, sig
    """
    df = results_df.copy()
    df["stem_prov"] = df["provider"].map(_stem)
    df["stem_cons"] = df["consumer"].map(_stem)

    # Build per-run per-agent series for the target metric
    def _per_run_series(agent: str) -> np.ndarray:
        if target_metric == "odrl_utility_self":
            as_p = df[df["stem_prov"] == agent][["run", "odrl_utility_provider"]].rename(
                columns={"odrl_utility_provider": "val"})
            as_c = df[df["stem_cons"] == agent][["run", "odrl_utility_consumer"]].rename(
                columns={"odrl_utility_consumer": "val"})
            merged = pd.concat([as_p, as_c])
        elif target_metric == "agreement_rate":
            as_p = df[df["stem_prov"] == agent][["run", "agreement"]].rename(
                columns={"agreement": "val"})
            as_c = df[df["stem_cons"] == agent][["run", "agreement"]].rename(
                columns={"agreement": "val"})
            merged = pd.concat([as_p, as_c])
            merged["val"] = merged["val"].astype(float)
        elif target_metric == "compliance_rate":
            # per-run compliance: fraction of own agreements with zero own violations
            as_p = df[(df["stem_prov"] == agent) & df["agreement"]][
                ["run", "provider_hard_violations"]].copy()
            as_p["val"] = (as_p["provider_hard_violations"] == 0).astype(float)
            as_p = as_p[["run", "val"]]
            as_c = df[(df["stem_cons"] == agent) & df["agreement"]][
                ["run", "consumer_hard_violations"]].copy()
            as_c["val"] = (as_c["consumer_hard_violations"] == 0).astype(float)
            as_c = as_c[["run", "val"]]
            merged = pd.concat([as_p, as_c])
        elif target_metric == "feasible_agreement_rate":
            as_p = df[df["stem_prov"] == agent][["run", "feasible_agreement"]].rename(
                columns={"feasible_agreement": "val"})
            as_c = df[df["stem_cons"] == agent][["run", "feasible_agreement"]].rename(
                columns={"feasible_agreement": "val"})
            merged = pd.concat([as_p, as_c])
            merged["val"] = merged["val"].astype(float)
        elif target_metric == "odrl_nash_product":
            as_p = df[df["stem_prov"] == agent][["run", "odrl_nash_product"]].rename(
                columns={"odrl_nash_product": "val"})
            as_c = df[df["stem_cons"] == agent][["run", "odrl_nash_product"]].rename(
                columns={"odrl_nash_product": "val"})
            merged = pd.concat([as_p, as_c])
        else:
            raise ValueError(f"Unsupported target_metric: {target_metric!r}")
        return merged.groupby("run")["val"].mean().values

    all_agents = sorted(set(df["stem_prov"]) | set(df["stem_cons"]))
    refs = reference_agents if reference_agents else all_agents

    rows: List[Dict] = []
    for a in refs:
        for b in all_agents:
            if a == b:
                continue
            xa = _per_run_series(a)
            xb = _per_run_series(b)
            if len(xa) < 4 or len(xb) < 4:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u_stat, p_val = sp_stats.mannwhitneyu(xa, xb, alternative="two-sided")
            r = rank_biserial_r(xa, xb)
            rows.append({
                "agent_a": a, "agent_b": b,
                "n_a": len(xa), "n_b": len(xb),
                "median_a": float(np.median(xa)), "median_b": float(np.median(xb)),
                "U_stat": float(u_stat), "p_value": float(p_val),
                "effect_r": r,
            })

    if not rows:
        return pd.DataFrame()

    pw = pd.DataFrame(rows)
    adj_p, _ = bh_correct(pw["p_value"].values, alpha=alpha)
    pw["p_adjusted"] = adj_p
    pw["sig"] = pw["p_adjusted"].map(_stars)
    return pw.sort_values(["agent_a", "p_adjusted"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multi-metric pairwise block
# ---------------------------------------------------------------------------

def multi_metric_pairwise(
    results_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    reference_agents: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run pairwise_stats for each metric and concatenate with a ``metric`` column."""
    if metrics is None:
        metrics = [
            "odrl_utility_self",
            "compliance_rate",
            "feasible_agreement_rate",
            "odrl_nash_product",
            "agreement_rate",
        ]
    frames: List[pd.DataFrame] = []
    for m in metrics:
        try:
            pw = pairwise_stats(results_df, reference_agents=reference_agents,
                                target_metric=m, alpha=alpha)
            if not pw.empty:
                pw.insert(0, "metric", m)
                frames.append(pw)
        except Exception as exc:
            print(f"[warn] pairwise_stats failed for metric {m!r}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Condensed console summary
# ---------------------------------------------------------------------------

def _console_summary(summary: pd.DataFrame, pw: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("AGENT SUMMARY (mean ± std)")
    print("=" * 70)
    cols = [
        ("agreement_rate_mean", "agreement_rate_std", "Agree"),
        ("feasible_agreement_rate_mean", "feasible_agreement_rate_std", "Feasible"),
        ("odrl_utility_self_mean", "odrl_utility_self_std", "Self-U"),
        ("odrl_nash_product_mean", "odrl_nash_product_std", "Nash"),
    ]
    header = f"{'Agent':<28}" + "".join(f"  {lbl:>18}" for _, _, lbl in cols)
    print(header)
    print("-" * len(header))
    for agent, row in summary.iterrows():
        line = f"{agent:<28}"
        for m_col, s_col, _ in cols:
            m, s = row.get(m_col, float("nan")), row.get(s_col, float("nan"))
            line += f"  {m:6.3f}±{s:5.3f}      " if not np.isnan(m) else f"  {'n/a':>18}"
        print(line)

    if not pw.empty:
        print("\n" + "=" * 70)
        print("PAIRWISE TESTS (BH-corrected, significant only)")
        print("=" * 70)
        sig = pw[pw["sig"].isin(["*", "**", "***"])]
        if sig.empty:
            print("  No significant differences after BH correction.")
        else:
            for _, r in sig.iterrows():
                direction = ">" if r["median_a"] > r["median_b"] else "<"
                print(f"  [{r['metric']:28s}] {r['agent_a']:24s} {direction} "
                      f"{r['agent_b']:24s}  "
                      f"p_adj={r['p_adjusted']:.4f}{r['sig']:3s}  r={r['effect_r']:.3f}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def full_stats_report(
    results_df: pd.DataFrame,
    out_dir: str,
    reference_agents: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> Dict[str, str]:
    """Compute all statistics and write CSV + JSON artefacts to *out_dir*.

    Parameters
    ----------
    results_df
        The DataFrame returned by ``run_full_benchmark``.
    out_dir
        Directory where output files are written (created if absent).
    reference_agents
        If given, pairwise tests are only run from these agents' perspective.
        Typically the Bambon variants: ``["bambon_balanced", ...]``.
    alpha
        FDR threshold for BH correction (default 0.05).

    Returns
    -------
    dict mapping ``"summary_csv"``, ``"pairwise_csv"``, ``"report_json"`` to
    their absolute paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Per-agent aggregate
    summary = agent_summary_stats(results_df)
    summary_path = os.path.join(out_dir, "stats_summary.csv")
    summary.to_csv(summary_path)

    # 2. Multi-metric pairwise
    pw = multi_metric_pairwise(
        results_df,
        metrics=["odrl_utility_self", "feasible_agreement_rate",
                 "odrl_nash_product", "agreement_rate"],
        reference_agents=reference_agents,
        alpha=alpha,
    )
    pw_path = os.path.join(out_dir, "pairwise_wilcoxon.csv")
    if not pw.empty:
        pw.to_csv(pw_path, index=False)

    # 3. Console
    _console_summary(summary, pw)

    # 4. JSON report
    report: Dict = {
        "alpha": alpha,
        "n_negotiations": int(len(results_df)),
        "n_pairings": int(results_df.groupby(["provider", "consumer"]).ngroups),
        "correction": "Benjamini-Hochberg FDR",
        "test": "Mann-Whitney U (two-sided), rank-biserial r effect size",
        "agents": {},
        "significant_differences": [],
    }
    for agent, row in summary.iterrows():
        report["agents"][agent] = {k: (None if np.isnan(v) else round(float(v), 6))
                                   for k, v in row.items()
                                   if isinstance(v, (float, int, np.floating, np.integer))}
    if not pw.empty:
        sig = pw[pw["sig"].isin(["*", "**", "***"])]
        for _, r in sig.iterrows():
            report["significant_differences"].append({
                "metric": r["metric"],
                "agent_a": r["agent_a"], "agent_b": r["agent_b"],
                "median_a": round(float(r["median_a"]), 6),
                "median_b": round(float(r["median_b"]), 6),
                "U_stat": round(float(r["U_stat"]), 2),
                "p_value": round(float(r["p_value"]), 6),
                "p_adjusted": round(float(r["p_adjusted"]), 6),
                "effect_r": round(float(r["effect_r"]), 4),
                "sig": r["sig"],
            })

    report_path = os.path.join(out_dir, "stats_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    paths = {
        "summary_csv": summary_path,
        "pairwise_csv": pw_path,
        "report_json": report_path,
    }
    print(f"[stats] Written: {summary_path}")
    print(f"[stats] Written: {pw_path}")
    print(f"[stats] Written: {report_path}")
    return paths
