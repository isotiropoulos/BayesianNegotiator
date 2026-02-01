import ast
import numpy as np
import pandas as pd

# Use non-interactive backend so it works even without Tk
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Order of profiles in plots
PROFILES = ["balanced", "aggressive", "conceder", "boulware"]


# -------------------------------------------------------------
# 1. Load and expand CSV (Dict → per-run rows)
# -------------------------------------------------------------

def load_results(csv_path: str = "output2.csv") -> pd.DataFrame:
    """
    Load the aggregated CSV produced by the tests and parse the Dict column.
    """
    df = pd.read_csv(csv_path)
    df["Dict"] = df["Dict"].apply(ast.literal_eval)
    return df


def expand_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    From one row per (Agent_1, Agent_2) with Dict of lists
    -> one row per *simulation run*.

    Robust to Dict lists having different lengths.
    """

    def safe_get(d, key, i, default=None):
        """Return d[key][i] if it exists and is long enough, else default."""
        if key not in d:
            return default
        v = d[key]
        if isinstance(v, list):
            return v[i] if i < len(v) else default
        # If for some reason it's a scalar
        return v

    rows = []
    for _, row in df.iterrows():
        d = row["Dict"]

        # Use the longest list as the number of runs
        list_lengths = [len(v) for v in d.values() if isinstance(v, list)]
        if not list_lengths:
            continue
        n = max(list_lengths)

        for i in range(n):
            agreement = safe_get(d, "agreements_flag", i, 0)
            num_rounds = safe_get(d, "num_rounds", i, None)

            rec = {
                "Agent_1": row["Agent_1"],     # provider
                "Agent_2": row["Agent_2"],     # initiator
                "run": i,
                "agreement": agreement,
                "num_rounds": num_rounds,
                "agent_1_score": safe_get(d, "agent_1_score", i, None),
                "agent_2_score": safe_get(d, "agent_2_score", i, None),
                "agent_1_severity": safe_get(d, "agent_1_severity", i, None),
                "agent_2_severity": safe_get(d, "agent_2_severity", i, None),
            }

            rows.append(rec)

    return pd.DataFrame(rows)


# -------------------------------------------------------------
# 2. Pair-wise metrics  (balanced vs aggressive, etc.)
# -------------------------------------------------------------

def compute_pair_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregated metrics per (Agent_1 profile, Agent_2 profile) pair.

    n_runs: number of simulations for that pair
    agreement_rate: fraction of runs with agreement
    mean_rounds_all: avg rounds over all runs
    mean_rounds_agree: avg rounds only where agreement == 1
    agent_*_score: mean context similarity per side
    agent_*_severity: mean severity per side
    """

    def agg_pair(g: pd.DataFrame) -> pd.Series:
        n_total = len(g)
        n_agree = g["agreement"].sum()

        agreement_rate = n_agree / n_total if n_total > 0 else np.nan
        mean_rounds_all = g["num_rounds"].mean()
        mean_rounds_agree = g.loc[g["agreement"] == 1, "num_rounds"].mean()

        m1_score = g["agent_1_score"].mean()
        m2_score = g["agent_2_score"].mean()
        m1_sev = g["agent_1_severity"].mean()
        m2_sev = g["agent_2_severity"].mean()

        return pd.Series(
            dict(
                n_runs=n_total,
                agreement_rate=agreement_rate,
                mean_rounds_all=mean_rounds_all,
                mean_rounds_agree=mean_rounds_agree,
                agent_1_score=m1_score,
                agent_2_score=m2_score,
                agent_1_severity=m1_sev,
                agent_2_severity=m2_sev,
            )
        )

    pair_df = (
        runs.groupby(["Agent_1", "Agent_2"], as_index=False)
        .apply(agg_pair)
        .reset_index(drop=True)
    )
    return pair_df


# -------------------------------------------------------------
# 3. Role-based metrics (profile as Agent_1 vs Agent_2)
# -------------------------------------------------------------

def build_role_long_df(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Long format with one row per (run, role, profile).
    Allows: "how does 'aggressive' behave as provider vs initiator?"
    """
    rows = []
    for _, r in runs.iterrows():
        # Provider side (Agent_1)
        rows.append(
            dict(
                profile=r["Agent_1"],
                role="provider",            # Agent_1
                agreement=r["agreement"],
                num_rounds=r["num_rounds"],
                score=r["agent_1_score"],
                severity=r["agent_1_severity"],
            )
        )
        # Initiator side (Agent_2)
        rows.append(
            dict(
                profile=r["Agent_2"],
                role="initiator",          # Agent_2 starts the negotiation
                agreement=r["agreement"],
                num_rounds=r["num_rounds"],
                score=r["agent_2_score"],
                severity=r["agent_2_severity"],
            )
        )
    return pd.DataFrame(rows)


def compute_role_metrics(role_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregated metrics per (profile, role).
    """
    role_metrics = (
        role_df.groupby(["profile", "role"])
        .agg(
            n_runs=("agreement", "size"),
            agreement_rate=("agreement", "mean"),
            mean_rounds=("num_rounds", "mean"),
            mean_score=("score", "mean"),
            mean_severity=("severity", "mean"),
        )
        .reset_index()
    )
    return role_metrics


# -------------------------------------------------------------
# 4. Plot helpers – Heatmaps and Bars for the paper
# -------------------------------------------------------------

def heatmap_pair(metric_df: pd.DataFrame,
                 value_col: str,
                 title: str,
                 cmap: str = "RdYlGn",
                 vmin=None,
                 vmax=None,
                 ax=None,
                 cbar_label: str | None = None):
    """
    Heatmap with Agent_1 on rows and Agent_2 on cols.

    If ax is None, creates its own figure, otherwise draws on the given ax.
    """
    pivot = (
        metric_df.pivot(index="Agent_1", columns="Agent_2", values=value_col)
        .reindex(index=PROFILES, columns=PROFILES)
    )

    if ax is None:
        plt.figure(figsize=(6, 5))
        ax = plt.gca()

    hm = sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": cbar_label or value_col.replace("_", " ").title()},
        ax=ax,
    )
    ax.set_xlabel("Agent 2 Profile (initiator)")
    ax.set_ylabel("Agent 1 Profile (provider)")
    ax.set_title(title)
    return hm


def plot_agreement_heatmap(pair_df: pd.DataFrame):
    """
    Heatmap of agreement rate per (Agent_1, Agent_2) profile pair.
    """
    plt.figure(figsize=(6, 5))
    heatmap_pair(
        pair_df,
        "agreement_rate",
        "Agreement Rate",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_label="Agreement Rate",
    )
    plt.tight_layout()
    plt.savefig("heatmap_agreement_rate.png", dpi=300)
    plt.close()


def plot_rounds_heatmap(pair_df: pd.DataFrame):
    """
    Heatmap of mean rounds to agreement (only successful negotiations).
    """
    plt.figure(figsize=(6, 5))
    heatmap_pair(
        pair_df,
        "mean_rounds_agree",
        "Mean Rounds (Agreements Only)",
        cmap="YlOrBr",
        cbar_label="Mean Rounds Agree",
    )
    plt.tight_layout()
    plt.savefig("heatmap_rounds_agree.png", dpi=300)
    plt.close()


def plot_score_heatmaps(pair_df: pd.DataFrame):
    """
    Two heatmaps:
    - Agent_1_score (provider)  = context similarity for provider
    - Agent_2_score (initiator) = context similarity for initiator
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    heatmap_pair(
        pair_df,
        "agent_1_score",
        "Mean Context Score (Agent 1 / provider)",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        ax=axes[0],
        cbar_label="Mean Score",
    )

    heatmap_pair(
        pair_df,
        "agent_2_score",
        "Mean Context Score (Agent 2 / initiator)",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        ax=axes[1],
        cbar_label="Mean Score",
    )

    plt.tight_layout()
    plt.savefig("heatmap_scores.png", dpi=300)
    plt.close()


def plot_severity_heatmaps(pair_df: pd.DataFrame):
    """
    Two heatmaps:
    - Agent_1_severity (provider)
    - Agent_2_severity (initiator)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    heatmap_pair(
        pair_df,
        "agent_1_severity",
        "Mean Severity (Agent 1 / provider)",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=axes[0],
        cbar_label="Mean Severity",
    )

    heatmap_pair(
        pair_df,
        "agent_2_severity",
        "Mean Severity (Agent 2 / initiator)",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=axes[1],
        cbar_label="Mean Severity",
    )

    plt.tight_layout()
    plt.savefig("heatmap_severity.png", dpi=300)
    plt.close()


def barplot_role(role_metrics: pd.DataFrame, metric: str, title: str, filename: str, ylim=None):
    """
    Barplot comparing profiles per role (provider vs initiator).
    """
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(
        data=role_metrics,
        x="profile",
        y=metric,
        hue="role",
        order=PROFILES,
    )
    ax.set_title(title)
    ax.set_xlabel("Profile")
    ax.set_ylabel(metric.replace("_", " ").title())
    if ylim is not None:
        ax.set(ylim=ylim)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def boxplot_rounds_by_profile(role_df: pd.DataFrame, filename: str):
    """
    Distribution of rounds per profile & role.
    """
    plt.figure(figsize=(8, 4))
    ax = sns.boxplot(
        data=role_df,
        x="profile",
        y="num_rounds",
        hue="role",
        order=PROFILES,
    )
    ax.set_title("Distribution of Rounds per Profile & Role")
    ax.set_xlabel("Profile")
    ax.set_ylabel("Number of Rounds")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# -------------------------------------------------------------
# 5. Main script
# -------------------------------------------------------------

if __name__ == "__main__":
    # Load & expand
    df = load_results("output2.csv")
    runs = expand_runs(df)

    # --- Pair-wise behaviour (profile vs profile) ---
    pair_df = compute_pair_metrics(runs)
    print("Pair-wise metrics:")
    print(pair_df.head())

    # Heatmaps for paper
    plot_agreement_heatmap(pair_df)
    plot_rounds_heatmap(pair_df)
    plot_score_heatmaps(pair_df)
    plot_severity_heatmaps(pair_df)

    # --- Role-based behaviour (same profile as provider vs initiator) ---
    role_df = build_role_long_df(runs)
    role_metrics = compute_role_metrics(role_df)
    print("\nRole-based metrics:")
    print(role_metrics)

    # Role-based barplots / boxes (optional but nice for paper)
    barplot_role(
        role_metrics,
        "agreement_rate",
        "Agreement Rate per Profile & Role",
        filename="bar_agreement_rate_role.png",
        ylim=(0, 1),
    )
    barplot_role(
        role_metrics,
        "mean_rounds",
        "Mean Rounds per Profile & Role",
        filename="bar_mean_rounds_role.png",
    )
    barplot_role(
        role_metrics,
        "mean_score",
        "Mean Context Score per Profile & Role",
        filename="bar_mean_score_role.png",
        ylim=(0, 1),
    )
    barplot_role(
        role_metrics,
        "mean_severity",
        "Mean Severity per Profile & Role",
        filename="bar_mean_severity_role.png",
        ylim=(0, 1),
    )
    boxplot_rounds_by_profile(role_df, filename="box_rounds_profile_role.png")
