"""Negotiation-dynamics analysis for the FULL ODRL scenario.

This ports the plots from ``Results Analysis Full.ipynb`` (which only ever lived
inside the notebook) into reusable functions that render PNG files and return a
markdown section explaining each one. It can be run standalone or imported by
``benchmark_baselines_full.py`` so the dynamics plots are embedded in, and
explained by, the same benchmark report.

Data source: a results JSON that records, for ONE pairing's ``full`` negotiation,
the value of every negotiated operand at every round, across many Monte-Carlo
runs. Each run is a dict with:

  - ``rounds``    : list of round indices (-1 = provider initial offer, 0 =
                    consumer initial offer, then odd = provider, even = consumer).
  - ``data``      : {"{rule}:{action}:{operand}": [value per round]}.
  - ``durations`` : {"provider": [...], "consumer": [...]} wall-times per side.

The plotting helpers are parameterised so the SAME figures can be produced for
any pairing: the two trajectory lines are labelled with the *actual* provider and
consumer agents (``prov_label`` / ``cons_label``), the saved files can be given a
scenario prefix (``fname_prefix``) and written flat (no ``plots/`` subfolder) via
``path_prefix``.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    import isodate
except Exception:  # pragma: no cover
    isodate = None

PROV_C, CONS_C = "#2c7fb8", "#d95f0e"

# Per-call saving configuration (set by build_section): controls where files are
# written-to in the returned markdown links, and an optional scenario filename
# prefix. Kept as a module global so the many small plot helpers don't each need
# the two extra arguments threaded through.
_SAVE = {"path_prefix": "plots/", "fname_prefix": ""}


# --------------------------------------------------------------------------
# parsing helpers (ported from the notebook)
# --------------------------------------------------------------------------
def _is_consumer_round(r: float) -> bool:
    return r % 2 == 0


def parse_pt_to_minutes(s: str) -> float:
    """ISO-8601 duration, e.g. 'PT21M55.5S' -> minutes (float)."""
    s = s.replace("PT", "")
    hours = minutes = seconds = 0.0
    if "H" in s:
        h, s = s.split("H", 1); hours = float(h)
    if "M" in s:
        m, s = s.split("M", 1); minutes = float(m)
    if "S" in s and s:
        sp = s.replace("S", "")
        if sp:
            seconds = float(sp)
    return hours * 60.0 + minutes + seconds / 60.0


def _normalize_cat(v):
    if isinstance(v, list):
        return "+".join(sorted(map(str, v)))
    return str(v)


def _smooth(xs, window=3):
    xs = np.asarray(xs, dtype=float)
    if window <= 1 or len(xs) < window:
        return xs.tolist()
    pad = window // 2
    xs_pad = np.pad(xs, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(xs_pad, kernel, mode="valid").tolist()


def _to_dt(x):
    s = str(x).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).replace(tzinfo=None)
    except ValueError:
        return datetime.strptime(s[:10], "%Y-%m-%d")


def _round95(runs, q=0.99):
    final = [run["rounds"][-1] for run in runs]
    if not final:
        return 0
    uniq = sorted(set(final))
    cum = [sum(f <= x for f in final) / len(final) for x in uniq]
    idx = next((i for i, c in enumerate(cum) if c >= q), len(uniq) - 1)
    return uniq[idx]


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def _save(fig, plot_dir, fname):
    """Save under ``plot_dir`` with the configured scenario prefix; return the
    markdown-relative path using the configured ``path_prefix``."""
    full_fname = f"{_SAVE['fname_prefix']}{fname}"
    path = os.path.join(plot_dir, full_fname)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return f"{_SAVE['path_prefix']}{full_fname}"


def _title(base, title_suffix):
    return f"{base}\n{title_suffix}" if title_suffix else base


def plot_convergence(runs, plot_dir, fname="T1_convergence_cdf.png", title_suffix=""):
    final = [run["rounds"][-1] for run in runs]
    uniq = sorted(set(final))
    cum = [sum(f <= x for f in final) / len(final) for x in uniq]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.step(uniq, cum, where="post", color="#225ea8", lw=2)
    ax1.set_xlabel("final negotiation round"); ax1.set_ylabel("fraction of runs completed")
    ax1.set_title("Convergence (CDF of final round)"); ax1.set_ylim(0, 1.02); ax1.grid(alpha=0.3)
    ax2.hist(final, bins=min(40, max(5, len(uniq))), color="#41b6c4", edgecolor="white")
    ax2.set_xlabel("final negotiation round"); ax2.set_ylabel("number of runs")
    ax2.set_title("Distribution of final round"); ax2.grid(axis="y", alpha=0.3)
    if title_suffix:
        fig.suptitle(title_suffix, fontsize=10)
    return _save(fig, plot_dir, fname)


def plot_numeric_trend(runs, key, plot_dir, fname, ylabel, round95,
                       transform=None, iso_minutes=False,
                       prov_label="provider", cons_label="consumer", title_suffix=""):
    prov, cons = defaultdict(list), defaultdict(list)
    for run in runs:
        if run["rounds"][-1] > round95:
            continue
        for r, v in zip(run["rounds"], run["data"].get(key, [])):
            if v is None:
                continue
            if transform:
                try:
                    v = transform(v)
                except Exception:
                    continue
            try:
                val = float(v)
            except Exception:
                continue
            (cons if _is_consumer_round(r) else prov)[r].append(val)

    def stats(d):
        xs = sorted(d)
        if not xs:
            return [], np.array([]), np.array([])
        nmax = max(len(d[r]) for r in xs)
        mean = np.array([np.mean(d[r]) for r in xs])
        std = np.array([
            (np.std(d[r], ddof=1) * np.sqrt(len(d[r]) / nmax)) if len(d[r]) >= 5 else 0.0
            for r in xs
        ])
        return xs, mean, std

    pr, pm, ps = stats(prov)
    cr, cm, cs = stats(cons)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    if pr:
        ax.plot(pr, pm, marker="o", color=PROV_C, label=prov_label)
        ax.fill_between(pr, pm - ps, pm + ps, color=PROV_C, alpha=0.15)
    if cr:
        ax.plot(cr, cm, marker="s", ls="--", color=CONS_C, label=cons_label)
        ax.fill_between(cr, cm - cs, cm + cs, color=CONS_C, alpha=0.15)
    if iso_minutes:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda m, _: f"{int(m)}m{int(round((m - int(m)) * 60)):02d}s"))
    ax.set_xlabel("round"); ax.set_ylabel(ylabel)
    ax.set_title(_title(f"Per-round trajectory — {key}", title_suffix))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    return _save(fig, plot_dir, fname)


def plot_date_trend(runs, key, plot_dir, fname, ylabel, round95,
                    prov_label="provider", cons_label="consumer", title_suffix=""):
    prov, cons = defaultdict(list), defaultdict(list)
    for run in runs:
        if run["rounds"][-1] > round95:
            continue
        for r, v in zip(run["rounds"], run["data"].get(key, [])):
            if v is None:
                continue
            try:
                dt = _to_dt(v)
            except Exception:
                continue
            (cons if _is_consumer_round(r) else prov)[r].append(dt.timestamp())

    def stats(d):
        xs = sorted(d)
        if not xs:
            return [], np.array([]), np.array([])
        nmax = max(len(d[r]) for r in xs)
        mean = np.array([np.mean(d[r]) for r in xs])
        std = np.array([
            (np.std(d[r], ddof=1) * np.sqrt(len(d[r]) / nmax)) if len(d[r]) >= 5 else 0.0
            for r in xs
        ])
        return xs, mean, std

    pr, pm, ps = stats(prov)
    cr, cm, cs = stats(cons)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    if pr:
        ax.plot(pr, [datetime.fromtimestamp(t) for t in pm], marker="o", color=PROV_C, label=prov_label)
        ax.fill_between(pr, [datetime.fromtimestamp(t) for t in pm - ps],
                        [datetime.fromtimestamp(t) for t in pm + ps], color=PROV_C, alpha=0.15)
    if cr:
        ax.plot(cr, [datetime.fromtimestamp(t) for t in cm], marker="s", ls="--", color=CONS_C, label=cons_label)
        ax.fill_between(cr, [datetime.fromtimestamp(t) for t in cm - cs],
                        [datetime.fromtimestamp(t) for t in cm + cs], color=CONS_C, alpha=0.15)
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.set_xlabel("round"); ax.set_ylabel(ylabel)
    ax.set_title(_title(f"Per-round trajectory — {key}", title_suffix))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    return _save(fig, plot_dir, fname)


def plot_duration(runs, plot_dir, fname="T3_counteroffer_duration.png",
                  prov_label="provider", cons_label="consumer", title_suffix=""):
    prov, cons = defaultdict(list), defaultdict(list)
    for run in runs:
        rounds = run["rounds"]
        pr = [r for r in rounds if not _is_consumer_round(r)]
        cr = [r for r in rounds if _is_consumer_round(r)]
        for r, d in zip(pr, run.get("durations", {}).get("provider", [])):
            prov[r].append(float(d))
        for r, d in zip(cr, run.get("durations", {}).get("consumer", [])):
            cons[r].append(float(d))
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for d, c, lbl, m, ls in ((prov, PROV_C, prov_label, "o", "-"),
                             (cons, CONS_C, cons_label, "s", "--")):
        xs = sorted(d)
        if not xs:
            continue
        med = np.array([np.median(d[r]) for r in xs])
        sd = np.array([np.std(d[r]) for r in xs])
        ax.plot(xs, med, marker=m, ls=ls, color=c, label=f"{lbl} (median)")
        ax.fill_between(xs, med - sd, med + sd, color=c, alpha=0.15)
    ax.set_xlabel("round"); ax.set_ylabel("seconds to produce a counter-offer")
    ax.set_title(_title("Per-round compute time of a counter-offer", title_suffix))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    return _save(fig, plot_dir, fname)


def plot_category_area(runs, key, plot_dir, fname, round95, smooth=True, min_share=0.02,
                       title_suffix=""):
    all_rounds = sorted({r for run in runs for r in run["rounds"] if r <= round95})
    cats = sorted({_normalize_cat(v) for run in runs for v in run["data"].get(key, []) if v is not None})
    frac = {c: [] for c in cats}
    for r in all_rounds:
        vals = []
        for run in runs:
            if run["rounds"][-1] > round95:
                continue
            if r in run["rounds"]:
                v = run["data"][key][run["rounds"].index(r)]
                if v is not None:
                    vals.append(_normalize_cat(v))
        cnt = Counter(vals); total = len(vals)
        for c in cats:
            frac[c].append(cnt.get(c, 0) / total if total else 0.0)
    if smooth:
        for c in cats:
            frac[c] = _smooth(frac[c], 3)
    cats = [c for c in cats if max(frac[c]) >= min_share]
    cats = sorted(cats, key=lambda c: frac[c][-1])
    Y = np.array([frac[c] for c in cats], dtype=float)
    if Y.size and Y.sum(axis=0).max() > 0:
        colsum = Y.sum(axis=0, keepdims=True); colsum[colsum == 0] = 1.0
        Y = Y / colsum
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    if Y.size:
        ax.stackplot(all_rounds, Y, labels=cats, alpha=0.85)
    ax.set_xlabel("round"); ax.set_ylabel("share of runs")
    ax.set_title(_title(f"Categorical evolution — {key}", title_suffix))
    ax.set_ylim(0, 1); ax.legend(fontsize=7, loc="upper left", ncol=2)
    return _save(fig, plot_dir, fname)


def plot_final_distribution(runs, key, plot_dir, fname, xlabel, kind="numeric", title_suffix=""):
    raw = [run["data"][key][-1] for run in runs if run["data"].get(key) and run["data"][key][-1] is not None]
    if kind == "duration":
        xs = np.array([parse_pt_to_minutes(v) if isinstance(v, str) else float(v) for v in raw], dtype=float)
        unit = "minutes"
    else:
        xs = np.array([float(v) for v in raw], dtype=float)
        unit = ""
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.boxplot(xs, vert=False, notch=True, showfliers=True, widths=0.4,
               patch_artist=True, boxprops=dict(facecolor="#9ecae1"))
    med = np.median(xs)
    ax.axvline(med, color="#d95f0e", ls="--", lw=1, label=f"median={med:.1f}{unit}")
    ax.set_yticks([]); ax.set_xlabel(f"{xlabel} ({unit})" if unit else xlabel)
    ax.set_title(_title(f"Final agreed value — {key}", title_suffix)); ax.legend(fontsize=8)
    return _save(fig, plot_dir, fname)


def _entropy(cnt: Counter) -> float:
    total = sum(cnt.values())
    if not total:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in cnt.values()) + 0.0


def robustness_rows(runs, keys):
    rows = []
    for key in keys:
        agreed = []
        for run in runs:
            seq = run["data"].get(key)
            if seq and seq[-1] is not None:
                agreed.append(tuple(sorted(seq[-1])) if isinstance(seq[-1], list) else (seq[-1],))
        cnt = Counter(agreed); total = sum(cnt.values())
        if not total:
            rows.append((key, "—", 0.0, 0, 0.0)); continue
        dom, dc = cnt.most_common(1)[0]
        dom = "{" + ", ".join(map(str, dom)) + "}" if len(dom) > 1 else str(dom[0])
        rows.append((key, dom, 100.0 * dc / total, len(cnt), _entropy(cnt)))
    return rows


# --------------------------------------------------------------------------
# top-level: build the markdown section + all figures
# --------------------------------------------------------------------------
def build_section(json_path, plot_dir, iterations=None,
                  prov_label="provider", cons_label="consumer",
                  fname_prefix="", path_prefix="plots/", title_suffix=""):
    """Generate every dynamics plot and return (markdown_lines, ok).

    Parameters
    ----------
    prov_label, cons_label : labels for the two trajectory lines — pass the actual
        provider/consumer agents (e.g. ``"bambon_balanced (provider)"``).
    fname_prefix : prefix added to every saved filename (e.g. the scenario slug),
        so per-pairing files are uniquely named.
    path_prefix : prefix used in the returned markdown image paths. Use ``""`` to
        reference files that sit next to the markdown (flat, no ``plots/`` folder).
    title_suffix : extra line appended to each figure title (e.g. the scenario).
    """
    if not os.path.exists(json_path):
        return ([f"_Dynamics analysis skipped: `{os.path.basename(json_path)}` not found._\n"], False)
    with open(json_path, "r", encoding="utf-8") as f:
        runs = json.load(f)
    if not runs:
        return (["_Dynamics analysis skipped: results file is empty._\n"], False)

    os.makedirs(plot_dir, exist_ok=True)
    _SAVE["path_prefix"] = path_prefix
    _SAVE["fname_prefix"] = fname_prefix
    r95 = _round95(runs)
    agree_rate = (len(runs) / iterations) if iterations else None

    L = dict(prov_label=prov_label, cons_label=cons_label, title_suffix=title_suffix)

    p = {}
    p["conv"] = plot_convergence(runs, plot_dir, title_suffix=title_suffix)
    p["pct"] = plot_numeric_trend(runs, "permission:use:percentage", plot_dir,
                                  "T2a_use_percentage.png", "percentage (%)", r95, **L)
    p["count"] = plot_numeric_trend(runs, "permission:use:count", plot_dir,
                                    "T2b_use_count.png", "count", r95, **L)
    p["delay"] = plot_numeric_trend(runs, "permission:use:delayPeriod", plot_dir,
                                    "T2c_use_delayPeriod.png", "delayPeriod", r95,
                                    transform=parse_pt_to_minutes, iso_minutes=True, **L)
    p["date"] = plot_date_trend(runs, "permission:use:absoluteTemporalPosition", plot_dir,
                                "T2d_use_absoluteTemporalPosition.png", "date", r95, **L)
    p["obl"] = plot_numeric_trend(runs, "obligation:anonymize:percentage", plot_dir,
                                  "T2e_obligation_percentage.png", "percentage (%)", r95, **L)
    p["prohib"] = plot_numeric_trend(runs, "prohibition:redistribute:count", plot_dir,
                                     "T2f_prohibition_count.png", "count", r95, **L)
    p["dur"] = plot_duration(runs, plot_dir, prov_label=prov_label, cons_label=cons_label,
                             title_suffix=title_suffix)
    cat_keys = [
        ("permission:use:fileFormat", "T4a_use_fileFormat.png"),
        ("permission:use:language", "T4b_use_language.png"),
        ("permission:use:purpose", "T4c_use_purpose.png"),
        ("prohibition:redistribute:purpose", "T4d_prohibition_purpose.png"),
    ]
    p["cats"] = [plot_category_area(runs, k, plot_dir, fn, r95, title_suffix=title_suffix)
                 for k, fn in cat_keys]
    p["dist_count"] = plot_final_distribution(runs, "permission:use:count", plot_dir,
                                              "T5a_final_count.png", "count", title_suffix=title_suffix)
    p["dist_delay"] = plot_final_distribution(runs, "permission:use:delayPeriod", plot_dir,
                                              "T5b_final_delayPeriod.png", "delayPeriod",
                                              kind="duration", title_suffix=title_suffix)
    rob = robustness_rows(runs, [k for k, _ in cat_keys] + ["permission:use:industry",
                                                            "obligation:anonymize:industry"])

    w = []
    add = w.append
    who = title_suffix or "BAMBON vs BAMBON"
    add(f"---\n\n## E. Negotiation dynamics ({who}, Monte-Carlo)\n")
    rate_txt = f" Agreement rate **{agree_rate:.1%}** ({len(runs)} agreed runs)." if agree_rate else f" {len(runs)} agreed runs."
    add("Sections A–D compare agents on *final* outcomes. This section instead opens the black box and traces "
        "*how* the negotiation evolves round by round, aggregated over many seeded Monte-Carlo "
        f"runs." + rate_txt + " Even/odd rounds belong to the consumer/provider "
        "respectively; round −1 is the pair of initial offers. Curves show the mean across runs and the shaded band "
        "the (run-count-scaled) standard deviation. We truncate at the 99th-percentile round "
        f"(**{r95}**) to keep the long deadlock tail from dominating the axes.\n")

    add("### E.1 Convergence\n")
    add(f"![convergence]({p['conv']})")
    add("*Figure T1.* Left: CDF of the final round — the fraction of runs concluded by round *x*. Right: histogram "
        "of final rounds. Most negotiations close in a small number of rounds; the thin tail corresponds to harder "
        "starting gaps that take longer to bridge.\n")

    add("### E.2 Numeric, temporal and duration operands converge from both sides\n")
    add("Across all rule types (permission, obligation, prohibition) and actions, the provider and consumer curves "
        "start at their opposed aspirations and **converge to a common agreed value**, demonstrating "
        "that BAMBON's belief-driven concession works identically regardless of operand type or rule type.\n")
    add(f"![use percentage]({p['pct']})")
    add("*Figure T2a.* `permission:use:percentage` — provider vs consumer aspirations meet.\n")
    add(f"![use count]({p['count']})")
    add("*Figure T2b.* `permission:use:count` — same dynamic on a large-magnitude integer operand.\n")
    add(f"![use delayPeriod]({p['delay']})")
    add("*Figure T2c.* `permission:use:delayPeriod` — an ISO-8601 **duration** operand (y-axis in m/s).\n")
    add(f"![use date]({p['date']})")
    add("*Figure T2d.* `permission:use:absoluteTemporalPosition` — an ISO **date** operand.\n")
    add(f"![obligation percentage]({p['obl']})")
    add("*Figure T2e.* `obligation:anonymize:percentage` — an **obligation** operand.\n")
    add(f"![prohibition count]({p['prohib']})")
    add("*Figure T2f.* `prohibition:redistribute:count` — a **prohibition** operand.\n")

    add("### E.3 Cost per round\n")
    add(f"![counter-offer duration]({p['dur']})")
    add("*Figure T3.* Median wall-time to compute one counter-offer per side.\n")

    add("### E.4 Categorical / set-valued operands\n")
    add("Categorical operands (file format, language, purpose, …) are negotiated as set membership. The stacked-area "
        "plots show the share of runs settling on each category as the negotiation proceeds.\n")
    for (k, _), path in zip(cat_keys, p["cats"]):
        add(f"![{k}]({path})")
        add(f"*Categorical evolution of* `{k}`.\n")

    add("### E.5 Final agreed-value distributions\n")
    add(f"![final count]({p['dist_count']})")
    add("*Figure T5a.* Distribution of the final agreed `permission:use:count` across runs.\n")
    add(f"![final delay]({p['dist_delay']})")
    add("*Figure T5b.* Distribution of the final agreed `permission:use:delayPeriod` (minutes).\n")

    add("### E.6 Robustness of categorical agreements\n")
    add("For each set/enum operand we report the most frequent final agreement (its dominant share across runs), "
        "the number of distinct agreements reached, and the Shannon entropy of the agreement distribution.\n")
    add("| Operand | Dominant agreement | Share (%) | #distinct | Entropy (bits) |")
    add("|---|---|---|---|---|")
    for key, dom, share, uniq, ent in rob:
        add(f"| `{key}` | {dom} | {share:.1f} | {uniq} | {ent:.2f} |")
    add("")
    return (w, True)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(here, "out_test", "full_results.json")
    out_dir = os.path.join(here, "out_full")
    plot_dir = os.path.join(out_dir, "plots")
    lines, ok = build_section(json_path, plot_dir, iterations=1000)
    if ok:
        with open(os.path.join(out_dir, "dynamics_report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("Dynamics report written:", os.path.join(out_dir, "dynamics_report.md"))
    else:
        print(lines[0])


if __name__ == "__main__":
    main()
