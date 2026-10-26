# Bambon Negotiator

**Bambon** (*Bayesian Adaptive Mixed-type Bilateral ODRL-policy Negotiator*) is a learning-based negotiator for **negotiating constraints over one or multiple actions** (e.g., `use`, `read`, `share`, `transfer`) expressed in an **ODRL-style policy format**.

Bambon observes opponent offers, updates Bayesian beliefs about opponent preferences (numeric / enum / set issues), and generates counter-offers until:
- agreement is reached (`odrl:Agreement`), or
- the negotiation becomes infeasible (`NegotiationInfeasible`)

---

## Features

- ✅ Negotiates **constraints per action** (single-action or multi-action offers)
- ✅ Supports **mixed-type issues**:
  - numeric constraints
  - categorical/enum constraints
  - set-valued constraints
- ✅ Bayesian belief models to learn opponent preferences over time
- ✅ Hard policy enforcement (non-negotiable constraints)
- ✅ Context similarity & severity scoring
- ✅ Hopelessness detection (fails fast if negotiation diverges)

---

## Repository Structure

```
src/bambon/
  negotiator.py          Bambon: observe / counter-offer / agreement logic
  distributions/
    belief.py            BeliefRegistry, one belief model per action and issue
    distributions.py     numeric Beta, categorical Dirichlet, set Beta-Bernoulli
  models/
    models.py            core dataclasses and NegotiationInfeasible
    expressions.py       expression helpers for constraints
  utils/
    helpers.py           constraint parsing and operator helpers
    context.py           similarity and severity scoring
    utility.py           the ODRL utility model
  baselines/veto.py      veto-and-repair shim, usable around any strategy
  negmas_negotiator.py   ODRLOutcomeSpace and the NegMAS adapter

tests/                   unit tests, and the scoring code the benchmark shares
data/50scenarios/        50 hand-authored scenarios across 10 sectors
data/generated/          the five generated corpora the tournament runs on
data/generator/          the seeded generator and its feasibility oracle
examples/                edc-dataspace-demo: the end-to-end EDC connector demo
paper/                   the ODRL 2026 paper and its sweeps
odrl-tournament/results/agentodrl/  raw results of the ESWA paper (p0, p32)
odrl-tournament/         the benchmark harness, as a submodule
archive/                 superseded runs and old baseline code; safe to delete
```

---

## Offer Format (ODRL-style)

Bambon expects offers in an ODRL-like JSON format:

```json
{
  "@id": "urn:uid:example",
  "@type": "odrl:Offer",
  "permission": [
    {
      "action": "use",
      "constraint": [
        {
          "leftOperand": "odrl:purpose",
          "operator": "odrl:eq",
          "rightOperand": "research"
        }
      ]
    }
  ]
}
```

Constraints are normalized internally (e.g., `odrl:` prefixes stripped during parsing).

---

## Quickstart

### 1) Create a negotiator

```python
from negotiator import Bambon

neg = Bambon(
    name="Provider",
    init_offer=init_offer,        # required
    hard_policy=hard_policy       # optional
)
```

Each action in `init_offer["permission"]` gets its own:
- belief registry
- negotiation state

---

### 2) Observe an opponent offer and generate a response

```python
signals, response_offer = neg.observe(opponent_offer)

print("Signals:", signals)
print("Response:", response_offer)
```

- `signals` is `None` while negotiating
- `signals` becomes non-null when an action is accepted and agreement is formed
- `response_offer` will be either:
  - a counter-offer (`odrl:Offer`), or
  - an agreement (`odrl:Agreement`)

---

## Running

### Run tests (recommended)

```bash
pytest -q
```

or:

```bash
python -m unittest -v
```

---

### Example Minimal Negotiation Loop

```python
from negotiator import Bambon

provider = Bambon(name="Provider", init_offer=provider_init, hard_policy=provider_hard)
consumer = Bambon(name="Consumer", init_offer=consumer_init)

offer = consumer_init
for round_i in range(50):
    sig_p, offer = provider.observe(offer)
    if offer.get("@type") == "odrl:Agreement":
        print("Agreement reached (provider -> consumer)")
        break

    sig_c, offer = consumer.observe(offer)
    if offer.get("@type") == "odrl:Agreement":
        print("Agreement reached (consumer -> provider)")
        break
else:
    print("No agreement reached within max rounds")
```

---

## Key Negotiation Parameters

Bambon behavior is governed by tunable controls, including:

- `lr` — learning rate for belief updates
- `inertia` — resistance to changing past proposals
- `tau_in`, `tau_out` — thresholds controlling adaptation vs acceptance
- `max_set_k` — maximum size for proposed sets
- `jaccard_snap_threshold` — snaps set proposals when overlap is high
- `no_agreement_until` — prevents early agreement before minimum rounds
- `max_rounds` — maximum rounds before infeasibility
- `hopeless_window`, `hopeless_patience` — detects non-converging negotiation

Negotiation terminates with `NegotiationInfeasible` when:

- maximum number of rounds is exceeded, or
- persistent low context and high severity indicate divergence

This enables early stopping in non-converging negotiations.

---
## Negotiation Semantics

Bambon operates under incomplete and asymmetric information, modeling opponent preferences through Bayesian belief updates.

Each issue is internally represented in a computational domain (numeric, categorical, or set-based), enabling:

- probabilistic acceptance decisions using Monte Carlo sampling
- operator-aware constraint compatibility checks
- adaptive proposal generation based on learned beliefs

Acceptance is not deterministic:
- numeric issues use credible intervals and probabilistic consistency
- categorical issues use Dirichlet-based argmax likelihood
- set issues use Beta-Bernoulli inclusion probabilities

Global negotiation signals:
- **context score** (overall similarity of offers)
- **severity** (degree of disagreement)

These signals influence acceptance thresholds dynamically.

---
## Multi-Action Negotiation

Each action (e.g., `use`, `read`, `share`) is negotiated independently:

- separate belief models per action
- separate convergence tracking
- combined into a single ODRL offer

An agreement is reached when all issues for at least one action are accepted.

---
## Output Types

Bambon produces ODRL-like structures:

- **Counter-offer**: `@type = "odrl:Offer"`
- **Agreement**: `@type = "odrl:Agreement"` with constraints filled using accepted values
---
## Acknowledgments

This work has been supported by the Horizon Europe research and innovation programme under the project CLIMRES (Grant Agreement No. 101147777). 

The content of this paper reflects only the authors’ views, and the European Commission is not responsible for any use that may be made of the information it contains.

---

## Experiments (ODRL 2026 Paper)

This repository accompanies the paper:

**“Constrained Adaptive Negotiation Agent of ODRL Offers Under Incomplete and Asymmetric Information”**  
2nd *ODRL and Beyond: Practical Applications and Challenges for Policy-Based Access and Usage Control*  
May 10–11, 2026, Dubrovnik, Croatia

### Running Experiments

To reproduce the experimental results:

1. Run negotiation simulations:
```bash
python test_negotiator.py
```
2. Then open and execute the notebook: `results_analysis.ipynb` 
The notebook performs the analysis and visualization of the negotiation outcomes.

---

## BAMBON Paper (Expert Systems with Applications, under review)

**BAMBON: A Policy-Compliant Bayesian Adaptive Negotiator of ODRL Agreements**
I. Sotiropoulos, I. Karvelas, E. Sarmas, S. Rizou, V. Marinakis.
Submitted to *Expert Systems with Applications*, 2026.

This section explains what the paper claims, where each part of it lives in this
repository, and exactly how to reproduce every table and figure. The paper is a
redesign of the agent presented at OPAL 2026 (previous section), not an extension:
it adds the formal model and compliance guarantees, the feasibility repair and
commit-time veto, new bidding and acceptance rules, five corpora built from
third-party ODRL offers, the comparison against published strategies, an ablation
study and a deployment between two EDC connectors.

### The problem

In a data space a connector will not transmit an asset until an ODRL agreement
exists, so producing that agreement is a bilateral, multi-issue negotiation. Each
party also holds a **hard policy**: a second ODRL document whose constraints come
from law, an ethics approval or a corporate mandate and cannot be traded away.

Published negotiators encode such limits as a **scalar reservation utility**. Over
the additive utilities they use, a scalar floor admits contracts in which a surplus
on one issue pays for a violation on another, so it cannot in general express a
conjunction of per-issue constraints. The agent then signs contracts its own
connector should refuse to execute. BAMBON instead treats the hard policy as a
constraint **checked on the negotiated contract itself**.

Formal model (paper, Section 3.1):

- An offer `O` induces its issue set `K(O) = {(rule type, action, leftOperand)}`.
  Issues are not declared in advance but read from the two exchanged documents, and
  each has a kind: `NUM`, `ENUM`, `SET` or `RANGE`. Dates and durations are mapped
  to numbers (days since epoch, seconds).
- A term is *feasible* for party `i` if it satisfies every constraint its hard
  policy `H_i` states on that issue, under the ODRL satisfaction predicate. `F_i`
  is the product of the per-issue feasible sets.
- An agreement `G` *complies* for `i` iff `G` is in `F_i`, i.e. the number of
  violated hard clauses `nu_i(G)` is 0.

Compliance is reported twice. `C^agr` is the share of the contracts an agent signed
that it may lawfully perform. `C^all` is the share of all encounters that did not
end in a breach, so a refusal counts as compliant. Both are maximised by never
agreeing, so the headline metric is **feasible deals closed** = agreement rate x
`C^agr`.

### The agent, mechanism by mechanism

One BAMBON round (Algorithm 1 in the paper):

| Step | Paper | What it does | Code |
|---|---|---|---|
| Induce issues | Sec. 3.1 | builds the issue set from the incoming offer and the agent's own aspiration | `src/bambon/utils/helpers.py`, `src/bambon/negmas_negotiator.py` (`ODRLOutcomeSpace`) |
| Update beliefs | Sec. 3.2 | per-issue, operator-aware posteriors over what the opponent would accept: Beta over an adaptive numeric support, Dirichlet for `ENUM`, one Beta-Bernoulli per label for `SET`. Updated only from clauses that are feasible for the agent | `src/bambon/distributions/distributions.py`, `src/bambon/distributions/belief.py` |
| Alignment signals | Sec. 3.3 | context `c_t` (mean similarity), severity `s_t` (worst stuck issue), gain `g_t = c_t(1 - s_t)`; they drive inertia, learning rate and set thresholds | `src/bambon/utils/context.py` |
| Bid | Sec. 3.4 | samples from the posterior truncated to the half-line the agent's own operator admits, or bids on the estimated acceptance frontier under a time-dependent demand `D(t)` | `src/bambon/negotiator.py`, `src/bambon/utils/utility.py` |
| Feasibility repair | Sec. 3.5 | projects every outgoing offer into `F_i` (for sets `Pi(S) = ((S n A) u R) \ F`, for numbers clamping to the hard bounds), then tests the exact document that would be signed; otherwise re-emits the last verified offer, then the opening offer, else withdraws | `src/bambon/negotiator.py`, `src/bambon/models/models.py` |
| Commit-time acceptance | Sec. 3.6 | per issue: veto if infeasible; accept if the term meets the opening demand or the two standing terms satisfy each other; otherwise accept when it clears `D(t)` or the posterior says holding out is not worth the waiting cost | same |
| Withdrawal | Sec. 3.7 | after half the deadline, withdraws when an unaccepted issue is blocked by the agent's own hard bound for several consecutive rounds | `src/bambon/negotiator.py` (`hopeless_mode="reservation"`) |

Two guarantees follow (Propositions 1 and 2, proofs in Appendix A): **every offer
BAMBON emits lies in `F_i`**, and **every agreement BAMBON is party to has
`nu_i(G) = 0`, whichever side accepts**. Neither depends on the beliefs, the
concession schedule, the elapsed time or any parameter. They do not promise that
an agreement is reached, and they say nothing about the opponent's compliance.

The compliance predicate (`count_hard_violations`) and the harness utility
(`odrl_utility`) live in `tests/benchmark_baselines_full.py` and are imported, not
copied, by the tournament, so the agent's guard and the metric that grades it are
the same function. BAMBON's published hyper-parameters are in
`odrl-tournament/odrl_tournament/profile.py`.

### Experimental design

- **Opponents.** Eight NegMAS 0.16 strategies, run unmodified: `Boulware`,
  `Linear` and `Conceder` (Faratin et al., 1998), `CAB`, `WAB`, `MiCRO` (de Jonge,
  2022), `NaiveTFT` and `Tough`. `NiceTFT` and `Hybrid` are excluded because they
  enumerate the outcome space, which here has 10^11 to 10^16 outcomes.
- **How the baselines receive their hard policy.** Two encodings, on the same
  scenarios and seeds. The *scalar-reservation setting* (`penalty: 0`) gives every
  agent a reserved value `r = 0.35`, calibrated against each scenario's witness
  contract. The *violation-penalised setting* (`penalty: 32`) charges
  `u'(w) = u(w) - lambda * nu(w) / k`. With 5 to 15 issues a violation can no
  longer be bought back once `lambda > k - 1`, so `lambda = 32` is past that
  threshold in every scenario. BAMBON ignores the penalty.
- **Corpora.** Built from 1,837 validated ODRL offers produced by the third-party
  AgentODRL generator (Zhong et al., AAAI 2026). Each offer becomes the provider's
  aspiration; the consumer's opening, both hard policies and a witness agreement
  are derived around it, and a feasibility oracle checks every scenario. Ordered
  issues are laid out as `a_p <= h_c <= h_p <= a_c`.

| Paper name | `--corpus` (directory in `data/generated_agentodrl/`) | Scenarios | Opening gap | ZOPA / gap | Split |
|---|---|---|---|---|---|
| FAR-WIDE | `far_wide` | 1837 | 1.00 | 0.25-1.00 | 50/50 |
| FAR-NARROW | `far_narrow` | 1837 | 1.00 | 0.10-0.25 | 50/50 |
| CLOSE-NARROW | `close_narrow` | 1837 | 0.25 | 0.10-0.25 | 50/50 |
| ASYMMETRIC | `far_narrow_asym` | 1837 | 1.00 | 0.10-0.25 | 20/80 |
| INFEASIBLE | `full_infeasible` | 1837 | 1.00 | empty | - |

- **Protocol.** Alternating offers, provider first, deadline `T = 100` moves
  (`steps: 50`), a 41-level value grid per numeric issue, `PYTHONHASHSEED=0`, and
  every ordered pairing of arms in both roles on every scenario.
- **Ablation arms** (Section 5), defined in
  `odrl-tournament/odrl_tournament/tournament.py`: `ScheduleBidding` (time
  schedule instead of posterior bidding), `ScheduleAcceptance` (schedule-only
  acceptance), `NoBeliefs` (both), `ContinuationAcceptance` (the OPAL 2026
  stopping rule) and `NoVeto` (commit-time tests and the mapping-layer re-check
  removed).

Two tournament runs produce every number in the paper:

| Run | Settings | Arms | Encounters per corpus | Feeds |
|---|---|---|---|---|
| `p0` | `--penalty 0 --ablation` | BAMBON + 5 ablations + 8 baselines = 14 | 14 x 14 x 1837 = 360,052 | Table 3 (restricted to the 9 non-ablation arms), Table B.1, Figures 2, 4 and 5 |
| `p32` | `--penalty 32` | BAMBON + 8 baselines = 9 | 9 x 9 x 1837 = 148,797 | Table 4, Figure 3 |

Table 3 drops the ablation arms both as arms and as opponents, so BAMBON does not
meet five compliant near-copies of itself while no baseline does; Table B.1 keeps
the whole 14-arm tournament so every ablation faces the same opponents.

### Main results

- BAMBON never signs a contract that breaches its own hard policy: `C^agr` =
  `C^all` = 1.000 on all five corpora in both settings (148,268 agreements at
  `lambda = 0`, 62,250 at `lambda = 32`). The published agents' `C^agr` ranges from
  0.013 to 0.583.
- At `lambda = 0`, on the four feasible corpora, BAMBON closes 0.483-0.543 feasible
  deals per encounter against at most 0.137 for any published agent (3.5-4.0x),
  and has the highest compliance-discounted utility `u^c` (2.0-2.7x the best
  baseline) and Jain index on all five corpora.
- On INFEASIBLE, CAB, WAB and Conceder still sign in 85-89% of encounters, and
  98.3-98.6% of those contracts violate the signer's own policy.
- At `lambda = 32`, five baselines rise to `C^agr` = 0.975-0.994 by giving up
  46-98% of their deals, none reaches 1.000, and BAMBON still closes the most
  compliant contracts on every feasible corpus.
- Ablations: only `NoVeto` moves compliance (down to 0.737-0.896), so the guarantee
  comes from the commit-time check. Removing the posterior from bidding closes
  fewer and less balanced deals. The acceptance-rule ablations move results by a
  few thousandths.
- Deployment (Section 6): two Eclipse Dataspace Components connectors exchanging
  Dataspace Protocol messages reached `FINALIZED` agreements with both hard
  policies respected. See `examples/edc-dataspace-demo/`.

### Reproducing the paper

Every command below starts from the repository root. Where Windows PowerShell and
bash differ, both are given. Keep the quotes around glob patterns and around
`{corpus}`: the scripts expand them themselves, and PowerShell would otherwise
read `{corpus}` as a script block.

#### 1. Environment

Python 3.10 or newer (the published runs used CPython 3.13 on Windows).

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r odrl-tournament\requirements.txt
pip install pydantic scipy
```

bash:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r odrl-tournament/requirements.txt
pip install pydantic scipy
```

The first install brings the harness dependencies (`negmas` 0.16.x, `pandas`,
`numpy`, `pyyaml`, `matplotlib`); the second brings BAMBON's own. Nothing needs to
be pip-installed from this repository: the harness finds BAMBON as the parent
directory of `odrl-tournament/`. Set the environment variable `BAMBON_ROOT` only
if you move the harness elsewhere.

#### 2. Smoke test (about a minute)

```bash
cd odrl-tournament
python -m odrl_tournament run --config configs/quick.yaml
```

This plays BAMBON, Boulware and Conceder on four `far_wide` scenarios (36
encounters) and writes `results/quick.csv`. Runs resume from their output file, so
delete `results/quick.csv` to run it again.

#### 3. Rebuild every table and figure from the shipped results

The raw per-encounter results of both runs ship in
`odrl-tournament/results/agentodrl/p0/` and `odrl-tournament/results/agentodrl/p32/`,
one CSV per corpus. Rebuilding needs no negotiation and takes a few minutes:

```bash
cd odrl-tournament
python analyse_results.py --results "results/agentodrl/p0/*.csv"  --out analysis/p0
python analyse_results.py --results "results/agentodrl/p32/*.csv" --out analysis/p32
python paper_figures.py --p0 analysis/p0 --p32 analysis/p32 --out paper_figs
```

Where each paper element ends up:

| Paper | File |
|---|---|
| Table 3 | `analysis/p0/tables/metrics_<corpus>.md` (LaTeX body: `analysis/p0/tables/metrics.tex`) |
| Table 4 | `analysis/p32/tables/metrics_<corpus>.md` (LaTeX body: `analysis/p32/tables/metrics.tex`) |
| Table B.1 | `analysis/p0/tables/ablation_<corpus>.md` (BAMBON's absolute row, then each ablation as a delta) |
| Figure 2 | `paper_figs/fig_frontier.pdf` |
| Figure 3 | `paper_figs/fig_penalty.pdf` |
| Figure 4 | `paper_figs/fig_ablation.pdf` |
| Figure 5 | `paper_figs/fig_ablation_quality.pdf` |

`analyse_results.py` also writes heatmaps and diagnostic plots under
`analysis/<run>/figures/`. The `ms` column of the metrics tables is wall-clock time
per encounter.

#### 4. Rerun the tournaments from scratch

`configs/eswa.yaml` is the paper's configuration file. It sets `penalty: 32`,
so the `p0` run overrides it with `--penalty 0`. Write to a new directory:
pointing `--out` at the shipped CSVs would make the runner treat every scenario
as already done.

PowerShell:

```powershell
cd odrl-tournament
$corpora = "far_wide", "far_narrow", "close_narrow", "far_narrow_asym", "full_infeasible"
foreach ($c in $corpora) {
    python -m odrl_tournament run --config configs/eswa.yaml --corpus $c `
        --penalty 0 --ablation --out "results/rerun/p0/{corpus}.csv"
}
foreach ($c in $corpora) {
    python -m odrl_tournament run --config configs/eswa.yaml --corpus $c `
        --penalty 32 --out "results/rerun/p32/{corpus}.csv"
}
```

bash:

```bash
cd odrl-tournament
for c in far_wide far_narrow close_narrow far_narrow_asym full_infeasible; do
  python -m odrl_tournament run --config configs/eswa.yaml --corpus "$c" \
    --penalty 0 --ablation --out "results/rerun/p0/{corpus}.csv"
done
for c in far_wide far_narrow close_narrow far_narrow_asym full_infeasible; do
  python -m odrl_tournament run --config configs/eswa.yaml --corpus "$c" \
    --penalty 32 --out "results/rerun/p32/{corpus}.csv"
done
```

Then build the tables and figures as in step 3, replacing `results/agentodrl/` with
`results/rerun/`.

**Cost.** Each corpus takes roughly 15 CPU-hours for `p0` and 17 for `p32` (the sum
of per-encounter wall time in the shipped `far_wide` files), so about 160
CPU-hours for both runs. Progress is checkpointed per scenario, and rerunning the
same command resumes where it stopped.

**Sharding.** Split a corpus by scenario index with `--lo`/`--hi` (1837 scenarios,
indices `0` to `1836`), give each shard its own `--out`, run the shards in separate
terminals, then concatenate. Two runs must never share an output file; the runner
locks it and refuses to start if the lock is held (`--force` overrides a stale
lock). Example, four shards of `far_wide` for `p0`:

```bash
cd odrl-tournament
# terminal 1 (terminals 2-4: --lo 460 --hi 920, --lo 920 --hi 1380, --lo 1380 --hi 1837)
python -m odrl_tournament run --config configs/eswa.yaml --corpus far_wide \
  --penalty 0 --ablation --lo 0 --hi 460 --out results/rerun/p0/shards/far_wide_0.csv

# after all four finish
python -c "import glob, pandas as pd; pd.concat(pd.read_csv(f) for f in sorted(glob.glob('results/rerun/p0/shards/far_wide_*.csv'))).to_csv('results/rerun/p0/far_wide.csv', index=False)"
```

In PowerShell, replace the trailing `\` with a backtick, or put the command on one
line. Keep shards in a subdirectory so the step-3 glob does not pick them up as
extra corpora.

**Reproducibility.** The runner re-launches itself with `PYTHONHASHSEED=0` and each
encounter is seeded from the scenario, so a rerun on the same platform and library
versions reproduces the shipped CSVs. On a different OS, Python or SciPy version a
small share of encounters can differ (a check of one `far_wide` scenario on Linux
with Python 3.10 matched 188 of its 196 encounters exactly). The compliance
guarantees do not depend on this; exact table values do, which is why the raw
results ship with the repository.

#### 5. Rebuild the corpora (optional)

The corpora need the AgentODRL offer set (`offers_negotiation.json`, built with
`ODRL_V3/build_offer_dataset.py --validate --synthesise-parties` in the AgentODRL
repository). Write them to a new directory so the shipped corpora stay intact:

```bash
python data/odrl_corpora/generate_from_odrl.py -i <path-to>/offers_negotiation.json -o data/my_corpora
cd odrl-tournament
python -m odrl_tournament run --config configs/eswa.yaml --corpus far_wide \
  --penalty 0 --ablation --scenarios-dir my_corpora --out "results/mine/p0/{corpus}.csv"
```

The default seed (`--seed 20260914`) is the one behind the published corpora.
`data/odrl_corpora/README.md` documents the geometry, the issue floor (padding to
4 ordered and 8 total issues) and the remaining options. To inspect a corpus:
`python -m odrl_tournament geometry far_wide`.

#### 6. EDC connector deployment (Section 6)

See `examples/edc-dataspace-demo/README.md`. With Docker, `./run_demo.sh` from that
directory brings up two connectors and runs the negotiation and data pull (on
Windows it needs Git Bash or WSL). Without connectors,
`python -m bambon_edc.consumer_flow --offline` from the same directory runs the
bilateral loop in-process and prints the EDC policy bodies it would post.
