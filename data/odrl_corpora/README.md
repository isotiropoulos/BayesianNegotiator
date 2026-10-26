# ZOPA corpora from real ODRL offers

`generate_from_odrl.py` builds the five BAMBON corpora from a corpus of ODRL
Offers produced by AgentODRL, instead of synthesising every document from the
operand catalogue as `data/generator/generate_scenarios.py` does.

Each input offer becomes the **provider's aspiration** `A_p`, and the remaining
five documents of a scenario are derived around it.

## Pipeline

```powershell
# 1. AgentODRL: MongoDB -> compact negotiation offers
cd ..\AgentODRL
python ODRL_V3\build_offer_dataset.py --validate --synthesise-parties

# 2. here: offers -> five ZOPA corpora
cd ..\BayesianNegotiator
python data\odrl_corpora\generate_from_odrl.py `
    -i ..\AgentODRL\ODRL_V3\result\offers_negotiation.json

# 3. tournament (reads data/generated_agentodrl by default)
cd odrl-tournament
python -m odrl_tournament run --corpus far_wide
```

Options: `--families far_wide,far_narrow` to restrict, `--seed` (default
20260914), `--limit N` for a smoke run, `--min-ordered` / `--min-issues` /
`--no-padding` for the issue floor below.

## Output

One directory per offer per family, named as in the published corpora, holding
the same six files:

```
data/generated_agentodrl/<family>/<NNNN>_<family>_<feasible|infeasible>/
    provider.json        the input offer (A_p), plus any padding
    consumer.json        the consumer's opening A_c, opposed on every issue
    hard_provider.json   H_p, the provider's non-negotiable policy
    hard_consumer.json   H_c, the consumer's non-negotiable policy
    witness.json         a point Agreement satisfying H_p and H_c
    meta.json            geometry, provenance, verification result
```

## Geometry

Every ordered issue is laid out as `a_p <= h_c <= h_p <= a_c` in the frame where
the provider prefers the small value: each party's ideal lies outside the other's
lawful range, while the lawful ranges still overlap on `[h_c, h_p]`.

| family | opening gap | ZOPA / gap | split |
|---|---|---|---|
| `far_wide` | 0.80 | 0.25-1.00 | 50/50 |
| `far_narrow` | 0.80 | 0.10-0.25 | 50/50 |
| `close_narrow` | 0.20 | 0.10-0.25 | 50/50 |
| `far_narrow_asym` | 0.80 | 0.10-0.25 | 20/80 |
| `full_infeasible` | 0.80 | empty | — |

The four feasible families share a per-case random stream, so their categorical
halves and their padding are identical and only the numeric geometry differs.

## The issue floor

A generated ODRL offer is often constraint-poor, and the ZOPA geometry lives
entirely on ordered (numeric, date, duration) issues: an offer with none has no
numeric window to place and can yield no scenario at all. By default each
scenario is padded up to **4 ordered and 8 total** issues, drawn from the same
operand catalogue the seeded generator uses and spread across permissions,
prohibitions and duties so every scenario carries all three rule types.

Padding is what makes the corpus complete: one scenario per input offer in every
family, `full_infeasible` included. `meta.json` records `n_issues_from_offer`
against `n_issues_padded` for every scenario, so the real share is always
visible. `--no-padding` uses only what the offers state, and then scenarios with
no ordered issue are dropped instead.

## Three deviations from the seeded generator

**The declared range is constructed, not read from a catalogue.** Real offers
state a value, not the operand's domain, so the range is built around `a_p` such
that it sits 10% in from its own end. That is what keeps the opening gap at 0.798
of the range. Operands with a genuine domain (`percentage`, `samplingRate`,
`confidenceLevel`) are clamped to it, and every operand is held inside its
natural bounds — durations and counts never go negative.

**`opening` is applied from the consumer's side only.** In the seeded generator
it shrinks both openings toward the centre of the declared range. Here the
provider's opening is fixed — it is the real offer's value — so scaling both ends
would leave the provider's clause in place while moving the consumer's, pushing
the lawful window off centre and inflating the gap. Anchoring on `a_p` keeps
`close_narrow` at a 50/50 split and its ZOPA fraction intact.

**Rendering is checked, not assumed.** An integer operand with a sub-unit window
rounds both hard bounds and the witness onto the same value, leaving a strict
`lt`/`gt` bound no room. Such a window is widened to two units, and if the
rendered witness still fails a bound the operator falls back to its non-strict
form, so a scenario is never emitted whose own witness does not verify.

## What the input governs

Rules with no constraints are dropped from all six documents, since they induce
no issue. `full_infeasible` needs one ordered issue whose hard bounds can cross;
categorical requirements are overlap tests and cannot be made provably
unsatisfiable, so with `--no-padding` an all-categorical offer is counted under
`skipped_no_ordered_issue` rather than written out mislabelled.

## Verification

Every scenario is checked against both hard policies as it is written, and the
result is recorded in `meta.json` (`witness_verified`, `witness_violations_*`).
Cross-check independently with the repository's own oracle:

```
python data/generator/verify_scenarios.py data/generated_agentodrl/far_wide --expect-meta
```

`structural problems: 0`, `oracle disagreements: 0` and `mislabelled: 0` is the
result to expect.

## Reading the corpus from the tournament

`odrl-tournament` reads `data/generated_agentodrl` by default. The fully
synthetic corpora stay one flag away:

```
python -m odrl_tournament run --corpus far_wide              # ODRL-derived
python -m odrl_tournament run --corpus far_wide --synthetic  # data/generated
python -m odrl_tournament run --corpus far_wide --scenarios-dir my_corpora
```

`BAMBON_SCENARIOS_DIR` sets the default for a shell, and a checkout that has
never run this generator falls back to `data/generated` automatically.
`odrl_tournament generate` always writes to `data/generated`, since it builds
the synthetic set.

One caveat when comparing to Table 1. `odrl_tournament geometry` normalises the
opening gap by the operand's range **in the seeded generator's catalogue**, which
these corpora do not use — their range is constructed around the offer's own
value. The `gap` column is therefore not comparable across the two corpus
families. `ZOPA/gap` and `split` are ratios within the scenario and are directly
comparable; on a 19-offer sample they measure 0.375 / 0.111 / 0.100 / 0.103 and
50/50, 50/50, 50/50, 20/80 against the paper's 0.386 / 0.107 / 0.108 / 0.107.
