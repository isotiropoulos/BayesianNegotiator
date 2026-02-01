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

- `negotiator.py`  
  Main implementation: `Bambon` (observe / counter-offer / agreement logic)

- `belief.py`  
  `BeliefRegistry` manages belief models per action & issue

- `distributions.py`  
  Bayesian belief distributions:
  - numeric beta-based model
  - Dirichlet model for categorical constraints
  - set belief handling

- `models.py`  
  Core dataclasses and `NegotiationInfeasible` exception

- `helpers.py`  
  Constraint parsing + operator helpers

- `context.py`  
  Similarity + severity scoring

- `expressions.py`  
  Expression helpers for constraints

- `test_negotiator.py`  
  Negotiation simulations / profiles (balanced, aggressive, exploratory)

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

## Example Minimal Negotiation Loop

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

If negotiation is judged impossible, Bambon raises:

```python
from models import NegotiationInfeasible
```

---

## Output Types

Bambon produces ODRL-like structures:

- **Counter-offer**: `@type = "odrl:Offer"`
- **Agreement**: `@type = "odrl:Agreement"` with constraints filled using accepted values

---
