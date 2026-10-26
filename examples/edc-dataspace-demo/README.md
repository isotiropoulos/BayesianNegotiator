# BAMBON between two data space connectors

Two [Eclipse Dataspace Components](https://eclipse-edc.github.io/docs/) connectors
running locally. One asks the other for its catalog, gets a single offer it cannot
live with, and counter-offers. [BAMBON](../../README.md) runs on **both** sides:
the provider counters back, the consumer counters again, and when they agree the
consumer pulls the dataset under that agreement.

---

## The problem this solves

EDC's contract negotiation is **take-it-or-leave-it**. A consumer must send a
`ContractRequestMessage` whose policy is byte-equal to one already published in
the provider's catalog; anything else is terminated. The Dataspace Protocol does
define an `OFFERED` state for provider counter-offers, but EDC does not implement
the iterative loop.

So in a stock data space, a consumer whose needs differ from the published offer
has exactly two options: accept terms that do not work, or walk away. There is no
bargaining.

BAMBON is a bargaining agent. This example shows where it fits, and — the part
that makes it more than a demo of an algorithm — that **the negotiated agreement
is the one the data plane honours**.

---

## The flow

```
 consumer connector                                   provider connector
 ──────────────────                                   ──────────────────
  1  ─── DSP catalog request ──────────────────────────────►
     ◄── one offer: purpose=research, ≤30d, EU ──────────────

  2  consumer BAMBON: violates our hard policy (commercial, ≥60d).
     Requesting this verbatim would bind us to unusable terms.

  3  ─── DSP ContractRequestMessage ───────────────────────►
         carrying the consumer's counter-offer
     ◄── TERMINATED  (policy ≠ published offer) ─────────────

  4                       provider BAMBON reads the rejected request from its
                          OWN connector, counters, and republishes its catalog
                          entry — that republication IS the counter-offer

  5  … 1–4 repeat, both agents updating beliefs, ~6 rounds …

  6  consumer BAMBON accepts, requests the published terms verbatim
     ◄── FINALIZED → contract agreement id ─────────────────

  7  ─── transfer process (HttpData-PULL, quoting that agreement) ──►
     ◄── EndpointDataReference ─────────────────────────────
     ─── GET the provider's data plane ────────────────────►
     ◄── the dataset ───────────────────────────────────────
```

Every proposal in steps 3–4 is real Dataspace Protocol traffic between the two
connectors. **The BAMBONs never talk to each other** — each only reads and writes
its own connector's Management API.

The counter-offer EDC lacks is emulated with the two operations it does support:
*request* (consumer side) and *republish* (provider side). The consumer's
proposals ride inside genuine `ContractRequestMessage`s; the provider's ride in
the catalog.

Step 7 quotes the contract agreement id from step 6. That matters: pulling data
after separately accepting the published offer would demonstrate nothing about
the negotiation.

---

## Running it

Requires Docker and Docker Compose. From this directory:

```bash
./run_demo.sh
```

That generates keystores, brings up two connectors, a RabbitMQ broker, edcpy's
consumer backend and both BAMBON services; seeds the provider with its restrictive
published offer; then runs the flow above and prints the transcript.

Clean up with `docker compose down -v`.

To re-run just the flow:

```bash
docker compose run --rm runner
```

**Re-seed between runs for a clean scenario.** The provider retires its previous
contract definition every time it counter-offers, so a second run without
re-seeding starts from wherever the last one left the catalog. The flow says so
on start-up when it notices, and prints the re-seed command.

On Windows this needs Git Bash or WSL — `run_demo.sh` is bash and shells out to
`openssl`. The `docker compose` commands underneath work anywhere.

Connector hostnames are Compose service names, so no `/etc/hosts`
`host.docker.internal` entry is needed (unlike the upstream example).

### Without a data space

The bargaining and the ODRL↔EDC normalisation need no connectors at all:

```bash
pip install -e ../..           # BAMBON
pip install numpy scipy pytest
pytest tests/ -q               # everything except the live connector test
python -m bambon_edc.consumer_flow --offline
```

`--offline` runs the bilateral loop in-process and prints the exact
`PolicyDefinition` and `ContractDefinition` bodies that would be POSTed. Add
`--dialect tractusx` to see the same agreement rendered for a Tractus-X connector
instead.

`tests/test_two_connector_flow.py` is the useful one here: it runs the **whole
seven-step flow** against fake connectors that enforce what real EDC enforces
(see below).

---

## Two arrangements, and which one to cite

| | `two_connector_flow` | `consumer_flow` + `service` |
| --- | --- | --- |
| where counter-offers travel | across DSP, between the connectors | private HTTP between the two BAMBONs |
| provider's counter-offer is | a republished catalog entry | a direct HTTP reply |
| ends with | contract agreement **+ dataset pull** | contract agreement + dataset pull |
| honest description | "BAMBON drove EDC contract negotiations" | "BAMBON agreed terms, then EDC negotiated them once" |

The side-channel variant is simpler and still useful — it exercises the bargaining
and the materialisation path without depending on how a given EDC build persists
rejected requests. Run it with
`docker compose run --rm runner-sidechannel`. But it is the weaker claim, and
`two_connector_flow` is the one to put in a paper.

---

## Connector dialects

"EDC" is not one policy format. `odrl_edc.py` carries a `Dialect` for each family,
because getting any field wrong yields either a 400 from the Management API or —
worse — a policy that is accepted and silently never matches.

| | `VANILLA` (upstream EDC) | `TRACTUSX` (Tractus-X 0.12+) |
| --- | --- | --- |
| policy `@context` | `w3.org/ns/odrl.jsonld` | Catena-X profile list |
| `leftOperand` | bare string | bare string |
| rule `constraint` | `[{"@type": "LogicalConstraint", "and": [...]}]` | `{"and": [...]}` |
| mandatory baseline | none | `Membership`, `FrameworkAgreement`, `UsagePurpose` |

Select with `BAMBON_EDC_DIALECT=vanilla|tractusx`. The Tractus-X shapes are taken
from a working deployment, not from documentation.

**The baseline is not negotiable.** Tractus-X rejects a use policy that lacks
those three constraints, so BAMBON bargains *on top of* them: the floor is
injected at materialisation time into every `use` permission, and the negotiated
operands sit alongside it. That is a real constraint on what a negotiator can
achieve in a Catena-X data space, and worth stating in any write-up — the agent is
free over the terms the profile leaves open, not over the whole policy.

---

## What a run looks like

Provider opens at research-only / 30 days / EU; consumer needs commercial for a
year:

```
  round 1
    catalog  <- permission use: purpose eq research, elapsedTime lteq 30, spatial isAnyOf ['EU']
    consumer -> permission use: purpose eq commercial, elapsedTime lteq 212.5, spatial isAnyOf ['EU']
    DSP      :: negotiation urn:uuid:691e420c… -> TERMINATED
    provider -> permission use: purpose eq research, elapsedTime lteq 30.0, spatial isAnyOf ['EU']   [read from own connector]
    provider :: republished as bambon-offer-r1-56b3e5
  …
  round 6
    catalog  <- permission use: purpose eq commercial, elapsedTime lteq 69.53, spatial isAnyOf ['EU']
    consumer -> permission use: purpose eq commercial, elapsedTime lteq 69.53, spatial isAnyOf ['EU']   [accepting]
    DSP      :: negotiation urn:uuid:57cff41c… -> FINALIZED

  transfer (agreement urn:uuid:a0688596…)
    pull     <- HTTP 200, 4271 bytes
```

Both hard policies survive: the provider never exceeds its 180-day ceiling or
leaves EU/EEA, and the consumer never accepts below its 60-day floor. The flow
prints `hard policies: both respected` and the tests pin both properties.

Keep the transcript. The operands that moved, the count of DSP contract requests,
the **contract agreement id** and the pulled byte count are the artifacts worth
citing.

---

## Layout

| Path | What it is |
| --- | --- |
| `bambon_edc/two_connector_flow.py` | **The entrypoint. The seven steps above.** |
| `bambon_edc/connector_negotiation.py` | The negotiation conducted through two EDC connectors. |
| `bambon_edc/transfer.py` | The consumer pull, bound to the negotiated agreement id. |
| `bambon_edc/odrl_edc.py` | ODRL ↔ EDC JSON-LD normalisation. Pure functions, no I/O. |
| `bambon_edc/management.py` | Thin async client for EDC's Management API. |
| `bambon_edc/profiles.py` | Both participants' opening offers and hard policies. |
| `bambon_edc/seed_provider.py` | Creates the asset and the published restrictive offer. |
| `bambon_edc/session.py` | The bilateral loop, transport-agnostic. |
| `bambon_edc/service.py` | FastAPI side channel. Symmetric — both roles run it. |
| `bambon_edc/consumer_flow.py` | The side-channel variant, for comparison. |
| `tests/fake_connectors.py` | A connector pair that enforces what EDC enforces. |
| `tests/test_two_connector_flow.py` | **The whole flow, offline, against those fakes.** |
| `tests/test_e2e_two_connectors.py` | **Two live connectors.** Skips without the stack. |
| `tests/test_odrl_edc.py` | Normalisation, both dialects. Offline. |
| `tests/test_offline_negotiation.py` | BAMBON convergence and hard-policy safety. Offline. |
| `tests/test_service_contract.py` | Static check that service and client agree on the wire format. |

---

## The two tests that carry the claim

### `tests/test_two_connector_flow.py` — offline, always runs

`tests/fake_connectors.py` implements the slice of the Management API the flow
touches, and reproduces the four behaviours that actually break demos:

1. **A fresh offer id per catalog fetch.** Real EDC mints a new base64 id every
   time, so code that caches one and quotes it later is terminated.
2. **Byte-equality on the requested policy** — FINALIZED only if the request
   matches the currently published offer; otherwise TERMINATED. This is the
   take-it-or-leave-it behaviour, enforced.
3. **Catalog = live contract definitions.** Every definition bound to the asset
   contributes a policy, so forgetting to retire the previous one produces two
   offers and the test fails.
4. **The provider sees the policy it rejected** via its own
   `/contractnegotiations/request` — and a second run with that turned off, so the
   relay fallback is tested too.

Policies travel through the real writer and parser, so the JSON-LD normalisation
is under test as well. What it does *not* prove: that EDC accepts the JSON-LD, or
that a data plane exists.

### `tests/test_e2e_two_connectors.py` — needs the stack

```bash
docker compose up -d --build --wait
docker compose run --rm --no-deps \
    -e EDC_CONNECTOR_HOST=provider-connector \
    -e EDC_CONNECTOR_MANAGEMENT_PORT=19193 \
    provider-negotiator python -m bambon_edc.seed_provider

pytest tests/test_e2e_two_connectors.py -v -s
```

It skips (never fails) when the stack is down.
`test_every_round_crossed_the_dataspace_protocol` asserts that each round produced
an EDC negotiation record and a catalog fetch, so a regression back to a private
side channel fails the suite.
`test_the_transfer_quoted_the_negotiated_agreement` asserts the pull named the
agreement id BAMBON's own contract request produced.

Run it from **inside** the compose network if you want the transfer, because the
broker and consumer backend are only reachable under their service names:

```bash
docker compose run --rm runner pytest tests/test_e2e_two_connectors.py -v -s
```

From the host, set `E2E_TRANSFER=0` — the transfer tests then skip and the
negotiation assertions still run.

### The one honest caveat, and how the tests surface it

Step 4 assumes the provider connector persists the policy of a request it
rejected. That is version-dependent. When it doesn't, the harness relays the
proposal to the provider instead and **says so on every affected round**:

```
provider read the request from its own connector in rounds: []
proposal was relayed out of band in rounds: [1, 2, 3, 4, 5]
-> describe this as 'BAMBON drove EDC contract negotiations',
   NOT 'the provider observed proposals purely via EDC'
```

Both modes still negotiate *through* the connectors — every proposal is still
carried by a real `ContractRequestMessage`, and the outcome is still a real
contract agreement. The relayed mode is just weaker about how the provider learned
what was asked. Check this line before writing the claim up.

---

## Caveats worth stating plainly

These are real limitations, not polish items. If this becomes a paper section,
they belong in it.

### 1. Agreement is not enforcement

The pull in step 7 proves the data plane honoured the *agreement*. It does not
prove the connector **evaluated the negotiated constraints** — see the next point.
Demonstrating enforcement needs a follow-up: one request the negotiated policy
permits, one it denies.

### 2. Policy functions

EDC's policy engine only evaluates a left operand it has a registered function for
(`PolicyEngine#registerFunction`). BAMBON will happily agree on `odrl:spatial` —
but if the provider connector has no `spatial` function, the published policy is
decoration.

`operands_used()` reports every operand in an agreement, and setting
`PROVIDER_REGISTERED_OPERANDS` makes the provider refuse to publish an agreement
that uses anything outside the list. Default is off, because the stock connector
image has no functions for the demo's operands — the demo shows the *negotiation,
publication and transfer* path working, not policy enforcement.

### 3. The provider's counter-offer channel is a workaround

Republishing the catalog entry is not a counter-offer in any specification. It
works because the consumer refetches every round.

Each republication is bound to a `participantId`-scoped access policy, so a
counter-offer meant for one consumer is not visible data-space wide — otherwise
the terms of a bilateral negotiation would leak. What the demo does *not* handle
is two consumers bargaining for the same asset concurrently: the provider runs one
`NegotiationSession` per process, so a second consumer would need its own session
keyed by participant id.

DSP's `OFFERED` state is the right place for all of this. Everything here is a
shim until connectors implement it.

### 4. Numeric agreements are continuous

BAMBON converges on real values, so an agreement may say `elapsedTime ≤ 69.53125`
days. EDC compares right operands as strings; `odrl_edc.py` drops a trailing `.0`
but does not round. The transcript rounds for display only. If your policy
functions parse integers, quantise before publishing.

### 5. JSON-LD shapes drift between EDC versions

Catalog output is compacted differently across releases — single-element arrays
collapse to bare objects, prefixes come and go, `leftOperand` is sometimes a string
and sometimes `{"@id": ...}`. The parser is deliberately defensive and the writer
is dialect-driven, with `id_form=True` available for connectors that want the
expanded node form.

`agmangas/edc-connector` publishes no semver tags — only `latest`, `main` and
commit SHAs. The compose file uses `latest`; set `CONNECTOR_TAG` to a SHA if you
need a reproducible build.

### 6. This demo is not your production stack

The compose stack runs **upstream/vanilla EDC** so it stays self-contained. The
INHERIT deployment is **Tractus-X 0.12.1** with Postgres, Hashicorp Vault and an
SSI wallet-stub, DSP at `/api/v1/dsp:8084`, and DID participant ids. `edcpy`
targets the upstream connector and will very likely *not* negotiate against
Tractus-X's IATP identity flow — the `edc-bridge` service uses raw `httpx` for
exactly that reason.

`BAMBON_EDC_DIALECT=tractusx` makes the *policy bodies* correct for that stack,
which is the part that matters for retargeting. The negotiation loop itself uses
`management.py` (raw Management API) so it retargets more easily than
`consumer_flow.py`, but the transfer in `transfer.py` goes through `edcpy` and
would need replacing.

### 7. Disjunctions are flattened

BAMBON models a conjunction of clauses per action. An `odrl:or` in a provider
policy is flattened into alternative clauses on the same operand, which its
merged-categorical routing handles — but a genuine disjunction across *different*
operands loses meaning. Policies using those should be filtered out before they
reach BAMBON.

### 8. Access policy vs contract policy

EDC asks two separate questions: the **access** policy decides who may see the
offer in the catalog, the **contract** policy decides what they may then do. Both
flows keep these separate — a participant-scoped access policy, plus a contract
policy carrying the negotiated terms. Reusing one document for both works on
upstream EDC but Tractus-X rejects it, so the split is deliberate rather than
tidiness.

### 9. One agreement, one contract definition

Each converged negotiation creates its own `PolicyDefinition` and
`ContractDefinition`, and each round of bargaining creates another pair. The
contract definitions are retired as the negotiation proceeds; the policy
definitions are not. A real deployment needs expiry and cleanup, which this demo
does not implement.

---

## Acknowledgments

BAMBON is supported by the Horizon Europe programme under CLIMRES (Grant Agreement
No. 101147777). `edcpy` and the `agmangas/edc-connector` image come from
[fundacionctic/connector-building-blocks](https://github.com/fundacionctic/connector-building-blocks)
(EUPL-1.2).
