from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple, List, Set

IssueKey = Tuple[str, str]  # (leftOperand, myOperator)

@dataclass
class BaseModel:
    def observe(self, x: Any, operator: str, lr: float = 1.0) -> None:
        """Update belief from evidence x under a given operator."""
        raise NotImplementedError

    def sample(
        self,
        x: Any = None,
        operator: str = "eq",
        inertia_val: Any = None,
        inertia: float = 0.0,
        **kwargs,
    ):
        """Propose a new value (or set/label)."""
        raise NotImplementedError

    def accept(self, x: Any, operator: str = "eq", **kwargs):
        """Decide whether to accept x under given operator."""
        raise NotImplementedError

@dataclass
class Issue:
    """Belief model container with adapters to/from rightOperand representation.

    A single Issue represents one negotiable unit for a given (action, operand).
    When ``merged`` is True it carries the combined view of *several* ODRL
    constraints that share the same operand within the same action:
      - ``operator`` / ``lastValue`` describe the inclusion clause (eq/isAnyOf/isAllOf)
      - ``forbidden`` holds the union of exclusion clauses (neq/isNoneOf)
    and the analogous opponent view is kept in ``oppOperator`` / ``oppLastValue`` /
    ``opp_forbidden``. Non-merged issues keep ``forbidden`` empty and behave exactly
    as before.
    """
    kind: str  # 'num' | 'enum' | 'set'
    operand: str
    operator: str
    initOperator: str
    oppOperator: str | None
    lastValue: Any
    initValue: Any
    oppLastValue: Any
    model: BaseModel
    to_internal: Any
    to_external: Any
    hard_rule: Any | None
    accepted: bool = False
    # The term actually agreed when `accepted` was set: the counterparty's
    # operator and value. Kept separately from `lastValue` so that accepting an
    # issue does not have to overwrite the agent's own standing demand. See
    # BAMBON_NONDESTRUCTIVE_ACCEPT.
    agreedOperator: Any | None = None
    agreedValue: Any | None = None

    # Per-issue alignment similarity sigma_i in [0,1], refreshed each round by
    # Bambon._context_and_severity. Used to detect a lone laggard issue so its
    # concession can be accelerated (1 - similarity = issue-level severity / gap).
    similarity: float = 1.0

    # --- merged-constraint support (multiple clauses on the same operand) ---
    merged: bool = False
    forbidden: Set[str] = field(default_factory=set)       # my exclusion labels
    opp_forbidden: Set[str] = field(default_factory=set)   # opponent exclusion labels

    # --- merged numeric range support (kind == 'numrange') ---
    model_hi: BaseModel | None = None   # belief for the upper bound (model holds the lower)
    lower_op: str | None = None         # 'gt' / 'gteq'
    upper_op: str | None = None         # 'lt' / 'lteq'

class HardConstraints:
    """Belief model container with adapters to/from rightOperand representation."""
    operand: str
    operator: str
    value: Any
    reason: str

class NegotiationInfeasible(Exception):
    pass

@dataclass
class ActionState:
    lr: float
    inertia: float
    tau_set: List[float, float] # [tau_in, tau_out]
    max_set_k: int
    jaccard_snap_threshold: float

    # Converge metrics
    context_score: float = None
    severity: float = None

    # --- utility-aware policy (opt-in; refreshed each round by Bambon) ---
    # Defaults keep the original behaviour (utility_mode off).
    utility_mode: bool = False
    aspiration: float = 1.0       # target self-utility to hold out for this round
    p_floor: float = 0.5          # min opponent-accept prob a demanding bid must clear
    round_frac: float = 0.0       # rounds_seen / deadline (always set, even when utility_mode is off)
    deadline_beta: float = 2.0    # shape of the deadline pressure on rho_eff (>1 = late ramp / Boulware)
    time_cost: float = 0.02       # per-round cost charged in accept-vs-wait (#7)

    # --- accelerated concession on a lone holdout issue (always active) ---
    # When the rest of the action has converged (context >= concede_ctx_min) but
    # one numeric issue is still far apart (issue-severity >= concede_sev_hot),
    # nudge that issue's proposal toward the opponent by an extra
    # clamp(concede_gain * s_i * round_frac, 0, concede_step_cap) fraction of the
    # remaining gap. Defaults are conservative; result is still clamped to hard.
    concede_gain: float = 0.5
    concede_step_cap: float = 0.34
    concede_ctx_min: float = 0.6
    concede_sev_hot: float = 0.5
    reservation_span: float = 0.5 # fallback reservation width (fraction of belief span)

    # --- ablation: where the *bid* comes from -----------------------------------
    # 'belief' (default, the full model) -- the proposal consults the opponent
    #     posterior: numeric bids walk down the utility grid until the estimated
    #     opponent-acceptance probability clears the round's floor, and
    #     categorical bids are drawn from the per-label posterior.
    # 'time'  -- the opponent model is removed from bidding *only*. The agent
    #     still updates its posteriors and still uses them to *accept*; the bid
    #     becomes a memoryless, time-scheduled concession computed from the
    #     agent's own aspiration, its own reservation and the opponent's last
    #     offer. Every hyper-parameter, the concession schedule, the repair
    #     projection and the acceptance test are unchanged, so the difference
    #     between the two settings isolates the contribution of Bayesian
    #     opponent modelling to *bidding*.
    bidding_mode: str = "belief"

    # --- ablation switch: opponent modelling in ACCEPTANCE (BAMBON-OM(acc)) ---
    # "belief" (default): the acceptance test is `issue.model.accept(...)`, which
    #     samples the posterior to decide whether holding out is likely to beat
    #     the term on the table.
    # "blind":  the posterior is never consulted to accept. The test becomes the
    #     classic time-dependent rule -- accept a term at least as good as the one
    #     I am currently asking for -- which is time-dependent because my own term
    #     rides the aspiration schedule A(t). Posteriors are still updated and are
    #     still used to *bid*, so the difference between the two settings isolates
    #     the contribution of Bayesian opponent modelling to *acceptance*, the
    #     complement of `bidding_mode`.
    # The hard-policy veto and the repair projection are untouched in both, so
    # Proposition 1 holds identically and compliance is not what this arm tests.
    accept_mode: str = "combi"

    # --- opt-in stricter commit-time semantics on categorical issues ---
    # False (default): a committed inclusion term satisfies a hard allow-list by
    #     *overlap*, as `sat` defines it and as Def. 2 scores it.
    # True: the committed label set must be *contained* in the allow-list. This
    #     removes the residual over-broad commitments BAMBON can otherwise sign
    #     (it accepts the opponent's clause verbatim), at the price of a
    #     different notion of violation from the one the paper reports.
    strict_categorical_commit: bool = False

    # --- ablation: the commit-time hard-policy veto (BAMBON-NoVeto) ---
    # True (default): a term breaching the agent's own hard policy is never
    #     signed. This is Prop. 3.2 and it is what makes C_agr 1.000.
    # False: the veto is removed while the belief, the bidding policy and the
    #     repair projection are untouched, so the arm answers the question a
    #     reader is entitled to ask -- is perfect compliance produced by the
    #     mechanism under test, or by the harness that wraps it?
    hard_veto: bool = True

    # rolling histories per action
    _hopeless_streak: int = 0
    _num_move_hist: List[float] = field(default_factory=list)
    _opp_set_sizes: List[int] = field(default_factory=list)

