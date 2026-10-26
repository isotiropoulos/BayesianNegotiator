"""Strategy-agnostic safety mechanisms for ODRL negotiation baselines.

This package exists to answer a specific methodological objection to the BAMBON
evaluation: BAMBON enforces its hard policy at commit time and repairs its own
proposals, while the five reference baselines do not, so *any* compliance gap
between them could be explained by the presence of that mechanism alone rather
than by BAMBON's Bayesian machinery.

:mod:`src.bambon.baselines.veto` supplies exactly that mechanism as a wrapper
that can be placed around **any** agent implementing the project's
``observe(offer) -> (accept, counter)`` protocol. It is deliberately
strategy-agnostic: it consults only (i) the agent's own hard policy and (ii) the
ODRL satisfaction predicate of the problem formulation. It contains no beliefs,
no opponent model, no utility function and no BAMBON code.
"""
from .veto import (  # noqa: F401
    HardPolicyVeto,
    feasibility_report,
    is_feasible,
    project_offer,
    veto_wrap,
)

__all__ = [
    "HardPolicyVeto",
    "feasibility_report",
    "is_feasible",
    "project_offer",
    "veto_wrap",
]
