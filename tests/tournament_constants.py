"""Constants for the ANL / BAMBON ODRL benchmark tournament."""
from __future__ import annotations

from typing import Dict, Tuple

from utility import UtilityConfig

RULE_TYPES = ["permission", "obligation", "prohibition"]
NUM_OPS = ("lt", "lteq", "gt", "gteq", "eq", "neq")

MAX_TURNS = 200
DEFAULT_N_RUNS = 100
SCENARIO = "full"
OUT_SUBDIR = "out_anl"

import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

ODRL_UTILITY_CONFIG = UtilityConfig()

BAMBON_PROFILES: Dict[str, Dict] = {
    "bambon_balanced": {
        "lr": 0.50, "inertia": 0.55, "tau_in": 0.60, "tau_out": 0.74,
        "max_set_k": 3, "no_agreement_until": 3, "hopeless_window": 16,
        "jaccard_snap_threshold": 0.45, "max_rounds": 150,
        "tau_cooling_threshold": 0.55, "tau_in_cooled": 0.60, "tau_out_cooled": 0.56,
        "lr_min": 0.32, "lr_max": 0.95, "vol_window": 5, "max_param_delta": 0.12,
        "hopeless_severity": 0.98, "hopeless_patience": 24, "hopeless_context": 0.06,
    },
    "bambon_aggressive": {
        "lr": 0.42, "inertia": 0.60, "tau_in": 0.72, "tau_out": 0.84,
        "max_set_k": 3, "no_agreement_until": 4, "hopeless_window": 20,
        "jaccard_snap_threshold": 0.46, "max_rounds": 160,
        "tau_cooling_threshold": 0.55, "tau_in_cooled": 0.62, "tau_out_cooled": 0.58,
        "lr_min": 0.32, "lr_max": 0.95, "vol_window": 5, "max_param_delta": 0.09,
        "hopeless_severity": 0.99, "hopeless_patience": 30, "hopeless_context": 0.04,
    },
    "bambon_exploratory": {
        "lr": 0.85, "inertia": 0.46, "tau_in": 0.58, "tau_out": 0.70,
        "max_set_k": 4, "no_agreement_until": 3, "hopeless_window": 14,
        "jaccard_snap_threshold": 0.42, "max_rounds": 150,
        "tau_cooling_threshold": 0.50, "tau_in_cooled": 0.58, "tau_out_cooled": 0.54,
        "lr_min": 0.36, "lr_max": 1.1, "vol_window": 4, "max_param_delta": 0.20,
        "hopeless_severity": 0.97, "hopeless_patience": 26, "hopeless_context": 0.07,
    },
}

ANL_DEADLINE = 80
N_NUMERIC_LEVELS = 6
MAX_OUTCOME_CARDINALITY = 500

ANL_WINNER_IMPORTS: Tuple[Tuple[str, str, str], ...] = (
    ("Shochan", "anl_agents.anl2024.takafam.shochan", "Shochan"),
    ("UOAgent", "anl_agents.anl2024.team_moto.uoagent", "UOAgent"),
    ("AgentRenting2024", "anl_agents.anl2024.team_renting.renting2024", "AgentRenting2024"),
)

AGENT_RENTING_MODEL_FILES = (
    "model1-v4_phase2.keras",
    "model1-v4_history_scaler.joblib",
    "model1-v4_features_scaler.joblib",
)
