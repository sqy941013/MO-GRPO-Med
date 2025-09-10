"""
Lightweight facade: re-exports reward APIs after splitting the original single-file
`reward_functions.py` into multiple modules under `models.rewards`.

Backwards compatibility: both `from reward_functions import ...` and
`from models.reward_functions import ...` will continue to work.
"""

from .rewards import (
    # rewards
    compute_r_struct_reward,
    compute_r_cover_reward,
    compute_r_medfact_reward,
    compute_r_style_reward,
    compute_total_reward,
    compute_think_format_reward,
    compute_think_gated_total_reward,
    # helpers
    calculate_risk_weighted_f1,
    calculate_combined_precision,
    calculate_r_cover,
    evaluate_medfact_for_ie_results,
    calculate_r_medfact,
    evaluate_style_for_text,
    calculate_advanced_style_score,
    compute_c_sec,
    compute_A_sec_gold,
    compute_final_r_struct,
    NINE_KEYS,
)

# Backward-compatible alias
total_reward = compute_total_reward

__all__ = [
    # rewards
    "compute_r_struct_reward",
    "compute_r_cover_reward",
    "compute_r_medfact_reward",
    "compute_r_style_reward",
    "compute_total_reward",
    "compute_think_format_reward",
    "compute_think_gated_total_reward",
    # alias
    "total_reward",
    # helpers
    "calculate_risk_weighted_f1",
    "calculate_combined_precision",
    "calculate_r_cover",
    "evaluate_medfact_for_ie_results",
    "calculate_r_medfact",
    "evaluate_style_for_text",
    "calculate_advanced_style_score",
    "compute_c_sec",
    "compute_A_sec_gold",
    "compute_final_r_struct",
    "NINE_KEYS",
]


