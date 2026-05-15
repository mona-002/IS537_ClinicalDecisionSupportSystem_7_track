"""
IS 537 - Intelligent Diagnostic System (Track 2: Clinical Decision Support)
Subsystem: Uncertainty Reasoning (Bayesian Network)
Module: Inference Demonstration

This script runs three diagnostic scenarios on the Bayesian Network defined in
`bayesian_network.py` and generates bar charts showing how the posterior
distribution over diseases shifts as new evidence is provided.

The three scenarios target the rubric requirement:
    "Demonstration of probability shifts based on input evidence."

Scenario A: Classic flu presentation (strong, consistent evidence)
Scenario B: Partial information, with evidence added incrementally
Scenario C: Conflicting evidence (winter season vs. allergy-like symptoms)
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from pgmpy.inference import VariableElimination

from bayesian_network import build_full_network

warnings.filterwarnings("ignore")  # silence pgmpy's internal FutureWarning

DISEASES = ["influenza", "common_cold", "allergic_rhinitis", "none"]
DISEASE_LABELS = ["Influenza", "Common Cold", "Allergic Rhinitis", "None"]
DISEASE_COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#7f7f7f"]

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Helper: run inference and return posterior as a dict
# =============================================================================

def query_disease(infer, evidence):
    """
    Run inference on the BN and return P(Disease | evidence) as a dict.
    """
    posterior = infer.query(
        variables=["Disease"],
        evidence=evidence,
        show_progress=False,
    )
    # pgmpy's TabularCPD stores values in the order of state_names
    state_order = posterior.state_names["Disease"]
    return {state_order[i]: float(posterior.values[i]) for i in range(len(state_order))}


def format_evidence(evidence):
    """Return a short human-readable string for the evidence dict."""
    if not evidence:
        return "no evidence"
    parts = []
    for k, v in evidence.items():
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def print_posterior_table(label, posterior):
    """Print a posterior distribution as a formatted table."""
    print(f"  [{label}]")
    for d, prob in posterior.items():
        bar = "#" * int(prob * 40)
        print(f"    {d:20s} {prob:.4f}  {bar}")
    print()


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_posterior_progression(scenario_title, step_labels, posteriors, filename):
    """
    Grouped bar chart showing how the posterior shifts across evidence steps.

    Args:
        scenario_title (str): chart title
        step_labels (list[str]): one label per evidence step
        posteriors (list[dict]): posterior dict for each step (same order)
        filename (str): output PNG name
    """
    n_steps = len(step_labels)
    n_diseases = len(DISEASES)
    x = np.arange(n_steps)
    width = 0.8 / n_diseases

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (disease, label, color) in enumerate(zip(DISEASES, DISEASE_LABELS, DISEASE_COLORS)):
        values = [post[disease] for post in posteriors]
        offset = (i - n_diseases / 2) * width + width / 2
        bars = ax.bar(x + offset, values, width, label=label, color=color, edgecolor="black", linewidth=0.5)
        # Annotate top bar in each step with probability value
        for bar, val in zip(bars, values):
            if val >= 0.03:  # only label bars tall enough to read
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(step_labels, fontsize=9)
    ax.set_ylabel("Posterior Probability  P(Disease | Evidence)")
    ax.set_ylim(0, 1.10)
    ax.set_title(scenario_title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved chart: {out_path}")


# =============================================================================
# SCENARIO A: Classic flu presentation
# =============================================================================

def scenario_a(infer):
    print("=" * 70)
    print("SCENARIO A: Classic Flu Presentation (Winter)")
    print("=" * 70)
    print("Goal: show the network correctly converges on Influenza when given")
    print("      strong, consistent evidence.\n")

    evidence_steps = [
        ("Prior\n(no evidence)",                 {}),
        ("+ Season=winter",                      {"Season": "winter"}),
        ("+ Fever=present",                      {"Season": "winter", "Fever": "present"}),
        ("+ MuscleJointPain=present",            {"Season": "winter", "Fever": "present",
                                                  "MuscleJointPain": "present"}),
        ("+ Cough=present",                      {"Season": "winter", "Fever": "present",
                                                  "MuscleJointPain": "present", "Cough": "present"}),
    ]

    posteriors = []
    labels = []
    for label, evidence in evidence_steps:
        post = query_disease(infer, evidence)
        posteriors.append(post)
        labels.append(label)
        print_posterior_table(label.replace("\n", " "), post)

    plot_posterior_progression(
        "Scenario A: Classic Flu Presentation - Posterior Shift as Evidence Accumulates",
        labels, posteriors, "scenario_A_flu.png",
    )


# =============================================================================
# SCENARIO B: Partial information, evidence added incrementally
# =============================================================================

def scenario_b(infer):
    print("=" * 70)
    print("SCENARIO B: Partial Information (Cold vs. Allergic Rhinitis)")
    print("=" * 70)
    print("Goal: demonstrate how the system handles missing/partial data, and")
    print("      how each new piece of evidence shifts the posterior.\n")

    evidence_steps = [
        ("Prior\n(no evidence)",                  {}),
        ("Sneezing+RunnyNose\n(season unknown)",  {"Sneezing": "present", "RunnyNose": "present"}),
        ("+ Season=spring/autumn",                {"Sneezing": "present", "RunnyNose": "present",
                                                   "Season": "spring_autumn"}),
        ("+ ItchyEyes=present",                   {"Sneezing": "present", "RunnyNose": "present",
                                                   "Season": "spring_autumn", "ItchyEyes": "present"}),
        ("+ Fever=absent",                        {"Sneezing": "present", "RunnyNose": "present",
                                                   "Season": "spring_autumn", "ItchyEyes": "present",
                                                   "Fever": "absent"}),
    ]

    posteriors = []
    labels = []
    for label, evidence in evidence_steps:
        post = query_disease(infer, evidence)
        posteriors.append(post)
        labels.append(label)
        print_posterior_table(label.replace("\n", " "), post)

    plot_posterior_progression(
        "Scenario B: Partial Information - Disambiguating Cold vs. Allergic Rhinitis",
        labels, posteriors, "scenario_B_partial.png",
    )


# =============================================================================
# SCENARIO C: Conflicting evidence (winter prior vs. allergy symptoms)
# =============================================================================

def scenario_c(infer):
    print("=" * 70)
    print("SCENARIO C: Conflicting Evidence (Winter Season vs. Allergy Symptoms)")
    print("=" * 70)
    print("Goal: show the network integrates evidence rather than relying on")
    print("      season alone - symptoms can override seasonal priors.\n")

    evidence_steps = [
        ("Season=winter\nonly",                   {"Season": "winter"}),
        ("+ Sneezing=present",                    {"Season": "winter", "Sneezing": "present"}),
        ("+ ItchyEyes=present",                   {"Season": "winter", "Sneezing": "present",
                                                   "ItchyEyes": "present"}),
        ("+ Fever=absent",                        {"Season": "winter", "Sneezing": "present",
                                                   "ItchyEyes": "present", "Fever": "absent"}),
        ("+ MuscleJointPain=absent",              {"Season": "winter", "Sneezing": "present",
                                                   "ItchyEyes": "present", "Fever": "absent",
                                                   "MuscleJointPain": "absent"}),
    ]

    posteriors = []
    labels = []
    for label, evidence in evidence_steps:
        post = query_disease(infer, evidence)
        posteriors.append(post)
        labels.append(label)
        print_posterior_table(label.replace("\n", " "), post)

    plot_posterior_progression(
        "Scenario C: Conflicting Evidence - Symptoms Override Seasonal Prior",
        labels, posteriors, "scenario_C_conflict.png",
    )


# =============================================================================
# Main
# =============================================================================

def main():
    print("\nBuilding Bayesian Network...\n")
    bn = build_full_network()
    infer = VariableElimination(bn)

    scenario_a(infer)
    scenario_b(infer)
    scenario_c(infer)

    print("=" * 70)
    print("All three scenarios complete.")
    print(f"Charts saved in: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
