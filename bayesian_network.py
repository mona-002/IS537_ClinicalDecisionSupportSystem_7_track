"""
IS 537 - Intelligent Diagnostic System (Track 2: Clinical Decision Support)
Subsystem: Uncertainty Reasoning (Bayesian Network)

This module builds a Bayesian Network for diagnosing three respiratory
conditions: Influenza, Common Cold, and Allergic Rhinitis.

Network structure:
    Season  ->  Disease  ->  {10 symptom nodes}

CPT values are evidence-based and grounded in peer-reviewed clinical
literature and authoritative public-health sources. Citations appear as
inline comments next to each probability and are summarized in the
references list at the bottom of this file.
"""

from pgmpy.models import BayesianNetwork as DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# =============================================================================
# STEP 1: NETWORK STRUCTURE
# =============================================================================

def build_network_structure():
    """
    Define the directed acyclic graph (DAG):
        Season -> Disease -> each symptom
    """
    symptoms = [
        "Fever",
        "MuscleJointPain",
        "Headache",
        "Cough",
        "SoreThroat",
        "Sneezing",
        "RunnyNose",
        "MildBodyAches",
        "ItchyEyes",
        "NasalCongestion",
    ]

    edges = [("Season", "Disease")]
    for symptom in symptoms:
        edges.append(("Disease", symptom))

    return DiscreteBayesianNetwork(edges), symptoms


# =============================================================================
# STEP 2: CONDITIONAL PROBABILITY TABLES (CPTs)
# =============================================================================

def build_season_cpt():
    """
    Prior over seasons. Uniform: a patient could present in any season.
    States: winter, spring_autumn, year_round.
    """
    return TabularCPD(
        variable="Season",
        variable_card=3,
        values=[[1 / 3], [1 / 3], [1 / 3]],
        state_names={"Season": ["winter", "spring_autumn", "year_round"]},
    )


def build_disease_cpt():
    """
    P(Disease | Season).

    Seasonality is grounded in:
      - WHO: influenza epidemics occur mainly during winter in temperate
        climates [Ref 1].
      - Mayo Clinic / ACAAI: allergic rhinitis (hay fever) is driven by
        pollen exposure in spring/summer/early fall [Ref 5, 6].
      - Harvard Health / CDC: rhinoviruses cause common colds year-round,
        with peaks in early fall and spring [Ref 3].

    Disease states: influenza, common_cold, allergic_rhinitis, none
    """
    # Columns: winter | spring_autumn | year_round
    values = [
        [0.40, 0.05, 0.10],   # influenza   - peaks in winter [WHO Ref 1]
        [0.30, 0.20, 0.40],   # common_cold - year-round, fall/spring peaks [Ref 3]
        [0.05, 0.50, 0.20],   # allergic_rhinitis - spring/autumn pollen [Ref 5]
        [0.25, 0.25, 0.30],   # none (healthy / other)
    ]
    return TabularCPD(
        variable="Disease",
        variable_card=4,
        values=values,
        evidence=["Season"],
        evidence_card=[3],
        state_names={
            "Disease": ["influenza", "common_cold", "allergic_rhinitis", "none"],
            "Season": ["winter", "spring_autumn", "year_round"],
        },
    )


def build_symptom_cpt(symptom_name, p_given_disease):
    """
    Build a CPT for a binary symptom node conditioned on Disease.
    States: ["present", "absent"].
    """
    diseases = ["influenza", "common_cold", "allergic_rhinitis", "none"]
    present_row = [p_given_disease[d] for d in diseases]
    absent_row = [1 - p for p in present_row]

    return TabularCPD(
        variable=symptom_name,
        variable_card=2,
        values=[present_row, absent_row],
        evidence=["Disease"],
        evidence_card=[4],
        state_names={
            symptom_name: ["present", "absent"],
            "Disease": diseases,
        },
    )


# -----------------------------------------------------------------------------
# Evidence-based symptom probabilities P(Symptom = present | Disease)
# Each value is annotated with the supporting clinical source.
# -----------------------------------------------------------------------------
SYMPTOM_PROBABILITIES = {
    # Fever
    #   Influenza: 62%-90% prevalence; ~38% of hospitalized adults present
    #     WITHOUT fever [Ref 7]. We use 0.85 to balance the WHO description
    #     "sudden onset of fever" [Ref 1] with real afebrile presentations.
    #   Common Cold: feverishness reported in <10% of adults [Ref 3].
    #   Allergic Rhinitis: hay fever does not cause fever [Ref 6].
    "Fever": {
        "influenza": 0.85,         # WHO [1]; tempered by Ref 7
        "common_cold": 0.08,       # Harvard Health / NIH [Ref 3]: <10% in adults
        "allergic_rhinitis": 0.01, # ACAAI [Ref 6]: hay fever does not cause fever
        "none": 0.03,              # baseline
    },

    # Muscle / Joint Pain
    "MuscleJointPain": {
        "influenza": 0.80,         # Novi Sad cohort [2]; WHO [1] lists as core
        "common_cold": 0.10,       # CDC: mild body aches occasionally [Ref 4]
        "allergic_rhinitis": 0.02, # ACAAI [Ref 6]: not a hay fever symptom
        "none": 0.03,
    },

    # Headache
    "Headache": {
        "influenza": 0.66,         # 12-season surveillance, n=8171 [Ref 8]
        "common_cold": 0.30,       # CDC common-cold profile [Ref 4]
        "allergic_rhinitis": 0.20, # Cleveland Clinic [Ref 6]
        "none": 0.10,
    },

    # Cough
    "Cough": {
        "influenza": 0.80,         # WHO [1]; clinical OR 5.35 [Ref 2]
        "common_cold": 0.50,       # NIH [Ref 3]: ~40% adults; CDC core symptom
        "allergic_rhinitis": 0.15, # Harvard [Ref 5]: postnasal-drip cough
        "none": 0.05,
    },

    # Sore Throat
    "SoreThroat": {
        "influenza": 0.65,         # WHO [1]; long-duration sore throat
        "common_cold": 0.55,       # Harvard 50% [Ref 3]; NIH 60% adults
        "allergic_rhinitis": 0.20, # Harvard [Ref 5]: itchy/sore throat possible
        "none": 0.05,
    },

    # Sneezing
    "Sneezing": {
        "influenza": 0.20,         # WHO [1] does not list as core
        "common_cold": 0.75,       # NIH [Ref 3]: ~60% adults; CDC [Ref 4]
        "allergic_rhinitis": 0.71, # 260-patient AR cohort [Ref 9]: 70.9%
        "none": 0.05,
    },

    # Runny Nose
    "RunnyNose": {
        "influenza": 0.45,         # WHO [1]; less prominent
        "common_cold": 0.85,       # NIH [Ref 3]; CDC core symptom [Ref 4]
        "allergic_rhinitis": 0.75, # 260-patient AR cohort [Ref 9]: 75.0%
        "none": 0.05,
    },

    # Mild Body Aches
    "MildBodyAches": {
        "influenza": 0.40,         # flu typically causes severe (not mild) aches
        "common_cold": 0.55,       # CDC [Ref 4]: listed as common cold symptom
        "allergic_rhinitis": 0.05, # not a feature [Ref 6]
        "none": 0.05,
    },

    # Itchy Eyes (allergic conjunctivitis)
    "ItchyEyes": {
        "influenza": 0.03,         # rare; conjunctivitis with some flu strains
        "common_cold": 0.10,       # NIH/CDC [Ref 3]: watery, occasionally itchy
        "allergic_rhinitis": 0.47, # 260-patient AR cohort [Ref 9]: 47.3%
        "none": 0.03,
    },

    # Nasal Congestion (blocked nose)
    "NasalCongestion": {
        "influenza": 0.35,         # WHO/PAHO [1]: common but not defining
        "common_cold": 0.80,       # NIH [Ref 3]: ~60% adults; CDC core [Ref 4]
        "allergic_rhinitis": 0.83, # 260-patient AR cohort [Ref 9]: 82.7%
        "none": 0.05,
    },
}


# =============================================================================
# STEP 3: ASSEMBLE THE NETWORK
# =============================================================================

def build_full_network():
    """Assemble the DAG, attach all CPTs, validate, and return the model."""
    model, symptoms = build_network_structure()

    model.add_cpds(build_season_cpt())
    model.add_cpds(build_disease_cpt())
    for symptom in symptoms:
        model.add_cpds(build_symptom_cpt(symptom, SYMPTOM_PROBABILITIES[symptom]))

    assert model.check_model(), "Bayesian Network failed validation"
    return model


# =============================================================================
# STEP 4: VERIFICATION + DEMO INFERENCE
# =============================================================================

def print_network_summary(model):
    print("=" * 70)
    print("BAYESIAN NETWORK SUMMARY (evidence-based CPTs)")
    print("=" * 70)
    print(f"\nNodes ({len(model.nodes())}):")
    for node in model.nodes():
        print(f"  - {node}")
    print(f"\nEdges ({len(model.edges())}):")
    for parent, child in model.edges():
        print(f"  {parent}  ->  {child}")
    print(f"\nCPDs attached: {len(model.get_cpds())}")
    print("Model valid:", model.check_model())


def quick_inference_check(model):
    print("\n" + "=" * 70)
    print("SANITY-CHECK INFERENCE")
    print("=" * 70)
    print("Evidence: Season=winter, Fever=present, MuscleJointPain=present")

    infer = VariableElimination(model)
    posterior = infer.query(
        variables=["Disease"],
        evidence={
            "Season": "winter",
            "Fever": "present",
            "MuscleJointPain": "present",
        },
        show_progress=False,
    )
    print("\nPosterior over Disease:")
    print(posterior)


if __name__ == "__main__":
    bn = build_full_network()
    print_network_summary(bn)
    quick_inference_check(bn)
    print("\n[OK] Evidence-based network built, validated, and inference works.")


# =============================================================================
# REFERENCES (IEEE format)
# =============================================================================
"""
[1] World Health Organization, "Influenza (Seasonal)," WHO Fact Sheet, 2018.
    [Online]. Available:
    https://www.who.int/news-room/fact-sheets/detail/influenza-(seasonal)

[2] V. Petrovic et al., "Diagnostic Significance of Influenza Symptoms and
    Signs, and Their Variation by Type/Subtype, in Outpatients Aged >=15
    Years: Novi Sad, Serbia," PMC, 2024. [Online]. Available:
    https://pmc.ncbi.nlm.nih.gov/articles/PMC11860240/

[3] R. B. Turner and R. Couch, "Rhinoviruses," in Mandell, Douglas, and
    Bennett's Principles and Practice of Infectious Diseases, NIH PMC,
    2010. [Online]. Available:
    https://pmc.ncbi.nlm.nih.gov/articles/PMC7150364/

[4] Centers for Disease Control and Prevention, "About Common Cold,"
    CDC, 2025. [Online]. Available:
    https://www.cdc.gov/common-cold/about/index.html

[5] Harvard Health Publishing, "Allergic Rhinitis: Your Nose Knows,"
    Harvard Medical School, 2020. [Online]. Available:
    https://www.health.harvard.edu/diseases-and-conditions/allergic-rhinitis-your-nose-knows

[6] American College of Allergy, Asthma & Immunology, "Hay Fever
    (Rhinitis): Symptoms & Treatment," ACAAI, 2023. [Online]. Available:
    https://acaai.org/allergies/allergic-conditions/hay-fever/

[7] Consensus Academic Search Engine summary citing hospital-based
    influenza-without-fever cohort data, 2024. [Online]. Available:
    https://consensus.app/questions/influenza-symptoms-without-fever/

[8] M. Ruiz-Sevilla et al., "Incidence and prevalence of headache in
    influenza: A 2010-2021 surveillance-based study," PMC, 2024.
    [Online]. Available:
    https://pmc.ncbi.nlm.nih.gov/articles/PMC11236060/

[9] N. Y. Lee et al., "Extranasal symptoms of allergic rhinitis are
    difficult to treat and affect quality of life," ScienceDirect, 2016.
    [Online]. Available:
    https://www.sciencedirect.com/science/article/pii/S1323893015002129
"""
