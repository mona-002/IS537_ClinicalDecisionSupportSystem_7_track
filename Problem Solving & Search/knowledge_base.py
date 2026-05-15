"""
Clinical knowledge base for the IDS Treatment-Space Search subsystem.



Alignment with the peer Prolog deliverable
------------------------------------------
The Prolog file (Group deliverable, Section 2: Symbolic Reasoning)
declares four high-level treatment facts:

    requires_treatment(influenza,         oseltamivir).
    requires_treatment(common_cold,       symptomatic_relief).
    requires_treatment(allergic_rhinitis, antihistamines).
    requires_treatment(allergic_rhinitis, allergen_avoidance).

A state-space search over only four actions would be trivial, so this
subsystem *refines* the abstract category `symptomatic_relief` into six
concrete OTC-level interventions (`paracetamol`, `ibuprofen`,
`cough_suppressant`, `throat_lozenge`, `decongestant`, `rest_hydration`).
The three named drugs (`oseltamivir`, `antihistamines`, `allergen_avoidance`)
are preserved verbatim from the Prolog KB. This is the standard
hierarchical AI design pattern: Prolog handles diagnosis and treatment
*classification*; search optimises the tactical *plan* within the class.

Integration roadmap (see README, "Swapping the KB for Prolog")
--------------------------------------------------------------
The four public functions at the bottom of this file --
`get_initial_symptoms`, `get_all_treatments`, `get_disease_list`,
`get_patient_symptoms` -- form the only interface the search engine
touches. Replacing this file with a thin wrapper around `pyswip` that
queries the Prolog KB is therefore a localised change.



References
----------
[1] S. J. Russell and P. Norvig, *Artificial Intelligence: A Modern
    Approach*, 4th ed. Pearson, 2021, ch. 3 ("Solving Problems by
    Searching").
"""



from dataclasses import dataclass, field
from typing import Dict, FrozenSet


@dataclass(frozen=True)
class Treatment:

    """An action in the treatment state-space.

    The action model is deliberately simple: applying a treatment clears the
    intersection of its `relieves` set with the patient's current symptom set
    in a single transition. Time-to-relief and tolerability burden are folded
    into a scalar `cost` used as the edge weight in the search graph.

    Attributes
    ----------
    name : str
        Atomic identifier matching the corresponding FOL term.
    relieves : FrozenSet[str]
        Symptoms cleared by a single application.
    days : int
        Expected days to symptomatic relief; contributes to g(n).
    side_effect_penalty : float
        Dimensionless tolerability burden in [0, 5]. Contributes to g(n).
        Calibrated from common clinical reasoning: 0.0 (none), 0.3 (mild),
        1.0 (moderate GI/CNS), 1.5+ (substantial nausea/drowsiness).
    contraindications : FrozenSet[str]
        Reserved for the "rich" extension that adds patient comorbidities.
        Always empty in the moderate-complexity baseline used here.
    """

    name: str
    relieves: FrozenSet[str]
    days: int
    side_effect_penalty: float
    contraindications: FrozenSet[str] = field(default_factory=frozenset)

    @property
    def cost(self) -> float:
        """Edge weight in the state-space graph: `days + side_effect_penalty`.

        Both components are non-negative, so step costs are non-negative --
        a precondition for the consistency proof of the heuristic
        (see `docs/ADMISSIBILITY.md`).
        """
        return self.days + self.side_effect_penalty


# ---------------------------------------------------------------------------
# Treatments (9 actions) -- one moderate-complexity instantiation of the
# Track 2 treatment space. Calibrated so that (a) at least two treatments
# cover most symptoms (multiple optimal paths exist), and (b) BFS and A*
# diverge measurably on at least one disease (see experiments/influenza).
# ---------------------------------------------------------------------------

_TREATMENTS: Dict[str, Treatment] = {
    t.name: t
    for t in [
        Treatment(
            name="oseltamivir",
            relieves=frozenset({"fever", "muscle_joint_pain", "cough"}),
            days=3,
            side_effect_penalty=1.5,
        ),
        Treatment(
            name="paracetamol",
            relieves=frozenset({"fever", "headache", "mild_body_aches"}),
            days=2,
            side_effect_penalty=0.3,
        ),
        Treatment(
            name="ibuprofen",
            relieves=frozenset({"fever", "muscle_joint_pain", "headache"}),
            days=2,
            side_effect_penalty=1.0,
        ),
        Treatment(
            name="cough_suppressant",
            relieves=frozenset({"cough", "sore_throat"}),
            days=2,
            side_effect_penalty=0.5,
        ),
        Treatment(
            # Peer's Prolog uses the plural spelling
            # (RequiresTreatment(allergic_rhinitis, antihistamines)), kept
            # identical here so integration through pyswip is mechanical.
            name="antihistamines",
            relieves=frozenset({"sneezing", "runny_nose", "itchy_eyes"}),
            days=2,
            side_effect_penalty=1.2,
        ),
        Treatment(
            name="decongestant",
            relieves=frozenset({"nasal_congestion", "runny_nose"}),
            days=2,
            side_effect_penalty=0.8,
        ),
        Treatment(
            name="throat_lozenge",
            relieves=frozenset({"sore_throat"}),
            days=1,
            side_effect_penalty=0.1,
        ),
        Treatment(
            name="allergen_avoidance",
            relieves=frozenset(
                {"itchy_eyes", "sneezing", "nasal_congestion", "runny_nose"}
            ),
            days=5,
            side_effect_penalty=0.0,
        ),
        Treatment(
            name="rest_hydration",
            relieves=frozenset({"mild_body_aches", "headache", "sore_throat"}),
            days=3,
            side_effect_penalty=0.0,
        ),
    ]
}


# ---------------------------------------------------------------------------
# Disease -> presenting symptoms. Sourced from the project's KR subsystem
# (peer deliverable, "Medical Domain Ontology Mapping").
# ---------------------------------------------------------------------------

_DISEASE_SYMPTOMS: Dict[str, FrozenSet[str]] = {
    "influenza": frozenset(
        {"fever", "muscle_joint_pain", "headache", "cough", "sore_throat"}
    ),
    "common_cold": frozenset(
        {"sneezing", "runny_nose", "mild_body_aches", "sore_throat"}
    ),
    "allergic_rhinitis": frozenset(
        {"sneezing", "runny_nose", "itchy_eyes", "nasal_congestion"}
    ),
}



_PATIENT_SYMPTOMS: Dict[str, FrozenSet[str]] = {
    "patient_ali": frozenset({"fever", "muscle_joint_pain"}),
    "patient_sarah": frozenset({"sneezing", "runny_nose", "sore_throat"}),
    "patient_ahmed": frozenset(
        {"itchy_eyes", "nasal_congestion", "sneezing"}
    ),
    # patient_khaled is the interesting case: peer's Prolog diagnoses him as
    # influenza via the (fever, muscle_joint_pain) rule, but he also reports
    # cold-like symptoms (runny_nose, sore_throat) -- an atypical or
    # comorbid presentation that exercises the search at depth 3.
    "patient_khaled": frozenset(
        {"fever", "muscle_joint_pain", "runny_nose", "sore_throat"}
    ),
}


# ---------------------------------------------------------------------------
# Public interface -- the only functions the search engine consumes.
# Replace these (e.g., with pyswip calls) to swap in a Prolog KB.
# ---------------------------------------------------------------------------

def get_initial_symptoms(disease: str) -> FrozenSet[str]:
    """Return the presenting symptom set for `disease`.

    Raises
    ------
    KeyError
        If `disease` is not modelled.
    """
    if disease not in _DISEASE_SYMPTOMS:
        raise KeyError(
            f"Unknown disease '{disease}'. "
            f"Modelled diseases: {sorted(_DISEASE_SYMPTOMS)}."
        )
    return _DISEASE_SYMPTOMS[disease]


def get_patient_symptoms(patient: str) -> FrozenSet[str]:
    """Return the symptoms reported by a specific patient.

    The patient identifiers (`patient_ali`, `patient_sarah`, ...) and their
    symptom lists are taken from the peer's Prolog `has_symptom/2` facts.
    Use this entry point when driving the search from clinical inputs
    rather than canonical disease presentations.

    Raises
    ------
    KeyError
        If `patient` is not registered.
    """
    if patient not in _PATIENT_SYMPTOMS:
        raise KeyError(
            f"Unknown patient '{patient}'. "
            f"Registered patients: {sorted(_PATIENT_SYMPTOMS)}."
        )
    return _PATIENT_SYMPTOMS[patient]


def get_all_treatments() -> Dict[str, Treatment]:
    """Return all treatments keyed by name (read-only view)."""
    return dict(_TREATMENTS)


def get_disease_list() -> list[str]:
    """Return the names of all modelled diseases, sorted."""
    return sorted(_DISEASE_SYMPTOMS)


def get_patient_list() -> list[str]:
    """Return the names of all registered patients, sorted."""
    return sorted(_PATIENT_SYMPTOMS)
