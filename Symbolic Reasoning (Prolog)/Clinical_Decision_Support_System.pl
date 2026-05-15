% The Facts

% Disease Facts
disease(influenza).
disease(common_cold).
disease(allergic_rhinitis).

% wrong fact for verification 
disease(fever).

% Symptom Facts
symptom(fever).
symptom(sore_throat).
symptom(itchy_eyes).
symptom(muscle_joint_pain).
symptom(headache).
symptom(cough).
symptom(sneezing).
symptom(runny_nose).
symptom(mild_body_aches).
symptom(nasal_congestion).

% Seasonality Facts
common_in_season(influenza, winter).
common_in_season(allergic_rhinitis, spring).
common_in_season(common_cold, year_round).
common_in_season(allergic_rhinitis, autumn).

% Severity Facts
severity(influenza, severe).
severity(common_cold, mild).
severity(allergic_rhinitis, mild_moderate).

% Causative Agent Facts
causative_agent(influenza, influenza_a_virus).
causative_agent(influenza, influenza_b_virus).
causative_agent(common_cold, rhinovirus).
causative_agent(common_cold, coronavirus).
causative_agent(common_cold, rsv).
causative_agent(allergic_rhinitis, pollen).
causative_agent(allergic_rhinitis, dust).
causative_agent(allergic_rhinitis, mold).

% Treatment Facts
requires_treatment(influenza, oseltamivir).
requires_treatment(common_cold, symptomatic_relief).
requires_treatment(allergic_rhinitis, antihistamines).
requires_treatment(allergic_rhinitis, allergen_avoidance).

% Contagious Facts
contagious(influenza).
contagious(common_cold).


% --------------------------------------------------------
% Patients Facts

% Patient ali (influenza)
has_symptom(patient_ali, fever).              
has_symptom(patient_ali, muscle_joint_pain).

% Patient sarah (common_cold)
has_symptom(patient_sarah, sneezing).         
has_symptom(patient_sarah, runny_nose).         
has_symptom(patient_sarah, sore_throat).


% Patient ahmed (allergic_rhinitis)
has_symptom(patient_ahmed, itchy_eyes).         
has_symptom(patient_ahmed, nasal_congestion).         
has_symptom(patient_ahmed, sneezing).

% Patient nourah
has_symptom(patient_nourah, cough).

% Patient khaled (influenza)
has_symptom(patient_khaled, fever).
has_symptom(patient_khaled, muscle_joint_pain).
has_symptom(patient_khaled, runny_nose).
has_symptom(patient_khaled, sore_throat).

% Contagion Transmission
infected_from(patient_sarah, patient_ali).  
infected_from(patient_khaled, patient_sarah).

% --------------------------------------------------------
% Rules

has_disease(P, influenza) :-
    has_symptom(P, fever),
    has_symptom(P, muscle_joint_pain).

has_disease(P, common_cold) :-
    has_symptom(P, sneezing),
    has_symptom(P, runny_nose),
    has_symptom(P, sore_throat).

has_disease(P, allergic_rhinitis) :-
    has_symptom(P, itchy_eyes),
    has_symptom(P, nasal_congestion),
    has_symptom(P, sneezing).

is_influenza(P) :-
    has_disease(P, Disease),
    causative_agent(Disease, Agent),
    (Agent = influenza_a_virus ; Agent = influenza_b_virus).

is_allergic_reaction(P) :-
    has_disease(P, allergic_rhinitis),
    causative_agent(allergic_rhinitis, Agent),
    (Agent = pollen ; Agent = dust ; Agent = mold).

is_common_cold(P) :-
    has_disease(P, Disease),
    causative_agent(Disease, Agent),
    (Agent = rsv ; Agent = coronavirus ; Agent = rhinovirus).

likely_diagnosis(P, influenza, winter) :-
    has_disease(P, influenza),
    common_in_season(influenza, winter).

likely_diagnosis(P, allergic_rhinitis, spring) :-
    has_disease(P, allergic_rhinitis),
    common_in_season(allergic_rhinitis, spring).

likely_diagnosis(P, common_cold, year_round) :-
    has_disease(P, common_cold),
    common_in_season(common_cold, year_round).

likely_diagnosis(P, allergic_rhinitis, autumn) :-
    has_disease(P, allergic_rhinitis),
    common_in_season(allergic_rhinitis, autumn).

is_critical_condition(P) :-
    has_disease(P, Disease),
    severity(Disease, severe).

has_fever(P) :-
    has_disease(P, influenza).

is_stable_condition(P) :-
    has_disease(P, Disease),
    severity(Disease, mild).

suggest_treatment(P, Treatment) :-
    has_disease(P, Disease),
    requires_treatment(Disease, Treatment).

needs_isolation(P) :-
    has_disease(P, Disease),
    contagious(Disease).

identified_agent(P, Agent) :-
    has_disease(P, Disease),
    causative_agent(Disease, Agent).

need_pain_killer(P) :-
    has_disease(P, influenza),
    severity(influenza, severe).

complex_diagnosis(P) :-
    has_disease(P, D1),
    has_disease(P, D2),
    D1 \= D2.

recommends_bed_rest(P) :-
    is_critical_condition(P).

precautionary_advice(P, avoid_public_places) :-
    needs_isolation(P).

% Recursive Rule
original_source(A, B) :- 
    infected_from(B, A).

original_source(A, C) :- 
    infected_from(B, A), 
    original_source(B, C).

% Verification Typing & Constraints Rules

is_valid_symptom(S) :- symptom(S).

is_valid_disease(D) :- disease(D).

check_disjoint(X) :-
    disease(X),
    symptom(X),
    write('Error: Conflict found!').