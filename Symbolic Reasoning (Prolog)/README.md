# Clinical Decision Support System

## Overview
This project is a Prolog-based Clinical Decision Support System that uses facts and rules to infer diseases, symptoms, treatments, isolation needs, and infection sources.
## Requirements
- SWI-Prolog
## How to Run
1. Install SWI-Prolog.
2. Save the Prolog file as:
   `clinical_decision_support.pl`
3. Open SWI-Prolog.
4. Load the file using:
```prolog
consult('clinical_decision_support.pl').
## If the file loads successfully, Prolog will return:
true.

## Example Queries
Check a patient diagnosis
has_disease(patient_ali, Disease).
## Expected output:
Disease = influenza.
Suggest treatment
suggest_treatment(patient_ali, Treatment).

## Expected output:
Treatment = oseltamivir.
Check isolation need
needs_isolation(patient_sarah).
## Expected output:
true.
Check infection source
original_source(Source, patient_khaled).
## Expected output:
Source = patient_sarah ;
Source = patient_ali.
## Notes
The file includes a verification example where fever is intentionally added as both a disease and a symptom. This can be checked using:
check_disjoint(fever).
## Expected output:
Error: Conflict found!
true.
## This will make your project look organized and easy to understand.
## Summary of Useful Queries
has_disease(patient_ali, Disease).
has_disease(patient_sarah, Disease).
has_disease(patient_ahmed, Disease).
suggest_treatment(patient_ali, Treatment).
needs_isolation(patient_sarah).
identified_agent(patient_ali, Agent).

likely_diagnosis(patient_ali, Disease, winter).
is_critical_condition(patient_ali).
is_stable_condition(patient_sarah).
original_source(Source, patient_khaled).
is_valid_symptom(fever).
is_valid_disease(influenza).
check_disjoint(fever).
