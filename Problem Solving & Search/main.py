"""
Entry point: run BFS and A* over the modelled diseases, a real patient case
from the Prolog KB, and a multi-morbidity stress test. Then print
one comparison table per scenario.

Usage
-----
    python main.py                # run all scenarios
    python main.py influenza      # run a single named scenario
    python main.py --quiet        # one-line summary per scenario

The scenarios were selected to expose four behaviours:
* `allergic_rhinitis` -- a 1-step solution exists; both algorithms agree.
* `influenza` and `common_cold` -- multi-step canonical presentations;
  the divergence in cost between BFS and A* shows the value of the
  heuristic.
* `patient_khaled` -- a real patient from the peer's Prolog
  `has_symptom/2` facts (Section 2.1 of the group deliverable). Diagnosed
  as influenza by the Prolog rule, but his reported symptoms include
  cold-like ones (runny_nose, sore_throat) -- an atypical/comorbid
  presentation that pushes the search to depth 3.
* `multi_morbidity` -- influenza + allergic rhinitis presenting together;
  branching factor and depth both grow, amplifying A*'s cost advantage.
"""

from __future__ import annotations

import sys
from typing import Dict

from astar import astar
from bfs import bfs
from knowledge_base import get_patient_symptoms
from metrics import SearchResult, format_comparison_table
from problem import TreatmentProblem


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def _build_scenarios() -> Dict[str, TreatmentProblem]:
   
    scenarios: Dict[str, TreatmentProblem] = {
        # Canonical (textbook) presentations -- initial state is the full
        # symptom set listed in the ontology for the disease.
        "influenza": TreatmentProblem(disease="influenza"),
        "common_cold": TreatmentProblem(disease="common_cold"),
        "allergic_rhinitis": TreatmentProblem(disease="allergic_rhinitis"),
        # Real patient from the Prolog `has_symptom/2` facts.
        # Demonstrates Mode B operation: initial state = reported symptoms,
        # not the canonical disease presentation.
        "patient_khaled": TreatmentProblem(
            initial_symptoms=get_patient_symptoms("patient_khaled")
        ),
        # Multi-morbidity stress test: union of influenza and allergic
        # rhinitis symptoms. Exercises a larger state graph and a higher
        # branching factor than any single-disease scenario.
        "multi_morbidity": TreatmentProblem(
            initial_symptoms=frozenset(
                {
                    "fever",
                    "muscle_joint_pain",
                    "headache",
                    "cough",
                    "sore_throat",
                    "itchy_eyes",
                    "nasal_congestion",
                }
            )
        ),
    }
    return scenarios


def _run_scenario(name: str, problem: TreatmentProblem) -> tuple[
    SearchResult, SearchResult
]:
    return bfs(problem), astar(problem)


def _print_full(name: str, bfs_r: SearchResult, astar_r: SearchResult) -> None:
    print()
    print(format_comparison_table(name, [bfs_r, astar_r]))


def _print_summary(
    name: str, bfs_r: SearchResult, astar_r: SearchResult
) -> None:
    print(
        f"{name:18s}  "
        f"BFS cost={bfs_r.cost:5.2f} exp={bfs_r.nodes_expanded:4d}  |  "
        f"A* cost={astar_r.cost:5.2f} exp={astar_r.nodes_expanded:4d}"
    )


def main(argv: list[str]) -> int:
    quiet = "--quiet" in argv
    args = [a for a in argv if not a.startswith("--")]

    scenarios = _build_scenarios()
    selected = args if args else list(scenarios.keys())

    unknown = [s for s in selected if s not in scenarios]
    if unknown:
        print(f"Unknown scenario(s): {unknown}", file=sys.stderr)
        print(f"Available: {list(scenarios)}", file=sys.stderr)
        return 2

    print("=" * 72)
    print("IS 537 Project | Track 2 | Treatment-Space Search | BFS vs A*")
    print("=" * 72)

    for name in selected:
        bfs_r, astar_r = _run_scenario(name, scenarios[name])
        if quiet:
            _print_summary(name, bfs_r, astar_r)
        else:
            _print_full(name, bfs_r, astar_r)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
