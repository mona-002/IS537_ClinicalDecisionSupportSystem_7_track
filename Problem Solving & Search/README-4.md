# IS 537 Project — Search Subsystem (Track 2)

This folder contains the source code for the **Search** part of our Intelligent Diagnostic System (Track 2: Clinical Decision Support). It implements and compares two search algorithms — **Breadth-First Search (BFS)** and **A\*** — on the task of planning a patient's treatment.

The report (in the main PDF submission) explains the design choices and results. This README only tells you how to run the code.

---

## Requirements

- **Python 3.10 or later.** Tested on Python 3.12.
- No extra packages need to be installed. The code uses only the Python standard library.

To check your Python version:
```bash
python3 --version
```

---

## How to run the code

Open a terminal inside this folder, then use any of the commands below.

**1. Run all five scenarios and show the full comparison tables:**
```bash
python3 main.py
```

**2. Run one scenario at a time:**
```bash
python3 main.py influenza
python3 main.py common_cold
python3 main.py allergic_rhinitis
python3 main.py patient_khaled
python3 main.py multi_morbidity
```

**3. Show a short one-line summary for each scenario:**
```bash
python3 main.py --quiet
```

**4. Run the test suite (8 tests, finishes in a few milliseconds):**
```bash
python3 -m unittest test_search.py -v
```

---

## What you should see

Running `python3 main.py --quiet` prints:

```
========================================================================
IS 537 Project | Track 2 | Treatment-Space Search | BFS vs A*
========================================================================
influenza           BFS cost= 7.50 exp=   2  |  A* cost= 5.50 exp=   5
common_cold         BFS cost= 6.20 exp=   4  |  A* cost= 6.20 exp=   6
allergic_rhinitis   BFS cost= 5.00 exp=   1  |  A* cost= 5.00 exp=   1
patient_khaled      BFS cost=10.20 exp=   6  |  A* cost= 6.90 exp=   7
multi_morbidity     BFS cost=12.50 exp=  15  |  A* cost=10.50 exp=  37
```

- `cost` = the total cost of the treatment plan (treatment days + side-effect penalty)
- `exp` = the number of nodes the algorithm had to expand during the search

---

## What each scenario shows

| Scenario | What it demonstrates |
|---|---|
| `influenza` | Full influenza symptoms. BFS and A\* both find a 2-step plan, but A\* finds a cheaper one. |
| `common_cold` | Full common cold symptoms. Both algorithms find the same cost. |
| `allergic_rhinitis` | Full allergic rhinitis symptoms. A one-step plan exists and both algorithms find it. |
| `patient_khaled` | A patient taken from the Prolog `has_symptom/2` facts. BFS plan costs 10.20; A\* plan costs 6.90 — a 32% improvement. |
| `multi_morbidity` | A patient showing symptoms of two diseases at once. Used as a stress test. |

---

## What each file does

| File | Purpose |
|---|---|
| `main.py` | The entry point. Runs the experiments. |
| `knowledge_base.py` | Lists the diseases, symptoms, patients, and treatments. |
| `problem.py` | Defines the search problem (states, actions, goal test). |
| `heuristic.py` | The heuristic used by A\*. |
| `bfs.py` | Breadth-First Search algorithm. |
| `astar.py` | A\* algorithm. |
| `metrics.py` | Helper code for formatting the output tables. |
| `test_search.py` | Eight unit tests that check correctness and verify the heuristic properties. |
| `docs/ADMISSIBILITY.md` | Proof that the heuristic is admissible and consistent. |
| `docs/ANALYSIS.md` | Theoretical and empirical analysis of BFS vs A\*. |

---

## Testing with partial information

The project brief asks each subsystem to handle partial information. For the search subsystem, you can pass any set of symptoms — not just a complete disease presentation — and the search still works:

```python
from problem import TreatmentProblem
from bfs import bfs
from astar import astar

# A patient with only two reported symptoms.
problem = TreatmentProblem(
    initial_symptoms=frozenset({"fever", "cough"})
)
print("BFS:", bfs(problem))
print("A* :", astar(problem))
```

This is what allows the search to connect to the Uncertainty subsystem later: any set of likely symptoms (after thresholding the Bayesian/CF output) can be passed in directly.

---

## A note on the heuristic

For every symptom `r` still in the patient's state, let `μ(r)` be the cost of the cheapest treatment that relieves `r`. The heuristic used by A\* is:

```
h(s) = max { μ(r) : r ∈ s },   with h(∅) = 0
```

This heuristic is **admissible** (it never overestimates the real remaining cost) and **consistent** (it satisfies `h(s) ≤ c(t) + h(s')` for every transition).

---

## References

[1] S. J. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. Pearson, 2021.

[2] J. Pearl, *Heuristics: Intelligent Search Strategies for Computer Problem Solving*. Addison-Wesley, 1984.

[3] P. E. Hart, N. J. Nilsson, and B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths," *IEEE Transactions on Systems Science and Cybernetics*, vol. 4, no. 2, pp. 100–107, 1968.

[4] L. Sterling and E. Shapiro, *The Art of Prolog*, 2nd ed. MIT Press, 1994.
