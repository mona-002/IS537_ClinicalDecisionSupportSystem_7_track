"""
Heuristic function for A* in the Treatment Space.

Strategy: *max-of-min cheapest-cover*.

For each un-relieved symptom `r` in the current state, compute the cost of
the cheapest treatment that relieves `r`. The heuristic value is the
maximum of these per-symptom lower bounds:

    h(s) = max_{r in s}  min { cost(t) : t relieves r }

"""

from functools import lru_cache
from typing import FrozenSet

from knowledge_base import get_all_treatments


def _build_min_cost_per_symptom() -> dict[str, float]:
    """For each symptom, the cost of the cheapest treatment relieving it.

    Computed once at import time and cached. If the KB is swapped at runtime
    (e.g., via pyswip), call `_min_cost_per_symptom_cache_clear()` first.
    """
    treatments = get_all_treatments().values()
    table: dict[str, float] = {}
    for t in treatments:
        for sym in t.relieves:
            if sym not in table or t.cost < table[sym]:
                table[sym] = t.cost
    return table


# Module-level cache. Recomputable via `reset_heuristic_cache()`.
_MIN_COST_PER_SYMPTOM: dict[str, float] = _build_min_cost_per_symptom()


def reset_heuristic_cache() -> None:
    """Re-derive the per-symptom min-cost table from the current KB.

    Call after swapping the knowledge base (e.g., when integrating pyswip)
    or after mutating treatments in tests."""
    global _MIN_COST_PER_SYMPTOM
    _MIN_COST_PER_SYMPTOM = _build_min_cost_per_symptom()
    h.cache_clear()


@lru_cache(maxsize=None)
def h(state: FrozenSet[str]) -> float:
    """Admissible & consistent heuristic for the Treatment Problem.

    Parameters
    ----------
    state : FrozenSet[str]
        The current symptom set (frozen so it is hashable for caching).

    Returns
    -------
    float
        A lower bound on the optimal cost to reach the goal from `state`.
        Returns 0.0 at the goal.
    """
    if not state:
        return 0.0
    # If a symptom has no covering treatment, the goal is unreachable; we
    # return +inf so A* will prefer any other branch.
    try:
        return max(_MIN_COST_PER_SYMPTOM[s] for s in state)
    except KeyError:
        return float("inf")
