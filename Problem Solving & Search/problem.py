"""
Search-problem formalisation for the Treatment Space.

"""

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Tuple

from knowledge_base import Treatment, get_all_treatments, get_initial_symptoms


# A state is the immutable set of symptoms still present in the patient.
State = FrozenSet[str]


@dataclass
class TreatmentProblem:
    """Concrete search problem over the treatment state-space.

    The state space is the powerset of the union of all symptoms appearing
    in the disease ontology. The branching factor at state `s` is the number
    of treatments whose `relieves` set has non-empty intersection with `s`.
    The goal is the empty set.

    Parameters
    ----------
    disease : str
        Modelled disease whose presenting symptoms form the initial state.
        For combined (multi-morbidity) scenarios, pass `initial_symptoms`
        directly instead.
    initial_symptoms : FrozenSet[str], optional
        Overrides `disease` if supplied. Useful for stress-testing the
        search with patients exhibiting symptoms from multiple diseases.
    """

    disease: str = ""
    initial_symptoms: State = frozenset()

    def __post_init__(self) -> None:
        if not self.initial_symptoms and not self.disease:
            raise ValueError(
                "Either `disease` or `initial_symptoms` must be provided."
            )
        if not self.initial_symptoms:
            object.__setattr__(
                self, "initial_symptoms", get_initial_symptoms(self.disease)
            )
        # Cache the treatment table once -- the search is hot on this lookup.
        self._treatments = get_all_treatments()

    # ------------------------------------------------------------------ API

    @property
    def initial_state(self) -> State:
        return self.initial_symptoms

    def is_goal(self, state: State) -> bool:
        """The patient is recovered iff no symptoms remain."""
        return len(state) == 0

    def actions(self, state: State) -> Iterable[Treatment]:
        """Treatments applicable at `state`.

        An action is applicable iff it clears at least one current symptom.
        This pruning rule keeps the state graph finite (no self-loops via
        no-op treatments) and is a sound restriction because any optimal
        plan can omit no-op actions without changing cost (step costs are
        strictly positive)."""
        for t in self._treatments.values():
            if t.relieves & state:
                yield t

    def result(self, state: State, action: Treatment) -> State:
        """Successor state after applying `action`.

        The transition function is: `s' = s \\ action.relieves`.
        """
        return state - action.relieves

    def step_cost(
        self, state: State, action: Treatment, next_state: State
    ) -> float:
        """Edge weight. State arguments retained for the standard signature;
        the cost depends only on the action under this model."""
        return action.cost


# ---------------------------------------------------------------------------
# Search node -- a lightweight wrapper used by both BFS and A*.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Node:
    """A node in the search tree (Russell & Norvig, ch. 3)."""

    state: State
    parent: "Node | None" = None
    action: Treatment | None = None
    path_cost: float = 0.0
    depth: int = 0

    def path(self) -> list["Node"]:
        """Return the path from the root to this node, root first."""
        rev: list[Node] = []
        n: Node | None = self
        while n is not None:
            rev.append(n)
            n = n.parent
        rev.reverse()
        return rev

    def action_sequence(self) -> list[str]:
        """Treatment names along the path, root excluded."""
        return [n.action.name for n in self.path()[1:] if n.action is not None]


def child_node(
    problem: TreatmentProblem, parent: Node, action: Treatment
) -> Node:
    """Construct the successor of `parent` via `action`."""
    s_next = problem.result(parent.state, action)
    cost = parent.path_cost + problem.step_cost(parent.state, action, s_next)
    return Node(
        state=s_next,
        parent=parent,
        action=action,
        path_cost=cost,
        depth=parent.depth + 1,
    )
