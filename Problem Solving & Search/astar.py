"""
A* graph search with an admissible & consistent heuristic.




Implementation notes
--------------------
* Frontier is a binary min-heap keyed on (f, tiebreak, node). The integer
  tiebreak preserves FIFO order among ties, which yields determinism in
  the comparison table.
* Nodes are compared by identity to avoid relying on `Node.__lt__`.
"""

import heapq
import itertools
from time import perf_counter

from heuristic import h
from metrics import SearchResult
from problem import Node, TreatmentProblem, child_node


def astar(problem: TreatmentProblem) -> SearchResult:
    """Run A* graph search on `problem`.

    Uses the heuristic `heuristic.h`. Returns a populated `SearchResult`.
    """
    t_start = perf_counter()
    counter = itertools.count()  # FIFO tie-breaker

    root = Node(state=problem.initial_state)
    f_root = root.path_cost + h(root.state)
    frontier: list[tuple[float, int, Node]] = [(f_root, next(counter), root)]
    # best_g maps state -> lowest g-value at which the state has been reached
    best_g: dict = {root.state: 0.0}

    nodes_generated = 1
    nodes_expanded = 0
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        f, _, node = heapq.heappop(frontier)

        # Skip if a strictly better g for this state has been recorded since
        # this entry was pushed (only possible with non-consistent h).
        if node.path_cost > best_g.get(node.state, float("inf")):
            continue

        if problem.is_goal(node.state):
            return SearchResult(
                algorithm="A*",
                found=True,
                solution=node.action_sequence(),
                cost=node.path_cost,
                depth=node.depth,
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier=max_frontier,
                elapsed_seconds=perf_counter() - t_start,
            )

        nodes_expanded += 1
        for action in problem.actions(node.state):
            child = child_node(problem, node, action)
            nodes_generated += 1

            if child.path_cost < best_g.get(child.state, float("inf")):
                best_g[child.state] = child.path_cost
                f_child = child.path_cost + h(child.state)
                heapq.heappush(frontier, (f_child, next(counter), child))

    return SearchResult(
        algorithm="A*",
        found=False,
        cost=float("inf"),
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier=max_frontier,
        elapsed_seconds=perf_counter() - t_start,
    )
