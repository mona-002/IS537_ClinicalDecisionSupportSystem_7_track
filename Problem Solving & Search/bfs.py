"""
Breadth-First Search (graph variant).



Note on optimality
------------------
BFS is optimal *in the number of actions* (i.e., shortest plan). It is
**not** guaranteed cost-optimal when step costs differ, which is exactly
the case in our treatment space. The comparison with A* in the report
illustrates this point on the influenza scenario.

"""

from collections import deque
from time import perf_counter

from metrics import SearchResult
from problem import Node, TreatmentProblem, child_node


def bfs(problem: TreatmentProblem) -> SearchResult:
    """Run breadth-first graph search on `problem`.

    Returns a populated `SearchResult` whether or not a goal is reached.
    """
    t_start = perf_counter()

    root = Node(state=problem.initial_state)
    if problem.is_goal(root.state):
        return SearchResult(
            algorithm="BFS",
            found=True,
            solution=[],
            cost=0.0,
            depth=0,
            nodes_expanded=0,
            nodes_generated=1,
            max_frontier=1,
            elapsed_seconds=perf_counter() - t_start,
        )

    frontier: deque[Node] = deque([root])
    # `reached` maps state -> True (set semantics) to avoid re-exploring.
    reached: set = {root.state}

    nodes_generated = 1
    nodes_expanded = 0
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        node = frontier.popleft()
        nodes_expanded += 1

        for action in problem.actions(node.state):
            child = child_node(problem, node, action)
            nodes_generated += 1

            if child.state in reached:
                continue

            if problem.is_goal(child.state):
                return SearchResult(
                    algorithm="BFS",
                    found=True,
                    solution=child.action_sequence(),
                    cost=child.path_cost,
                    depth=child.depth,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    max_frontier=max_frontier,
                    elapsed_seconds=perf_counter() - t_start,
                )

            reached.add(child.state)
            frontier.append(child)

    return SearchResult(
        algorithm="BFS",
        found=False,
        cost=float("inf"),
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier=max_frontier,
        elapsed_seconds=perf_counter() - t_start,
    )
