"""
Search result container 
"""

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class SearchResult:
    """Empirical output of a single search run.

    Attributes
    ----------
    algorithm : str
        Identifier such as 'BFS' or 'A*'.
    found : bool
        Whether a goal state was reached.
    solution : list[str]
        Names of the actions applied along the discovered path.
        Empty if `found` is False.
    cost : float
        Total path cost (sum of step costs). `math.inf` if not found.
    depth : int
        Number of actions on the discovered path.
    nodes_expanded : int
        Count of nodes whose successors were generated.
    nodes_generated : int
        Count of `Node` objects constructed (children of expanded nodes,
        plus the root).
    max_frontier : int
        Peak size of the frontier (fringe) during the search.
    elapsed_seconds : float
        Wall-clock time for the call.
    """

    algorithm: str
    found: bool
    solution: list[str] = field(default_factory=list)
    cost: float = 0.0
    depth: int = 0
    nodes_expanded: int = 0
    nodes_generated: int = 0
    max_frontier: int = 0
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt_path(solution: Sequence[str]) -> str:
    if not solution:
        return "(no solution)"
    return " -> ".join(solution)


def format_comparison_table(
    scenario: str, results: Sequence[SearchResult]
) -> str:
    """Render a fixed-width table comparing several algorithms on one scenario.

    Designed so that copy-pasting into a monospace block in the report
    preserves alignment.
    """
    header_cols = [
        "Algorithm",
        "Cost",
        "Depth",
        "Expanded",
        "Generated",
        "Max |Frontier|",
        "Time (ms)",
    ]
    widths = [12, 7, 6, 10, 10, 15, 10]

    def row(values: Sequence[str]) -> str:
        return " | ".join(v.ljust(w) for v, w in zip(values, widths))

    lines = [
        f"Scenario: {scenario}",
        "-" * (sum(widths) + 3 * (len(widths) - 1)),
        row(header_cols),
        "-" * (sum(widths) + 3 * (len(widths) - 1)),
    ]
    for r in results:
        lines.append(
            row(
                [
                    r.algorithm,
                    f"{r.cost:.2f}" if r.found else "-",
                    str(r.depth) if r.found else "-",
                    str(r.nodes_expanded),
                    str(r.nodes_generated),
                    str(r.max_frontier),
                    f"{r.elapsed_seconds * 1000:.2f}",
                ]
            )
        )
    lines.append("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for r in results:
        lines.append(f"  {r.algorithm} path: {_fmt_path(r.solution)}")
    return "\n".join(lines)
