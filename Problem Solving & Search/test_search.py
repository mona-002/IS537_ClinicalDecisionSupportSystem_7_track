"""
Unit tests for the Treatment-Space search module.

Run from the project root:
    python -m unittest test_search.py -v

"""

import unittest

from astar import astar
from bfs import bfs
from heuristic import h
from knowledge_base import get_all_treatments, get_disease_list
from problem import TreatmentProblem


class TestSearchSoundness(unittest.TestCase):
    """Both algorithms must produce valid plans for every disease."""

    def test_bfs_reaches_goal_for_each_disease(self) -> None:
        for disease in get_disease_list():
            with self.subTest(disease=disease):
                problem = TreatmentProblem(disease=disease)
                result = bfs(problem)
                self.assertTrue(result.found, f"BFS failed on {disease}")
                self._assert_valid_plan(problem, result.solution)

    def test_astar_reaches_goal_for_each_disease(self) -> None:
        for disease in get_disease_list():
            with self.subTest(disease=disease):
                problem = TreatmentProblem(disease=disease)
                result = astar(problem)
                self.assertTrue(result.found, f"A* failed on {disease}")
                self._assert_valid_plan(problem, result.solution)

    def _assert_valid_plan(
        self, problem: TreatmentProblem, plan: list[str]
    ) -> None:
        treatments = get_all_treatments()
        state = problem.initial_state
        for action_name in plan:
            self.assertIn(action_name, treatments)
            action = treatments[action_name]
            self.assertTrue(
                action.relieves & state,
                f"Action {action_name} not applicable in state {set(state)}",
            )
            state = problem.result(state, action)
        self.assertTrue(problem.is_goal(state), "Plan does not reach goal")


class TestAStarOptimality(unittest.TestCase):
    """A* with an admissible heuristic finds an optimal-cost plan.

    BFS finds an optimal-in-depth plan, so cost(A*) <= cost(BFS)."""

    def test_astar_no_worse_than_bfs(self) -> None:
        for disease in get_disease_list():
            with self.subTest(disease=disease):
                problem = TreatmentProblem(disease=disease)
                self.assertLessEqual(
                    astar(problem).cost,
                    bfs(problem).cost + 1e-9,
                )

    def test_astar_strictly_cheaper_on_influenza(self) -> None:
        """The treatment-cost calibration ensures the influenza scenario
        admits multiple depth-2 plans of differing cost. BFS, which
        terminates on the first goal generated, returns a higher-cost
        plan than A*. This is the textbook illustration of why BFS is
        only optimal in path *length*, not path *cost* (Russell &
        Norvig, sec. 3.4.1)."""
        problem = TreatmentProblem(disease="influenza")
        bfs_r, astar_r = bfs(problem), astar(problem)
        self.assertLess(astar_r.cost, bfs_r.cost)
        # Both find a 2-step plan (BFS is optimal-in-depth).
        self.assertEqual(bfs_r.depth, 2)
        self.assertEqual(astar_r.depth, 2)


class TestHeuristicProperties(unittest.TestCase):
    """Sanity-check admissibility and consistency on reachable states."""

    def test_h_goal_is_zero(self) -> None:
        self.assertEqual(h(frozenset()), 0.0)

    def test_h_non_negative(self) -> None:
        for disease in get_disease_list():
            self.assertGreaterEqual(
                h(TreatmentProblem(disease=disease).initial_state), 0.0
            )

    def test_consistency_on_reachable_states(self) -> None:
        """For every transition s -> s', verify h(s) <= cost(a) + h(s')."""
        treatments = get_all_treatments().values()
        for disease in get_disease_list():
            problem = TreatmentProblem(disease=disease)
            # BFS-style enumeration over all reachable states.
            frontier: list = [problem.initial_state]
            seen = {problem.initial_state}
            while frontier:
                state = frontier.pop()
                for t in treatments:
                    if not (t.relieves & state):
                        continue
                    nxt = problem.result(state, t)
                    self.assertLessEqual(
                        h(state),
                        t.cost + h(nxt) + 1e-9,
                        f"Consistency violated at {set(state)} via {t.name}",
                    )
                    if nxt not in seen:
                        seen.add(nxt)
                        frontier.append(nxt)

    def test_admissibility_via_astar(self) -> None:
        """A* with an admissible heuristic finds an optimal plan, and any
        plan's cost is an upper bound on the true h*(initial_state). So
        h(initial_state) <= cost(A*-plan) must hold for every scenario.

        (This is necessary but not sufficient for admissibility; the
        consistency check above is the strong condition.)
        """
        for disease in get_disease_list():
            with self.subTest(disease=disease):
                problem = TreatmentProblem(disease=disease)
                self.assertLessEqual(
                    h(problem.initial_state),
                    astar(problem).cost + 1e-9,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
