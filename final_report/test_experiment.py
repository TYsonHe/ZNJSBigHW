import tempfile
import unittest
from pathlib import Path

import experiment as e


class ExperimentTest(unittest.TestCase):
    def test_distance_matrix_and_route_length(self):
        points = [(0, 0), (3, 4), (6, 0)]
        dist = e.build_distance_matrix(points)
        self.assertEqual(round(dist[0][1], 6), 5)
        self.assertEqual(round(e.route_length([0, 1, 2], dist), 6), 16)

    def test_instances_have_expected_sizes(self):
        self.assertEqual(len(e.random_instance(30, 20260608)), 30)
        self.assertEqual(len(e.fixed_instance()), 20)

    def test_ga_returns_valid_route_and_curve(self):
        points = e.random_instance(12, 1)
        best, route, curve = e.run_ga(points, 1, generations=5, pop_size=20)
        self.assertGreater(best, 0)
        self.assertEqual(sorted(route), list(range(12)))
        self.assertEqual(len(curve), 5)

    def test_aco_returns_valid_route_and_curve(self):
        points = e.random_instance(12, 1)
        best, route, curve = e.run_aco(points, 1, iterations=5, ants=10)
        self.assertGreater(best, 0)
        self.assertEqual(sorted(route), list(range(12)))
        self.assertEqual(len(curve), 5)

    def test_main_writes_expected_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original = e.RESULTS_DIR
            e.RESULTS_DIR = Path(temp_dir)
            try:
                e.main()
                self.assertTrue((e.RESULTS_DIR / "summary.csv").exists())
                self.assertTrue((e.RESULTS_DIR / "convergence.svg").exists())
                self.assertTrue((e.RESULTS_DIR / "routes.svg").exists())
                self.assertIn("random_30", (e.RESULTS_DIR / "summary.csv").read_text(encoding="utf-8"))
            finally:
                e.RESULTS_DIR = original


if __name__ == "__main__":
    unittest.main()
