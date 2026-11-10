import unittest
import numpy as np
from parallel_science.simulation import Simulation, diffuse, initial_grid, run


class SimulationTests(unittest.TestCase):
    def test_single_step_stencil(self):
        grid = np.zeros((5, 5))
        grid[2, 2] = 1
        expected = np.zeros_like(grid)
        expected[2, 2] = 0.2
        expected[1, 2] = expected[3, 2] = 0.2
        expected[2, 1] = expected[2, 3] = 0.2
        np.testing.assert_allclose(diffuse(grid, 1, 0.2), expected)

    def test_boundaries_and_dissipation(self):
        grid = initial_grid(16, 42)
        result = diffuse(grid, 100, 0.2)
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), grid.max())
        self.assertLess(result.sum(), grid.sum())
        np.testing.assert_array_equal(result[[0, -1]], 0)
        np.testing.assert_array_equal(result[:, [0, -1]], 0)

    def test_zero_steps_and_input_preserved(self):
        grid = initial_grid(8, 1)
        original = grid.copy()
        np.testing.assert_array_equal(diffuse(grid, 0, 0.2), original)
        diffuse(grid, 4, 0.2)
        np.testing.assert_array_equal(grid, original)

    def test_parallel_matches_sequential(self):
        config = Simulation(12, 10)
        expected = run(config, 5, 17)
        np.testing.assert_array_equal(run(config, 5, 17, 2), expected)
        np.testing.assert_array_equal(run(config, 2, 17, 3), expected[:2])

    def test_seed_reproducibility(self):
        np.testing.assert_array_equal(initial_grid(8, 9), initial_grid(8, 9))
        self.assertFalse(np.array_equal(initial_grid(8, 9), initial_grid(8, 10)))

    def test_validation(self):
        for args in [(2, 10, 0.2), (8, -1, 0.2), (8, 10, 0.26)]:
            with self.assertRaises(ValueError):
                Simulation(*args)
        for kwargs in [dict(trials=0), dict(workers=0), dict(seed=-1)]:
            with self.assertRaises(ValueError):
                run(Simulation(), **kwargs)


if __name__ == "__main__":
    unittest.main()
