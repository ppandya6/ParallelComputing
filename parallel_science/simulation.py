"""Explicit finite-difference heat diffusion on a unit-spaced square grid."""
from dataclasses import dataclass
import multiprocessing as mp
import numpy as np


@dataclass(frozen=True)
class Simulation:
    size: int = 128
    steps: int = 200
    alpha: float = 0.2

    def __post_init__(self):
        if not isinstance(self.size, int) or self.size < 3:
            raise ValueError("size must be an integer >= 3")
        if not isinstance(self.steps, int) or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if not 0 < self.alpha <= 0.25:
            raise ValueError("alpha must be in (0, 0.25] for a stable 2D scheme")


def initial_grid(size, seed):
    """Seed each trial independently of worker assignment."""
    grid = np.zeros((size, size), dtype=np.float64)
    grid[1:-1, 1:-1] = np.random.default_rng(seed).random((size - 2, size - 2))
    return grid


def diffuse(grid, steps, alpha):
    """Return evolved temperatures with fixed zero-temperature boundaries."""
    config = Simulation(grid.shape[0], steps, alpha)
    if grid.ndim != 2 or grid.shape != (config.size, config.size):
        raise ValueError("grid must be square")
    current = np.array(grid, dtype=np.float64, copy=True)
    current[[0, -1], :] = 0
    current[:, [0, -1]] = 0
    other = np.zeros_like(current)
    for _ in range(steps):
        center = current[1:-1, 1:-1]
        other[1:-1, 1:-1] = center + alpha * (
            current[:-2, 1:-1] + current[2:, 1:-1]
            + current[1:-1, :-2] + current[1:-1, 2:] - 4 * center
        )
        current, other = other, current
    return current


def _trial(task):
    config, seed = task
    grid = diffuse(initial_grid(config.size, seed), config.steps, config.alpha)
    # Return compact observables rather than transferring whole grids.
    return float(grid.mean()), float(np.square(grid).sum())


def run(config, trials=8, seed=42, workers=1):
    """Run an ensemble; one worker is the sequential baseline.

    Multiple workers use spawn for portability. Pool creation and result
    collection are included by callers timing this function.
    """
    if trials < 1 or workers < 1 or seed < 0:
        raise ValueError("trials and workers must be positive; seed nonnegative")
    tasks = [(config, seed + index) for index in range(trials)]
    if workers == 1:
        return np.asarray([_trial(task) for task in tasks])
    with mp.get_context("spawn").Pool(min(workers, trials)) as pool:
        return np.asarray(pool.map(_trial, tasks))
