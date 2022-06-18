import sys
import time
from typing import Callable

import numpy as np
from numpy import ndarray
from tqdm import tqdm

from Backtracking.PuzzleState import PuzzleState


class PSOSolver:
    def __init__(self, constraints, error_fun: Callable[[PuzzleState], float]):
        self.constraints = constraints
        self.error_fun = error_fun

    def solve(self, n_particles: int = 4096,
              iterations: int = 20,
              w: float = 0.7298,
              alpha1: float = 1.49618,
              alpha2: float = 1.49618) -> ...:

        height, width = self.constraints.height, self.constraints.width

        # Prepare variables
        velocity = np.zeros((n_particles, height, width))

        best_local_error = np.zeros(n_particles)
        best_local_error[:] = sys.maxsize
        best_local_particle = np.zeros((n_particles, height, width))

        global_best_error = sys.maxsize
        global_best_particle = np.zeros((height, width))

        # Initialize particles
        particles = np.random.random((n_particles, height, width))

        self.start_time = time.perf_counter()

        # Run iterations
        for i in tqdm(range(iterations), desc="PSO solver progress"):
            # Calculate error for each particle
            for p in range(n_particles):
                # Compute error
                error = self.error_fun(self.__from_representation(particles[p]))

                # Update local best errors and particles
                if error < best_local_error[p]:
                    best_local_particle[p] = particles[p]
                    best_local_error[p] = error

            # Update global best error and particle
            if np.min(best_local_error) < global_best_error:
                global_best_error = np.min(best_local_error)
                global_best_particle = best_local_particle[np.argmin(best_local_error)]

            # Update velocity and position of each particle
            for p in range(n_particles):
                r1 = np.random.rand(1)
                r2 = np.random.rand(1)

                velocity[p] = w * velocity[p] \
                              + alpha1 * r1 * (best_local_particle[p] - particles[p]) \
                              + alpha2 * r2 * (global_best_particle - particles[p])

                particles[p] = particles[p] + velocity[p]

        self.end_time = time.perf_counter()

        return [self.__from_representation(global_best_particle)]

    @staticmethod
    def __to_representation__(puzzle: PuzzleState) -> ndarray:
        return NotImplementedError

    def __from_representation(self, rep: ndarray) -> PuzzleState:
        puzzle = PuzzleState(self.constraints)

        height, width = rep.shape

        for row in range(height):
            for col in range(width):
                puzzle.set(row, col, 0 if round(rep[row, col]) < 1 else 1)

        return puzzle
