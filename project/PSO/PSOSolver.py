import sys
import time

import numpy as np
from numpy import ndarray

from project.Backtracking.PuzzleState import PuzzleState


def error_function(state: PuzzleState) -> float:
    # TODO find a proper error_function
    return state.constraint_violations()


class PSOSolver:
    def __init__(self, constraints):
        self.constraints = constraints

    def solve(self, n_particles: int = 1000,
              iterations: int = 100,
              w: float = 0.5,
              alpha1: float = 0.25,
              alpha2: float = 0.25) -> ...:

        height, width = self.constraints.height, self.constraints.width

        # Prepare variables
        velocity = np.random.random((n_particles, height, width))

        best_local_error = np.zeros(n_particles)
        best_local_error[:] = sys.maxsize
        best_local_particle = np.zeros((n_particles, height, width))

        global_best_error = sys.maxsize
        global_best_particle = np.zeros((height, width))

        # Initialize particles
        particles = np.random.random((n_particles, height, width))

        self.start_time = time.perf_counter()

        # Run iterations
        for i in range(iterations):
            # Calculate error for each particle
            for p in range(n_particles):
                # Compute error
                error = error_function(self.__from_representation(particles[p]))

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
