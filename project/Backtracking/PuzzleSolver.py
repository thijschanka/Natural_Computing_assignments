from Backtracking.PuzzleState import PuzzleState as State
from Backtracking.Constraints import Constraints
from Backtracking.Permutations import Permutations
import copy
import time

class PuzzleSolver:
    def __init__(self, constraints):
        self.constraints = constraints
        self.permutations = Permutations(constraints)
        
        self.solved = None
        self.end_time = None
        self.start_time = None
        self.used_iter = None

    def _depth_first_search(self, row):
        self.nodes += 1
        if row > self.max_row:
            self.max_row = row

        if not self.state.validate(row):
            return

        if row + 1 == self.constraints.height:
            self.solutions.append(copy.deepcopy(self.state))
            return

        for perm in self.permutations.get_permutations(row+1):
            self.state.set_row(row+1, perm)
            self._depth_first_search(row+1)
            
        self.state.set_row(row+1, [None for _ in range(self.constraints.width)])

    def solve(self):
        self.state = State(self.constraints)
        self.solutions = []
        
        self.nodes = -1
        self.max_row = 0
        self.start_time = time.perf_counter()
        self._depth_first_search(-1)
        self.end_time = time.perf_counter()
        
        self.used_iter = self.nodes + 1
        return self.solutions
    
    def get_metrics(self):
        return self.start_time, self.end_time, self.used_iter, self.solved