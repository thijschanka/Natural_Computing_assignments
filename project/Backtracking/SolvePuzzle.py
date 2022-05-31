
import sys
import os
import json
from Constraints import Constraints
from project.PSOSolver import PSOSolver as Solver
from PuzzleState import PuzzleState as State

def process_puzzle(path) -> None:
    f = open(path)
    json_object = json.load(f)


    errors, instance = Constraints.get_constraints(json_object)
    solver = Solver(instance)
    solutions = solver.solve()
    print("Time to find solution: {} seconds".format(round(solver.end_time - solver.start_time, 4)))
    
    first = True
    for index, solution in enumerate(solutions):
        if not first:
            print()
        first = False
        print("Solution {}/{}".format(index + 1, len(solutions)))
        print(solution)
    
def main():
    if len(sys.argv) < 2 or \
       any(help_command in sys.argv for help_command in ["--help", "-h", "-?"]):
        return

    puzzles = len(sys.argv) - 1
    for index, path in enumerate(sys.argv[1:]):
        print("Processing puzzle {} of {}".format(index + 1, puzzles))
        process_puzzle(path)

if __name__ == "__main__":
    main()
