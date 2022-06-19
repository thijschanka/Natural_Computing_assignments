from nonoGram import NonoGram
from Backtracking.Constraints import Constraints
from Backtracking.PuzzleSolver import PuzzleSolver as BTSolver
from Backtracking.PuzzleState import PuzzleState as State

from PSO.PSOSolver import PSOSolver
from EvolutionaryAlgorithm.EA import EASolver

import sys
import os
import json

import pandas as pd
import click

import numpy as np

@click.command()
@click.option(
    "--eval_json",
    "-e",
    type=str,
    required=True,
    help="The relative location of the eval file"
)
@click.option(
    "--accepted_key",
    "-k",
    type=str,
    default=[""],
    required=False,
    multiple=True,
    help="the key of evaluation.json to use, if none given it will run all"
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=None,
    required=False,
    help="wheter to use a seed"
)
@click.option(
    "--algorithm",
    "-a",
    type=click.Choice(['pso', 'ea', 'bt']),
    default='pso',
    required=False,
    help="which algorithm to use, pso=particle swarm optimisation, ea=evolutionary algorithm, bt=backtracking"
)
@click.option(
    "--particles",
    "-p",
    type=int,
    default=4096,
    required=False,
    help="number of particles or popsize"
)
@click.option(
    "--iterations",
    "-i",
    type=int,
    default=100,
    required=False,
    help="number of iterations to run"
)
def main(eval_json, accepted_key=None, seed=None, algorithm='pso', particles=4096, iterations=100):
    if seed:
        np.random.seed(seed)
    
    # print("Processing puzzle {} of {}".format(index + 1, puzzles))
    f = open(eval_json)
    json_object = json.load(f)
    
    base_path = json_object['relative_path']        
    if accepted_key[0] == "":
        accepted_key = None
        
    tasks = {'pso': [True, False, False], 'ea': [False, False, True], 'bt': [False, True, False]}

    for key in json_object:
        if key != "relative_path" and (accepted_key is None or key in accepted_key):
            if os.path.exists(f'./results/{key}_{algorithm}_results.csv'):
                result_df = pd.read_csv(f'./results/{key}_{algorithm}_results.csv', header=0, index_col=0)
            else:
                result_df = pd.DataFrame()
                
            print("Processing puzzles {}".format(key))
            for index, puzzle_file in enumerate(json_object[key]):
                print("Processing puzzle {} of {}".format(puzzle_file, key))
                full_path = '/'.join([base_path, puzzle_file])
                result_df = process_puzzle(full_path, result_df, key, puzzle_file, iterations=iterations, pso=tasks[algorithm][0], backtrack=tasks[algorithm][1], ea=tasks[algorithm][2], n_particles=particles)
                result_df.to_csv(f'./results/{key}_{algorithm}_results.csv')


def process_puzzle(path, result_df, key, puzzle_file ,pso=True, ea=False, backtrack=False, iterations=100, n_particles=4096):
    puzzle = NonoGram(path)
    constraints = puzzle.getConstraints()

    if backtrack:
        puzlle_backtrack_solver = BTSolver(constraints)
        solutions = puzlle_backtrack_solver.solve()
        puzzle.setBTSoluiton(solutions)
        
        results = puzlle_backtrack_solver.get_metrics()
        final_time = results[1] - results[0]
        result_df = result_df.append([{'task':key, 'file_name':puzzle_file.split('.')[0], 'time':final_time, 'iteration':results[2], 'n_solutions':len(solutions)}])
    if pso:
        error_fun = lambda s: s.error_2()
        
        puzzle_pso_solver = PSOSolver(constraints, error_fun)
        PSOsolution = puzzle_pso_solver.solve(
            n_particles = n_particles,
            iterations = iterations)
        puzzle.setPSOSoluiton(PSOsolution)
        
        results = puzzle_pso_solver.get_metrics()
        final_time = results[1] - results[0]
        result_df = result_df.append([{'task':key, 'file_name':puzzle_file.split('.')[0], 'time':final_time, 'error':results[3], 'iteration':results[2], 'particles':n_particles}])
    if ea:
        puzzle_ea_solver = EASolver(constraints)
        EAsolution = puzzle_ea_solver.solve(populationSize=n_particles, maxIter=iterations)
        puzzle.setEASoluiton(EAsolution)
        
        results = puzzle_ea_solver.get_metrics()
        final_time = results[1] - results[0]
        result_df = result_df.append([{'task':key, 'file_name':puzzle_file.split('.')[0], 'time':final_time, 'error':results[3], 'iteration':results[2], 'population':n_particles}])
    
    print(puzzle.get_numpy_results())
    print(puzzle.getHints())
    print(puzzle.printSolutions())
    return result_df
    
if __name__ == "__main__":
    main()