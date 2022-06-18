#Filename: d:\master_opleiding\NC\Natural_Computing_assignments\Project\\nonoGram.py
#Path: d:\master_opleiding\NC\Natural_Computing_assignments\Project
#Created Date: Tuesday, May 10th 2022, 1:39:36 pm
#Author: thijschanka
#
#Copyright (c) 2022 Your Company

import numpy as np
import json
from Backtracking.Constraints import Constraints
from Backtracking.PuzzleSolver import PuzzleSolver as Solver
from Backtracking.PuzzleState import PuzzleState as State

from PSO.PSOSolver import PSOSolver
from EvolutionaryAlgorithm.EA import GeneticAlgorithm

class NonoGram:
    def __init__(self, json_file = False):
        self.hints = None
        self.results = None
        
        if json_file is not False:
            self.fromJson(json_file)
        
    def calculate_fitness(self, solution):
        return solution - self.result
    
    def getHints(self):
        return self.hints
    
    def getResults(self):
        return self.results
    
    def fromJson(self, path):
        with open(path, 'r') as f:
            json_object = json.load(f)
            
        self.hints = np.array([json_object["rows"], json_object["columns"]])
        
        errors, instance = Constraints.get_constraints(json_object)
        solver = Solver(instance)
        solutions = solver.solve()
        self.brute_force_sol = solutions

        error_fun = lambda s: s.error_2()

        self.geneticAlgorithmSolutions = GeneticAlgorithm(instance, 100)
        self.psoSolutions = PSOSolver(instance, error_fun).solve()
        
        first = True
        self.results = []
        for index, solution in enumerate(solutions):
            const = solution.get_constraints()
            result = np.zeros((const.width, const.height))
            
            for j in range(const.width):
                for i in range(const.height):
                    if solution._state[i][j]:
                        result[i][j] = 1
            
            self.results.append(result)
        
    def printSolutions(self):
        print("Brute force solutions:")
        for i in self.brute_force_sol:
            print(i)

        print("Genetic algorithm solutions:")
        for j in self.geneticAlgorithmSolutions:
            print(j)

        print("PSO solutions:")
        for k in self.psoSolutions:
            print(k)
