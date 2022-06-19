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
    def __init__(self, json_file = False, puzzle_state = False):
        self.hints = None
        self.results = None
        self.numpy_results = []
        self.constraints = None

        self.geneticAlgorithmSolutions = None
        self.PSOSolutions = None
        self.bruteForceSolutions = None
        
        self.geneticAlgorithmMetrics = None
        self.PSOMetrics = None
        self.bruteForceMetrics = None
        
        if json_file is not False:
            self.fromJson(json_file)
        elif puzzle_state is not False:
            self.from_puzzle_state(json_file)        
                
    def fromJson(self, path, ga=False, bf=False, pso=False):
        with open(path, 'r') as f:
            json_object = json.load(f)
            
        self.hints = np.array([json_object["rows"], json_object["columns"]])
        self.constraints = Constraints(json_object['width'], json_object["height"], json_object["rows"], json_object["columns"])

        errors, instance = Constraints.get_constraints(json_object)
        if bf:
            solver = Solver(instance)
            solutions = solver.solve()
            self.bruteForceSolutions = solutions
            
            for index, solution in enumerate(solutions):
                const = solution.get_constraints()
                result = np.zeros((const.width, const.height))
                
                for j in range(const.width):
                    for i in range(const.height):
                        if solution._state[i][j]:
                            result[i][j] = 1
                self.numpy_results.append(result)
        if ga:
            self.geneticAlgorithmSolutions = GeneticAlgorithm(instance, 20)
        if pso:
            error_fun = lambda s: s.error_2()
            self.psoSolutions = PSOSolver(instance, error_fun).solve()   
            
    def state_to_numpy(self, solution_state):
        const = solution_state.get_constraints()
        np_state_result = np.zeros((const.width, const.height))
        
        old_dff = (const.width * const.height)+1
        
        for j in range(const.width):
            for i in range(const.height):
                if solution_state._state[i][j]:
                    np_state_result[i][j] = 1
        
        for np_result in self.numpy_results:
            if np.sum(np.abs(np_state_result - np_result)) < old_dff:
                acc = 1 - (np.sum(np.abs(np_state_result-np_result)) / (const.width * const.height))
                
                
                tp = (np_state_result+np_result)
                tp[tp < 2] = 0
                tp[tp == 2] = 1
                
                fp = (np_state_result-np_result)
                fp[fp < 0] = 0
                
                precision = (np.sum(tp) / np.sum(tp+fp))
                recall = (np.sum(tp) / np.sum(np_result))
        return acc, precision, recall, np_state_result, np_result
        
    def printSolutions(self):
        if self.bruteForceSolutions is not None:
            print("Brute force solutions:")
            for i in self.bruteForceSolutions:
                print(i)

        if self.geneticAlgorithmSolutions is not None:
            print("Genetic algorithm solutions:")
            for j in self.geneticAlgorithmSolutions:
                print(j)
                
        if self.PSOSolutions is not None:
            print("PSO solutions:")
            for k in self.PSOSolutions:
                print(k)

    def calculate_fitness(self, solution):
        return np.abs(solution - self.result)
    
    def getHints(self):
        return self.hints
    
    def getResults(self):
        return self.numpy_results
    
    def getConstraints(self):
        return self.constraints
    
    def get_numpy_results(self):
        return self.numpy_results
    
    def setPSOSoluiton(self, state):
        self.PSOSolutions = state
        
    def setEASoluiton(self, state):
        self.geneticAlgorithmSolutions = state
        
    def setBTSoluiton(self, state):
        self.bruteForceSolutions = state
        for index, solution in enumerate(state):
            const = solution.get_constraints()
            result = np.zeros((const.width, const.height))
            
            for j in range(const.width-1):
                for i in range(const.height-1):
                    if solution._state[i][j]:
                        result[i][j] = 1
            self.numpy_results.append(result)