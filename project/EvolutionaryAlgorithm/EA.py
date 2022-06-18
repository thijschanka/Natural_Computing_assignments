from re import S
from Backtracking.Constraints import Constraints
from Backtracking.PuzzleState import PuzzleState

import numpy as np
import sys
from numpy import random

iterations = 0
def GeneticAlgorithm(constraints, populationSize):
    global iterations
    population = randomSolutions(constraints, populationSize)
    while not converge(population) and iterations < 10000:
        crossoverPopulation = crossover(population, constraints, populationSize)
        mutationPopulation = mutation(crossoverPopulation, constraints)
        population = select(population, mutationPopulation, populationSize)
        iterations += 1
        print("Iterations = ", iterations)
        print("Best fitness:", error_function(population[0]))

    return best(population)

def randomSolutions(constraints, populationSize):
    solutions = []

    for _ in range(populationSize):
        solution = PuzzleState(constraints)
        for j in range(constraints.width):
            for i in range(constraints.height):
                if random.random() >= 0.5:
                    solution.set(i, j , 1)
                else:
                    solution.set(i, j , 0)
        solutions += [solution]
    return solutions
                    
def crossover(population, constraints, populationSize):
    crossoverPopulation = []

    population = sorted(population, key = lambda s : (error_function(s), random.random()))
    n = (populationSize*(populationSize+1))/2
    prob = [i/n for i in range(1, populationSize+1)]

    for _ in range(populationSize):
        child1 = PuzzleState(constraints)
        child2 = PuzzleState(constraints)
        parent1, parent2 = random.choice(population, p=prob, replace=False, size=2)
        for i in range (constraints.height):
            for j in range(constraints.width):
                if random.random() <= 0.5:
                    child1.set(i,j, parent1.get(i,j))
                    child2.set(i,j, parent2.get(i,j))
                else:
                    child1.set(i,j, parent2.get(i,j))
                    child2.set(i,j, parent1.get(i,j))
        crossoverPopulation += [child1, child2]
    return crossoverPopulation

def mutation(population, constraints):
    mutationPopulation = []

    for solution in population:
        prob = 0.1/100
        
        for i in range(constraints.height):
            for j in range(constraints.width):
                if random.random() < prob:
                    if(solution.get(i,j) == 1):
                        solution.set(i,j,0)
                    else:
                        solution.set(i,j,1)
    
        mutationPopulation += [solution]
    
    return mutationPopulation

def select(crossoverPopulation, mutationPopulation, populationSize):
    crossoverPopulation = sorted(crossoverPopulation, key= lambda s : (error_function(s), random.random()), reverse=False)
    mutationPopulation = sorted(mutationPopulation, key= lambda s : (error_function(s), random.random()), reverse=False)

    numberOfParents = int(2*populationSize/10)+1
    numberOfChildren = int(2*populationSize/10)+1
    numberOfRandom = populationSize - numberOfParents - numberOfChildren

    bestSolutions = crossoverPopulation[:numberOfParents] + mutationPopulation[:numberOfChildren]
    otherSolutions = crossoverPopulation[numberOfParents:] + mutationPopulation[numberOfChildren:]

    nextPopulation = bestSolutions + np.ndarray.tolist(random.choice(otherSolutions, size=numberOfRandom, replace=False))

    return nextPopulation

def converge(population):

    for solution in population:
        if error_function(solution) == 0:
            print("Found solution!")
            return True
    
    for i in range(len(population)-1):
        if population[i].get_state() != population[i+1].get_state():
            return False

    print("All solutions converged..")
    return True

def best(population):
    bestSolutions = []
    for solution in population:
        if error_function(solution) == 0:
            bestSolutions += [solution]
    
    if len(bestSolutions) > 0:
        return bestSolutions
    else:
        return [population[0]]

def  error_function(solution: PuzzleState) -> float:
    return solution.error_2()
    
