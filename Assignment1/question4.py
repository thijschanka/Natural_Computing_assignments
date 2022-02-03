
#Natural Computing - Assignment 1 - Question 4
#Sven Berberich, Thijs Schoppema en Gerhard van der Knijff

#Imports
from bitarray import bitarray
from copy import *
from random import *
import numpy as np
import matplotlib.pyplot as plt

#Constants
LENGTH = 100
MUTATION_RATE = 1/LENGTH
ITERATIONS = 1500
RUNS = 10
COLORS = ['blue', 'black', 'yellow', 'green', 'purple', 'orange', 'grey', 'red', 'brown', 'pink' ]

#Results
iterationsArray = []
fitnessArray = []
numberOfIterations = []

def runAlgorithm(isC = False):
    iterationsArray.clear()
    fitnessArray.clear()
    foundOptimum = False
    X = bitarray([randrange(2) for _ in range(LENGTH)])
    print("Initial bitarray:")
    print(X)
    for i in range(ITERATIONS):
        if(sum(X) == LENGTH):
            numberOfIterations.append(i-1)
            foundOptimum = True
            print("Found optimum!")
            break
        Xm = deepcopy(X)
        for j in range(LENGTH):
            randomDouble = random()
            if randomDouble < MUTATION_RATE:
                Xm.invert(j)
        if isC or sum(Xm) > sum(X):
            iterationsArray.append(i)
            fitnessArray.append(sum(Xm))
            X = deepcopy(Xm)
    if foundOptimum == False:
        print("Optimum not found!")

def plotResults(runNumber):
    plt.plot(iterationsArray, fitnessArray, color=COLORS[runNumber])
    plt.xlabel("Iterations")
    plt.ylabel("Best fitness")
    plt.xlim([0, 1500])
    plt.ylim([0,100])

def runA():
    runAlgorithm()
    plotResults(0)
    print("Number of iterations to find the optimum: " + str(np.mean(numberOfIterations)))
    plt.show()

def runB():
    for i in range(RUNS):
        runAlgorithm()
        plotResults(i)
    print("Average number of iterations to find the optimum: " + str(np.mean(numberOfIterations)))
    plt.show()

def runC():
    for i in range(RUNS):
        runAlgorithm(True)
        plotResults(i)
    print("Average number of iterations to find the optimum: " + str(np.mean(numberOfIterations)))
    plt.show()
    


#runA()
#runB()
#runC()
