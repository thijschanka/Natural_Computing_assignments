#Natural Computing - Assignment 2 - Question 2
#Sven Berberich, Thijs Schoppema en Gerhard van der Knijff

#Imports
import matplotlib.pyplot as plt

#Constants
optimalValue = 0

#Variables
velocity = 10
xPos = 20
xBest = 20

#trajectory
xPositions = [xPos]

def fitness(x):
    return pow(x, 2)

def computeVelocity(omega, alpha, r):
    return omega * velocity + alpha*r*(xBest - xPos) + alpha * r * (xBest - xPos)

def runAlgorithm(omega, alpha, r):
    global xPos
    global xBest
    global velocity
    iterations = 0
    while(xPos != optimalValue and iterations < 10000):
        velocity = computeVelocity(omega, alpha, r)
        xPos += velocity
        if fitness(xPos) < fitness(xBest):
            xBest = xPos
        xPositions.append(xPos)
        iterations += 1
    fitnesses = list(map(fitness, xPositions))
    plt.plot(xPositions, fitnesses, '->')
    plt.xlabel("Position")
    plt.ylabel("Fitness")
    plt.plot(0,0, marker='x', color='red')
    plt.show()

#runAlgorithm(0.5, 1.5, 0.5) #settings set 1
#runAlgorithm(0.7, 1.5, 1)   #settings set 2
