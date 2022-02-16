
#constants
optimalValue = 0

#variables
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
    while(xPos != optimalValue):
        velocity = computeVelocity(omega, alpha, r)
        xPos += velocity
        if fitness(xPos) < xBest:
            xBest = xPos
        xPositions.append(xPos)


runAlgorithm(0.5, 1.5, 0.5)
plt.plot(xPositions, '->')
plt.show()

runAlgorithm(0.7, 1.5, 1)
