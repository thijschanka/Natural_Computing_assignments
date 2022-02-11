from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor
import numpy as np
import matplotlib.pyplot as plt

x = np.array(
    [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
     1.0]).reshape(-1, 1)
y = [0,
     -0.1629,
     -0.2624,
     -0.3129,
     -0.3264,
     -0.3125,
     -0.2784,
     -0.2289,
     -0.1664,
     -0.0909,
     0.0,
     0.1111,
     0.2496,
     0.4251,
     0.6496,
     0.9375,
     1.3056,
     1.7731,
     2.3616,
     3.0951,
     4.0000]


def _protected_exp(x1):
    with np.errstate(invalid='ignore'):
        return np.where(x1 < 10, np.exp(x1), 22026.46)


exp_fun = make_function(function=_protected_exp, name='exp', arity=1)

functions = ['add', 'sub', 'mul', 'div', 'log', exp_fun, 'sin', 'cos']

sae_fitness = make_fitness(function=lambda y, y_pred, w: -np.sum(np.abs(np.array(y) - np.array(y_pred)) * w),
                           greater_is_better=True)


gp = SymbolicRegressor(population_size=1000,
                       generations=50,
                       function_set=functions,
                       metric=sae_fitness,
                       p_crossover=0.7,
                       p_hoist_mutation=0,
                       p_point_mutation=0,
                       p_subtree_mutation=0)
gp.fit(x, y)

plt.plot(range(50), gp.run_details_['best_fitness'])
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness of best solutions')
plt.savefig('question8_fitness.pdf')
plt.show()

# For this fill between to work one needs to adjust the genetic.py file of the gplearn package to also track the min
# and max length.
plt.fill_between(range(50), gp.run_details_['min_length'], gp.run_details_['max_length'], label='Between', alpha=0.5,
                 color='orange')
plt.plot(range(50), gp.run_details_['average_length'], label='Average', color='red')
plt.plot(range(50), gp.run_details_['best_length'], label='Best')
plt.xlabel('Generation')
plt.ylabel('Length')
plt.title('Length of generations')
plt.legend()
plt.savefig('question8_length.pdf')
plt.show()

print(gp)
