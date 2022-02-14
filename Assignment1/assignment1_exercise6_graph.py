# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 08:31:36 2022

@author: thijs
"""

import os
from matplotlib import pyplot as plt
import numpy as np


x = np.arange(1500)



def plot_results(axs, problem, alg, j=0):
    path = "problem{}\\{}\\".format(problem, alg)
    for i, file in enumerate(os.listdir(path)):
        result = np.loadtxt(path+file, dtype=float, delimiter=';')
            
        axs[i, j].plot(x, result[:, 0], label='Best Result')
        axs[i, j].plot(x, result[:, 1], label='Average Result', c='r')
        axs[i, j].set_title("Run {} {}".format(str(i), alg), fontsize=15)
        axs[i, j].set_xlabel("Iterations", fontsize=15)
        axs[i, j].set_ylabel("Fitness", fontsize=15)
    
    
# First for problem 1
fig, axs = plt.subplots(10,2, figsize=(22,22))
#fig.set_figheight(30)
#fig.set_figwidth(25)

plot_results(axs, '1', 'GE', 0)
plot_results(axs, '1', 'ME', 1)

fig.suptitle('Result for problem {}'.format(str(1)) , fontsize=20)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=20)
plt.tight_layout()
fig.savefig('6_problem1.png')

# Now problem 2
fig, axs = plt.subplots(10,2, figsize=(20,20))
#fig.set_figheight(30)
#fig.set_figwidth(25)

plot_results(axs, '2', 'GE', 0)
plot_results(axs, '2', 'ME', 1)

fig.suptitle('Result for problem {}'.format(str(2)), fontsize=20)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=20)
plt.tight_layout()
fig.savefig('6_problem2.png')