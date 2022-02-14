# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 09:29:11 2022

@author: thijs
"""

#TSP http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/att48.tsp
import numpy as np
import copy
import random
import os

# Distance metric
def euclideanDist(route, vals):
    totalDist = 0
    for j, city1 in enumerate(route[:-1]):
        city2 = route[j]
        totalDist += np.linalg.norm(vals[city1]-vals[city2])
    city1 = route[-1]
    city2 = route[0]    
    
    totalDist += np.linalg.norm(vals[city1]-vals[city2])
    return totalDist

# Swap procedure
def twooptswap(route, i, k):
    step1 = route[0:i-1]
    step2 = route[i-1:k]
    step2 = step2[::-1]
    return np.concatenate((step1, step2, route[k:]))

# Inner loop of  the algorithm
def twooptloop(existing_route, vals, best_dist):
    for i in range(1, len(existing_route)-1):
        for k in range(2, len(existing_route)):
            new_route = twooptswap(existing_route, i, k)
            new_dist = euclideanDist(new_route, vals)
            
            if new_dist < best_dist:
                return new_route, new_dist
    return existing_route, best_dist

# Outer loop of the two-opt algorithm
def two_opt_localsearch(existing_route, vals):
    best_dist = euclideanDist(existing_route, vals)
    start_dist = 0
    while best_dist != start_dist:
        start_dist = copy.copy(best_dist)
        existing_route, best_dist = twooptloop(existing_route, vals, best_dist)

    return existing_route
    
# Crossover between 2 parents
def crossover(p1, p2):
    point1 = random.randint(0, len(p1)-1)
    point2 = random.randint(point1, len(p1)+1)
    
    mid_child1 = p2[point1:point2]
    mid_child2 = p1[point1:point2]
    
    rest_c1 = p1[~np.in1d(p1,mid_child1)]
    rest_c2 = p2[~np.in1d(p2,mid_child2)]
    
    child1 = np.concatenate((rest_c1[:point1], mid_child1, rest_c1[point1:]))
    child2 = np.concatenate((rest_c2[:point1], mid_child2, rest_c2[point1:]))
    
    return child1, child2

# Mutation of parents
def mutation(parent):
    point1, point2 = random.sample(range(len(parent)), 2)
    child = np.copy(parent)
    child[point1] = parent[point2]
    child[point2] = parent[point1]
    return child

#Tournament selection of the candidates
def tournament_selection(candidates, cand_vals, k):

    possible_cands = random.sample(range(len(candidates)), k)
    best_val = np.min(cand_vals)
    best_cand = None

    for i in possible_cands:
        if cand_vals[i] >= best_val:
            best_val = cand_vals[i]
            best_cand = candidates[i]

    return best_cand

# fitness = 1/dist
# Crossover = order
# Mutation = swap
# Selection = Tournament
# Replacement = Generational
def ge(vals, n_candidates=4, k=2, MA=False, Pc = 1, Pm = 0.001, epochs=30):
    # Random init candiates
    candidates = np.zeros((n_candidates, len(vals)), dtype=int)
    cand_vals = np.zeros(n_candidates, dtype=float)
    for i in range(n_candidates):
        cand = np.arange(len(vals))
        np.random.shuffle(cand)
        # LOCAL SEARCH for candidate
        if MA:
            candidates[i] = two_opt_localsearch(cand, vals)
        else:
            candidates[i] = cand
        # Evaluate candidate
        cand_vals[i] = 1/euclideanDist(candidates[i], vals)

    
    # Repeat until termination cond:
    best_list = []
    worst_list = []
    avg_list = []
    counter = 0
    current_best = 0
    for e in range(epochs):
        if e % 100 == 0: print(e)
        elif e < 10: print(e)
        # Select Parents for rep
        selected_cands = np.zeros((n_candidates, len(vals)), dtype=int)
        selected_vals = np.zeros(n_candidates, dtype=float)
        for i in range(n_candidates):
            selected_cands[i] = tournament_selection(candidates, cand_vals, k)
        # Recombine Selection
        
        changed = [False]*n_candidates

        cs = random.sample(range(len(candidates)), 2)
        if random.random() <= Pc:
            selected_cands[cs[0]], selected_cands[cs[1]] = crossover(selected_cands[cs[0]], selected_cands[cs[1]])
            
            changed[cs[0]] = True
            changed[cs[1]] = True
            
        # Mutate Selection
        for i in range(len(selected_cands)):
            if random.random() <= Pm:
                selected_cands[i] = mutation(selected_cands[i])
                changed[i] = True
        
        # LOCAL SEARCH for candidate
        for i in range(n_candidates):

            if MA and changed[i]:
                selected_cands[i] = two_opt_localsearch(selected_cands[i], vals)
                        
            # Evaluate candidate
            selected_vals[i] = 1/euclideanDist(selected_cands[i], vals)
            
            assert len(np.unique(selected_cands[i])) == len(selected_cands[i])
        
        # Select new generation
        candiates = selected_cands
        cand_vals = selected_vals
        
        best_list.append(np.max(cand_vals))
        worst_list.append(np.min(cand_vals))
        avg_list.append(np.mean(cand_vals))
        
        #Stop critereon
        if best_list[-1] <= current_best: counter += 1
        else: 
            current_best = best_list[-1] 
            counter = 0    
            
        #if counter == 10: 
        #    best_val = np.max(cand_vals)
        #    best_cand = candiates[np.argmax(cand_vals)]
            
        #    return best_cand, best_val, best_list, worst_list, avg_list
    
    best_val = np.max(cand_vals)
    best_cand = candiates[np.argmax(cand_vals)]
    
    return best_cand, best_val, best_list, worst_list, avg_list


values = np.loadtxt("file-tsp.txt", dtype=float)
values2 = np.loadtxt("file2-tsp.txt", dtype=float)
values2 = values2[:,1:]

import time


print("ME")

for i in range(10):
    print("run", i+1)
    
    start = time.time()
    _, _, best, worst, avg = ge(values, n_candidates=10, k=2, MA=False, Pc = 1, Pm = 0.001, epochs=1500)
    end = time.time()
    print("problem1", end-start)
    
    results = np.stack((best, worst, avg), axis=1)
    with open("problem1\\GE\\run{}.txt".format(str(i+1)), 'w+') as f:
        np.savetxt(f, results, delimiter=';', fmt='%1.5f')
      
    start = time.time()
    _, _, best, worst, avg = ge(values2, n_candidates=10, k=2, MA=False, Pc = 1, Pm = 0.001, epochs=1500)
    end = time.time()
    print("problem2", end-start)
    
    
    results = np.stack((best, worst, avg), axis=1)
    with open("problem2\\GE\\run{}.txt".format(str(i+1)), 'w+') as f:
        np.savetxt(f, results, delimiter=';', fmt='%1.5f')


print("ME")
for i in range(10):
    print("run", i+1)
    
    start = time.time()
    _, _, best, worst, avg = ge(values, n_candidates=10, k=2, MA=True, Pc = 0.7, Pm = 0.001, epochs=1500)
    end = time.time()
    print("problem1", end-start)
    
    results = np.stack((best, worst, avg), axis=1)
    with open("problem1\\ME\\run{}.txt".format(str(i+1)), 'w+') as f:
        np.savetxt(f, results, delimiter=';', fmt='%1.5f')
        
    start = time.time()
    _, _, best, worst, avg = ge(values2, n_candidates=10, k=2, MA=True, Pc = 0.7, Pm = 0.001, epochs=1500)
    end = time.time()
    print("problem2", end-start)
    
    results = np.stack((best, worst, avg), axis=1)
    with open("problem2\\ME\\run{}.txt".format(str(i+1)), 'w+') as f:
        np.savetxt(f, results, delimiter=';', fmt='%1.5f')
