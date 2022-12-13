NUM_QUEENS = 8
POPULATION_SIZE = 10
MIXING_NUMBER = 2
MUTATION_RATE = 0.26

def fitness_score(seq):
    score = 0
    
    for row in range(NUM_QUEENS):
        col = seq[row]
        
        for other_row in range(NUM_QUEENS):
            
            
            if other_row == row:
                continue
            if seq[other_row] == col:
                continue
            if other_row + seq[other_row] == row + col:
                continue
            if other_row - seq[other_row] == row - col:
                continue
            
            score += 1
    
    
    return score/2

import random

from scipy import special as sc

def selection(population):
    parents = []
    
    for ind in population:
        
        if random.randrange(sc.comb(NUM_QUEENS, 2)*2) < fitness_score(ind):
            parents.append(ind)
            
    
    return parents

import itertools

def crossover(parents):
    
   
    cross_points = random.sample(range(NUM_QUEENS), MIXING_NUMBER - 1)
    offsprings = []
    
    permutations = list(itertools.permutations(parents, MIXING_NUMBER))
    
    for perm in permutations:
        offspring = []
        
        
        start_pt = 0
        
        for parent_idx, cross_point in enumerate(cross_points): 
            
            
            parent_part = perm[parent_idx][start_pt:cross_point]
            offspring.append(parent_part)
            
            
            start_pt = cross_point
            
        
        last_parent = perm[-1]
        parent_part = last_parent[cross_point:]
        offspring.append(parent_part)
        
        
        offsprings.append(list(itertools.chain(*offspring)))
    
    return offsprings

def mutate(seq):
    for row in range(len(seq)):
        if random.random() < MUTATION_RATE:
            seq[row] = random.randrange(NUM_QUEENS)
    
    return seq

def print_found_goal(population, to_print=True):
    for ind in population:
        score = fitness_score(ind)
        if to_print:
            print(f'{ind}. Score: {score}')
        if score == sc.comb(NUM_QUEENS, 2):
            if to_print:
                print('Solution found')
            return True
    
    if to_print:
        print('Solution not found')
    return False

def evolution(population):
    
    parents = selection(population)

    
    offsprings = crossover(parents)

    
    offsprings = list(map(mutate, offsprings))

    
    new_gen = offsprings

    for ind in population:
        new_gen.append(ind)

    new_gen = sorted(new_gen, key=lambda ind: fitness_score(ind), reverse=True)[:POPULATION_SIZE]

    return new_gen

def generate_population():
    population = []

    for individual in range(POPULATION_SIZE):
        new = [random.randrange(NUM_QUEENS) for idx in range(NUM_QUEENS)]
        population.append(new)
    
    return population

generation = 0


population = generate_population()
    





gens = []
for run in range(100):
    generation = 0
    population = generate_population()
    print(f'Run: {run}')
    while not print_found_goal(population, to_print=False):
        population = evolution(population)
        generation += 1

    gens.append(generation)


import numpy as np
"""
print(f'Mean: {np.mean(gens)}')
print(f'St. dev: {np.std(gens)}')
print(f'Min: {min(gens)}')
print(f'Max: {max(gens)}')



"""

generation = 0


population = generate_population()
    
while not print_found_goal(population):
    print(f'Generation: {generation}')
    print_found_goal(population)
    population = evolution(population)
    generation += 1


