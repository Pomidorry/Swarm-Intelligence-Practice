import numpy as np
import pandas as pd
import random as rd
from random import randint
import matplotlib.pyplot as plt

item_number = np.arange(1, 21)
weight = np.random.randint(1, 20, size=20)
value = np.random.randint(10, 750, size=20)
knapsack_threshold = 50    # Максимальна вага, яку може вмістити рюкзак
print('Список такий:')
print('Номер товару  Вага  Значення')
for i in range(item_number.shape[0]):
    print('{0}             {1}      {2}\n'.format(item_number[i], weight[i], value[i]))

solutions_per_pop = 50
pop_size = (solutions_per_pop, item_number.shape[0])
print('Розмір популяції = {}'.format(pop_size))
initial_population = np.random.randint(2, size=pop_size)
initial_population = initial_population.astype(int)
num_generations = 70
print('Початкова популяція: \n{}'.format(initial_population))


def cal_fitness(weight, value, population, threshold):
    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * value)
        S2 = np.sum(population[i] * weight)
        if S2 <= threshold:
            fitness[i] = S1
        else:
            fitness[i] = 0
    return fitness.astype(int)


def selection(fitness, num_parents, population):
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        parents[i, :] = population[max_fitness_idx[0][0], :]
        fitness[max_fitness_idx[0][0]] = -999999
    return parents


def crossover(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = int(parents.shape[1] / 2)
    crossover_rate = 0.8
    i = 0
    while (parents.shape[0] < num_offsprings):
        parent1_index = i % parents.shape[0]
        parent2_index = (i + 1) % parents.shape[0]
        x = rd.random()
        if x > crossover_rate:
            continue
        parent1_index = i % parents.shape[0]
        parent2_index = (i + 1) % parents.shape[0]
        offsprings[i, 0:crossover_point] = parents[parent1_index, 0:crossover_point]
        offsprings[i, crossover_point:] = parents[parent2_index, crossover_point:]
        i = +1
    return offsprings


def mutation(offsprings):
    mutants = np.empty((offsprings.shape))
    mutation_rate = 0.4
    for i in range(mutants.shape[0]):
        random_value = rd.random()
        mutants[i, :] = offsprings[i, :]
        if random_value > mutation_rate:
            continue
        int_random_value = randint(0, offsprings.shape[1] - 1)
        if mutants[i, int_random_value] == 0:
            mutants[i, int_random_value] = 1
        else:
            mutants[i, int_random_value] = 0
    return mutants


def optimize(weight, value, population, pop_size, num_generations, threshold):
    parameters, fitness_history = [], []
    num_parents = int(pop_size[0] / 2)
    num_offsprings = pop_size[0] - num_parents
    for i in range(num_generations):
        fitness = cal_fitness(weight, value, population, threshold)
        fitness_history.append(fitness)
        parents = selection(fitness, num_parents, population)
        offsprings = crossover(parents, num_offsprings)
        mutants = mutation(offsprings)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants
        # Візуалізація гістограми популяції
        if (i + 1) % K == 0:
            plt.figure()
            plt.bar(np.arange(len(fitness)), fitness)
            plt.title('Гістограма популяції після {} ітерацій'.format(i + 1))
            plt.xlabel('Номер хромосоми')
            plt.ylabel('Значення цільової функції')
            plt.show()

    print('Остання популяція: \n{}\n'.format(population))
    fitness_last_gen = cal_fitness(weight, value, population, threshold)
    print('Фітнес останньої популяції: \n{}\n'.format(fitness_last_gen))
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    parameters.append(population[max_fitness[0][0], :])
    return parameters, fitness_history


K = 10  # Кількість ітерацій між візуалізаціями
parameters, fitness_history = optimize(weight, value, initial_population, pop_size, num_generations, knapsack_threshold)
print('Оптимальні параметри для вказаних вхідних даних: \n{}'.format(parameters))
selected_items = item_number * parameters
print('\nВибрані товари, які максимізують рюкзак без його розриву:')
for i in range(selected_items.shape[1]):
    if selected_items[0][i] != 0:
        print('{}\n'.format(selected_items[0][i]))

fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
fitness_history_max = [np.max(fitness) for fitness in fitness_history]
plt.plot(list(range(num_generations)), fitness_history_mean, label='Середній фітнес')
plt.plot(list(range(num_generations)), fitness_history_max, label='Максимальний фітнес')
plt.legend()
plt.title('Фітнес протягом поколінь')
plt.xlabel('Покоління')
plt.ylabel('Фітнес')
plt.show()
print(np.asarray(fitness_history).shape)
