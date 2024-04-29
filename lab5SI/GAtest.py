import numpy as np
import random as rd
from random import randint
import matplotlib.pyplot as plt
import networkx as nx

# Read data from the text file
with open('E:\\SI\\lab5SI\\Data2.txt', 'r') as file:
    lines = file.readlines()

knapsack_threshold = int(lines[0].strip())  # Maximum weight the knapsack can hold
data = [list(map(int, line.strip().split())) for line in lines[1:]]  # Weight-value pairs of items

weight = np.array([item[0] for item in data])
value = np.array([item[1] for item in data])

item_number = np.arange(1, len(weight) + 1)

print('Список такий:')
print('Номер товару  Вага  Значення')
for i in range(len(item_number)):
    print('{0}             {1}      {2}\n'.format(item_number[i], weight[i], value[i]))

solutions_per_pop = 250
pop_size = (solutions_per_pop, len(item_number))
print('Розмір популяції = {}'.format(pop_size))

def create_individual(): 
    # Generate a random binary string of the same length as the items list
    return [rd.randint(0, 1) for i in range(len(data))]

def create_population():  
    # Create a population of random individuals
    return [create_individual() for _ in range(solutions_per_pop)]
#initial_population = np.random.randint(2, size=pop_size)
initial_population = np.array(create_population())
num_generations = 100
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
        fitness[max_fitness_idx[0][0]] = 0
    return parents


def crossover(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]), dtype=int)  # Задаем тип данных int для массива потомков
    crossover_point = int(parents.shape[1] / 2)
    crossover_rate = 0.8
    i = 0
    while (i < num_offsprings):  # Исправлено условие завершения цикла
        parent1_index = i % parents.shape[0]
        parent2_index = (i + 1) % parents.shape[0]
        x = rd.random()
        if x > crossover_rate:
            continue
        for j in range(parents.shape[1]):  # Проходим по всем генам потомка
            if j < crossover_point:
                offsprings[i, j] = parents[parent1_index, j]  # Гены до точки скрещивания берутся от первого родителя
            else:
                offsprings[i, j] = parents[parent2_index, j]  # Гены после точки скрещивания берутся от второго родителя
        i += 1  # Исправлено увеличение счетчика
    #print(offsprings)    
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


def optimize(weight, value, population, pop_size, num_generations, threshold, K):
    parameters, fitness_history = [], []
    best_fitness_history = []
    best_solution = None
    best_fitness = 0

    num_parents = int(pop_size[0] / 2)
    num_offsprings = pop_size[0] - num_parents

    for i in range(num_generations):
        fitness = cal_fitness(weight, value, population, threshold)
        fitness_history.append(fitness)

        if np.max(fitness) > best_fitness:
            best_fitness = np.max(fitness)
            best_solution = population[np.argmax(fitness)]

        if (i + 1) % K == 0:
            best_fitness_history.append(best_fitness)
            plt.figure()
            plt.bar(np.arange(len(fitness)), fitness)
            plt.title('Population histogram after {} iterations'.format(i + 1))
            plt.xlabel('Chromosome Number')
            plt.ylabel('Fitness Value')
            plt.show()

        parents = selection(fitness, num_parents, population)
        offsprings = crossover(parents, num_offsprings)
        mutants = mutation(offsprings)

        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants

    print('Last population: \n{}\n'.format(population))
    print(population[0])
    fitness_last_gen = cal_fitness(weight, value, population, threshold)
    print('Fitness of the last population: \n{}\n'.format(fitness_last_gen))
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    parameters.append(population[max_fitness[0][0], :])
    return parameters, fitness_history, best_fitness_history, best_solution


def create_graph(solution):
    graph = nx.Graph()
    for i in range(len(solution) - 1):
        graph.add_edge(solution[i], solution[i + 1])
    graph.add_edge(solution[-1], solution[0])  # Connect last and first nodes to create a cycle
    return graph


K = 10  # Number of iterations between visualizations
parameters, fitness_history, best_fitness_history, best_solution = optimize(weight, value, initial_population, pop_size, num_generations, knapsack_threshold, K)

# Convert the best solution to a Hamiltonian cycle
hamiltonian_cycle = best_solution.nonzero()[0] + 1  # Convert indices to item numbers

# Plotting the Hamiltonian cycle
graph = create_graph(hamiltonian_cycle)
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='black', linewidths=1,
        font_size=10, font_color='black')
plt.title('Hamiltonian Cycle')
plt.show()

print('Optimal parameters for the given input data: \n{}'.format(parameters))
selected_items = item_number * parameters
print('\nSelected items maximizing the knapsack without tearing it apart:')
for i in range(selected_items.shape[1]):
    if selected_items[0][i] != 0:
        print('{}\n'.format(selected_items[0][i]))

fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
fitness_history_max = [np.max(fitness) for fitness in fitness_history]
plt.plot(list(range(num_generations)), fitness_history_mean, label='Mean Fitness')
plt.plot(list(range(num_generations)), fitness_history_max, label='Max Fitness')
plt.legend()
plt.title('Fitness over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()

# Plotting best fitness history
plt.plot(list(range(0, num_generations, K)), best_fitness_history, label='Best Fitness')
plt.title('Best Fitness over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.legend()
plt.show()
