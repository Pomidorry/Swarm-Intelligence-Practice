import matplotlib.pyplot as plt
import numpy as np
import random

n_population = 200
crossover_per = 0.8
mutation_per = 0.2
n_generations = 800

def generate_random_points_on_circle(radius, center, num_points):
    """
    Generate random points on the perimeter of a circle.
    Input:
    1. radius: radius of the circle
    2. center: center coordinates of the circle (x, y)
    3. num_points: number of points to generate
    Output:
    List of random points on the perimeter of the circle
    """
    points = []
    for _ in range(num_points):
        theta = random.uniform(0, 2*np.pi)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        points.append((x, y))
    return points

def generate_random_points_within_circle(radius, center, num_points):
    """
    Generate random points within a circle.
    Input:
    1. radius: radius of the circle
    2. center: center coordinates of the circle (x, y)
    3. num_points: number of points to generate
    Output:
    List of random points within the circle
    """
    points = []
    for _ in range(num_points):
        theta = random.uniform(0, 2*np.pi)
        r = radius * np.sqrt(random.uniform(0, 1))
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        points.append((x, y))
    return points

def dist_two_points(point_1, point_2):
    """
    Calculate the Euclidean distance between two points.
    Input:
    1. point_1: coordinates of the first point (x, y)
    2. point_2: coordinates of the second point (x, y)
    Output:
    Euclidean distance between the two points
    """
    return np.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)

def initial_population(points_list, n_population=250):
    """
    Generate initial population of points randomly selected from the given list.
    Input:
    1. points_list: List of points
    2. n_population: Number of individuals in the population
    Output:
    Generated lists of points
    """
    population = []
    for _ in range(n_population):
        population.append(random.sample(points_list, len(points_list)))
    return population

def total_dist_individual(individual):
    """
    Calculate the total distance traveled by an individual in the TSP.
    Input:
    1. Individual: List of points representing the order of cities
    Output:
    Total distance traveled by the individual
    """
    total_dist = 0
    for i in range(len(individual)):
        total_dist += dist_two_points(individual[i], individual[(i+1)%len(individual)])
    return total_dist

def fitness_prob(population):
    """
    Calculate the fitness probability of the population for selection.
    Input:
    1. Population: List of individuals (lists of points)
    Output:
    Population fitness probabilities
    """
    total_dist_all_individuals = [total_dist_individual(individual) for individual in population]
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - np.array(total_dist_all_individuals)
    population_fitness_sum = sum(population_fitness)
    population_fitness_probs = population_fitness / population_fitness_sum
    return population_fitness_probs

def roulette_wheel(population, fitness_probs):
    """
    Perform selection using the roulette wheel method.
    Input:
    1. Population: List of individuals (lists of points)
    2. Fitness_probs: Fitness probabilities of the population
    Output:
    Selected individual
    """
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    selected_individual_index = np.searchsorted(population_fitness_probs_cumsum, np.random.rand())
    return population[selected_individual_index]

def crossover(parent_1, parent_2):
    """
    Implement mating strategy using simple crossover between two parents.
    Input:
    1. parent_1: First parent (list of points)
    2. parent_2: Second parent (list of points)
    Output:
    Offspring after crossover (two lists of points)
    """
    cut = random.randint(0, len(parent_1)-1)
    offspring_1 = parent_1[0:cut]
    offspring_1 += [city for city in parent_2 if city not in offspring_1]

    offspring_2 = parent_2[0:cut]
    offspring_2 += [city for city in parent_1 if city not in offspring_2]

    return offspring_1, offspring_2

def mutation(offspring):
    """
    Implement mutation strategy in a single offspring.
    Input:
    1. offspring: Individual to undergo mutation (list of points)
    Output:
    Mutated offspring individual (list of points)
    """
    index_1, index_2 = random.sample(range(len(offspring)), 2)
    offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
    return offspring

def run_ga(points_list, n_population, n_generations, crossover_per, mutation_per):
    """
    Run the Genetic Algorithm for TSP.
    Input:
    1. points_list: List of points representing cities
    2. n_population: Number of individuals in the population
    3. n_generations: Number of generations
    4. crossover_per: Crossover probability
    5. mutation_per: Mutation probability
    Output:
    Best individual found after the specified number of generations
    """
    population = initial_population(points_list, n_population)
    best_individual = None
    best_distance = float('inf')

    for generation in range(n_generations):
        fitness_probs = fitness_prob(population)
        parents_list = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]
        offspring_list = []

        for i in range(0, len(parents_list), 2):
            offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i+1])
            if random.random() < mutation_per:
                offspring_1 = mutation(offspring_1)
            if random.random() < mutation_per:
                offspring_2 = mutation(offspring_2)
            offspring_list.extend([offspring_1, offspring_2])

        mixed_offspring = parents_list + offspring_list
        population = random.sample(mixed_offspring, n_population)

        # Find the best individual in the current generation
        for individual in population:
            distance = total_dist_individual(individual)
            if distance < best_distance:
                best_distance = distance
                best_individual = individual

    return best_individual

# Circle parameters
radius = 10
center = (5, 5)
num_points = 30

# Generate random points within the circle
random_points = generate_random_points_on_circle(radius, center, num_points)

# Run GA to solve TSP
best_path = run_ga(random_points, n_population, n_generations, crossover_per, mutation_per)

# Plot the circle and best path
fig, ax = plt.subplots(figsize=(10, 10))
circle = plt.Circle(center, radius, color='blue', fill=False)
ax.add_artist(circle)

# Plot all possible paths as thin gray lines
# for i in range(len(random_points)):
#     for j in range(i+1, len(random_points)):
#         x_values = [random_points[i][0], random_points[j][0]]
#         y_values = [random_points[i][1], random_points[j][1]]
#         ax.plot(x_values, y_values, color='gray', alpha=0.3)

x_best = [point[0] for point in best_path]
y_best = [point[1] for point in best_path]

ax.plot(x_best, y_best, label='Best Route', c='r', linewidth=1)
ax.plot(x_best+[x_best[0]], y_best+[y_best[0]], c='r', linewidth=1)  # Connect last point to the first

plt.scatter([point[0] for point in random_points], [point[1] for point in random_points], c='r', marker='o')
# for i, txt in enumerate(best_path):
#     ax.annotate(str(i+1), (x_best[i], y_best[i]), fontsize=8)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('TSP Solution Using GA')
plt.legend()

# Adjust the axis limits to ensure the entire circle is visible
ax.set_xlim((center[0] - radius - 5), center[0] + radius + 5)
ax.set_ylim(center[1] - radius - 5, center[1] + radius + 5)

# Set aspect ratio to 'equal' to ensure the circle is circular
ax.set_aspect('equal')

plt.grid(True)
plt.show()
