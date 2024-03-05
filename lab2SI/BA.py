import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def bee_waggle_dance(solution, ngh, minVal, maxVal):
    # Perform a waggle dance with the solution, neighborhood radius, and min/max values.
    random_offset = (2 * ngh * np.random.rand(solution.size)) - ngh
    new_solution = solution + random_offset
    new_solution = np.clip(new_solution, minVal, maxVal)  # Ensure new_solution is within bounds
    return new_solution

def generate_random_solution(maxParameters, minVal, maxVal):
    return np.random.uniform(low=minVal, high=maxVal, size=maxParameters)

    
def evaluate_fitness(x):
    condition = np.array(x[0])**2 + np.array(x[1])**2 >= 2
    
    func_values = np.array((1-x[0])**2 + 100*(x[1]-x[0]**2)**2)
    func_values[condition] = 10000
    return func_values

population_show = np.zeros((30, 2 + 1))
maxParameters = 2
minVal = np.array([-1.25, -1.25])
maxVal = np.array([1.25, 1.25])


for i in range(30):
        solution = generate_random_solution(maxParameters, minVal, maxVal)
        population_show[i, 0:maxParameters] = solution
        population_show[i, maxParameters] = evaluate_fitness(solution)

def GBA(population_show):
    # Set the problem parameters
    maxIteration = 200
    maxParameters = 2
    minVal = np.array([-1.25, -1.25])
    maxVal = np.array([1.25, 1.25])

    # Set the grouped bees algorithm (GBA) parameters
    R_ngh = 1
    n = 30
    nGroups = 20

    # GBA's automatic parameter settings
    k = 3 * n / ((nGroups + 1) ** 3 - 1)
    groups = np.zeros(nGroups)
    recruited_bees = np.zeros(nGroups)
    a = (((maxVal - minVal) / 2) - R_ngh) / (nGroups ** 2 - 1)
    b = R_ngh - a

    for i in range(1, nGroups + 1):
        groups[i - 1] = np.floor(k * i ** 2)
        if groups[i - 1] == 0:
            groups[i - 1] = 1
        recruited_bees[i - 1] = (nGroups + 1 - i) ** 2
        ngh = a * i ** 2 + b

    group_random = n - np.sum(groups)
    group_random = max(group_random, 0)

    # Initialize the population matrix

    #sorted_population = population_show[population_show[:, maxParameters].argsort()]

    # Iterations of the grouped bees algorithm
    beeIndex = 0
    for g in range(nGroups):
        for j in range(int(groups[g])):
            beeIndex += 1

            for _ in range(int(recruited_bees[g])):
                solution = bee_waggle_dance(population_show[beeIndex, 0:maxParameters], ngh, minVal, maxVal)
                fit = evaluate_fitness(solution)

                if fit < population_show[beeIndex, maxParameters]:
                    population_show[beeIndex, 0:maxParameters] = solution
                    population_show[beeIndex, maxParameters] = fit
    for j in range(int(group_random)):
        if beeIndex >= n:
            break
        solution = generate_random_solution(maxParameters, minVal, maxVal)
        fit = evaluate_fitness(solution)
        population_show[beeIndex, 0:maxParameters] = solution
        population_show[beeIndex, maxParameters] = fit
        beeIndex += 1

    population_show = population_show[population_show[:, maxParameters].argsort()]

    return population_show

MinFitnessValues = []
generationCounter=0

def animate(i):
    global generationCounter, dim, particles_show, fitnessValues_show

    max_iter=30 

    if generationCounter >= max_iter:
        return
    
    plt.clf()
    if maxParameters == 1:
        ax = fig.add_subplot(111)
        X = np.arange(0, 3, 0.01)
        Y = evaluate_fitness([X])
        ax.plot(X, Y, color='blue', label='Function Curve')
        for individ, val in zip(population_show[:, 0], population_show[:, 1]):
            ax.scatter(individ, val, marker='*', edgecolors='red')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('GA')
        plt.legend()
    elif maxParameters == 2:
        ax = fig.add_subplot(111, projection='3d')
        X = np.arange(-1.25, 1.25, 0.01)
        Y = np.arange(-1.25, 1.25, 0.01)
        X, Y = np.meshgrid(X, Y)
        Z = evaluate_fitness([X, Y])
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.2)
        for individ0,individ1, val in zip(population_show[:,0],population_show[:,1], population_show[:,2]):
            ax.scatter(individ0,individ1, val, marker='*', edgecolors='red')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Fitness')
        ax.set_title('GA')
    elif maxParameters >= 3:
        plt.plot(MinFitnessValues[int(max_iter * 0.1):], color='red')
        plt.ylabel('Min пристосованість')
        plt.ylabel('Покоління')
        plt.title('Залежність min')

    # population_show[:] = GBA(population_show)
    # print(population_show[:, 0], population_show[:, 1])
    population_show[:] = GBA(population_show)
    minFitness = np.min(population_show[:, 2])
    min_index = np.argmin(population_show[:, 2])
    print("MIN: ",population_show[min_index])
    # MinFitnessValues.append(minFitness)

    generationCounter += 1


fig = plt.figure()

ani = animation.FuncAnimation(fig, animate, interval=900)

plt.show()

# population_show[:] = GBA(population_show)
# print(population_show[:, 0], population_show[:, 1])
# minFitness = min(population_show[:, 1])
# print(population_show)   
# print(population_show[:, 0])     




