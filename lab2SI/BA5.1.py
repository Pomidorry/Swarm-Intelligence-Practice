import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
dim=7
def bee_waggle_dance(solution, ngh, minVal, maxVal):
    # Perform a waggle dance with the solution, neighborhood radius, and min/max values.
    random_offset = (2 * ngh * np.random.rand(solution.size)) - ngh
    new_solution = solution + random_offset
    new_solution = np.clip(new_solution, minVal, maxVal)  # Ensure new_solution is within bounds
    return new_solution

def generate_random_solution(maxParameters, minVal, maxVal):
    return np.random.uniform(low=minVal, high=maxVal, size=maxParameters)

    
def f(x):
    g1=27 / (x[0] * x[1] ** 2 * x[2]) - 1>0
    g2=397.5 / (x[0] * x[1] ** 2 * x[2] ** 2) - 1>0
    g3=1.93 * x[3] ** 3 / (x[1] * x[2] * x[5] ** 4) - 1>0
    g4=1.93 / (x[1] * x[2] * x[6] ** 4) - 1>0
    g5=1.0 / (110 * x[5] ** 3) * np.sqrt((745.0 * x[3] / (x[1] * x[2])) ** 2 + 16.9 * 10 ** 6) - 1>0
    g6=1.0 / (85 * x[6] ** 3) * np.sqrt((745.0 * x[4] / (x[1] * x[2])) ** 2 + 157.5 * 10 ** 6) - 1>0
    g7=x[1] * x[2] / 40 - 1>0
    g8=1 * x[1] / x[0] - 1>0
    g9=x[0] / (12 * x[1]) - 1>0
    g10=(1.5 * x[5] + 1.9) / x[3] - 1>0
    g11=(1.1 * x[6] + 1.9) / x[4] - 1>0
    g12=x[0]<2.6
    g13=x[0]>3.6
    g14=x[1]<0.7
    g15=x[1]>0.8
    g16=np.round(x[2])<17
    g17=np.round(x[2])>28
    g18=x[3]<7.3
    g19=x[3]>8.3
    g20=x[4]<7.8
    g21=x[4]>8.3
    g22=x[5]<2.9
    g23=x[5]>3.9
    g24=x[6]<5.0
    g25=x[6]>5.5
    func=0.7854*x[0]*x[1]**2*(3.3333*np.round(x[2])**2 + 14.9334*np.round(x[2]) - 43.0934) - 1.508*x[0]*(x[5]**2 + x[6]**2) + 7.4777*(x[5]**3 + x[6]**3) + 0.7854*(x[3]*x[5]**2 + x[4]*x[6]**2)
    #func[g1 or g2 or g3 or g4 or g5 or g6 or g7 or g8 or g9 or g10 or g11]=10000
    if g1 or g2 or g3 or g4 or g5 or g6 or g7 or g8 or g9 or g10 or g11 or g12 or g13 or g14 or g15 or g16 or g17 or g18 or g19 or g20 or g21 or g22 or g23 or g24 or g25:
        func = 10000
    return func

population_show = np.zeros((30, dim + 1))
minVal = np.array([2.6, 0.7, 17, 7.3, 7.8, 2.9, 5.0])
maxVal = np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5])


for i in range(30):
        solution = generate_random_solution(dim, minVal, maxVal)
        population_show[i, 0:dim] = solution
        population_show[i, 2]=np.round(population_show[i, 2])
        population_show[i, dim] = f(solution)

def GBA(population_show):
    # Set the problem parameters
    maxIteration = 200
    maxParameters = 7
    minVal = np.array([2.6, 0.7, 17, 7.3, 7.8, 2.9, 5.0])
    maxVal = np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5])

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
                fit = f(solution)

                if fit < population_show[beeIndex, maxParameters]:
                    population_show[beeIndex, 0:maxParameters] = solution
                    population_show[beeIndex, 2]=np.round(population_show[beeIndex, 2])
                    population_show[beeIndex, maxParameters] = fit
    for j in range(int(group_random)):
        if beeIndex >= n:
            break
        solution = generate_random_solution(maxParameters, minVal, maxVal)
        fit = f(solution)
        population_show[beeIndex, 0:maxParameters] = solution
        population_show[beeIndex, 2]=np.round(population_show[beeIndex, 2])
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
    if dim == 1:
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
    elif dim == 2:
        ax = fig.add_subplot(111, projection='3d')
        X = np.arange(-10, 10, 0.01)
        Y = np.arange(-10, 10, 0.01)
        X, Y = np.meshgrid(X, Y)
        Z = evaluate_fitness([X, Y])
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.2)
        for individ0,individ1, val in zip(population_show[:,0],population_show[:,1], population_show[:,2]):
            ax.scatter(individ0,individ1, val, marker='*', edgecolors='red')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Fitness')
        ax.set_title('GA')
    elif dim >= 3:
        plt.plot(MinFitnessValues[int(max_iter * 0.1):], color='red')
        plt.ylabel('Min пристосованість')
        plt.ylabel('Покоління')
        plt.title('Залежність min')

    # population_show[:] = GBA(population_show)
    # print(population_show[:, 0], population_show[:, 1])
    population_show[:] = GBA(population_show)
    fitnessValues_show=list(map(f, population_show))
    minFitness = min(fitnessValues_show)
    MinFitnessValues.append(minFitness)

    print(f"Generation {generationCounter}: Min Fitness = {minFitness}")

    best_index = fitnessValues_show.index(min(fitnessValues_show))
    print("Best individual = ", *population_show[best_index, 0:7], "\n")
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




