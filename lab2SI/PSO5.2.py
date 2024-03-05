import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dim = 3
# Define the Rastrigin function
# def rastrigin(x):
#     return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0]-np.pi) ** 2 -(x[1]-np.pi) ** 2)

# def harmonic(x):
#     return (x[0]**3) * ((3-x[0])**5) * np.sin(10 * np.pi * x[0])

def rastrigin(x):
    # Суммируем только значения, для которых выполнено условие
    return sum([2*xi*xi for xi in x])

# def constraint(x):
#    if(x[0]**2 - x[1]**2-(1+0.2*np.cos(8*np.arctan(x[0]/x[1])))**2>0):
#     return x[0]**2 - x[1]**2-(1+0.2*np.cos(8*np.arctan(x[0]/x[1])))**2
#    else:
#     return None 

def f(x):
    func=(x[2] + 2) * x[1] * (x[0] ** 2)
    g1 = 1 - (((x[1] ** 3) * x[2]) / (7.178 * (x[0] ** 4))) <=0
    g2 = ((4 * (x[1] ** 2) - x[0] * x[1]) / (12.566 * (x[1] * (x[0] ** 3)) - x[0] ** 4)) + (1 / (5.108 * (x[0] ** 2))) - 1 <= 0
    g3 = 1 - ((140.45 * x[0]) / ((x[1] ** 2) * x[2])) <= 0
    g4 = ((x[0] + x[1]) / 1.5) - 1<=0
    g5 = x[0] >= 0.005
    g6 = x[0] <= 2.0
    g7 = x[1] >= 0.25
    g8 = x[1] <= 1.3
    g9 = x[2] >= 2.0
    g10=x[2] <= 15.0
    if g1 and g2 and g3 and g4 and g5 and g6 and g7 and g8 and g9 and g10: 
        return func  # Penalty for violating constraints
    else:
        return 10000

# lower_boundaries=[2.6, 0.7, 17, 7.3, 7.8, 2.9, 5.0]
# upper_boundaries=[3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]
# particles_show=np.random.uniform(lower_boundaries, upper_boundaries)   
# print(particles_show)
# lower_boundaries = np.random.uniform(2.6, 0.7, 17, 7.3, 7.8, 2.9, 5.0, (250, dim))

# # Generate random values for upper boundaries
# upper_boundaries = np.random.uniform(3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5, (250, dim))

# Create random values between lower_boundaries and upper_boundaries
ranges=[(0.005, 2.0), (0.25, 1.3), (2, 15)]
#particles_show = np.random.uniform(lower_boundaries, upper_boundaries)
particles_show = np.random.uniform([r[0] for r in ranges], [r[1] for r in ranges], size=(250, 3))
particles_show[:, 2] = np.round(particles_show[:, 2])
#print(particles_show)
#particles_show = np.random.uniform(-5.12, 5.12, (250, dim))
fitnessValues_show = list(map(f, particles_show))

# Define the PSO algorithm
# def pso(cost_func, particles, dim=2, num_particles=30, max_iter=100, w=0.5, c1=1, c2=2):
#     # Initialize particles and velocities
  

#     # Return the best solution found by the PSO algorithm
#     return swarm_best_position, swarm_best_fitness, particles


# dim = 2

# solution, fitness = pso(rastrigin, dim=dim)

# print('Solution:', solution)
# print('Fitness:', fitness)

# x = np.linspace(-5.12, 5.12, 100)
# y = np.linspace(-5.12, 5.12, 100)
# X, Y = np.meshgrid(x, y)
# Z = rastrigin([X, Y])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# ax.scatter(solution[0], solution[1], fitness, color='red')
# plt.show()

def pso(particles_show, fitnessValues_show, num_particles=250, w=0.5, c1=1, c2=2):
    particles = particles_show
    velocities = np.zeros((num_particles, dim))

    # Initialize the best positions and fitness values
    best_positions = np.copy(particles)
    best_fitness = np.array([f(p) for p in particles])
    swarm_best_position = best_positions[np.argmin(best_fitness)]
    swarm_best_fitness = np.min(best_fitness)

    # Iterate through the specified number of iterations, updating the velocity and position of each particle at each iteration
    print(generationCounter)
    # Update velocities
    r1 = np.random.uniform(0, 1, (num_particles, dim))
    r2 = np.random.uniform(0, 1, (num_particles, dim))
    velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)

        # Update positions
    particles += velocities
    particles[:, 2] = np.round(particles[:, 2])
        # Evaluate fitness of each particle
    fitness_values = np.array([f(p) for p in particles])
    particles_show[:]=particles
    fitnessValues_show[:] = fitness_values
        # Update best positions and fitness values
    improved_indices = np.where(fitness_values < best_fitness)
    best_positions[improved_indices] = particles[improved_indices]
    best_fitness[improved_indices] = fitness_values[improved_indices]
    if np.min(fitness_values) < swarm_best_fitness:
        swarm_best_position = particles[np.argmin(fitness_values)]
        swarm_best_fitness = np.min(fitness_values)    
    print('Solution:', swarm_best_position)
    print('Fitness:', swarm_best_fitness)
    return particles_show, fitnessValues_show

MinFitnessValues = []

generationCounter=0

def animate(i):
    global generationCounter, dim, particles_show, fitnessValues_show

    max_iter=30 

    if generationCounter >= max_iter:
        return
    
    plt.clf()
    # if dim == 1:
    #     ax = fig.add_subplot(111)
    #     X = np.arange(-5.12, 5.12, 0.01)
    #     Y = rastrigin([X])
    #     ax.plot(X, Y, color='blue', label='Function Curve')
    #     for individ, val in zip(particles_show, fitnessValues_show):
    #         ax.scatter(individ[0], val, marker='*', edgecolors='red')
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_title('PSO')
    #     plt.legend()
    # elif dim == 2:
    #     ax = fig.add_subplot(111, projection='3d')
    #     X = np.arange(-5.12, 5.12, 0.01)
    #     Y = np.arange(-5.12, 5.12, 0.01)
    #     X, Y = np.meshgrid(X, Y)
    #     Z = f([X, Y])
    #     ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.2)
    #     for individ, val in zip(particles_show, fitnessValues_show):
    #         ax.scatter(individ[0], individ[1], val, marker='*', edgecolors='red')
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Fitness')
    #     ax.set_title('PSO')
    if dim >= 3:
        plt.plot(MinFitnessValues[int(max_iter * 0.1):], color='red')
        plt.ylabel('Min пристосованість')
        plt.ylabel('Покоління')
        plt.title('Залежність min')
    temp=[]
    particles_show[:], temp[:] = pso(particles_show, fitnessValues_show)
    #fitnessValues_show[:] = list(map(constraint, particles_show))
    #fitnessValues_show[:] = list(map(rastrigin, particles_show))
    minFitness = min(fitnessValues_show)
    MinFitnessValues.append(minFitness)

    generationCounter += 1

          
   
fig = plt.figure()

ani = animation.FuncAnimation(fig, animate, interval=400)

plt.show()
