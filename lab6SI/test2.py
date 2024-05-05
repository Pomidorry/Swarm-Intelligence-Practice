# differential evolution search of the two-dimensional sphere objective function
from numpy.random import rand
from numpy.random import choice
from numpy import asarray
from numpy import clip
from numpy import argmin
from numpy import min
from numpy import around
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf

def load_data():
    # Завантаження даних з xlsx-файлу
    global Data, xTrain, yTrain, xTest, yTest, x, y
    Data = pd.read_excel("E:\\SI\\lab6SI\\DataRegression.xlsx", sheet_name="Var10")
    
    # Перемішування рядків з даними, що отримані з xlsx-файлу
    Data = Data.sample(frac=1).reset_index(drop=True)
    
    # Розділення на ознаки (x) і цільову змінну (y)
    x = Data.iloc[:, 1]  # Ознака - другий стовпець
    y = Data.iloc[:, 0]  # Цільова змінна - перший стовпець
    
    # Розділення на навчальну та тестову вибірки
    TrainPercent = 75
    LenDataTrain = round(len(Data) * TrainPercent / 100)
    xTrain, xTest = x[:LenDataTrain], x[LenDataTrain:]
    yTrain, yTest = y[:LenDataTrain], y[LenDataTrain:]

load_data()

y_f=y.tolist()
x_f=x.tolist()
# define objective function
def obj(b):
    b1, b2 = b
    return sum([(y_f[i] - (b1*(1-1/np.sqrt(1+2*b2*x_f[i]))))**2 for i in range(len(x_f))])

# define mutation operation
def mutation(x, F):
    return x[0] + F * (x[1] - x[2])


# define boundary check operation
def check_bounds(mutated, bounds):
    mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return mutated_bound


# define crossover operation
def crossover(mutated, target, dims, cr):
    # generate a uniform random value for every dimension
    p = rand(dims)
    # generate trial vector by binomial crossover
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial

def model(b, x):
    # Кількість точок x та кількість параметрів b
    N = len(x)
    M = len(b)

    # Ініціалізуємо матрицю Y з нулями
    Y = np.zeros((N, M))
    print(b)
    # Обчислюємо Y для кожного параметра b
    Y = b[0]*(1-1/np.sqrt(1+2*b[1]*x))

    return Y

def mse_loss(y_true, y_pred):
  """Calculates Mean Squared Error"""
  return (((np.array(y_true) - np.array(y_pred)) ** 2))/len(y_true)

def func(x, b1, b2):
    return b1*(1-1/np.sqrt(1+2*b2*x))

def differential_evolution(pop_size, bounds, iter, F, cr, k):
    # initialise population of candidate solutions randomly within the specified bounds
    pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
    # evaluate initial population of candidate solutions
    obj_all = [obj(ind) for ind in pop]
    # find the best performing vector of initial population
    best_vector = pop[argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj
    # run iterations of the algorithm
    for i in range(iter):
        # iterate over all candidate solutions
        for j in range(pop_size):
            # choose three candidates, a, b and c, that are not the current one
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[choice(candidates, 3, replace=False)]
            # perform mutation
            mutated = mutation([a, b, c], F)
            # check that lower and upper bounds are retained after mutation
            mutated = check_bounds(mutated, bounds)
            # perform crossover
            trial = crossover(mutated, pop[j], len(bounds), cr)
            # compute objective function value for target vector
            obj_target = obj(pop[j])
            # compute objective function value for trial vector
            obj_trial = obj(trial)
            # perform selection
            if obj_trial < obj_target:
                # replace the target vector with the trial vector
                pop[j] = trial
                # store the new objective function value
                obj_all[j] = obj_trial
        # find the best performing vector at each iteration
        best_obj = min(obj_all)
        # store the lowest objective function value
        if best_obj < prev_obj:
            best_vector = pop[argmin(obj_all)]
            prev_obj = best_obj
            # report progress at each iteration
        print('Iteration: %d f([%s]) = %.5f' % (i, around(best_vector, decimals=5), best_obj))
        if i % k == 0:  # Додано умову для відображення кожні k ітерацій
            all_b = best_vector
            b_list = all_b.tolist()
            b1, b2 = b_list[0], b_list[1]

            x_values = np.linspace(0, max(x_f), 14)
            y_values_func = func(x_values, b1, b2)

            plt.scatter(x_f, y_f, label='Data')
            plt.plot(x_values, y_values_func, color='red', label='Function')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Regression plot')
            plt.legend()
            plt.grid(True)
            plt.text(0.05, 0.95, f'Iteration: {i}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
            plt.show()

            x_values_loss = np.linspace(0, max(x_f), 14)
            y_values_func_loss = func(x_values_loss, b1, b2)

            mse_losses = mse_loss(y_values_func_loss.tolist(), y_f)
            plt.plot(y_values_func_loss, mse_losses, label='MSE')
            plt.xlabel('Predicted Value (y_pred)')
            plt.ylabel('Loss')
            plt.text(0.05, 0.95, f'Iteration: {i}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
            plt.show()

            YModelTrain = model(b_list, xTrain)

            # Получение предсказанных значений для тестового набора данных
            YModelTest = model(b_list, xTest)

            plt.plot(YModelTrain, yTrain, '.r', markersize=15)
            plt.plot(np.arange(min(yTrain), max(yTrain), 0.03), np.arange(min(yTrain), max(yTrain), 0.03))
            plt.xlabel('YModelTrain')
            plt.ylabel('yTrain (Real Data)')
            plt.grid(True)
            #print(YModelTrain)
            plt.axis([min(YModelTrain), max(YModelTrain), min(yTrain), max(yTrain)])
            plt.text(0.05, 0.95, f'Iteration: {i}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
            plt.show()
            # График для тестовых данных

            plt.plot(YModelTest, yTest, '.', markersize=15)
            plt.plot(np.arange(min(yTest), max(yTest), 0.03), np.arange(min(yTest), max(yTest), 0.03))
            plt.xlabel('YModelTest')
            plt.ylabel('yTest (Real Data)')
            plt.grid(True)
            plt.axis([min(YModelTest), max(YModelTest), min(yTest), max(yTest)])
            plt.text(0.05, 0.95, f'Iteration: {i}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
            plt.show()
    return best_vector, best_obj

def pso(cost_func, k=20, dim=3, num_particles=400, max_iter=80, w=0.5, c1=1, c2=2):
    # Initialize particles and velocities
    #particles = np.random.uniform(-5.12, 5.12, (num_particles, dim))
    dim_ranges = asarray([(-10000, -1000), (1, 100), (0,1)])  # Default ranges for 2 dimensions
    particles = dim_ranges[:, 0] + (rand(num_particles, len(dim_ranges)) * (dim_ranges[:, 1] - dim_ranges[:, 0]))
    velocities = np.zeros((num_particles, dim))
    #velocities = np.zeros((num_particles, dim))

    # Initialize the best positions and fitness values
    best_positions = np.copy(particles)
    best_fitness = np.array([cost_func(p) for p in particles])
    swarm_best_position = best_positions[np.argmin(best_fitness)]
    swarm_best_fitness = np.min(best_fitness)
    min_values = np.array([-10000, 1, 0])
    max_values = np.array([-1000, 100, 1])
    # Iterate through the specified number of iterations, updating the velocity and position of each particle at each iteration
    for i in range(max_iter):
        # Update velocities
        r1 = np.random.uniform(0, 1, (num_particles, dim))
        r2 = np.random.uniform(0, 1, (num_particles, dim))
        velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)

        # Update positions
        particles += velocities
        for d in range(dim):
            particles[:, d] = np.clip(particles[:, d], min_values[d], max_values[d])
        # Evaluate fitness of each particle
        fitness_values = np.array([cost_func(p) for p in particles])

        # Update best positions and fitness values
        improved_indices = np.where(fitness_values < best_fitness)
        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)]
            swarm_best_fitness = np.min(fitness_values)

        print(swarm_best_position, swarm_best_fitness)
        if i % k == 0:  # Додано умову для відображення кожні k ітерацій
                all_b = swarm_best_position
                b_list = all_b.tolist()
                b1, b2, b3 = b_list[0], b_list[1], b_list[2]

                x_values = np.linspace(0, max(x_f), 154)
                y_values_func = func(x_values, b1, b2, b3)

                plt.scatter(x_f, y_f, label='Data')
                plt.plot(x_values, y_values_func, color='red', label='Function')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Regression plot')
                plt.legend()
                plt.grid(True)
                plt.text(0.05, 0.95, f'Iteration: {i}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
                plt.show()

                x_values_loss = np.linspace(0, max(x_f), 154)
                y_values_func_loss = func(x_values_loss, b1, b2, b3)

                mse_losses = mse_loss(y_values_func_loss.tolist(), y_f)
                plt.plot(y_values_func_loss, mse_losses, label='MSE')
                plt.xlabel('Predicted Value (y_pred)')
                plt.ylabel('Loss')
                plt.text(0.05, 0.95, f'Iteration: {i}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
                plt.show()

                YModelTrain = model(b_list, xTrain)

                # Получение предсказанных значений для тестового набора данных
                YModelTest = model(b_list, xTest)

                plt.plot(YModelTrain, yTrain, '.r', markersize=15)
                plt.plot(np.arange(min(yTrain), max(yTrain), 0.03), np.arange(min(yTrain), max(yTrain), 0.03))
                plt.xlabel('YModelTrain')
                plt.ylabel('yTrain (Real Data)')
                plt.grid(True)
                #print(YModelTrain)
                plt.axis([min(YModelTrain), max(YModelTrain), min(yTrain), max(yTrain)])
                plt.text(0.05, 0.95, f'Iteration: {i}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
                plt.show()
                # График для тестовых данных

                plt.plot(YModelTest, yTest, '.', markersize=15)
                plt.plot(np.arange(min(yTest), max(yTest), 0.01), np.arange(min(yTest), max(yTest), 0.01))
                plt.xlabel('YModelTest')
                plt.ylabel('yTest (Real Data)')
                plt.grid(True)
                plt.axis([min(YModelTest), max(YModelTest), min(yTest), max(yTest)])
                plt.text(0.05, 0.95, f'Iteration: {i}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
                plt.show()
    # Return the best solution found by the PSO algorithm
    return swarm_best_position, swarm_best_fitness


pop_size = 950
# define lower and upper bounds for every dimension
bounds = asarray([[100, 1000], [0,1]])
# define number of iterations
iter = 101
# define scale factor for mutation
F = 0.5
# define crossover rate for recombination
cr = 0.7
# define the frequency of updating plots
k = 20  # кожні k ітерацій оновлюємо графіки
 
# perform differential evolution
solution = differential_evolution(pop_size, bounds, iter, F, cr, k)
print(solution)

# solution1=pso(obj)
# print(solution1)