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
    Data = pd.read_excel("E:\\SI\\lab7SI\\TestRegData.xlsx", sheet_name="Var7")
    
    # Перемішування рядків з даними, що отримані з xlsx-файлу
    Data = Data.sample(frac=1).reset_index(drop=True)
    
    # Розділення на ознаки (x) і цільову змінну (y)
    x = Data.iloc[:, :-1]  
    y = Data.iloc[:, -1]  # Цільова змінна - перший стовпець
    
    # Розділення на навчальну та тестову вибірки
    TrainPercent = 75
    LenDataTrain = round(len(Data) * TrainPercent / 100)
    xTrain, xTest = x[:LenDataTrain], x[LenDataTrain:]
    yTrain, yTest = y[:LenDataTrain], y[LenDataTrain:]

load_data()
x_f = x.values

y_f = y.values
one_col=np.array([1]*len(y_f))
x_f1=np.insert(x_f, 0, one_col, axis=1)
#print(len(x_f[0]), len(y_f))
# print("Масив x:")
# print(x_array)

# print("\n Масив y:")
# print(y_array)
# define objective function

def model(a, xm):
    # Кількість точок x та кількість параметрів b
    N = len(xm)
    M = len(a)
    #one_col=np.array([1]*len(x))
    x1=np.insert(x, 0, one_col, axis=1)
    # Ініціалізуємо матрицю Y з нулями
    Y = []
    #print(xm[0], N, M)
    temp=0
    for i in range(len(xm)):
        for j in range(len(xm[0])):
                if j!=0:
                    temp+=a[j]*x1[i][j]
        temp+=a[0]            
        Y.append(temp)
        temp=0    
    #print(Y)
    return Y

def objL2(a):
    gamma=0.000001
    return sum([(y_f[i] - (a[0]+sum([(a[j]*x_f1[i][j]) for j  in range(len(x_f1[0])) if j!=0])))**2 for i in range(len(y_f)) ]) + gamma*sum([(a[i])**2 for i in range(len(x_f[0]))])

def objL1(a):
    gamma=0.01
    return sum([(y_f[i] - (a[0]+sum([(a[j]*x_f1[i][j]) for j  in range(len(x_f1[0])) if j!=0])))**2 for i in range(len(y_f)) ]) + gamma*sum([abs(a[i]) for i in range(len(x_f[0]))])

def elastic_net(a):
    gamma1=0.01
    gamma2=0.005
    return sum([(y_f[i] - (a[0]+sum([(a[j]*x_f1[i][j]) for j  in range(len(x_f1[0])) if j!=0])))**2 for i in range(len(y_f)) ]) + gamma1*sum([abs(a[i]) for i in range(len(x_f[0]))])+gamma2*sum([(a[i])**2 for i in range(len(x_f[0]))])

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


def differential_evolution(pop_size, bounds, iter, F, cr, k=50):
    # initialise population of candidate solutions randomly within the specified bounds
    pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
    #pop= [pop1[i][0] for i in range(len(pop1))]
    #print(pop)
    # evaluate initial population of candidate solutions
    obj_all = [objL2(ind) for ind in pop]
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
            obj_target = objL2(pop[j])
            # compute objective function value for trial vector
            obj_trial = objL2(trial)
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
        if i % k == 0:
            plt.figure(figsize=(8,8))
            plt.subplot(2,1,1)
            YModelTrain = model(best_vector, xTrain.values)
            plt.plot(YModelTrain, yTrain, '.r', markersize=15)
            plt.plot(np.arange(min(yTrain), max(yTrain), 0.03), np.arange(min(yTrain), max(yTrain), 0.03))
            plt.grid(); plt.tight_layout(); plt.legend()
            plt.xlabel('Measured'); plt.ylabel('Predicted')

            plt.subplot(2,1,2)
            #plt.plot([-400,400],[-400,400],'b--')
            YModelTest = model(best_vector, xTest.values)
            plt.plot(YModelTest, yTest, '.r', markersize=15)
            plt.plot(np.arange(min(yTest), max(yTest), 0.03), np.arange(min(yTest), max(yTest), 0.03))
            plt.grid(); plt.tight_layout(); plt.legend()
            plt.xlabel('Measured'); plt.ylabel('Predicted')
            plt.savefig('parity.png',dpi=300)
            plt.show()    
    return best_vector, best_obj

def pso(cost_func, k=10, dim=16, num_particles=230, max_iter=81, w=0.5, c1=1, c2=2):
    # Initialize particles and velocities
    #particles = np.random.uniform(-5.12, 5.12, (num_particles, dim))
    dim_ranges = asarray([(-10, 10)]*16) # Default ranges for 2 dimensions
    particles = dim_ranges[:, 0] + (rand(num_particles, len(dim_ranges)) * (dim_ranges[:, 1] - dim_ranges[:, 0]))
    velocities = np.zeros((num_particles, dim))
    #velocities = np.zeros((num_particles, dim))

    # Initialize the best positions and fitness values
    best_positions = np.copy(particles)
    best_fitness = np.array([cost_func(p) for p in particles])
    swarm_best_position = best_positions[np.argmin(best_fitness)]
    swarm_best_fitness = np.min(best_fitness)
    # min_values = np.array([0, 0])
    # max_values = np.array([100, 1])
    # Iterate through the specified number of iterations, updating the velocity and position of each particle at each iteration
    for i in range(max_iter):
        # Update velocities
        r1 = np.random.uniform(0, 1, (num_particles, dim))
        r2 = np.random.uniform(0, 1, (num_particles, dim))
        velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)

        # Update positions
        particles += velocities
        # for d in range(dim):
        #     particles[:, d] = np.clip(particles[:, d], min_values[d], max_values[d])
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
        if i % k == 0:
            plt.figure(figsize=(8,8))
            plt.subplot(2,1,1)
            YModelTrain = model(swarm_best_position, xTrain.values)
            plt.plot(YModelTrain, yTrain, '.r', markersize=15)
            plt.plot(np.arange(min(yTrain), max(yTrain), 0.03), np.arange(min(yTrain), max(yTrain), 0.03))
            plt.grid(); plt.tight_layout(); plt.legend()
            plt.xlabel('Measured'); plt.ylabel('Predicted')

            plt.subplot(2,1,2)
            #plt.plot([-400,400],[-400,400],'b--')
            YModelTest = model(swarm_best_position, xTest.values)
            plt.plot(YModelTest, yTest, '.r', markersize=15)
            plt.plot(np.arange(min(yTest), max(yTest), 0.03), np.arange(min(yTest), max(yTest), 0.03))
            plt.grid(); plt.tight_layout(); plt.legend()
            plt.xlabel('Measured'); plt.ylabel('Predicted')
            plt.savefig('parity.png',dpi=300)
            plt.show()

    # Return the best solution found by the PSO algorithm
    return swarm_best_position, swarm_best_fitness

pop_size = 70
# define lower and upper bounds for every dimension
bounds = asarray([(-10, 10)]*16)
# define number of iterations
iter = 401
# define scale factor for mutation
F = 0.4
# define crossover rate for recombination
cr = 0.6
# define the frequency of updating plots
k = 10  # кожні k ітерацій оновлюємо графіки
 
# perform differential evolution

solution = differential_evolution(pop_size, bounds, iter, F, cr, k)
# print(solution)

# solution1=pso(objL2)
# print(solution1)

one_col=np.array([1]*len(y_f))
x_f1=np.insert(x_f, 0, one_col, axis=1)
# print(x_f1)
# x_f_t=np.transpose(x_f1)
# print(x_f_t)
# I_m=np.eye(len(x_f[0])+1)
# print(I_m)
# gamma=0.0001
# K=((x_f_t.dot(x_f1)+gamma*I_m)**(-1))
# print(K)
# P=K.dot(x_f_t)
# print(P)
# C=P.dot(y_f)

# print(objL2(C))

def ridge_regression(X, Y, gamma):
    # Рассчитываем X^T*X
    XTX = np.dot(X.T, X)
    
    # Добавляем регуляризацию
    XTX += gamma * np.eye(XTX.shape[0])
    
    # Вычисляем обратную матрицу (X^T*X + gamma*I)^-1
    inverse_XTX = np.linalg.inv(XTX)
    
    # Рассчитываем X^T*Y
    XTY = np.dot(X.T, Y)
    
    # Вычисляем коэффициенты a
    a = np.dot(inverse_XTX, XTY)
    
    return a


# k=ridge_regression(x_f1, y_f, 0.01)
# print(k)
# print(objL2(k))




