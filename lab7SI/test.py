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
    Data = pd.read_excel("E:\\SI\\lab7SI\\TestRegData.xlsx", sheet_name="Var2")
    
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
print(len(x_f[0]), len(y_f))
# print("Масив x:")
# print(x_array)

# print("\n Масив y:")
# print(y_array)

# define objective function
def obj(a):
    gamma=0.01
    return sum([(y_f[i] - (a[0]-sum([(a[j]*x_f[i][j]) for j  in range(len(x_f[0])) if j != 0])))**2 for i in range(len(y_f)) if i != 0]) + gamma*sum([(a[i])**2 for i in range(len(x_f[0]))])

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


def differential_evolution(pop_size, bounds, iter, F, cr, k):
    # initialise population of candidate solutions randomly within the specified bounds
    pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
    #pop= [pop1[i][0] for i in range(len(pop1))]
    #print(pop)
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
    return best_vector, best_obj


pop_size = 40
# define lower and upper bounds for every dimension
bounds = asarray([(-1000, 1000)]*9)
# define number of iterations
iter = 1000
# define scale factor for mutation
F = 0.5
# define crossover rate for recombination
cr = 0.7
# define the frequency of updating plots
k = 10  # кожні k ітерацій оновлюємо графіки
 
# perform differential evolution
solution = differential_evolution(pop_size, bounds, iter, F, cr, k)


